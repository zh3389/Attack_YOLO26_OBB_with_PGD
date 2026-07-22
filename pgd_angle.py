import argparse
import os
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


# ==========================================
# 1. 自定义 YOLO OBB 旋转角度 PGD 攻击器 (核心)
# ==========================================
class YOLOBB_AnglePGD:
    """
    针对 Ultralytics YOLO OBB 旋转角度输出的 PGD 攻击。
    参考 torchattacks.PGD 文档: eps=8/255, alpha=2/255, steps=10, random_start=True
    攻击目标：最大化 top-k 高置信度 anchor 的角度绝对值，使旋转框预测角度发生偏移。

    与漏检攻击 / 分类攻击的区别：
    - 漏检攻击：压低所有 anchor 的类别置信度 → 隐藏目标
    - 分类攻击：让 anchor 的预测类别偏离正确类别 → 检测到但分错类
    - 角度攻击：偏转 anchor 的旋转角度 → 旋转框角度错误，影响定位精度
    """
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=True, topk=50):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.topk = topk  # 只攻击 top-k 高置信度 anchor 的角度

    def __call__(self, images, targets=None):
        return self.forward(images, targets)

    def forward(self, images, targets=None):
        """PGD 攻击 forward 方法，与 torchattacks.PGD.forward(images, labels) 接口一致"""
        images = images.clone().detach()
        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            # 1. 前向传播 (获取 NMS 之前的 Raw Output)
            outputs = self.model(adv_images)

            # 2. 计算角度对抗损失
            cost = self.compute_angle_loss(outputs)

            # 3. 反向传播
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            # 4. 梯度上升 (PGD)
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def compute_angle_loss(self, outputs):
        """
        解析 YOLO OBB 训练模式下的 raw predictions 并计算角度损失。

        ultralytics OBB 在 train() 模式下返回:
          {'one2many': {'boxes': [B,4,N], 'scores': [B,nc,N], 'angle': [B,1,N], 'feats': [...]},
           'one2one':   {'boxes': [B,4,N], 'scores': [B,nc,N], 'angle': [B,1,N], 'feats': [...]}}

        攻击目标 (angle)：选取 top-k 高置信度 anchor，最大化其角度绝对值。
        PGD 梯度上升公式: x += alpha * sign(grad)，因此 loss 取负，
        让梯度上升实际增大 |angle|，使旋转框角度偏离。
        """
        if isinstance(outputs, dict):
            loss = 0.0
            for head_name in ['one2many', 'one2one']:
                if head_name in outputs and isinstance(outputs[head_name], dict):
                    scores = outputs[head_name]['scores']  # [B, nc, N]
                    angle = outputs[head_name]['angle']    # [B, 1, N]

                    B, _, N = scores.shape
                    max_scores, _ = scores.max(dim=1)  # [B, N]

                    # 选取 top-k 高置信度 anchor
                    k = min(self.topk, N)
                    _, topk_indices = max_scores.topk(k, dim=1)  # [B, K]

                    for b in range(B):
                        idx = topk_indices[b]  # [K]
                        angle_b = angle[b, 0, idx]  # [K]
                        # 最大化角度绝对值 → 让角度偏离原有值
                        loss -= angle_b.abs().mean()

            return loss
        else:
            raise ValueError(f"无法解析 OBB 模型输出，期望训练模式下的 dict 格式，实际为 {type(outputs)}")


# ==========================================
# 2. 图像数据集 (适配 YOLO 输入)
# ==========================================
class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, img_size=640):
        self.image_paths = image_paths
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.random.randint(0, 256, (self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))

        tensor_img = self.transform(img)
        return tensor_img, path


def main():
    parser = argparse.ArgumentParser(description="YOLO OBB Angle Head PGD Attack Tool")
    parser.add_argument('--weight_path', type=str, required=True, help='YOLO OBB .pt 权重路径 (如 yolov8n-obb.pt)')
    parser.add_argument('--image_dir', type=str, default='./images', help='包含测试图片的文件夹路径')
    parser.add_argument('--img_size', type=int, default=640, help='YOLO 输入尺寸 (通常为 640 或 1024)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (OBB 显存占用大，建议 1-2)')

    # PGD 参数
    parser.add_argument('--eps', type=float, default=8.0/255.0)
    parser.add_argument('--alpha', type=float, default=2.0/255.0)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--topk', type=int, default=50, help='攻击 top-k 高置信度 anchor 的角度')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==========================================
    # 3. 加载 Ultralytics YOLO OBB 模型
    # ==========================================
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请先安装 ultralytics: pip install ultralytics")

    print(f"Loading YOLO OBB model from {args.weight_path}...")
    yolo_wrapper = YOLO(args.weight_path)

    nn_model = yolo_wrapper.model
    nn_model.to(device)
    nn_model.train()

    # ==========================================
    # 4. 准备数据
    # ==========================================
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)
        print(f"[Warning] 目录 {args.image_dir} 不存在，已创建。请放入 jpg/png 图片后重新运行。")
        return

    image_paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not image_paths:
        print(f"[Error] 在 {args.image_dir} 中未找到图片。")
        return

    dataset = SimpleImageDataset(image_paths, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ==========================================
    # 5. 初始化角度 PGD 攻击
    # ==========================================
    atk = YOLOBB_AnglePGD(nn_model, eps=args.eps, alpha=args.alpha, steps=args.steps, topk=args.topk)

    print("Starting YOLO OBB Angle PGD Attack...")
    print(f"Attack target: deflect rotation angle of top-{args.topk} high-confidence anchors")
    os.makedirs("adv_outputs", exist_ok=True)

    # ==========================================
    # 6. 执行攻击与保存
    # ==========================================
    for batch_idx, (images, paths) in enumerate(dataloader):
        images = images.to(device)

        # 1. 干净样本的角度情况
        with torch.no_grad():
            clean_out = nn_model(images)
            if isinstance(clean_out, dict) and 'one2many' in clean_out:
                clean_scores = clean_out['one2many']['scores']  # [B, nc, N]
                clean_angle = clean_out['one2many']['angle']    # [B, 1, N]

                max_scores, _ = clean_scores.max(dim=1)  # [B, N]
                k = min(args.topk, max_scores.shape[1])
                _, topk_idx = max_scores.topk(k, dim=1)  # [B, K]

                clean_angle_abs_means = []
                for b in range(images.shape[0]):
                    idx = topk_idx[b]
                    clean_angle_abs_means.append(clean_angle[b, 0, idx].abs().mean().item())
            else:
                clean_angle_abs_means = None

        # 2. 生成对抗样本
        adv_images = atk(images)

        # 3. 对抗样本的角度情况
        with torch.no_grad():
            adv_out = nn_model(adv_images)
            if isinstance(adv_out, dict) and 'one2many' in adv_out:
                adv_scores = adv_out['one2many']['scores']
                adv_angle = adv_out['one2many']['angle']

                max_scores_adv, _ = adv_scores.max(dim=1)
                k = min(args.topk, max_scores_adv.shape[1])
                _, topk_idx_adv = max_scores_adv.topk(k, dim=1)

                adv_angle_abs_means = []
                for b in range(images.shape[0]):
                    idx = topk_idx_adv[b]
                    adv_angle_abs_means.append(adv_angle[b, 0, idx].abs().mean().item())

                # 打印统计
                for b in range(images.shape[0]):
                    c_abs = clean_angle_abs_means[b] if clean_angle_abs_means else 0
                    a_abs = adv_angle_abs_means[b]
                    delta = a_abs - c_abs
                    print(f"  Batch {batch_idx}, Image {b}: "
                          f"Angle |mean|: {c_abs:.4f} -> {a_abs:.4f} (Δ: {delta:+.4f})")
            else:
                print(f"  Batch {batch_idx}: 无法解析输出")

        # 4. 保存对抗图片
        for i in range(adv_images.shape[0]):
            adv_np = adv_images[i].cpu().permute(1, 2, 0).numpy()
            adv_np = (adv_np * 255).clip(0, 255).astype(np.uint8)
            adv_np = cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR)

            save_name = os.path.basename(paths[i])
            cv2.imwrite(f"adv_outputs/angle_{save_name}", adv_np)

    print(f"\nAttack finished! Adversarial images saved to ./adv_outputs/")


if __name__ == '__main__':
    main()


"""
python pgd角度攻击.py \
    --weight_path yolo_obb.pt \
    --image_dir ./test_images \
    --img_size 1024 \
    --batch_size 2 \
    --steps 10 \
    --topk 50
"""
