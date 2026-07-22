import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. 自定义 YOLO OBB PGD 攻击器 (核心)
# ==========================================
# 不再继承 torchattacks.Attack（torchattacks 3.5.1 已移除 Attack 基类）
# 改为独立实现，遵循 torchattacks PGD 文档中的参数规范
class YOLOBB_PGD:
    """
    针对 Ultralytics YOLO OBB (旋转框) 的 PGD 攻击。
    参考 torchattacks.PGD 文档: eps=8/255, alpha=2/255, steps=10, random_start=True
    攻击目标：压低所有 anchor 的最大类别置信度，实现"漏检"（隐藏目标）。
    """
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=True):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

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

            # 2. 计算 OBB 对抗损失
            cost = self.compute_obb_loss(outputs)

            # 3. 反向传播
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            # 4. 梯度上升 (PGD)
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def compute_obb_loss(self, outputs):
        """
        解析 YOLO OBB 训练模式下的 raw predictions 并计算损失。

        ultralytics OBB 在 train() 模式下返回:
          {'one2many': {'boxes': [B,4,N], 'scores': [B,nc,N], 'angle': [B,1,N], 'feats': [...]},
           'one2one':   {'boxes': [B,4,N], 'scores': [B,nc,N], 'angle': [B,1,N], 'feats': [...]}}

        攻击目标 (evasion)：降低所有 anchor 的类别置信度，使模型漏检。
        PGD 梯度上升公式: x += alpha * sign(grad)，因此需要 loss 取负，
        让梯度上升实际降低 scores。
        """
        if isinstance(outputs, dict):
            loss = 0.0
            for head_name in ['one2many', 'one2one']:
                if head_name in outputs and isinstance(outputs[head_name], dict):
                    scores = outputs[head_name]['scores']  # [B, nc, N]
                    # 每个 anchor 的最大类别得分（logits）
                    max_scores, _ = scores.max(dim=1)  # [B, N]
                    loss -= max_scores.mean()  # 负号：PGD 梯度上升 → 降低 scores
            return loss
        else:
            raise ValueError(f"无法解析 OBB 模型输出，期望训练模式下的 dict 格式，实际为 {type(outputs)}")


# ==========================================
# 2. 图像数据集 (适配 YOLO 640x640 输入)
# ==========================================
class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, img_size=640):
        self.image_paths = image_paths
        self.img_size = img_size
        # YOLO 期望输入是 [0, 1] 的 RGB tensor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # 使用 cv2 读取并转换为 RGB
        img = cv2.imread(path)
        if img is None:
            # 如果读不到，生成随机噪声图
            img = np.random.randint(0, 256, (self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))

        tensor_img = self.transform(img)
        return tensor_img, path


def main():
    parser = argparse.ArgumentParser(description="YOLO OBB PGD Attack Tool")
    parser.add_argument('--weight_path', type=str, required=True, help='YOLO OBB .pt 权重路径 (如 yolov8n-obb.pt)')
    parser.add_argument('--image_dir', type=str, default='./images', help='包含测试图片的文件夹路径')
    parser.add_argument('--img_size', type=int, default=640, help='YOLO 输入尺寸 (通常为 640 或 1024)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (OBB 显存占用大，建议 1-2)')

    # PGD 参数
    parser.add_argument('--eps', type=float, default=8.0/255.0)
    parser.add_argument('--alpha', type=float, default=2.0/255.0)
    parser.add_argument('--steps', type=int, default=10)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==========================================
    # 3. 加载 Ultralytics YOLO OBB 模型 (关键)
    # ==========================================
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请先安装 ultralytics: pip install ultralytics")

    print(f"Loading YOLO OBB model from {args.weight_path}...")
    yolo_wrapper = YOLO(args.weight_path)

    # 提取底层 OBBModel（nn.Module），攻击时需要 train() 模式以获取可微的 raw predictions
    nn_model = yolo_wrapper.model
    nn_model.to(device)
    # 攻击时使用 train() 以获取带梯度的 raw prediction dict
    nn_model.train()

    # ==========================================
    # 4. 准备数据
    # ==========================================
    import os
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
    # 5. 初始化 OBB PGD 攻击
    # ==========================================
    atk = YOLOBB_PGD(nn_model, eps=args.eps, alpha=args.alpha, steps=args.steps)

    print("Starting YOLO OBB PGD Attack...")
    os.makedirs("adv_outputs", exist_ok=True)

    # ==========================================
    # 6. 执行攻击与保存
    # ==========================================
    for batch_idx, (images, paths) in enumerate(dataloader):
        images = images.to(device)

        # 1. 干净样本的置信度 (train 模式获取 raw scores)
        with torch.no_grad():
            clean_out = nn_model(images)
            if isinstance(clean_out, dict) and 'one2many' in clean_out:
                clean_scores = clean_out['one2many']['scores'].max(dim=1)[0].mean().item()
            else:
                clean_scores = 0.0

        # 2. 生成对抗样本 (train 模式下进行 PGD)
        adv_images = atk(images)

        # 3. 对抗样本的置信度 (同样 train 模式)
        with torch.no_grad():
            adv_out = nn_model(adv_images)
            if isinstance(adv_out, dict) and 'one2many' in adv_out:
                adv_scores = adv_out['one2many']['scores'].max(dim=1)[0].mean().item()
            else:
                adv_scores = 0.0

        print(f"Batch {batch_idx}: Clean Score: {clean_scores:.4f} -> Adv Score: {adv_scores:.4f} (Drop: {clean_scores - adv_scores:.4f})")

        # 4. 保存对抗图片 (转换回 [0, 255] 并保存)
        for i in range(adv_images.shape[0]):
            adv_np = adv_images[i].cpu().permute(1, 2, 0).numpy()
            adv_np = (adv_np * 255).clip(0, 255).astype(np.uint8)
            adv_np = cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR)

            save_name = os.path.basename(paths[i])
            cv2.imwrite(f"adv_outputs/adv_{save_name}", adv_np)

    print(f"\nAttack finished! Adversarial images saved to ./adv_outputs/")

if __name__ == '__main__':
    main()


"""
python main.py \
    --weight_path yolo_obb.pt \
    --image_dir ./test_images \
    --img_size 1024 \
    --batch_size 2 \
    --steps 10
"""
