"""
PGD 攻击 YOLO OBB 旋转框检测脚本。

支持三种攻击模式:
  --mode miss  : 漏检攻击 (压低所有 anchor 置信度，隐藏目标)
  --mode cls   : 分类攻击 (让 top-k anchor 预测类别偏离)
  --mode angle : 角度攻击 (偏转 top-k anchor 的旋转角度)

用法示例:
  python pgd_obb.py \
      --weight_path yolo_obb.pt \
      --image_dir ./test_images \
      --img_size 1024 \
      --mode miss \
      --steps 10

  python pgd_obb.py \
      --weight_path yolo_obb.pt \
      --image_dir ./test_images \
      --img_size 1024 \
      --mode cls \
      --steps 10 \
      --topk 50

  python pgd_obb.py \
      --weight_path yolo_obb.pt \
      --image_dir ./test_images \
      --img_size 1024 \
      --mode angle \
      --steps 10 \
      --topk 50
"""

import argparse
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


# ==========================================
# 1. PGD 攻击器 (支持三种攻击目标)
# ==========================================
class YOLOBB_PGD:
    """
    针对 Ultralytics YOLO OBB 的 PGD 攻击。

    参考 torchattacks.PGD 文档: eps=8/255, alpha=2/255, steps=10, random_start=True
    攻击时需要 model.train() 模式以获取可微的 raw prediction dict。
    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10,
                 random_start=True, mode='miss', topk=50):
        """
        Args:
            model: YOLO OBB nn.Module (需处于 train() 模式)
            eps: 最大扰动幅度
            alpha: 每步步长
            steps: 迭代次数
            random_start: 是否随机初始化扰动
            mode: 攻击模式 - 'miss' (漏检) / 'cls' (分类) / 'angle' (角度)
            topk: cls/angle 模式下选取的 top-k 高置信度 anchor 数量
        """
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.mode = mode
        self.topk = topk

        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

    def __call__(self, images, targets=None):
        return self.forward(images, targets)

    def forward(self, images, targets=None):
        """PGD 攻击 forward 方法"""
        images = images.clone().detach()
        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            # 1. 前向传播 (获取 NMS 之前的可微 Raw Output)
            outputs = self.model(adv_images)

            # 2. 根据模式计算对抗损失
            cost = self._compute_loss(outputs)

            # 3. 反向传播求梯度
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            # 4. 梯度上升更新 (PGD)
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def _compute_loss(self, outputs):
        """
        根据 self.mode 解析 YOLO OBB 训练模式 raw predictions 并计算损失。

        ultralytics OBB train() 模式返回:
          {'one2many': {'boxes': [B,4,N], 'scores': [B,nc,N], 'angle': [B,1,N], 'feats': [...]},
           'one2one':   {'boxes': [B,4,N], 'scores': [B,nc,N], 'angle': [B,1,N], 'feats': [...]}}

        三种模式:
          - miss:  压低所有 anchor 置信度 (负号让梯度上升时降低 scores)
          - cls:   最大化 top-k anchor 的 CE loss，使分类错误
          - angle: 最大化 top-k anchor 的角度绝对值
        """
        if not isinstance(outputs, dict):
            raise ValueError(f"无法解析 OBB 模型输出，期望训练模式的 dict，实际 {type(outputs)}")

        loss = 0.0
        for head_name in ['one2many', 'one2one']:
            if head_name not in outputs or not isinstance(outputs[head_name], dict):
                continue

            head = outputs[head_name]
            scores = head['scores']  # [B, nc, N]
            B, nc, N = scores.shape

            max_scores, pred_labels = scores.max(dim=1)  # [B, N] each

            if self.mode == 'miss':
                # 漏检攻击: 降低所有 anchor 的置信度
                loss -= max_scores.mean()

            elif self.mode == 'cls':
                # 分类攻击: 最大化 top-k anchor 的 CE loss
                k = min(self.topk, N)
                _, topk_idx = max_scores.topk(k, dim=1)  # [B, K]
                for b in range(B):
                    idx = topk_idx[b]
                    logits_b = scores[b, :, idx].permute(1, 0)  # [K, nc]
                    labels_b = pred_labels[b, idx]              # [K]
                    loss -= self.ce_loss(logits_b, labels_b)

            elif self.mode == 'angle':
                # 角度攻击: 最大化 top-k anchor 角度绝对值
                angle = head['angle']  # [B, 1, N]
                k = min(self.topk, N)
                _, topk_idx = max_scores.topk(k, dim=1)
                for b in range(B):
                    idx = topk_idx[b]
                    loss -= angle[b, 0, idx].abs().mean()

        return loss


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
        return self.transform(img), path


# ==========================================
# 3. 评估函数
# ==========================================
def evaluate_attack(nn_model, images, adv_images, args, device):
    """根据不同 attack 模式评估攻击效果并打印统计信息"""
    with torch.no_grad():
        clean_out = nn_model(images)
        adv_out = nn_model(adv_images)

    if not (isinstance(clean_out, dict) and 'one2many' in clean_out):
        print("  [Warn] 无法解析模型输出")
        return

    clean_head = clean_out['one2many']
    adv_head = adv_out['one2many']
    clean_scores = clean_head['scores']  # [B, nc, N]
    adv_scores = adv_head['scores']

    B = images.shape[0]
    max_clean, pred_clean = clean_scores.max(dim=1)  # [B, N]
    max_adv, pred_adv = adv_scores.max(dim=1)

    if args.mode == 'miss':
        # 漏检: 打印置信度下降幅度
        for b in range(B):
            c = max_clean[b].mean().item()
            a = max_adv[b].mean().item()
            print(f"  Image {b}: Clean Score {c:.4f} -> Adv Score {a:.4f} (Drop: {c - a:.4f})")

    elif args.mode == 'cls':
        # 分类: 统计 top-k anchor 类别翻转率
        k = min(args.topk, max_clean.shape[1])
        _, topk_idx = max_clean.topk(k, dim=1)  # [B, K]
        for b in range(B):
            idx = topk_idx[b]
            changed = (pred_clean[b, idx] != pred_adv[b, idx]).sum().item()
            print(f"  Image {b}: Top-{k} class flip rate = {changed / k:.2%}")

    elif args.mode == 'angle':
        # 角度: 打印角度绝对值变化
        clean_angle = clean_head['angle']  # [B, 1, N]
        adv_angle = adv_head['angle']

        k = min(args.topk, max_clean.shape[1])
        _, topk_idx_clean = max_clean.topk(k, dim=1)
        _, topk_idx_adv = max_adv.topk(k, dim=1)
        for b in range(B):
            c_abs = clean_angle[b, 0, topk_idx_clean[b]].abs().mean().item()
            a_abs = adv_angle[b, 0, topk_idx_adv[b]].abs().mean().item()
            print(f"  Image {b}: Angle |mean| {c_abs:.4f} -> {a_abs:.4f} (Δ: {a_abs - c_abs:+.4f})")


# ==========================================
# 4. 主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="YOLO OBB PGD Attack Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 漏检攻击
  python pgd_obb.py --weight_path yolo_obb.pt --image_dir ./test_images --mode miss

  # 分类攻击 (top-50 anchor)
  python pgd_obb.py --weight_path yolo_obb.pt --image_dir ./test_images --mode cls --topk 50

  # 角度攻击
  python pgd_obb.py --weight_path yolo_obb.pt --image_dir ./test_images --mode angle --topk 50
        """
    )
    parser.add_argument('--weight_path', type=str, required=True,
                        help='YOLO OBB .pt 权重路径')
    parser.add_argument('--image_dir', type=str, default='./test_images',
                        help='测试图片目录')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='YOLO 输入尺寸 (640/1024)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (OBB 显存大，建议 1-2)')

    # 攻击模式
    parser.add_argument('--mode', type=str, default='miss',
                        choices=['miss', 'cls', 'angle'],
                        help='攻击模式: miss=漏检, cls=分类混淆, angle=角度偏转')

    # PGD 参数
    parser.add_argument('--eps', type=float, default=8.0 / 255.0,
                        help='最大扰动幅度')
    parser.add_argument('--alpha', type=float, default=2.0 / 255.0,
                        help='每步步长')
    parser.add_argument('--steps', type=int, default=10,
                        help='迭代步数')
    parser.add_argument('--topk', type=int, default=50,
                        help='cls/angle 模式下的 top-k anchor 数量')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Mode: {args.mode} | Steps: {args.steps}")

    # ---- 加载模型 ----
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请安装 ultralytics: pip install ultralytics")

    print(f"Loading YOLO OBB model: {args.weight_path}")
    yolo_wrapper = YOLO(args.weight_path)
    nn_model = yolo_wrapper.model
    nn_model.to(device)
    nn_model.train()  # train 模式获取可微 raw prediction

    # ---- 准备数据 ----
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)
        print(f"[Warn] {args.image_dir} 不存在，已创建。请放入图片后重试。")
        return

    image_paths = sorted([
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if not image_paths:
        print(f"[Error] {args.image_dir} 中未找到图片。")
        return
    print(f"Found {len(image_paths)} images")

    dataset = SimpleImageDataset(image_paths, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ---- 初始化 PGD 攻击 ----
    atk = YOLOBB_PGD(
        nn_model,
        eps=args.eps,
        alpha=args.alpha,
        steps=args.steps,
        random_start=True,
        mode=args.mode,
        topk=args.topk,
    )

    mode_desc = {'miss': '漏检 (evasion)', 'cls': '分类混淆 (misclassification)',
                 'angle': '角度偏转 (angle deflection)'}
    print(f"\nPGD Attack: {mode_desc[args.mode]}")
    print(f"Eps={args.eps:.4f}  Alpha={args.alpha:.4f}  Steps={args.steps}")

    os.makedirs("adv_outputs", exist_ok=True)

    # ---- 执行攻击 ----
    for batch_idx, (images, paths) in enumerate(dataloader):
        images = images.to(device)

        print(f"\n--- Batch {batch_idx} ---")
        # 攻击
        adv_images = atk(images)

        # 评估
        evaluate_attack(nn_model, images, adv_images, args, device)

        # 保存对抗图片
        for i in range(adv_images.shape[0]):
            adv_np = adv_images[i].cpu().permute(1, 2, 0).numpy()
            adv_np = (adv_np * 255).clip(0, 255).astype(np.uint8)
            adv_np = cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR)

            save_name = os.path.basename(paths[i])
            cv2.imwrite(f"adv_outputs/pgd_{args.mode}_{save_name}", adv_np)

    print(f"\nDone! Results saved to ./adv_outputs/pgd_{args.mode}_*")


if __name__ == '__main__':
    main()
