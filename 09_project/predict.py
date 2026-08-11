"""
实战项目推理脚本：加载训练好的 checkpoint 对单张图片进行分类
=============================================================
运行方式：
  uv run python 09_project/predict.py --ckpt checkpoints/cifar10-xxx.ckpt
"""

import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import CIFAR10Model

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

TEST_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="checkpoint 路径")
    parser.add_argument("--image", required=True, help="待分类图片路径")
    args = parser.parse_args()

    # 从 checkpoint 恢复模型（超参已保存）
    model = CIFAR10Model.load_from_checkpoint(args.ckpt)
    model.eval()

    img = Image.open(args.image).convert("RGB")
    tensor = TEST_TRANSFORM(img).unsqueeze(0)  # (1, 3, 32, 32)

    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(dim=1).item()

    print(f"预测类别: {CIFAR10_CLASSES[pred]} (index={pred})")


if __name__ == "__main__":
    main()