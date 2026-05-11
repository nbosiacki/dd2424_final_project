import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms

# 预训练 ResNet 要求的标准化参数 (这是数学要求，保证颜色数据分布均匀)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_weak_transform():
    """
    弱增强 (Weak Augmentation)：用于生成伪标签。
    仅包含基础的缩放和 50% 概率的水平翻转。
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])


def get_strong_transform():
    """
    强增强 (Strong Augmentation)：FixMatch 的核心，用于无标签数据的训练。
    包含疯狂的扭曲、变色、旋转，逼迫模型学习。
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),  # 随机旋转最多30度
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 随机色彩扭曲
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])


def _unnormalize_and_show(tensor, title):
    """
    内部辅助函数：因为 Normalize 会把图片颜色弄乱，
    这个函数负责把张量还原成肉眼能看懂的正常图片并展示。
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, NORMALIZE_MEAN, NORMALIZE_STD):
        t.mul_(s).add_(m)  # 反标准化计算

    tensor = tensor.clamp(0, 1)  # 限制数值范围
    img = tensor.permute(1, 2, 0).numpy()  # 转换成 matplotlib 认识的格式

    plt.imshow(img)
    plt.title(title)
    plt.axis('off')


def test_my_augmentations():
    """
    测试函数：去 dataset 文件夹里随便抓一张图，让你看看强弱增强的区别。
    """
    # 这个路径是根据你们 README.md 里的结构写的
    image_dir = os.path.join('dataset', 'oxford-iiit-pet', 'images')

    if not os.path.exists(image_dir):
        print(f"报错啦：找不到文件夹 {image_dir}！请确保你是在 final_project 根目录下运行的。")
        return

    # 找一张随机的 .jpg 图片
    all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    test_image_path = os.path.join(image_dir, random.choice(all_images))
    print(f"抓取测试图片成功: {test_image_path}")

    # 读取原图
    original_img = Image.open(test_image_path).convert('RGB')

    # 获取你定义的强弱流水线
    weak_tf = get_weak_transform()
    strong_tf = get_strong_transform()

    # 把原图扔进流水线加工
    weak_tensor = weak_tf(original_img)
    strong_tensor = strong_tf(original_img)

    # 画图对比 (一行三列)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img.resize((224, 224)))
    plt.title("1. Original Image (原图)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    _unnormalize_and_show(weak_tensor, "2. Weak Augmentation (弱增强)")

    plt.subplot(1, 3, 3)
    _unnormalize_and_show(strong_tensor, "3. Strong Augmentation (强增强)")

    plt.tight_layout()
    plt.show()


# 只有当你直接运行这个文件时，下面这两行才会执行
if __name__ == "__main__":
    print("=== 开始测试我的数据增强模块 ===")
    test_my_augmentations()