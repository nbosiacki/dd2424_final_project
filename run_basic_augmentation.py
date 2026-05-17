import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import base_project as bp

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def run_basic_augmentation_experiment():
    print("=== Starting E-grade Task: Basic Data Augmentation Experiment ===")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Basic data augmentation pipeline
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load the dataset using the specific OxfordIIITPet class
    print("Loading Oxford-IIIT Pet dataset...")
    # 使用专门的类，它会自动处理你们这种全部混在一起的图片格式！
    # 把末尾的 download=False 改成了 download=True
    train_dataset = datasets.OxfordIIITPet(root='./dataset', split='trainval', transform=train_transform, download=True)
    test_dataset = datasets.OxfordIIITPet(root='./dataset', split='test', transform=test_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Call the model and training functions
    print("Loading model...")
    model = bp.get_model_finetune(num_classes=37)
    bp.unfreeze_layers(model, 2)
    model = model.to(device)

    optimizer = bp.get_optimizer(model, 2, base_lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Starting training (running for 5 epochs)...")
    train_losses, test_losses, train_accs, test_accs, best_model, best_acc = bp.train_with_history(
        model, train_loader, test_loader, optimizer, criterion, device, num_epochs=5
    )

    print(f"\nExperiment complete! Highest test accuracy with augmentation: {best_acc:.4f}")

    # Plot and show the curves
    bp.plot_training_curves(train_losses, test_losses, train_accs, test_accs)


if __name__ == "__main__":
    run_basic_augmentation_experiment()