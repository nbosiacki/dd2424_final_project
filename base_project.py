import torch
import torchvision.models as models
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import copy

from torch.utils.data import Dataset


CAT_BREEDS = {
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair',
    'Egyptian Mau', 'Maine Coon', 'Persian', 'Ragdoll', 'Russian Blue',
    'Siamese', 'Sphynx'
}

RESNET34_LAYER_GROUPS = ['layer4', 'layer3', 'layer2', 'layer1']

#
#---BINARY LABELS---
#

class BinaryPetDataset(Dataset):
    """
    Class for binary label version of the dataset
    """
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.classes = base_dataset.classes

    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, index):
        image, label = self.base[index]
        binary_label = _make_binary_label(label, self.classes)
        return image, binary_label


def _make_binary_label(original_label, classes):
    """
    helper function to return binary labels (dog/cat) based on existing breed label
    """
    breed = classes[original_label]
    return 1 if breed in CAT_BREEDS else 0

#
#---TRAINING---
#
def train_epoch_binary(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = (torch.sigmoid(logits) > 0.5).long()
        correct += (predictions == labels.long()).sum().item()
        total += labels.size(0)

    return total_loss /len(dataloader), correct / total

def evaluate_epoch_binary(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            predictions = (torch.sigmoid(logits) > 0.5).long()
            correct += (predictions == labels.long()).sum().item()
            total += labels.size(0)
    
    return total_loss /len(dataloader), correct / total

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss /len(dataloader), correct / total

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return total_loss /len(dataloader), correct / total

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(model, l, base_lr=1e-4):
    param_groups = []
    for i, layer_name in enumerate(RESNET34_LAYER_GROUPS[:l]):
        lr = base_lr * (0.1 ** i)
        param_groups.append({
            'params': getattr(model, layer_name).parameters(),
            'lr': lr
        })
    param_groups.append({'params': model.fc.parameters(), 'lr': base_lr})
    return torch.optim.Adam(param_groups)

# return a fresh ResNet34 with replaced fc and frozen layers
def get_model_finetune(num_classes=37):
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(512, num_classes)
    for param in model.parameters():
        param.requires_grad = False
    return model

# unfreeze the last l layers and fc
def unfreeze_layers(model, l):
    for param in model.parameters():
        param.requires_grad = False
    for layer_name in RESNET34_LAYER_GROUPS[:l]:
        for param in getattr(model, layer_name).parameters():
            param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

def train_with_history(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=10, scheduler=None):
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_acc = 0.0
    best_model = None

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc   = evaluate_epoch(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if scheduler is not None:
            scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Test loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    return train_losses, test_losses, train_accs, test_accs, best_model, best_acc

def plot_training_curves(train_losses, test_losses, train_accs, test_accs):
    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.xticks(range(1, len(train_losses) + 1, 2))
    plt.legend()
    plt.show()

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.xticks(range(1, len(train_losses) + 1, 2))
    plt.legend()
    plt.show()

def get_scheduler(optimizer, step_size=10, gamma=0.1):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)