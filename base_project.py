import torch
import torchvision.models as models

from torch.utils.data import Dataset


CAT_BREEDS = {
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair',
    'Egyptian Mau', 'Maine Coon', 'Persian', 'Ragdoll', 'Russian Blue',
    'Siamese', 'Sphynx'
}

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