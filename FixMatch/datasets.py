import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

import sys, os
sys.path.append(os.path.abspath('..'))
import base_project as bp


# Transform

def get_weak_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_strong_transform(magnitude=9):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=magnitude),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# no augmentation
def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# Datasets

class LabelledDataset(Dataset):
    #applies weak augmentation to labelled images
    def __init__(self, subset, weak_transform):
        self.subset         = subset
        self.weak_transform = weak_transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.weak_transform(img), label


class UnlabelledDataset(Dataset):
    # return weak and strong augmented pair without label
    def __init__(self, subset, weak_transform, strong_transform):
        self.subset           = subset
        self.weak_transform   = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, _ = self.subset[idx]
        return self.weak_transform(img), self.strong_transform(img)


# Split
# Splits train_data into labelled and unlabelled sets
def get_fixmatch_dataloaders(train_data, test_data, labelled_frac, batch_size=64, mu=7, 
                             num_workers=2, seed=42, magnitude=9):
    
    weak_transform   = get_weak_transform()
    strong_transform = get_strong_transform(magnitude=magnitude)
    test_transform   = get_test_transform()

    # stratified split
    labelled_subset    = bp.get_stratified_subset(train_data, labelled_frac, seed=seed)
    labelled_indices   = set(labelled_subset.indices)
    unlabelled_indices = list(set(range(len(train_data))) - labelled_indices)
    unlabelled_subset  = Subset(train_data, unlabelled_indices)

    print(f"Labelled:   {len(labelled_subset)} samples ({labelled_frac*100:.0f}%)")
    print(f"Unlabelled: {len(unlabelled_subset)} samples ({(1-labelled_frac)*100:.0f}%)")
    print(f"Test:       {len(test_data)} samples")

    # wrap with transforms
    labelled_dataset   = LabelledDataset(labelled_subset, weak_transform)
    unlabelled_dataset = UnlabelledDataset(unlabelled_subset, weak_transform, strong_transform)
    test_dataset       = datasets.OxfordIIITPet(
                            root=test_data.root,
                            split='test',
                            download=False,
                            transform=test_transform
                         )

    # dataloaders
    labelled_loader   = DataLoader(labelled_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   drop_last=True)

    unlabelled_loader = DataLoader(unlabelled_dataset,
                                   batch_size=batch_size * mu,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   drop_last=True)

    test_loader       = DataLoader(test_dataset,
                                   batch_size=32,
                                   shuffle=False,
                                   num_workers=num_workers)

    return labelled_loader, unlabelled_loader, test_loader