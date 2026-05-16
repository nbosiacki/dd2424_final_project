import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class PseudoLabelDataset(Dataset):
    def __init__(self, labelled_subset, unlabelled_subset, pseudo_indices, pseudo_labels):
        self.labelled        = labelled_subset
        self.unlabelled      = unlabelled_subset
        self.pseudo_indices  = pseudo_indices
        self.pseudo_labels   = pseudo_labels

    def __len__(self):
        return len(self.labelled) + len(self.pseudo_indices)

    def __getitem__(self, idx):
        if idx < len(self.labelled):
            return self.labelled[idx]
        else:
            pseudo_idx   = idx - len(self.labelled)
            image, _     = self.unlabelled[self.pseudo_indices[pseudo_idx]]
            pseudo_label = self.pseudo_labels[pseudo_idx]
            return image, pseudo_label


def generate_pseudo_labels(model, unlabelled_loader, device, confidence_threshold=0.95):
    model.eval()
    pseudo_indices = []
    pseudo_labels  = []

    with torch.no_grad():
        idx = 0
        for images, _ in unlabelled_loader:
            images = images.to(device)
            probs  = torch.softmax(model(images), dim=1)
            confs, preds = probs.max(dim=1)

            for i in range(len(images)):
                if confs[i].item() >= confidence_threshold:
                    pseudo_indices.append(idx + i)
                    pseudo_labels.append(preds[i].item())
            idx += len(images)

    return pseudo_indices, pseudo_labels