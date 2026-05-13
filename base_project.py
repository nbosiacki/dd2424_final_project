import torch
import torchvision.models as models
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle

from torch.utils.data import DataLoader, Dataset, Subset
from collections import defaultdict


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



def get_stratified_subset(dataset, fraction, seed=42):
    if fraction >= 1.0:
        return dataset

    rng = random.Random(seed)
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[int(label)].append(idx)

    selected = []
    for label, indices in class_indices.items():
        rng.shuffle(indices)
        n_keep = max(1, int(len(indices) * fraction))
        selected.extend(indices[:n_keep])

    return Subset(dataset, selected)

def print_summary(strategy1_results, strategy2_results, fractions):
    print("\n====== SUMMARY TABLE ======")
    print(f"{'Fraction':<10} {'S1 l=1':<10} {'S1 l=2':<10} {'S1 l=3':<10} {'S1 l=4':<10} {'S2 grad':<10}")
    print("-" * 60)
    for frac in fractions:
        s1 = strategy1_results[frac]
        s2 = strategy2_results[frac]
        print(f"{frac*100:>6.0f}%   "
              f"{s1[1]:.4f}    {s1[2]:.4f}    {s1[3]:.4f}    {s1[4]:.4f}    {s2:.4f}")

def plot_fraction_comparison(strategy1_results, strategy2_results, fractions):
    labels = [f"{int(f*100)}%" for f in fractions]
    x      = np.arange(len(fractions))
    width  = 0.15

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, l in enumerate(range(1, 5)):
        vals = [strategy1_results[f][l] for f in fractions]
        ax.bar(x + i*width, vals, width, label=f'S1 l={l}')

    s2_vals = [strategy2_results[f] for f in fractions]
    ax.bar(x + 4*width, s2_vals, width, label='S2 gradual', color='gray')

    ax.set_xticks(x + 2*width)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Training data fraction')
    ax.set_ylabel('Best test accuracy')
    ax.set_title('Transfer learning strategies vs. data fraction')
    ax.legend()
    plt.tight_layout()
    plt.show()

def save_limited_data_results(strategy1_results,
                              strategy2_results,
                              fractions,
                              filename="limited_data_results.pkl"):

    results = {
        "strategy1": strategy1_results,
        "strategy2": strategy2_results,
        "fractions": fractions
    }

    with open(filename, "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved successfully to {filename}")

def load_limited_data_results(
    filename="limited_data_results.pkl"
):

    import pickle

    with open(filename, "rb") as f:
        results = pickle.load(f)

    return (
        results["strategy1"],
        results["strategy2"],
        results["fractions"]
    )

def get_optimizer_l2(model, l, base_lr=1e-4, weight_decay=1e-4):
    param_groups = []
    for i, layer_name in enumerate(RESNET34_LAYER_GROUPS[:l]):
        lr = base_lr * (0.1 ** i)
        param_groups.append({
            'params': getattr(model, layer_name).parameters(),
            'lr': lr,
            'weight_decay': weight_decay
        })
    param_groups.append({
        'params': model.fc.parameters(),
        'lr': base_lr,
        'weight_decay': weight_decay
    })
    return torch.optim.Adam(param_groups)

def print_l2_summary(l2_results, fractions, weight_decays):
    print("\n====== L2 SUMMARY TABLE ======")
    print(f"{'Fraction':<10}", end="")
    for wd in weight_decays:
        print(f"  wd={wd:<8}", end="")
    print()
    print("-" * 60)
    for frac in fractions:
        print(f"{frac*100:>6.0f}%   ", end="")
        for wd in weight_decays:
            print(f"  {l2_results[frac][wd]:.4f}    ", end="")
        print()

def print_l2_s1_summary(l2_s1_results, fractions, weight_decays):
    for frac in fractions:
        print(f"\n====== FRACTION: {frac*100:.0f}% ======")
        print(f"{'WD':<12} {'S1 l=1':<10} {'S1 l=2':<10} {'S1 l=3':<10} {'S1 l=4':<10}")
        print("-" * 52)
        for wd in weight_decays:
            s1 = l2_s1_results[frac][wd]
            print(f"{wd:<12}  {s1[1]:.4f}    {s1[2]:.4f}    {s1[3]:.4f}    {s1[4]:.4f}")

#
#---IMBALANCED CLASSES---
#
def get_imbalanced_subset(dataset, cat_fraction=0.2, seed=42):
    rng = random.Random(seed)
    class_indices = defaultdict(list)
    for index in range(len(dataset)):
        _, label = dataset[index]
        class_indices[int(label)].append(index)

    selected = []
    for label, indices in class_indices.items():
        breed = dataset.classes[label]
        rng.shuffle(indices)
        n_keep = max(1, int(len(indices) * cat_fraction)) if breed in CAT_BREEDS else len(indices)
        selected.extend(indices[:n_keep])

    return Subset(dataset, selected)

def _get_labels_fast(dataset):
    """grabs integer labels without decoding the image"""
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, '_labels'):
        return [dataset.dataset._labels[i] for i in dataset.indices]
    return [int(dataset[i][1]) for i in range(len(dataset))]

def evaluate_per_class(model, dataloader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        per_class = {}

    for c, name in enumerate(class_names):
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[name] = {
            'precision': float(precision),
            'recall':    float(recall),
            'f1':        float(f1),
            'support':   int((all_labels == c).sum())
        }

    macro_f1    = float(np.mean([v['f1'] for v in per_class.values()]))
    total       = len(all_labels)
    weighted_f1 = float(sum(v[f'f1'] * v['support'] for v in per_class.values()) / total)
    overall_acc = float((all_preds == all_labels).mean())

    return per_class, macro_f1, weighted_f1, overall_acc

def get_class_weights(dataset, num_classes, device):
    """helper to compute inverse-frequency weights"""
    counts = torch.zeros(num_classes)
    for label in _get_labels_fast(dataset):
        counts[label] += 1
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights.to(device)

def get_oversampled_loader(dataset, batch_size=32, num_workers=2):
    """returns loader with sample distribution weighted in favor of rarer classes"""
    labels = _get_labels_fast(dataset)
    counts = defaultdict(int)
    for label in labels:
        counts[label] += 1

    sample_weights = [1.0 / counts[label] for label in labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(dataset), replacement=True
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

def print_per_class_results(per_class, cat_breeds):
    print(f"\n{'Class':<30} {'Type':<6} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("-" * 76)
    for name in sorted(per_class):
        m = per_class[name]
        t = 'cat' if name in cat_breeds else 'dog'
        print(f"{name:<30} {t:<6} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['support']:>8}")

def plot_per_class_f1(per_class, cat_breeds, title='Per-class F1 Score'):
    from matplotlib.patches import Patch
    names  = sorted(per_class)
    f1s    = [per_class[n]['f1'] for n in names]
    colors = ['tab:orange' if n in cat_breeds else 'tab:blue' for n in names]

    plt.figure(figsize=(16, 5))
    plt.bar(range(len(names)), f1s, color=colors)
    plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
    plt.ylim(0, 1)
    plt.ylabel('F1 Score')
    plt.title(title)
    plt.legend(handles=[
        Patch(facecolor='tab:orange', label='Cat (underrepresented)'),
        Patch(facecolor='tab:blue',   label='Dog'),
    ])
    plt.tight_layout()
    plt.show()

def print_imbalanced_summary(results):
    # results: dict  strategy_name -> (per_class, macro_f1, weighted_f1, overall_acc)
    print(f"\n{'Strategy':<25} {'Overall Acc':>12} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 62)
    for name, (_, macro_f1, weighted_f1, overall_acc) in results.items():
        print(f"{name:<25} {overall_acc:>12.4f} {macro_f1:>10.4f} {weighted_f1:>12.4f}")

    print(f"\n{'Strategy':<25} {'Cat avg F1':>12} {'Dog avg F1':>12}")
    print("-" * 52)
    for name, (per_class, _, _, _) in results.items():
        cat_f1 = np.mean([v['f1'] for k, v in per_class.items() if k in CAT_BREEDS])
        dog_f1 = np.mean([v['f1'] for k, v in per_class.items() if k not in CAT_BREEDS])
        print(f"{name:<25} {cat_f1:>12.4f} {dog_f1:>12.4f}")

def plot_imbalanced_comparison(results, cat_breeds):
    from matplotlib.patches import Patch
    strategy_names = list(results.keys())
    all_classes    = sorted(next(iter(results.values()))[0].keys())
    x              = np.arange(len(all_classes))
    width          = 0.8 / len(strategy_names)
    colors_strat   = ['tab:gray', 'tab:green', 'tab:red']

    fig, ax = plt.subplots(figsize=(18, 6))
    for i, (name, (per_class, _, _, _)) in enumerate(results.items()):
        f1s = [per_class[c]['f1'] for c in all_classes]
        ax.bar(x + i * width, f1s, width, label=name, color=colors_strat[i], alpha=0.8)

    for j, cls in enumerate(all_classes):
        if cls in cat_breeds:
            ax.axvspan(j - 0.4, j + 0.4 + width * (len(strategy_names) - 1),
                       color='tab:orange', alpha=0.08)

    ax.set_xticks(x + width * (len(strategy_names) - 1) / 2)
    ax.set_xticklabels(all_classes, rotation=90, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-class F1: Baseline vs. Weighted CE vs. Oversampling')
    ax.legend()
    plt.tight_layout()
    plt.show()