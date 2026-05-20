import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath('..'))
import base_project as bp
from fixmatch import train_epoch_fixmatch, train_epoch_fixmatch_rc

# for train and evaluation

# def train_fixmatch()
def train_fixmatch(model, labelled_loader, unlabelled_loader, test_loader,
                   optimizer, scheduler, device,
                   num_epochs=30, tau=0.95, lambda_u=1.0):
    history = {
        'loss_s': [], 'loss_u': [], 'train_acc': [],
        'test_acc': [], 'mask_ratio': [],
    }
    best_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        loss_s, loss_u, train_acc, mask_ratio = train_epoch_fixmatch(
            model, labelled_loader, unlabelled_loader,
            optimizer, device, tau=tau, lambda_u=lambda_u
        )
        if scheduler is not None:
            scheduler.step()

        test_acc = evaluate(model, test_loader, device)

        history['loss_s'].append(loss_s)
        history['loss_u'].append(loss_u)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['mask_ratio'].append(mask_ratio)

        print(f"Epoch {epoch:>3} | loss_s {loss_s:.4f} | loss_u {loss_u:.4f} "
              f"| train {train_acc:.4f} | test {test_acc:.4f} | mask {mask_ratio:.2%}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return history, best_state, best_acc

def run_ablation(labelled_loader, unlabelled_loader, test_loader,
                tau, recompute_every, device,
                num_epochs=30, base_lr=0.03, num_classes=37):
    model = bp.get_model_finetune(num_classes=num_classes)
    bp.unfreeze_layers(model, l=2)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    history   = {'loss_s': [], 'loss_u': [], 'train_acc': [], 'test_acc': [], 'mask_ratio': []}

    for epoch in range(1, num_epochs + 1):
        loss_s, loss_u, train_acc, mask_ratio = train_epoch_fixmatch_rc(
            model, labelled_loader, unlabelled_loader, optimizer, device,
            tau=tau, lambda_u=1.0, recompute_every=recompute_every,
        )   
        scheduler.step()
        test_acc = evaluate(model, test_loader, device)
        for k, v in zip(history, [loss_s, loss_u, train_acc, test_acc, mask_ratio]):
            history[k].append(v) 
        print(f"  Epoch {epoch:>3} | loss_s {loss_s:.4f} | loss_u {loss_u:.4f} "
            f"| train {train_acc:.4f} | test {test_acc:.4f} | mask {mask_ratio:.2%}")
        if test_acc > best_acc:
            best_acc   = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return history, best_acc

# def evaluate()
def evaluate(model, test_loader, device):
    _, acc = bp.evaluate_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    return acc

def plot_fixmatch_curves(history):
    epochs = range(1, len(history['loss_s']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(epochs, history['loss_s'], label='Supervised (ℓ_s)')
    axes[0].plot(epochs, history['loss_u'], label='Unsupervised (ℓ_u)')
    axes[0].set_title('FixMatch Losses')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(epochs, history['train_acc'], label='Train (labelled)')
    axes[1].plot(epochs, history['test_acc'],  label='Test')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    axes[2].plot(epochs, history['mask_ratio'])
    axes[2].set_title('Pseudo-label mask ratio (fraction above τ)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
