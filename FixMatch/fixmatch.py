import torch
import torch.nn as nn

# for fixmatch function implementation

# def fixmatch_loss()
def train_epoch_fixmatch(model, labelled_loader, unlabelled_loader,
                         optimizer, device, tau=0.95, lambda_u=1.0):
    model.train()
    total_loss_s = 0.0
    total_loss_u = 0.0
    correct      = 0
    total        = 0
    total_masked = 0
    total_unlab  = 0

    unlabelled_iter = iter(unlabelled_loader)
    criterion_unreduced = nn.CrossEntropyLoss(reduction='none')

    for imgs_weak_l, labels in labelled_loader:
        try:
            imgs_weak_u, imgs_strong_u = next(unlabelled_iter)
        except StopIteration:
            unlabelled_iter = iter(unlabelled_loader)
            imgs_weak_u, imgs_strong_u = next(unlabelled_iter)

        imgs_weak_l   = imgs_weak_l.to(device)
        labels        = labels.to(device)
        imgs_weak_u   = imgs_weak_u.to(device)
        imgs_strong_u = imgs_strong_u.to(device)

        # supervised loss equation 3
        logits_l = model(imgs_weak_l)
        loss_s = nn.CrossEntropyLoss()(logits_l, labels)

        # generate pseudo labels
        with torch.no_grad():
            probs_u = torch.softmax(model(imgs_weak_u), dim=1)
            conf, psuedo = probs_u.max(dim=1)
            mask = (conf >= tau).float()

        # unsupervised loss equation 4
        logits_u = model(imgs_strong_u)
        loss_u_per = criterion_unreduced(logits_u, psuedo)
        loss_u = (mask * loss_u_per).mean()

        # combined loss
        loss = loss_s + lambda_u * loss_u

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_s += loss_s.item()
        total_loss_u += loss_u.item()
        correct += (logits_l.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        total_masked += mask.sum().item()
        total_unlab += mask.size(0)

    n_batches = len(labelled_loader)
    mask_ratio = total_masked / total_unlab if total_unlab > 0 else 0.0
    return(
        total_loss_s / n_batches,
        total_loss_u / n_batches,
        correct / total,
        mask_ratio
    )

def train_epoch_fixmatch_rc(model, labelled_loader, unlabelled_loader,
                            optimizer, device, tau=0.95, lambda_u=1.0,
                            recompute_every=1):
    model.train()
    total_loss_s, total_loss_u = 0.0, 0.0
    correct, total = 0, 0
    total_masked, total_unlab = 0, 0
    unlabelled_iter     = iter(unlabelled_loader)
    criterion_unreduced = nn.CrossEntropyLoss(reduction='none')
    cached_pseudo = cached_mask = cached_strong = None

    def next_unlabelled():
        nonlocal unlabelled_iter
        try:
            return next(unlabelled_iter)
        except StopIteration:
            unlabelled_iter = iter(unlabelled_loader)
            return next(unlabelled_iter)

    for batch_idx, (imgs_weak_l, labels) in enumerate(labelled_loader):
        if batch_idx % recompute_every == 0:
            imgs_weak_u, imgs_strong_u = next_unlabelled()
            imgs_weak_u   = imgs_weak_u.to(device)
            cached_strong = imgs_strong_u.to(device)
            with torch.no_grad():
                probs_u = torch.softmax(model(imgs_weak_u), dim=1)
                conf, cached_pseudo = probs_u.max(dim=1)
                cached_mask = (conf >= tau).float()

        imgs_weak_l = imgs_weak_l.to(device)
        labels      = labels.to(device)
        logits_l    = model(imgs_weak_l)
        loss_s      = nn.CrossEntropyLoss()(logits_l, labels)
        logits_u    = model(cached_strong)
        loss_u      = (cached_mask * criterion_unreduced(logits_u, cached_pseudo)).mean()
        loss        = loss_s + lambda_u * loss_u

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        total_loss_s += loss_s.item();  total_loss_u += loss_u.item()
        correct      += (logits_l.argmax(dim=1) == labels).sum().item()
        total        += labels.size(0)
        total_masked += cached_mask.sum().item()
        total_unlab  += cached_mask.size(0)

    n_batches  = len(labelled_loader)
    mask_ratio = total_masked / total_unlab if total_unlab > 0 else 0.0
    return total_loss_s/n_batches, total_loss_u/n_batches, correct/total, mask_ratio
