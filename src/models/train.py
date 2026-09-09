"""
Training and validation loops.
"""
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from src.metrics import CombinedLoss

def train_one_epoch(model, loader, optimizer, criterion, epoch, epochs, device='cuda'):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs} [TRAIN]", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(len(loader), 1)

@torch.no_grad()
def validate(model, loader, criterion, num_classes, ignore_index, device='cuda'):
    model.eval()
    total_loss = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    correct = 0
    total = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outs = model(imgs)
        total_loss += criterion(outs, masks).item()
        preds = outs.argmax(1)
        valid = masks != ignore_index
        p, g = preds[valid], masks[valid]
        idx = g * num_classes + p
        confusion += torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
        correct += (p == g).sum().item()
        total += valid.sum().item()

    iou = []
    for c in range(num_classes):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        d = tp + fp + fn
        iou.append(tp / d if d > 0 else float("nan"))
    valid_iou = [v for v in iou if not math.isnan(v)]
    miou = float(np.mean(valid_iou)) if valid_iou else 0.0
    pa = correct / max(total, 1)
    return total_loss / max(len(loader), 1), miou, pa
