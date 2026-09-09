"""
Segmentation metrics: mIoU, Pixel Accuracy, Dice Loss, Combined Loss.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dataset import NUM_CLASSES, IGNORE_INDEX

# ─── mIoU ──────────────────────────────────────────────────
@torch.no_grad()
def compute_miou(model, loader, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, device='cuda'):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs).argmax(1)
        valid = masks != ignore_index
        p = preds[valid]
        g = masks[valid]
        idx = g * num_classes + p
        confusion += torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)

    iou = []
    for c in range(num_classes):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        d = tp + fp + fn
        iou.append(tp / d if d > 0 else float("nan"))

    valid_iou = [v for v in iou if not math.isnan(v)]
    return float(np.mean(valid_iou)) if valid_iou else 0.0, np.array(iou)

# ─── Pixel Accuracy ────────────────────────────────────────
@torch.no_grad()
def compute_pixel_accuracy(model, loader, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, device='cuda'):
    model.eval()
    correct_total = 0
    valid_total = 0
    correct_cls = torch.zeros(num_classes)
    total_cls = torch.zeros(num_classes)

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs).argmax(1)
        valid = masks != ignore_index
        correct_total += (preds[valid] == masks[valid]).sum().item()
        valid_total += valid.sum().item()
        for c in range(num_classes):
            gt_c = (masks == c) & valid
            correct_cls[c] += ((preds == c) & gt_c).sum().item()
            total_cls[c] += gt_c.sum().item()

    pa = correct_total / max(valid_total, 1)
    mpa = (correct_cls / (total_cls + 1e-7)).mean().item()
    return pa, mpa

# ─── Dice Loss ─────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=IGNORE_INDEX, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        probs = torch.softmax(logits, dim=1)
        mask = targets != self.ignore_index
        t_safe = targets.clone()
        t_safe[~mask] = 0
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, t_safe.unsqueeze(1), 1.0)
        m = mask.unsqueeze(1).expand_as(probs)
        probs = probs * m
        one_hot = one_hot * m
        axes = (0, 2, 3)
        inter = (probs * one_hot).sum(dim=axes)
        denom = probs.sum(dim=axes) + one_hot.sum(dim=axes)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        present = (one_hot.sum(dim=axes) > 0)
        if present.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return 1.0 - dice[present].mean()

# ─── Combined Loss (CE + Dice) ────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=IGNORE_INDEX, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index)

    def forward(self, logits, targets):
        return self.alpha * self.ce(logits, targets) + (1 - self.alpha) * self.dice(logits, targets)
