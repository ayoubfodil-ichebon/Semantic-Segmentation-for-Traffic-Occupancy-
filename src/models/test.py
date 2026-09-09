"""
Test evaluation: loss, mIoU, per‑class IoU.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from src.dataset import NUM_CLASSES, IGNORE_INDEX, CLASSES

def evaluate_test(loader, model, device='cuda', num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    model.eval()
    total_loss = 0.0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Test"):
            imgs, masks = imgs.to(device), masks.to(device)
            outs = model(imgs)
            total_loss += criterion(outs, masks).item()
            pred_np = outs.argmax(1).cpu().numpy()
            gt_np = masks.cpu().numpy()
            for b in range(gt_np.shape[0]):
                valid = gt_np[b] != ignore_index
                np.add.at(confusion, (gt_np[b][valid].ravel(), pred_np[b][valid].ravel()), 1)

    avg_loss = total_loss / len(loader)
    iou_per_class = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        d = tp + fp + fn
        iou_per_class.append(tp / d if d > 0 else float("nan"))

    valid_ious = [v for v in iou_per_class if not math.isnan(v)]
    avg_miou = float(np.mean(valid_ious)) if valid_ious else 0.0

    # Print table
    print(f"\n{'='*58}")
    print(f"  Test Loss : {avg_loss:.4f}  |  Test mIoU : {avg_miou:.4f}")
    print(f"{'='*58}")
    print(f"{'ID':>3} {'Classe':<22} {'IoU':>8}  Barre")
    print("-" * 50)
    for i, ((cls_name, _), iou) in enumerate(zip(CLASSES, iou_per_class)):
        bar = "█" * int((iou if not math.isnan(iou) else 0) * 20)
        s = f"{iou:.4f}" if not math.isnan(iou) else "  N/A "
        print(f"{i:3d}  {cls_name:<22} {s}  {bar}")
    print(f"{'='*58}")
    print(f"  mIoU (classes présentes) : {avg_miou:.4f}")
    return avg_loss, avg_miou, iou_per_class, confusion
