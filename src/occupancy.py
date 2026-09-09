"""
Traffic occupancy estimation (vehicle pixel ratio).
"""
import numpy as np
import torch
from tqdm import tqdm
from src.dataset import CLASSES, IGNORE_INDEX, ID2NAME

# ─── Vehicle classes from CamVid ──────────────────────────
VEHICLE_IDS = []
VEHICLE_NAMES = []
for i, (name, _) in enumerate(CLASSES):
    if name in ("Car", "Bicyclist", "SUVPickupTruck", "Truck_Bus",
                "Train", "MotorcycleScooter", "OtherMoving",
                "Pedestrian", "Child"):
        VEHICLE_IDS.append(i)
        VEHICLE_NAMES.append(name)

def compute_occupancy(mask_np, vehicle_ids=VEHICLE_IDS, ignore=IGNORE_INDEX):
    """Return (occupancy_ratio, dict{pixels_per_class})."""
    total_valid = int(np.sum(mask_np != ignore))
    if total_valid == 0:
        return 0.0, {}
    vehicle_pixels = 0
    detail = {}
    for vid in vehicle_ids:
        cnt = int(np.sum(mask_np == vid))
        detail[ID2NAME[vid]] = cnt
        vehicle_pixels += cnt
    return vehicle_pixels / total_valid, detail

def evaluate_occupancy(loader, model, device='cuda', max_batches=None):
    """Evaluate occupancy on a DataLoader. Returns (gt_occ, pred_occ, mae, rmse, corr)."""
    model.eval()
    occ_gt_all, occ_pred_all = [], []

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(tqdm(loader, desc="Occupancy Eval")):
            if max_batches and batch_idx >= max_batches:
                break
            preds = model(imgs.to(device)).argmax(1).cpu().numpy()
            masks = masks.numpy()
            for b in range(masks.shape[0]):
                og, _ = compute_occupancy(masks[b])
                op, _ = compute_occupancy(preds[b])
                occ_gt_all.append(og)
                occ_pred_all.append(op)

    occ_gt = np.array(occ_gt_all)
    occ_pred = np.array(occ_pred_all)
    mae = float(np.mean(np.abs(occ_gt - occ_pred)))
    rmse = float(np.sqrt(np.mean((occ_gt - occ_pred)**2)))
    corr = float(np.corrcoef(occ_gt, occ_pred)[0, 1]) if len(occ_gt) > 1 else 0.0
    return occ_gt, occ_pred, mae, rmse, corr
