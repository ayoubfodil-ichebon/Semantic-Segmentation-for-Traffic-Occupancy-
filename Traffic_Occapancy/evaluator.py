"""
Traffic Occupancy Evaluator
Compute occupancy rate and metrics (MAE, RMSE, correlation).
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataset import CLASSES, IGNORE_INDEX, ID2NAME

# ─── Vehicle classes (CamVid IDs) ───────────────────────────
VEHICLE_IDS = []
VEHICLE_NAMES = []
for i, (name, _) in enumerate(CLASSES):
    if name in ("Car", "Bicyclist", "SUVPickupTruck", "Truck_Bus",
                "Train", "MotorcycleScooter", "OtherMoving",
                "Pedestrian", "Child"):
        VEHICLE_IDS.append(i)
        VEHICLE_NAMES.append(name)

print(f"🚦 Vehicle classes for occupancy: {VEHICLE_NAMES}")


def compute_occupancy(mask_np, vehicle_ids=VEHICLE_IDS, ignore=IGNORE_INDEX):
    """
    Compute traffic occupancy rate from a segmentation mask.
    
    Args:
        mask_np: (H, W) numpy array of class IDs
        vehicle_ids: list of class IDs considered as "vehicles"
        ignore: ignore index (255)
    
    Returns:
        occupancy: float in [0, 1]
        detail: dict {class_name: pixel_count}
    """
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


@torch.no_grad()
def evaluate_occupancy(model, dataloader, device='cuda', max_batches=None):
    """
    Evaluate traffic occupancy on a full dataloader.
    
    Args:
        model: PyTorch model (segmentation)
        dataloader: DataLoader for test/val set
        device: 'cuda' or 'cpu'
        max_batches: limit number of batches (for quick testing)
    
    Returns:
        occ_gt: array of ground truth occupancy rates
        occ_pred: array of predicted occupancy rates
        metrics: dict with 'mae', 'rmse', 'corr'
        details: list of dicts with per-class pixel counts
    """
    model.eval()
    occ_gt_all = []
    occ_pred_all = []
    details_gt = []
    details_pred = []
    
    desc = "🚦 Occupancy Evaluation"
    for batch_idx, (imgs, masks) in enumerate(tqdm(dataloader, desc=desc)):
        if max_batches and batch_idx >= max_batches:
            break
        
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().numpy()
        masks = masks.numpy()
        
        for b in range(masks.shape[0]):
            gt_occ, det_gt = compute_occupancy(masks[b])
            pred_occ, det_pred = compute_occupancy(preds[b])
            
            occ_gt_all.append(gt_occ)
            occ_pred_all.append(pred_occ)
            details_gt.append(det_gt)
            details_pred.append(det_pred)
    
    occ_gt = np.array(occ_gt_all)
    occ_pred = np.array(occ_pred_all)
    
    mae = float(np.mean(np.abs(occ_gt - occ_pred)))
    rmse = float(np.sqrt(np.mean((occ_gt - occ_pred)**2)))
    corr = float(np.corrcoef(occ_gt, occ_pred)[0, 1]) if len(occ_gt) > 1 else 0.0
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'corr': corr,
        'gt_mean': float(np.mean(occ_gt)),
        'pred_mean': float(np.mean(occ_pred)),
        'n_samples': len(occ_gt)
    }
    
    return occ_gt, occ_pred, metrics, details_gt, details_pred


def plot_occupancy_results(occ_gt, occ_pred, metrics, model_name, save_path=None):
    """
    Generate histogram and scatter plot for occupancy results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Traffic Occupancy – {model_name}", fontsize=14, fontweight='bold')
    
    # ─── Histogram ─────────────────────────────────────────
    axes[0].hist(occ_gt * 100, bins=30, alpha=0.6, label="GT", color="steelblue", edgecolor='black')
    axes[0].hist(occ_pred * 100, bins=30, alpha=0.6, label="Pred", color="coral", edgecolor='black')
    axes[0].set_xlabel("Occupancy (%)")
    axes[0].set_ylabel("Number of images")
    axes[0].set_title("Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ─── Scatter ──────────────────────────────────────────
    lim = max(occ_gt.max(), occ_pred.max()) * 100 + 2
    axes[1].scatter(occ_gt * 100, occ_pred * 100, alpha=0.5, s=15, color="mediumpurple")
    axes[1].plot([0, lim], [0, lim], "r--", lw=1.5, label="Ideal (y=x)")
    axes[1].set_xlabel("GT Occupancy (%)")
    axes[1].set_ylabel("Pred Occupancy (%)")
    axes[1].set_title(f"MAE = {metrics['mae']*100:.2f}%  |  r = {metrics['corr']:.4f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, lim)
    axes[1].set_ylim(0, lim)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved: {save_path}")
    plt.show()


def print_occupancy_summary(metrics, model_name="Model"):
    """
    Print formatted occupancy metrics.
    """
    print("\n" + "="*60)
    print(f"  TRAFFIC OCCUPANCY — {model_name}")
    print("="*60)
    print(f"  Samples           : {metrics['n_samples']}")
    print(f"  GT mean           : {metrics['gt_mean']*100:.2f}%")
    print(f"  Pred mean         : {metrics['pred_mean']*100:.2f}%")
    print(f"  MAE               : {metrics['mae']*100:.2f} points")
    print(f"  RMSE              : {metrics['rmse']*100:.2f} points")
    print(f"  Pearson r         : {metrics['corr']:.4f}")
    print("="*60)
