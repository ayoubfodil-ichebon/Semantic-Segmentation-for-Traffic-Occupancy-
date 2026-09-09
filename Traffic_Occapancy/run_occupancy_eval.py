"""
Run Traffic Occupancy Evaluation for all trained models.
Loads best weights from outputs/models/ and computes occupancy metrics.
"""
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CamVidPairsDataset, NUM_CLASSES, IGNORE_INDEX
from src.models import UNet, SegNet, VMambaUNet, FCN8s
from src.metrics import compute_miou, compute_pixel_accuracy
from src.traffic_occupancy import evaluate_occupancy, plot_occupancy_results, print_occupancy_summary

# ─── Load config ─────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() and config["device"]["use_cuda"] else "cpu"
print(f"🔧 Using device: {device}")

# ─── Load test data ─────────────────────────────────────────
ROOT = config["data"]["root"]
IMG_DIR = os.path.join(ROOT, config["data"]["img_dir"])
MASK_DIR = os.path.join(ROOT, config["data"]["mask_dir"])
TRAIN_SIZE = tuple(config["data"]["train_size"])

def img_base(name): return os.path.splitext(name)[0]
def mask_base(name):
    n = os.path.splitext(name)[0]
    return n[:-2] if n.endswith("_L") else n

imgs_all = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])
masks_all = sorted([f for f in os.listdir(MASK_DIR) if f.endswith(".png")])
imgs_map = {img_base(f): f for f in imgs_all}
masks_map = {mask_base(f): f for f in masks_all}
common = [(imgs_map[k], masks_map[k]) for k in imgs_map if k in masks_map]

np.random.seed(config["split"]["seed"])
indices = np.random.permutation(len(common))
n = len(common)
tr, va, te = int(0.7*n), int(0.85*n)
test_pairs = [common[i] for i in indices[va:te]]

test_ds = CamVidPairsDataset(IMG_DIR, MASK_DIR, test_pairs, TRAIN_SIZE, augment=False)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

# ─── Model mapping ──────────────────────────────────────────
MODEL_CLASSES = {
    "UNet": (UNet, {}),
    "SegNet": (SegNet, {}),
    "VMambaUNet": (VMambaUNet, {"C": 64, "n_vss_enc": 2, "n_vss_bot": 2, "n_vss_dec": 2}),
    "FCN": (FCN8s, {"pretrained": False})
}

results = {}

# ─── Evaluate each model ────────────────────────────────────
for model_name, (cls, params) in MODEL_CLASSES.items():
    if not config["models"].get(model_name, {}).get("enabled", False):
        print(f"⚠️  {model_name} is disabled in config, skipping.")
        continue

    weight_path = os.path.join(config["training"]["save_dir"], f"best_{model_name}_camvid.pth")
    if not os.path.exists(weight_path):
        print(f"❌ Weights not found: {weight_path}")
        continue

    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")

    # Load model
    model = cls(NUM_CLASSES, **params).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # ── Segmentation metrics ──────────────────────────────
    miou, _ = compute_miou(model, test_loader, device=device)
    pa, mpa = compute_pixel_accuracy(model, test_loader, device=device)

    # ── Occupancy metrics ─────────────────────────────────
    occ_gt, occ_pred, occ_metrics, _, _ = evaluate_occupancy(
        model, test_loader, device=device, max_batches=None
    )

    print_occupancy_summary(occ_metrics, model_name)

    # Store
    results[model_name] = {
        "miou": miou,
        "pa": pa,
        "mpa": mpa,
        "occ_gt": occ_gt,
        "occ_pred": occ_pred,
        **occ_metrics  # mae, rmse, corr, gt_mean, pred_mean, n_samples
    }

    # ── Individual plot for this model ────────────────────
    fig_dir = config["evaluation"]["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    plot_occupancy_results(
        occ_gt, occ_pred, occ_metrics, model_name,
        save_path=os.path.join(fig_dir, f"occupancy_{model_name}.png")
    )

# ─── Global comparison table ───────────────────────────────
print("\n" + "="*90)
print("  FINAL COMPARISON — Segmentation + Traffic Occupancy")
print("="*90)
print(f"{'Model':<14} {'mIoU':>8} {'PA':>8} {'MAE (%)':>10} {'RMSE (%)':>10} {'Corr r':>10}")
print("-"*70)
for name, r in results.items():
    print(f"{name:<14} {r['miou']:8.4f} {r['pa']:8.4f} {r['mae']*100:10.2f} {r['rmse']*100:10.2f} {r['corr']:10.4f}")
print("="*90)

# ─── Comparison bar chart ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
names = list(results.keys())
colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

axes[0].bar(names, [r["miou"] for r in results.values()], color=colors)
axes[0].set_title("Test mIoU", fontweight="bold")
axes[0].set_ylabel("mIoU")
axes[0].grid(axis="y", alpha=0.3)
axes[0].set_ylim(0, 1)

axes[1].bar(names, [r["mae"]*100 for r in results.values()], color=colors)
axes[1].set_title("Occupancy MAE (percentage points)", fontweight="bold")
axes[1].set_ylabel("MAE (%)")
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("CamVid – Segmentation & Traffic Occupancy", fontsize=14, fontweight="bold")
plt.tight_layout()
comp_path = os.path.join(config["evaluation"]["figures_dir"], "occupancy_comparison.png")
plt.savefig(comp_path, dpi=150)
plt.show()
print(f"\n✅ Comparison chart saved: {comp_path}")

# ─── Best model scatter (highest correlation) ──────────────
best_model = max(results, key=lambda k: results[k]["corr"])
best_r = results[best_model]
print(f"\n🏆 Best model for occupancy: {best_model} (r={best_r['corr']:.4f})")

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(best_r["occ_gt"]*100, best_r["occ_pred"]*100, alpha=0.5, s=20, color="mediumpurple")
lim = max(best_r["occ_gt"].max(), best_r["occ_pred"].max()) * 100 + 2
ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Ideal (y=x)")
ax.set_xlabel("GT Occupancy (%)")
ax.set_ylabel("Pred Occupancy (%)")
ax.set_title(f"{best_model}  |  MAE={best_r['mae']*100:.2f}%  r={best_r['corr']:.4f}", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
best_path = os.path.join(config["evaluation"]["figures_dir"], "occupancy_best_scatter.png")
plt.savefig(best_path, dpi=150)
plt.show()
print(f"✅ Best scatter plot saved: {best_path}")
