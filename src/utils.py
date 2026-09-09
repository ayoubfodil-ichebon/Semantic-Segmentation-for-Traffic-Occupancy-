"""
Visualisation utilities.
"""
import numpy as np
from src.dataset import ID2COLOR, IGNORE_INDEX

def decode_segmap(mask_np):
    """
    Convert class ID mask to RGB image using CamVid palette.
    mask_np: (H, W) int array. Pixels with IGNORE_INDEX become black.
    Returns: (H, W, 3) uint8 RGB.
    """
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for cls_id, color in ID2COLOR.items():
        rgb[mask_np == cls_id] = color
    return rgb
