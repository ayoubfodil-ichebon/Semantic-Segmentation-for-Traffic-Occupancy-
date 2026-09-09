# CamVid Dataset

This folder contains the raw **CamVid** (Cambridge Driving Labeled Video Database) data.  
Images are in `.png` format, and segmentation masks are in `.png` with the `_L` suffix.

---

## 📁 Expected Folder Structure

```text
data/
├── 701_StillsRaw_full/          # Input RGB images
│   ├── 0001TP_006690.png
│   ├── 0001TP_006720.png
│   └── ...
└── LabeledApproved_full/        # Segmentation masks (RGB)
    ├── 0001TP_006690_L.png
    ├── 0001TP_006720_L.png
    └── ...
