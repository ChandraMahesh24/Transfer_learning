# 🚀 Hybrid YOLOv8 Training — Custom Dataset + COCO Pretrained Model

This repository provides a complete YOLOv8 hybrid training pipeline that combines:

- COCO-pretrained YOLOv8 model  
- Custom dataset from Roboflow  
- Advanced data augmentation  
- Fine-tuning with optimized hyperparameters  
- Full training/validation visualization  

It is ideal for:
- Helmet vs. No-Helmet detection  
- Multi-object human-related detection  
- Backpack/handbag/suitcase recognition  
- Any custom dataset merged with COCO knowledge  

---

# 📂 Project Structure

├── train.py
├── README.md
├── dataset_final/
│ ├── images/
│ ├── labels/
│ ├── data.yaml
└── hybrid_yolov8_training/



---

# 🔧 Installation

```bash
pip install ultralytics roboflow albumentations torch torchvision opencv-python seaborn matplotlib --quiet
```


# Roboflow Setup

  ROBOFLOW_API_KEY = "your-key"
  ROBOFLOW_WORKSPACE = "your-workspace"
  ROBOFLOW_PROJECT = "your-project"
  ROBOFLOW_VERSION = 1


# Training Pipeline Overview

  1. The training pipeline includes:
  2. Automatic dataset download (Roboflow)
  3. YOLO dataset structure builder
  4. Auto-creation of data.yaml
  5. COCO-pretrained YOLOv8 model loading
  6. Hybrid fine-tuning with optimized hyperparameters
  7. Training metrics visualization
  8. Validation + mAP, Precision, Recall evaluation
  9. Final model export

⚙️ YOLOv8 Training Hyperparameters (Explained)

Below is a detailed explanation of every hyperparameter used in this project.
  
  🧠 Training Configuration
  Hyperparameter	Description
  epochs: 30	Total number of full training cycles. More epochs = better accuracy but longer training.
  imgsz: 640	Image size used for training. Higher = more accuracy but more VRAM usage.
  batch: 32	Number of images per batch. Lower if GPU has low VRAM.
  device: cuda/cpu	Automatically selects GPU if available.
```bash
  | Hyperparameter         | Meaning & Reason                                                         |
| ---------------------- | ------------------------------------------------------------------------ |
| `pretrained: True`     | Loads COCO-pretrained weights for faster, better training.               |
| `freeze: 10`           | Freezes first 10 layers of backbone to keep COCO knowledge intact.       |
| `optimizer: AdamW`     | More stable & smoother convergence than SGD for custom datasets.         |
| `lr0: 0.002`           | Initial learning rate. Higher = faster learning but risk of instability. |
| `lrf: 0.01`            | Final LR fraction for cosine decay. Ensures low LR near end.             |
| `momentum: 0.937`      | Helps optimizer maintain direction.                                      |
| `weight_decay: 0.0005` | Regularization to prevent overfitting.                                   |
| `cos_lr: True`         | Enables cosine learning rate scheduling.                                 |
| `warmup_epochs: 3.0`   | Slow warmup to avoid unstable early training.                            |
| `warmup_momentum: 0.8` | Helps stabilize gradient updates during warmup.                          |


| Hyperparameter        | Description                                       |
| --------------------- | ------------------------------------------------- |
| `hsv_h: 0.015`        | Hue adjustment to handle color variations.        |
| `hsv_s: 0.7`          | Saturation shift for lighting variations.         |
| `hsv_v: 0.4`          | Brightness variations (day/night).                |
| `degrees: 10.0`       | Small rotations to generalize object orientation. |
| `translate: 0.1`      | Random shifting of objects in the frame.          |
| `scale: 0.5`          | Zoom-in/out scaling for size robustness.          |
| `shear: 2.0`          | Slight shear distortion improves robustness.      |
| `perspective: 0.0005` | Very small 3D perspective distortion.             |
| `fliplr: 0.5`         | 50% chance of horizontal flip (mirroring).        |
| `mosaic: 1.0`         | Mixes 4 images into one; powerful augmentation.   |
| `mixup: 0.2`          | Blends two images for increased robustness.       |
| `copy_paste: 0.1`     | Copies objects between images for balancing.      |



| Hyperparameter         | Explanation                                          |
| ---------------------- | ---------------------------------------------------- |
| `label_smoothing: 0.1` | Reduces overconfidence; helps if classes overlap.    |
| `rect: False`          | Uses square training images for better batch mixing. |
| `amp: True`            | Enables mixed precision (faster training).           |
| `patience: 20`         | Early stopping patience threshold.                   |


| Parameter                         | Meaning                              |
| --------------------------------- | ------------------------------------ |
| `save: True`                      | Saves best and last model.           |
| `plots: True`                     | Saves training graphs automatically. |
| `project: hybrid_yolov8_training` | Folder for saving logs.              |
| `name: hybrid_focus6_m`           | Training run name.                   |
| `exist_ok: True`                  | Overwrite allowed.                   |
| `verbose: True`                   | Detailed console logs.               |
 ```
📈 Output Metrics

After training, the script prints:
  mAP50
  mAP50-95
  Precision
  Recall
  Confusion matrix
  PR curve
  Loss curves
  
```bash
python train.py
```

```bash
from ultralytics import YOLO
model = YOLO("yolov8_hybrid_focus6_final.pt")
res = model("test.jpg", save=True)
```

```bash
yolov8_hybrid_focus6_final.pt
```

      
