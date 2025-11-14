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




# 📘 **Detailed Explanation of YOLOv8 Hyperparameters (Why, When, Importance)**

YOLO models depend heavily on hyperparameters. Each one controls part of the training process such as optimization, augmentation, or regularization. Understanding them helps you **improve accuracy**, **prevent overfitting**, and **stabilize training**.


---

# 🧠 1. **Core Training Hyperparameters**

---

## **1.1 epochs: 30**

**What it is:**
Number of full passes through the entire dataset.

**Why it's important:**
More epochs → more learning.
But too many epochs → overfitting.

**When to increase:**

* Small datasets
* Slow learning
* Underfitting

**When to decrease:**

* Large datasets
* Overfitting (validation loss increases)

---

## **1.2 imgsz: 640**

**What it does:**
Resizes images to 640×640 for training.

**Why important:**
Higher resolution → detects small objects better, but uses more GPU memory.

**Increase when:**

* Objects are small (helmet, phone, etc.)

**Decrease when:**

* You have low VRAM
* Training is too slow

---

## **1.3 batch: 32**

**What it means:**
Number of images processed together per step.

**Why important:**
Large batch = stable gradients and faster training
Small batch = noisy gradients but uses less VRAM

**Increase when:**

* GPU has high VRAM
* You want smoother training

**Decrease when:**

* RAM/GPU issues
* Out-of-memory errors

---

## **1.4 device: "cuda" / "cpu"**

Automatically uses GPU if available.

**Importance:**
Training YOLO on CPU is extremely slow. GPU highly recommended.

---

# 🔧 2. **Optimization Hyperparameters**

---

## **2.1 pretrained: True**

**What it means:**
Starts training using COCO-pretrained YOLO weights.

**Why important:**

* Faster convergence
* Needs less data
* Higher accuracy
* Better generalization

**Always recommended unless training from scratch.**

---

## **2.2 freeze: 10**

**What it does:**
Freezes the first 10 backbone layers.

**Why important:**

* Keeps COCO features (edges, shapes, textures)
* Prevents catastrophic forgetting
* Allows training to focus on NEW classes

**Increase if:**

* Dataset is small
* Dataset is similar to COCO

**Set to 0 if:**

* Dataset very different from COCO (satellite images, medical, etc.)

---

## **2.3 optimizer: "AdamW"**

**What it means:**
Optimization algorithm used to update weights.

**Why AdamW?**

AdamW advantages:

* Stable training
* Faster convergence
* Reduces overfitting
* Better for small datasets

YOLO default is SGD, but **AdamW performs better on custom datasets**.

---

## **2.4 lr0: 0.002**

Initial learning rate.

**Importance:**
Controls the speed of learning.

* Too high → unstable
* Too low → slow learning

**Good for AdamW.**

---

## **2.5 lrf: 0.01**

Final learning rate fraction for cosine decay.

**Meaning:**
At end of training, LR = lr0 × lrf.

**Why:**
Prevents model from making large changes late in training.

---

## **2.6 momentum: 0.937**

Momentum controls how much past gradient direction affects current updates.

**Why important:**
Higher momentum → smoother movement
Lower momentum → noisy updates

---

## **2.7 weight_decay: 0.0005**

Regularization term to prevent overfitting.

**Helps by:**

* Reducing large weight values
* Improving generalization

---

## **2.8 cos_lr: True**

Cosine Learning Rate Scheduler.

**Why important:**

* Starts with rapid learning
* Slowly reduces LR
* Helps reach stable, optimal point

This logic mimics human learning (fast at first, slow at end).

---

## **2.9 warmup_epochs: 3**

Gradually increases LR for first few epochs.

**Why:**
If LR starts too high → model diverges.

Warmup stabilizes training early on.

---

## **2.10 warmup_momentum: 0.8**

Low momentum at start → smoother warmup process.

---

# 🖼 3. **Data Augmentation Hyperparameters**

Augmentation helps the model generalize to more real-world conditions.

---

## **3.1 hsv_h, hsv_s, hsv_v**

Color augmentations.

| Param | Meaning    | Why Important                        |
| ----- | ---------- | ------------------------------------ |
| hsv_h | Hue        | Lighting changes, color-shifts       |
| hsv_s | Saturation | Handles dull or oversaturated images |
| hsv_v | Brightness | Day/night variations                 |

Makes the model robust to real-world lighting.

---

## **3.2 degrees: 10.0**

Small rotation.

* Helps detect rotated objects.
* Useful for helmets backpack, human objects.

---

## **3.3 translate: 0.1**

Moves objects around image slightly.

**Why:**
Prevents overfitting to exact object position.

---

## **3.4 scale: 0.5**

Zoom-in / zoom-out randomly.

**Why:**
Teaches model to recognize objects at different distances.

---

## **3.5 shear: 2.0**

Slight geometric distortion.

**Why:**
Provides robustness to angled camera views.

---

## **3.6 perspective: 0.0005**

3D tilt applied to images.

**Why:**
Improves detection on security/CCTV footage with perspective distortion.

---

## **3.7 fliplr: 0.5**

50% chance horizontal flip.

**Why:**
Doubles dataset variations (left-right).

---

## **3.8 mosaic: 1.0**

Combines 4 images into 1.

**Most powerful augmentation in YOLO.**

Advantages:

* Helps detect small objects
* Improves scene understanding
* Increases dataset diversity

---

## **3.9 mixup: 0.2**

Blends two images.

**Use when:**
Dataset is small
Classes have imbalance

---

## **3.10 copy_paste: 0.1**

Copies objects between images.

**Why:**

* Good for class balancing
* Useful when some classes appear rarely

---

# 🧩 4. **Regularization Hyperparameters**

---

## **4.1 label_smoothing: 0.1**

Smooths labels (instead of hard 1 or 0).

**Why:**

* Prevents overconfidence
* Helps overlapping classes (person vs. backpack)

---

## **4.2 rect: False**

If True → rectangular training
If False → square training

Square training improves batch mixing.

---

## **4.3 amp: True**

Automatic mixed precision training.

**Why:**

* Faster training
* Less VRAM usage
* No accuracy loss

---

## **4.4 patience: 20**

Used for early stopping.

**If validation stops improving:**
Training ends automatically.

**Prevents overfitting.**

---

# 🗂 5. **Logging & Saving**

---

## **save: True**

Stores:

* Last model
* Best model
* Training artifacts

---

## **plots: True**

Saves:

* Loss curves
* mAP curves
* PR curves
* Confusion matrices

These are essential for understanding model performance.

---

## **project & name**

Defines where training outputs are stored.

---

# 🎯 Summary — Why Hyperparameters Matter

Hyperparameters control:

### ✔ How fast the model learns

### ✔ How much it generalizes

### ✔ How it handles variations in data

### ✔ How stable and accurate the training becomes

### ✔ How much the model remembers from COCO

A correct hyperparameter setup:

* Reduces overfitting
* Increases mAP
* Speeds up training
* Produces stable, reliable detection in real-world scenarios



      
