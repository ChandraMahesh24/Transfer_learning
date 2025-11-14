"""
===========================================================================
 🚀 HYBRID YOLOv8 MODEL TRAINING SCRIPT
 COCO Pretrained + Custom Dataset (Roboflow)
 Classes: backpack, handbag, helmet, person, suitcase, without helmet
===========================================================================

 This script performs:

 1. Download dataset from Roboflow
 2. Prepare YOLOv8 dataset structure
 3. Auto-generate data.yaml
 4. Load COCO-pretrained YOLOv8 model
 5. Hybrid training (freeze + fine-tune)
 6. Plot training metrics
 7. Evaluate model on test set
 8. Compute optimal confidence threshold
 9. Compare COCO vs Custom (visual comparison)
10. Export final model (.pt)

 Perfect for GitHub repositories & reproducible research.
"""

# ========================================================================
# 🛠 INSTALL REQUIRED LIBRARIES
# ========================================================================
!pip install ultralytics roboflow albumentations torch torchvision opencv-python seaborn matplotlib --quiet


# ========================================================================
# 📦 IMPORT LIBRARIES
# ========================================================================
import os
import yaml
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from roboflow import Roboflow
from IPython.display import display, Image


# ========================================================================
# ⚙️ CONFIGURATION
# ========================================================================
ROBOFLOW_API_KEY = "<YOUR_API_KEY>"
ROBOFLOW_WORKSPACE = "<YOUR_WORKSPACE>"
ROBOFLOW_PROJECT = "<YOUR_PROJECT>"
ROBOFLOW_VERSION = 8

FINAL_DATASET_DIR = "/kaggle/working/dataset_final"    # Final dataset path

MODEL_SIZE = "m"                                       # Options: n, s, m, l, x

# Custom classes
TARGET_CLASSES = [
    'backpack',
    'handbag',
    'helmet',
    'person',
]


# ========================================================================
# 📥 DOWNLOAD DATASET FROM ROBOFLOW
# ========================================================================
print("📥 Downloading dataset from Roboflow...")

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
version = project.version(ROBOFLOW_VERSION)
dataset = version.download("yolov8")

DATASET_PATH = dataset.location
print(f"✅ Dataset downloaded at: {DATASET_PATH}")


# ========================================================================
# 📂 PREPARE YOLOv8 DATASET STRUCTURE
# ========================================================================
def prepare_dataset():
    """
    Copies images & labels from Roboflow → YOLO folder structure.

       dataset_final/
          images/train
          images/valid
          images/test
          labels/train
          labels/valid
          labels/test
    """
    os.makedirs(FINAL_DATASET_DIR, exist_ok=True)

    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(FINAL_DATASET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(FINAL_DATASET_DIR, 'labels', split), exist_ok=True)

        src_img = os.path.join(DATASET_PATH, split, 'images')
        src_lbl = os.path.join(DATASET_PATH, split, 'labels')

        # Copy images
        for f in os.listdir(src_img):
            shutil.copy2(
                os.path.join(src_img, f),
                os.path.join(FINAL_DATASET_DIR, 'images', split, f)
            )

        # Copy label files
        for f in os.listdir(src_lbl):
            shutil.copy2(
                os.path.join(src_lbl, f),
                os.path.join(FINAL_DATASET_DIR, 'labels', split, f)
            )

    print("✅ Dataset structure prepared successfully!")


prepare_dataset()


# ========================================================================
# 📝 CREATE data.yaml (Auto-generated)
# ========================================================================
def create_final_yaml():
    """
    Creates YOLOv8 data.yaml dynamically using custom classes.
    """

    print("📝 Creating data.yaml...")

    yaml_dict = {
        'path': FINAL_DATASET_DIR,
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'names': TARGET_CLASSES,
        'nc': len(TARGET_CLASSES)
    }

    yaml_path = os.path.join(FINAL_DATASET_DIR, 'data.yaml')

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)

    print(f"✅ data.yaml created at: {yaml_path}")
    return yaml_path


FINAL_DATA_YAML_PATH = create_final_yaml()


# ========================================================================
# 🔄 LOAD COCO-PRETRAINED YOLOv8 MODEL
# ========================================================================
print("\n🔄 Loading COCO-pretrained YOLOv8 model...")

model = YOLO(f"yolov8{MODEL_SIZE}.pt")

print(f"✅ Loaded model: YOLOv8{MODEL_SIZE} ({len(model.names)} COCO classes)")


# ========================================================================
# ⚙️ HYBRID TRAINING CONFIGURATION
# ========================================================================
training_config = {

    # Data
    'data': FINAL_DATA_YAML_PATH,
    'epochs': 30,
    'imgsz': 640,
    'batch': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Optimization settings
    'pretrained': True,
    'freeze': 10,
    'optimizer': 'AdamW',
    'lr0': 0.002,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'cos_lr': True,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,

    # Data augmentation
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 2.0,
    'perspective': 0.0005,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.2,
    'copy_paste': 0.1,

    # Regularization
    'label_smoothing': 0.1,
    'rect': False,
    'amp': True,
    'patience': 20,

    # Logging
    'save': True,
    'save_period': 10,
    'plots': True,
    'project': 'hybrid_yolov8_training',
    'name': f'hybrid_focus6_{MODEL_SIZE}',
    'exist_ok': True,
    'verbose': True
}


# ========================================================================
# 🚀 TRAIN THE MODEL
# ========================================================================
print("\n🎯 Starting hybrid training (COCO pretrained + custom dataset)...")

results = model.train(**training_config)

print("\n✅ Training completed!")


# ========================================================================
# 📊 PLOT TRAINING METRICS
# ========================================================================
print("\n📈 Generating training graphs...")

results_csv = os.path.join(results.save_dir, "results.csv")

if os.path.exists(results_csv):

    df = pd.read_csv(results_csv)

    # Prepare 6-subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("YOLOv8 Training Metrics", fontsize=16, fontweight="bold")

    metrics = [
        ('train/box_loss', 'val/box_loss', 'Box Loss'),
        ('train/cls_loss', 'val/cls_loss', 'Classification Loss'),
        ('metrics/precision(B)', 'metrics/recall(B)', 'Precision & Recall'),
        ('metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'mAP50 / mAP50-95'),
        ('lr/pg0', None, 'Learning Rate'),
        ('train/obj_loss', 'val/obj_loss', 'Objectness Loss')
    ]

    # Plot each metric
    for ax, (m1, m2, title) in zip(axes.flatten(), metrics):
        if m1 in df.columns:
            ax.plot(df[m1], label=m1.split('/')[-1], lw=2)

        if m2 and m2 in df.columns:
            ax.plot(df[m2], label=m2.split('/')[-1], lw=2, ls='--')

        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

else:
    print("⚠️ results.csv not found → metrics skipped.")


# ========================================================================
# 🧪 VALIDATION ON TEST SET
# ========================================================================
print("\n📊 Running final validation...")

metrics = model.val(
    data=FINAL_DATA_YAML_PATH,
    split='test',
    conf=0.25,
    iou=0.6,
    plots=True
)

print("\n📈 FINAL VALIDATION METRICS")
print("=" * 60)
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")

f1 = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-16)
print(f"F1 Score  : {f1:.4f}")
print("=" * 60)


# ========================================================================
# 🎯 OPTIMIZE CONFIDENCE THRESHOLD
# ========================================================================
def optimize_confidence(model):
    """
    Tests confidence thresholds (0.1–0.9) and selects the best F1 score.
    """

    confs = np.arange(0.1, 0.9, 0.1)
    f1_scores = []

    for conf in confs:
        m = model.val(
            data=FINAL_DATA_YAML_PATH,
            conf=conf,
            iou=0.5,
            plots=False,
            verbose=False
        )

        p, r = m.box.mp, m.box.mr
        f1_val = 2 * (p * r) / (p + r + 1e-16)
        f1_scores.append(f1_val)

    best_conf = confs[np.argmax(f1_scores)]

    # Plot F1 curve
    plt.figure(figsize=(8, 5))
    plt.plot(confs, f1_scores, 'bo-', lw=2)
    plt.axvline(best_conf, color='r', ls='--', label=f'Best Conf = {best_conf:.2f}')
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score")
    plt.title("Optimal Confidence Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"🎯 Best Confidence Threshold: {best_conf:.3f}")
    return best_conf


best_conf = optimize_confidence(model)


# ========================================================================
# 🧩 VISUAL COMPARISON: COCO MODEL VS HYBRID MODEL
# ========================================================================
print("\n🧩 Comparing COCO vs Fine-tuned model...")

coco_model = YOLO(f"yolov8{MODEL_SIZE}.pt")

test_dir = os.path.join(FINAL_DATASET_DIR, 'images', 'test')
test_images = [
    os.path.join(test_dir, f)
    for f in os.listdir(test_dir)
    if f.endswith(('.jpg', '.png'))
][:3]   # Limit to first 3 images

for img_path in test_images:

    print(f"\n🔍 Image: {os.path.basename(img_path)}")

    base_pred = coco_model(img_path, conf=0.25, save=True, project='compare_coco', name='pred')
    hybrid_pred = model(img_path, conf=best_conf, save=True, project='compare_hybrid', name='pred')

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(plt.imread(base_pred[0].save_dir / os.path.basename(img_path)))
    axes[0].set_title("COCO Base Model")

    axes[1].imshow(plt.imread(hybrid_pred[0].save_dir / os.path.basename(img_path)))
    axes[1].set_title("Hybrid Fine-tuned Model")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ========================================================================
# 💾 SAVE FINAL MODEL
# ========================================================================
FINAL_MODEL_PATH = "yolov8_hybrid_focus6_final.pt"
model.save(FINAL_MODEL_PATH)

print("\n💾 Final hybrid model saved at:", FINAL_MODEL_PATH)
print("\n🎉 Training complete!")
print("Your YOLOv8 Hybrid Model includes:")
print(" ✔ COCO Pretrained Knowledge")
print(" ✔ Custom Dataset Fine-tuning")
print(" ✔ Full Visualization + Comparison Tools")
