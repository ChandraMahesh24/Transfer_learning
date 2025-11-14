!pip install ultralytics roboflow albumentations torch torchvision opencv-python seaborn matplotlib --quiet

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

# =====================================================
# ✅ CONFIGURATION
# =====================================================
ROBOFLOW_API_KEY = "Api"
ROBOFLOW_WORKSPACE = ""
ROBOFLOW_PROJECT = ""
ROBOFLOW_VERSION = 8

FINAL_DATASET_DIR = "/kaggle/working/dataset_final"
MODEL_SIZE = "m"  # choose n, s, m, l, x depending on GPU power

TARGET_CLASSES = [
    'backpack',
    'handbag',
    'helmet',
    'person',
    'suitcase',
    'without helmet'
    
]

# =====================================================
# ✅ DOWNLOAD DATASET FROM ROBOFLOW
# =====================================================
print("📥 Downloading dataset from Roboflow...")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
version = project.version(ROBOFLOW_VERSION)
dataset = version.download("yolov8")
DATASET_PATH = dataset.location
print(f"✅ Dataset downloaded at: {DATASET_PATH}")

# =====================================================
# ✅ PREPARE YOLOv8 DATASET STRUCTURE
# =====================================================
def prepare_dataset():
    os.makedirs(FINAL_DATASET_DIR, exist_ok=True)
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(FINAL_DATASET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(FINAL_DATASET_DIR, 'labels', split), exist_ok=True)
        src_img = os.path.join(DATASET_PATH, split, 'images')
        src_lbl = os.path.join(DATASET_PATH, split, 'labels')
        for f in os.listdir(src_img):
            shutil.copy2(os.path.join(src_img, f), os.path.join(FINAL_DATASET_DIR, 'images', split, f))
        for f in os.listdir(src_lbl):
            shutil.copy2(os.path.join(src_lbl, f), os.path.join(FINAL_DATASET_DIR, 'labels', split, f))
    print("✅ Dataset structure prepared successfully!")

prepare_dataset()

# =====================================================
# ✅ CREATE data.yaml FOR 6 CLASSES
# =====================================================
def create_final_yaml():
    print("📝 Creating data.yaml for 6 classes...")
    data_yaml = {
        'path': FINAL_DATASET_DIR,
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'names': TARGET_CLASSES,
        'nc': len(TARGET_CLASSES)
    }
    yaml_path = os.path.join(FINAL_DATASET_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"✅ data.yaml created at: {yaml_path}")
    return yaml_path

FINAL_DATA_YAML_PATH = create_final_yaml()

# =====================================================
# ✅ LOAD COCO-PRETRAINED YOLOv8 MODEL
# =====================================================
print("\n🔄 Loading COCO pretrained YOLOv8 model...")
model = YOLO(f"yolov8{MODEL_SIZE}.pt")
print(f"✅ Loaded YOLOv8{MODEL_SIZE} pretrained on COCO ({len(model.names)} classes)")

# =====================================================
# ⚙️ HYBRID TRAINING CONFIGURATION
# =====================================================
training_config = {
    'data': FINAL_DATA_YAML_PATH,
    'epochs': 30,
    'imgsz': 640,
    'batch': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Optimization
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

    # Augmentations
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

# =====================================================
# 🚀 TRAIN MODEL
# =====================================================
print("\n🎯 Starting hybrid training (COCO pretrained + Custom Dataset)...")
results = model.train(**training_config)
print("\n✅ Hybrid fine-tuning completed successfully!")

# =====================================================
# 📊 PLOT TRAINING METRICS
# =====================================================
print("\n📈 Generating training graphs...")
results_dir = results.save_dir
results_csv = os.path.join(results_dir, 'results.csv')

if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"YOLOv8 Hybrid Model Training Metrics ({MODEL_SIZE})", fontsize=16, fontweight="bold")

    metrics = [
        ('train/box_loss', 'val/box_loss', 'Box Loss'),
        ('train/cls_loss', 'val/cls_loss', 'Classification Loss'),
        ('metrics/precision(B)', 'metrics/recall(B)', 'Precision & Recall'),
        ('metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'mAP50 / mAP50-95'),
        ('lr/pg0', None, 'Learning Rate'),
        ('train/obj_loss', 'val/obj_loss', 'Object Loss')
    ]

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
    print("⚠️ Results CSV not found — skipping graphs")

# =====================================================
# 📊 VALIDATION
# =====================================================
print("\n📊 Running validation...")
metrics = model.val(data=FINAL_DATA_YAML_PATH, split='test', conf=0.25, iou=0.6, plots=True)

print("\n📈 FINAL VALIDATION RESULTS:")
print("="*60)
print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
f1 = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-16)
print(f"F1 Score:  {f1:.4f}")
print("="*60)

# =====================================================
# ⚙️ OPTIMAL CONFIDENCE THRESHOLD
# =====================================================
def optimize_confidence(model):
    confs = np.arange(0.1, 0.9, 0.1)
    f1_scores = []
    for conf in confs:
        m = model.val(data=FINAL_DATA_YAML_PATH, conf=conf, iou=0.5, plots=False, verbose=False)
        p, r = m.box.mp, m.box.mr
        f1 = 2 * (p * r) / (p + r + 1e-16)
        f1_scores.append(f1)
    best_conf = confs[np.argmax(f1_scores)]

    plt.figure(figsize=(8,5))
    plt.plot(confs, f1_scores, 'bo-', lw=2)
    plt.axvline(best_conf, color='r', ls='--', label=f'Best Conf: {best_conf:.2f}')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('Optimal Confidence Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"🎯 Best Confidence Threshold: {best_conf:.3f}")
    return best_conf

best_conf = optimize_confidence(model)

# =====================================================
# 🧠 COMPARE COCO MODEL VS HYBRID MODEL VISUALLY
# =====================================================
print("\n🧩 Visual comparison: COCO vs Hybrid model")

coco_model = YOLO(f"yolov8{MODEL_SIZE}.pt")
test_dir = os.path.join(FINAL_DATASET_DIR, 'images', 'test')
test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))][:3]

for img_path in test_images:
    print(f"\n🔍 Image: {os.path.basename(img_path)}")

    # Run COCO base
    coco_result = coco_model(img_path, conf=0.25, save=True, project='compare_coco', name='pred')
    hybrid_result = model(img_path, conf=best_conf, save=True, project='compare_hybrid', name='pred')

    print("✅ Inference complete - showing comparison:")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(plt.imread(coco_result[0].save_dir / os.path.basename(img_path)))
    axes[0].set_title("COCO Base Model")
    axes[1].imshow(plt.imread(hybrid_result[0].save_dir / os.path.basename(img_path)))
    axes[1].set_title("Hybrid Fine-tuned Model")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()

# =====================================================
# 💾 EXPORT FINAL MODEL
# =====================================================
FINAL_MODEL_PATH = "yolov8_hybrid_focus6_final.pt"
model.save(FINAL_MODEL_PATH)
print(f"\n💾 Final hybrid model saved at: {FINAL_MODEL_PATH}")

print("\n🎉 Training and visualization complete!")
print("✅ Your hybrid YOLOv8 model now includes:")
print("   • COCO pretrained knowledge")
print("   • Custom fine-tuned understanding of helmets & no-helmet")
print("   • Training graphs and visual comparisons generated.")
