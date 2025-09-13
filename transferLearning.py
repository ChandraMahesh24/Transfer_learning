# === Step 1: Install Dependencies (if needed) ===
!pip install roboflow ultralytics

# === Step 2: Imports ===
import os
import shutil
import yaml
from roboflow import Roboflow
from ultralytics import YOLO
import torch
from collections import Counter

# === Step 3: Settings ===
ROBOFLOW_API_KEY = "h9neD8fbB62pwYy8wykA"  # Replace with your key
ROBOFLOW_WORKSPACE = "transline-technologies"
ROBOFLOW_PROJECT = "muthoot-qcon8"
ROBOFLOW_VERSION = 2

DATASET_DIR = "/kaggle/working/dataset"
FINAL_DATASET_DIR = "/kaggle/working/dataset_final"
DATA_YAML_PATH = os.path.join(FINAL_DATASET_DIR, "data.yaml")

FINAL_CLASS_NAMES = ["backpack", "helmet", "person", "without helmet"]

# === Step 4: Download Dataset from Roboflow ===
print("📥 Downloading dataset from Roboflow...")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
version = project.version(ROBOFLOW_VERSION)
dataset = version.download("yolov8", location=DATASET_DIR)
print(f"✅ Dataset downloaded to: {DATASET_DIR}")

# === Step 5: Copy Dataset as-is (No Remapping) ===
print("📦 Copying dataset (no remapping needed)...")

splits = ['train', 'valid', 'test']
for split in splits:
    src_label_dir = os.path.join(DATASET_DIR, split, "labels")
    dst_label_dir = os.path.join(FINAL_DATASET_DIR, split, "labels")
    src_img_dir = os.path.join(DATASET_DIR, split, "images")
    dst_img_dir = os.path.join(FINAL_DATASET_DIR, split, "images")

    if not os.path.exists(src_label_dir):
        print(f"⚠️ Skipping {split} - no labels found")
        continue

    os.makedirs(dst_label_dir, exist_ok=True)
    os.makedirs(dst_img_dir, exist_ok=True)

    shutil.copytree(src_label_dir, dst_label_dir, dirs_exist_ok=True)
    shutil.copytree(src_img_dir, dst_img_dir, dirs_exist_ok=True)
    print(f"✅ Copied: {split}")

# === Step 6: Check Class Distribution (Debugging Step) ===
def print_class_distribution(label_folder):
    all_classes = []
    for file in os.listdir(label_folder):
        if file.endswith(".txt"):
            with open(os.path.join(label_folder, file), 'r') as f:
                for line in f:
                    cls_id = int(line.strip().split()[0])
                    all_classes.append(cls_id)
    counts = Counter(all_classes)
    print(f"📊 Class distribution in {label_folder}: {counts}")

print_class_distribution(os.path.join(FINAL_DATASET_DIR, "train", "labels"))

# === Step 7: Create data.yaml ===
yaml_data = {
    "train": os.path.join(FINAL_DATASET_DIR, "train", "images"),
    "val": os.path.join(FINAL_DATASET_DIR, "valid", "images"),
    "nc": len(FINAL_CLASS_NAMES),
    "names": FINAL_CLASS_NAMES
}

with open(DATA_YAML_PATH, 'w') as f:
    yaml.dump(yaml_data, f)
print(f"✅ data.yaml created at {DATA_YAML_PATH}")

# === Step 8: Train YOLOv8 with Transfer Learning ===
print("🚀 Starting YOLOv8 training (transfer learning)...")

model = YOLO("yolov8n.pt")  # Use yolov8n, yolov8s, etc.

device = 0 if torch.cuda.is_available() else "cpu"

model.train(
    data=DATA_YAML_PATH,
    epochs=10,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=1e-3,
    lrf=0.01,
    warmup_epochs=5,
    weight_decay=0.001,
    label_smoothing=0.1,
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.1,
    freeze=10,  # Freeze backbone layers for transfer learning
    device=device,
    seed=42,
    project="/kaggle/working/runs/train",
    name="custom_no_remap"
)

print("✅ Training complete!")

# === Step 9: Validate Best Model ===
best_model_path = "/kaggle/working/runs/train/custom_no_remap/weights/best.pt"
if os.path.exists(best_model_path):
    print("📊 Validating best model...")
    best_model = YOLO(best_model_path)
    metrics = best_model.val()
    print("📈 Validation metrics:", metrics)
else:
    print("❌ best.pt not found. Skipping validation.")
