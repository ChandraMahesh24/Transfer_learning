!pip install  roboflow ultralytics 
# === Step 2: Imports ===
import os
import shutil
import yaml
import matplotlib.pyplot as plt
from roboflow import Roboflow
from ultralytics import YOLO
import torch
from collections import Counter

# === Step 3: Settings ===
ROBOFLOW_API_KEY = "key"  # Replace with your key
ROBOFLOW_WORKSPACE = "transline-technologies"
ROBOFLOW_PROJECT = "muthoot-qcon8"
ROBOFLOW_VERSION = 3

DATASET_DIR = "/kaggle/working/dataset"
FINAL_DATASET_DIR = "/kaggle/working/dataset_final"
DATA_YAML_PATH = os.path.join(FINAL_DATASET_DIR, "data.yaml")

FINAL_CLASS_NAMES = ["backpack", "handbag", "helmet", "person", "suitcase", "without helmet"]

# === Step 4: Download Dataset from Roboflow ===
print("📥 Downloading dataset from Roboflow...")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
version = project.version(ROBOFLOW_VERSION)
dataset = version.download("yolov8", location=DATASET_DIR)
print(f"✅ Dataset downloaded to: {DATASET_DIR}")

# === Step 5: Copy Dataset (No Remapping) ===
print("📦 Copying dataset...")

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

# === Step 6: Check and Visualize Class Distribution ===
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
    return counts

train_labels = os.path.join(FINAL_DATASET_DIR, "train", "labels")
counts = print_class_distribution(train_labels)

# Visualization
plt.bar([FINAL_CLASS_NAMES[i] for i in counts.keys()], counts.values())
plt.xticks(rotation=45)
plt.title("Class Distribution in Train Set")
plt.show()

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

# === Step 8: Two-Stage Transfer Learning ===
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8s.pt")

# === Phase 1: Freeze backbone ===
# === Phase 1: Freeze backbone ===
print("🚀 Phase 1: Training with frozen backbone...")
model.train(
    data=DATA_YAML_PATH,
    epochs=10,
    freeze=5,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=1e-3,
    weight_decay=0.001,
    label_smoothing=0.1,
    mosaic=1.0,
    mixup=0.4,
    copy_paste=0.3,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    device=device,
    project="/kaggle/working/runs/train",
    name="custom_transfer",
    seed=42
)

# === Phase 2: Finetune entire model ===
print("🔁 Phase 2: Finetuning entire model...")

# ✅ Load best.pt from phase 1
best_model_path = "/kaggle/working/runs/train/custom_transfer/weights/best.pt"
model = YOLO(best_model_path)  # ← Load trained model

model.train(
    data=DATA_YAML_PATH,
    epochs=30,        # Total desired epochs for Phase 2
    freeze=0,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=5e-4,
    weight_decay=0.0005,
    device=device,
    project="/kaggle/working/runs/train",
    name="custom_transfer_phase2"
)


print("✅ Training complete!")

# === Step 9: Validate Best Model ===
best_model_path = "/kaggle/working/runs/train/custom_transfer/weights/best.pt"
if os.path.exists(best_model_path):
    print("📊 Validating best model...")
    best_model = YOLO(best_model_path)
    metrics = best_model.val()

    # Print useful metrics
    print(f"📈 mAP50: {metrics.box.map50:.4f}")
    print(f"📈 mAP50-95: {metrics.box.map:.4f}")
    for idx, ap in enumerate(metrics.box.ap):
        print(f" - {FINAL_CLASS_NAMES[idx]} AP: {ap:.4f}")
else:
    print("❌ best.pt not found. Skipping validation.")

# === Step 10: (Optional) Export Trained Model ===
export_path = best_model.export(format="onnx")
print(f"📦 Exported model to ONNX format: {export_path}")
