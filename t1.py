!pip install roboflow ultralytics 
# === Step 2: Imports ===
import os
import shutil
import yaml
import matplotlib.pyplot as plt
import numpy as np
from roboflow import Roboflow
from ultralytics import YOLO
import torch
from collections import Counter
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split

# === Step 3: Settings ===
ROBOFLOW_API_KEY = "api"  # Replace with your key
ROBOFLOW_WORKSPACE = ""
ROBOFLOW_PROJECT = ""
ROBOFLOW_VERSION = 5
YOLO_FORMAT = "yolov8"

DATASET_DIR = "/kaggle/working/dataset"
FINAL_DATASET_DIR = "/kaggle/working/dataset_final"

# === Step 4: Dataset Download ===
print(f"📥 Downloading dataset from Roboflow: {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION}...")
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download(YOLO_FORMAT)
    DATA_YAML_PATH = os.path.join(dataset.location, "data.yaml")
    print(f"✅ Dataset downloaded to: {dataset.location}")
except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    exit()

# === Step 5: Validate data.yaml for Class Names ===
try:
    with open(DATA_YAML_PATH, 'r') as file:
        data_yaml = yaml.safe_load(file)

    num_classes = data_yaml.get('nc', 0)
    class_names = data_yaml.get('names', [])
    print(f"📊 Number of classes: {num_classes}")
    print(f"📊 Class names: {class_names}")
    
    if num_classes == 0 or not class_names:
        print("❌ Error: No classes found in data.yaml. Check Roboflow dataset configuration.")
        exit()
        
    # Create class name mapping
    FINAL_CLASS_NAMES = class_names
    print(f"✅ Using class names from dataset: {FINAL_CLASS_NAMES}")
    
except Exception as e:
    print(f"❌ Error reading data.yaml: {e}")
    exit()

# === Step 6: Enhanced Dataset Preparation ===
print("🔄 Preparing dataset structure with enhancements...")

# Create final dataset directory structure
os.makedirs(FINAL_DATASET_DIR, exist_ok=True)

# Copy and reorganize the dataset
for split in ['train', 'valid', 'test']:
    # Create directories
    os.makedirs(os.path.join(FINAL_DATASET_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(FINAL_DATASET_DIR, 'labels', split), exist_ok=True)
    
    # Copy images and labels
    src_img_dir = os.path.join(dataset.location, split, 'images')
    src_lbl_dir = os.path.join(dataset.location, split, 'labels')
    
    if os.path.exists(src_img_dir):
        for file in os.listdir(src_img_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy2(os.path.join(src_img_dir, file), 
                           os.path.join(FINAL_DATASET_DIR, 'images', split, file))
    
    if os.path.exists(src_lbl_dir):
        for file in os.listdir(src_lbl_dir):
            if file.endswith('.txt'):
                shutil.copy2(os.path.join(src_lbl_dir, file), 
                           os.path.join(FINAL_DATASET_DIR, 'labels', split, file))

# === Step 7: Dataset Analysis and Augmentation Check ===
print("🔍 Analyzing dataset...")

def analyze_dataset():
    """Analyze dataset statistics"""
    stats = {
        'train': {'images': 0, 'labels': 0, 'objects': 0},
        'valid': {'images': 0, 'labels': 0, 'objects': 0},
        'test': {'images': 0, 'labels': 0, 'objects': 0}
    }
    
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(FINAL_DATASET_DIR, 'images', split)
        lbl_dir = os.path.join(FINAL_DATASET_DIR, 'labels', split)
        
        if os.path.exists(img_dir):
            stats[split]['images'] = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if os.path.exists(lbl_dir):
            labels = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
            stats[split]['labels'] = len(labels)
            
            # Count objects
            object_count = 0
            for label_file in labels:
                with open(os.path.join(lbl_dir, label_file), 'r') as f:
                    object_count += len(f.readlines())
            stats[split]['objects'] = object_count
    
    return stats

dataset_stats = analyze_dataset()
print("📊 Dataset Statistics:")
for split, data in dataset_stats.items():
    print(f"   {split.upper()}: {data['images']} images, {data['labels']} labels, {data['objects']} objects")

# === Step 8: Create data.yaml ===
print("📝 Creating data.yaml...")

# Update data.yaml with correct paths
data_yaml['path'] = FINAL_DATASET_DIR
data_yaml['train'] = 'images/train'
data_yaml['val'] = 'images/valid'
data_yaml['test'] = 'images/test'

FINAL_DATA_YAML_PATH = os.path.join(FINAL_DATASET_DIR, "data.yaml")
with open(FINAL_DATA_YAML_PATH, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"✅ data.yaml created at: {FINAL_DATA_YAML_PATH}")

# === Step 9: Optimized Training Configuration ===
print("🚀 Starting OPTIMIZED YOLOv8 training...")

# Use YOLOv8 model (correct model name)
MODEL_SIZE = 'n'  # Options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
model = YOLO(f'yolov8{MODEL_SIZE}.pt')

# Advanced training configuration with optimizations
training_config = {
    'data': FINAL_DATA_YAML_PATH,
    'epochs': 80,  # Increased epochs for better convergence
    'imgsz': 640,
    'batch': 8 if MODEL_SIZE in ['l', 'x'] else 16,  # Adjust batch based on model size
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'workers': 4,  # Reduced for stability
    'optimizer': 'AdamW',  # Better optimizer
    'lr0': 0.001,  # Lower initial learning rate
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 7.5,  # Adjust loss weights
    'cls': 0.5,
    'dfl': 1.5,
    'close_mosaic': 10,  # Disable mosaic at the end
    'patience': 100,  # Increased patience
    'save': True,
    'save_period': 25,
    'project': 'yolov8_sackbags_optimized',
    'name': f'sackbags_detection_v2_{MODEL_SIZE}',
    'exist_ok': True,
    'pretrained': True,
    'verbose': True,
    'amp': True,  # Automatic Mixed Precision
    'overlap_mask': True,
    'mask_ratio': 4,
    'dropout': 0.0,
    'val': True,  # Validate during training
    'plots': True,  # Generate plots
    'cos_lr': True,  # Cosine learning rate scheduler
    'label_smoothing': 0.1,  # Regularization
    'nbs': 64,  # Nominal batch size
}

# Add data augmentation based on dataset size
if dataset_stats['train']['images'] < 1000:
    training_config.update({
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,    # HSV-Saturation augmentation
        'hsv_v': 0.4,    # HSV-Value augmentation
        'degrees': 10.0, # Rotation degrees
        'translate': 0.1,# Translation
        'scale': 0.5,    # Scale augmentation
        'shear': 2.0,    # Shear augmentation
        'perspective': 0.0001,  # Perspective
        'flipud': 0.0,   # Flip up-down
        'fliplr': 0.5,   # Flip left-right
        'mosaic': 1.0,   # Mosaic augmentation
        'mixup': 0.0,    # Mixup augmentation
    })
else:
    # Less augmentation for larger datasets
    training_config.update({
        'hsv_h': 0.01,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 5.0,
        'fliplr': 0.3,
    })

print(f"📊 Using YOLOv8{MODEL_SIZE} model with optimized training parameters")

# Train the model
results = model.train(**training_config)

print("✅ Training completed!")

# === Step 10: Enhanced Metrics Visualization ===
print("📊 Plotting enhanced training metrics...")

def plot_enhanced_metrics(results):
    """Plot comprehensive training and validation metrics for YOLOv8"""
    # Read the results.csv file that YOLOv8 generates
    results_file = os.path.join(results.save_dir, 'results.csv')
    
    if not os.path.exists(results_file):
        print("❌ Results CSV file not found. Cannot plot metrics.")
        return
    
    # Read the CSV file
    import pandas as pd
    df = pd.read_csv(results_file)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'YOLOv8{MODEL_SIZE} - Enhanced Training Metrics', fontsize=16, fontweight='bold')
    
    # Loss curves
    if 'train/box_loss' in df.columns:
        axes[0, 0].plot(df['train/box_loss'], label='Train Box Loss', linewidth=2, color='blue')
    if 'val/box_loss' in df.columns:
        axes[0, 0].plot(df['val/box_loss'], label='Val Box Loss', linewidth=2, color='red', linestyle='--')
    axes[0, 0].set_title('Box Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    if 'train/cls_loss' in df.columns:
        axes[0, 1].plot(df['train/cls_loss'], label='Train Cls Loss', linewidth=2, color='green')
    if 'val/cls_loss' in df.columns:
        axes[0, 1].plot(df['val/cls_loss'], label='Val Cls Loss', linewidth=2, color='orange', linestyle='--')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    if 'train/dfl_loss' in df.columns:
        axes[0, 2].plot(df['train/dfl_loss'], label='Train DFL Loss', linewidth=2, color='purple')
    if 'val/dfl_loss' in df.columns:
        axes[0, 2].plot(df['val/dfl_loss'], label='Val DFL Loss', linewidth=2, color='brown', linestyle='--')
    axes[0, 2].set_title('DFL Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # Metrics
    if 'metrics/precision(B)' in df.columns:
        axes[1, 0].plot(df['metrics/precision(B)'], label='Precision', linewidth=2, color='orange')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    if 'metrics/recall(B)' in df.columns:
        axes[1, 1].plot(df['metrics/recall(B)'], label='Recall', linewidth=2, color='purple')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    if 'metrics/mAP50(B)' in df.columns:
        axes[1, 2].plot(df['metrics/mAP50(B)'], label='mAP50', linewidth=2, color='brown')
    axes[1, 2].set_title('mAP50')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    # Learning rate
    if 'lr/pg0' in df.columns:
        axes[2, 0].plot(df['lr/pg0'], label='Learning Rate', linewidth=2, color='magenta')
        axes[2, 0].set_title('Learning Rate Schedule')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('LR')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].legend()
    
    # mAP50-95
    if 'metrics/mAP50-95(B)' in df.columns:
        axes[2, 1].plot(df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2, color='cyan')
        axes[2, 1].set_title('mAP50-95')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Value')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend()
    
    # Empty subplot for future use
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('enhanced_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_enhanced_metrics(results)

# === Step 11: Advanced Validation ===
print("📊 Running advanced validation...")

# Load the best model
best_model_path = os.path.join('yolov8_sackbags_optimized', f'sackbags_detection_v2_{MODEL_SIZE}', 'weights', 'best.pt')
model = YOLO(best_model_path)

# Run comprehensive validation
metrics = model.val(
    data=FINAL_DATA_YAML_PATH,
    split='test',  # Validate on test set
    conf=0.001,    # Low confidence threshold for comprehensive evaluation
    iou=0.6,       # Standard IoU threshold
    plots=True,    # Generate validation plots
    save_json=True,# Save JSON results
    save_hybrid=True
)

print("📈 Validation Results:")
print(f"   mAP50-95: {metrics.box.map:.4f}")
print(f"   mAP50: {metrics.box.map50:.4f}")
print(f"   Precision: {metrics.box.mp:.4f}")
print(f"   Recall: {metrics.box.mr:.4f}")
print(f"   F1 Score: {2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-16):.4f}")

# === Step 12: Confidence Threshold Optimization ===
print("⚙️ Optimizing confidence threshold...")

def optimize_confidence_threshold(model, data_yaml_path):
    """Find optimal confidence threshold"""
    conf_thresholds = np.arange(0.1, 0.9, 0.1)
    f1_scores = []
    
    for conf in conf_thresholds:
        temp_metrics = model.val(data=data_yaml_path, conf=conf, iou=0.5, plots=False, verbose=False)
        precision = temp_metrics.box.mp
        recall = temp_metrics.box.mr
        f1 = 2 * (precision * recall) / (precision + recall + 1e-16)
        f1_scores.append(f1)
    
    optimal_conf = conf_thresholds[np.argmax(f1_scores)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(conf_thresholds, f1_scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(optimal_conf, color='red', linestyle='--', label=f'Optimal: {optimal_conf:.2f}')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('Confidence Threshold Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('confidence_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_conf

optimal_conf = optimize_confidence_threshold(model, FINAL_DATA_YAML_PATH)
print(f"🎯 Optimal confidence threshold: {optimal_conf:.3f}")

# === Step 13: Enhanced Test Inference ===
print("🧪 Running enhanced inference tests...")

test_image_dir = os.path.join(FINAL_DATASET_DIR, 'images', 'test')
if os.path.exists(test_image_dir) and os.listdir(test_image_dir):
    # Test on multiple images
    test_images = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))][:3]  # Test on first 3 images
    
    for test_image in test_images:
        print(f"\n🔍 Testing on: {os.path.basename(test_image)}")
        
        # Run inference with optimal confidence
        results = model(test_image, conf=optimal_conf, save=True, save_txt=True)
        
        for result in results:
            if result.boxes is not None:
                print(f"   Detected {len(result.boxes)} objects")
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0].item())
                    confidence = box.conf[0].item()
                    bbox = box.xyxy[0].cpu().numpy()
                    print(f"      {FINAL_CLASS_NAMES[class_id]}: conf={confidence:.3f}, bbox={bbox}")
            else:
                print("   ❌ No objects detected")
        
        # Display results
        result_image_path = os.path.join('runs', 'detect', 'predict', os.path.basename(test_image))
        if os.path.exists(result_image_path):
            img = plt.imread(result_image_path)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title(f'Detection Results - {os.path.basename(test_image)}')
            plt.axis('off')
            plt.show()

# === Step 14: Model Export and Deployment ===
print("💾 Exporting model for deployment...")

# Save the final model
model.save('yolov8_sackbags_optimized_final.pt')

# Export to multiple formats
export_formats = ['onnx', 'engine']  # Add 'tflite' if needed
for fmt in export_formats:
    try:
        model.export(format=fmt)
        print(f"✅ Exported to {fmt.upper()} format")
    except Exception as e:
        print(f"⚠️  Could not export to {fmt}: {e}")

# === Step 15: Comprehensive Summary ===
print("\n" + "="*60)
print("🎉 OPTIMIZED TRAINING SUMMARY")
print("="*60)
print(f"📊 Model: YOLOv8{MODEL_SIZE}")
print(f"📊 Final mAP50: {metrics.box.map50:.4f}")
print(f"📊 Final mAP50-95: {metrics.box.map:.4f}")
print(f"📊 Precision: {metrics.box.mp:.4f}")
print(f"📊 Recall: {metrics.box.mr:.4f}")
print(f"🎯 Optimal confidence threshold: {optimal_conf:.3f}")
print(f"📁 Model saved: yolov8_sackbags_optimized_final.pt")
print(f"📁 Training logs: yolov8_sackbags_optimized/sackbags_detection_v2_{MODEL_SIZE}/")
print("="*60)

print("🎉 All optimized steps completed successfully!")
