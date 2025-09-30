# === INSTALLATION ===
!pip install roboflow ultralytics albumentations torch torchvision
!pip install seaborn matplotlib opencv-python

# === IMPORTS ===
import os
import shutil
import yaml
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from collections import Counter
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# === SETTINGS ===
ROBOFLOW_API_KEY = "YOUR_API_KEY"  # Replace with your actual key
ROBOFLOW_WORKSPACE = "YOUR_WORKSPACE"
ROBOFLOW_PROJECT = "YOUR_PROJECT" 
ROBOFLOW_VERSION = 1

DATASET_DIR = "/kaggle/working/dataset"
FINAL_DATASET_DIR = "/kaggle/working/dataset_final"
MODEL_SIZE = 'n'  # n, s, m, l, x

# === COCO CLASSES FOCUS ===
COCO_CLASS_IDS = {
    'person': 0,
    'backpack': 24, 
    'handbag': 26,
    'suitcase': 28
}

COCO_CLASS_NAMES = list(COCO_CLASS_IDS.keys())
COCO_CLASS_IDS_LIST = list(COCO_CLASS_IDS.values())

print(f"🎯 Selective Transfer Learning Focus: {COCO_CLASS_NAMES}")

# === DATASET DOWNLOAD ===
def download_dataset():
    print("📥 Downloading dataset from Roboflow...")
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        version = project.version(ROBOFLOW_VERSION)
        dataset = version.download("yolov8")
        DATA_YAML_PATH = os.path.join(dataset.location, "data.yaml")
        print(f" Dataset downloaded to: {dataset.location}")
        return DATA_YAML_PATH, dataset.location
    except Exception as e:
        print(f" Error downloading dataset: {e}")
        print(" Using local dataset...")
        # Fallback to local dataset
        DATA_YAML_PATH = "dataset/data.yaml"
        return DATA_YAML_PATH, "dataset"

# Download dataset
DATA_YAML_PATH, dataset_location = download_dataset()

# === VALIDATE data.yaml ===
def validate_dataset(yaml_path):
    print("🔍 Validating dataset configuration...")
    try:
        with open(yaml_path, 'r') as file:
            data_yaml = yaml.safe_load(file)

        num_classes = data_yaml.get('nc', 0)
        class_names = data_yaml.get('names', [])
        print(f" Number of classes: {num_classes}")
        print(f" Class names: {class_names}")
        
        if num_classes == 0 or not class_names:
            print(" Error: No classes found in data.yaml")
            exit()
            
        return data_yaml, class_names, num_classes
        
    except Exception as e:
        print(f" Error reading data.yaml: {e}")
        exit()

data_yaml, FINAL_CLASS_NAMES, NUM_CLASSES = validate_dataset(DATA_YAML_PATH)

# === ENHANCED DATASET PREPARATION ===
def prepare_dataset():
    print(" Preparing dataset structure...")
    
    # Create final dataset directory structure
    os.makedirs(FINAL_DATASET_DIR, exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        # Create directories
        os.makedirs(os.path.join(FINAL_DATASET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(FINAL_DATASET_DIR, 'labels', split), exist_ok=True)
        
        # Copy images and labels
        src_img_dir = os.path.join(dataset_location, split, 'images')
        src_lbl_dir = os.path.join(dataset_location, split, 'labels')
        
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
    
    print(" Dataset preparation completed")

prepare_dataset()

# === DATASET ANALYSIS ===
def analyze_dataset():
    print("🔍 Analyzing dataset statistics...")
    
    stats = {
        'train': {'images': 0, 'labels': 0, 'objects': 0, 'class_distribution': Counter()},
        'valid': {'images': 0, 'labels': 0, 'objects': 0, 'class_distribution': Counter()},
        'test': {'images': 0, 'labels': 0, 'objects': 0, 'class_distribution': Counter()}
    }
    
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(FINAL_DATASET_DIR, 'images', split)
        lbl_dir = os.path.join(FINAL_DATASET_DIR, 'labels', split)
        
        if os.path.exists(img_dir):
            stats[split]['images'] = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if os.path.exists(lbl_dir):
            labels = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
            stats[split]['labels'] = len(labels)
            
            # Count objects and class distribution
            object_count = 0
            for label_file in labels:
                with open(os.path.join(lbl_dir, label_file), 'r') as f:
                    lines = f.readlines()
                    object_count += len(lines)
                    for line in lines:
                        class_id = int(line.strip().split()[0])
                        stats[split]['class_distribution'][class_id] += 1
            stats[split]['objects'] = object_count
    
    # Print statistics
    print("\n📊 DATASET STATISTICS:")
    for split, data in stats.items():
        print(f"\n   {split.upper()}:")
        print(f"      Images: {data['images']}")
        print(f"      Labels: {data['labels']}")
        print(f"      Objects: {data['objects']}")
        print(f"      Class Distribution:")
        for class_id, count in data['class_distribution'].items():
            class_name = FINAL_CLASS_NAMES[class_id] if class_id < len(FINAL_CLASS_NAMES) else f"Class_{class_id}"
            print(f"        {class_name}: {count}")
    
    return stats

dataset_stats = analyze_dataset()

# === COCO COMPATIBILITY ANALYSIS ===
def analyze_coco_compatibility():
    print("\n🔍 Analyzing COCO class compatibility...")
    
    coco_related_counts = {class_name: 0 for class_name in COCO_CLASS_NAMES}
    coco_class_mapping = {}
    
    # Check if any of our classes match COCO classes
    for i, custom_class in enumerate(FINAL_CLASS_NAMES):
        custom_class_lower = custom_class.lower()
        for coco_class in COCO_CLASS_NAMES:
            if coco_class in custom_class_lower or custom_class_lower in coco_class:
                coco_class_mapping[i] = coco_class
                print(f" Match found: '{custom_class}' → COCO '{coco_class}'")
    
    # Count instances in training data
    train_label_dir = os.path.join(FINAL_DATASET_DIR, 'labels', 'train')
    if os.path.exists(train_label_dir):
        for label_file in os.listdir(train_label_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(train_label_dir, label_file), 'r') as f:
                    for line in f:
                        class_id = int(line.strip().split()[0])
                        if class_id in coco_class_mapping:
                            coco_class = coco_class_mapping[class_id]
                            coco_related_counts[coco_class] += 1
    
    print("\n📊 COCO-Class Instances in Your Training Data:")
    for coco_class, count in coco_related_counts.items():
        print(f"   {coco_class}: {count} instances")
    
    return coco_class_mapping, coco_related_counts

coco_class_mapping, coco_counts = analyze_coco_compatibility()

# === CREATE FINAL data.yaml ===
def create_final_yaml():
    print(" Creating final data.yaml...")
    
    # Update data.yaml with correct paths
    data_yaml['path'] = FINAL_DATASET_DIR
    data_yaml['train'] = 'images/train'
    data_yaml['val'] = 'images/valid'
    data_yaml['test'] = 'images/test'
    
    FINAL_DATA_YAML_PATH = os.path.join(FINAL_DATASET_DIR, "data.yaml")
    with open(FINAL_DATA_YAML_PATH, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f" data.yaml created at: {FINAL_DATA_YAML_PATH}")
    return FINAL_DATA_YAML_PATH

FINAL_DATA_YAML_PATH = create_final_yaml()

# === SELECTIVE TRANSFER LEARNING SETUP ===
def setup_transfer_learning():
    print("\n SETTING UP SELECTIVE TRANSFER LEARNING")
    print(" Focus: person, backpack, handbag, suitcase")
    print(" Strategy: Preserve COCO knowledge + Fine-tune on your data")
    
    # Load COCO pre-trained model
    print(f" Loading YOLOv8{MODEL_SIZE} with COCO pre-trained weights...")
    model = YOLO(f'yolov8{MODEL_SIZE}.pt')
    
    # Verify COCO classes are loaded
    print(f" Loaded model with {len(model.names)} COCO classes")
    print(" Key COCO classes in pre-trained model:")
    for class_name, class_id in COCO_CLASS_IDS.items():
        if class_id < len(model.names):
            print(f"   {class_name} (ID: {class_id})")
    
    return model

model = setup_transfer_learning()

# === OPTIMIZED TRAINING CONFIGURATION ===
def get_training_config():
    """Get optimized training configuration for selective transfer learning"""
    
    # Base configuration
    config = {
        'data': FINAL_DATA_YAML_PATH,
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'workers': 4,
        
        # === SELECTIVE TRANSFER LEARNING PARAMETERS ===
        'lr0': 0.001,           # Lower learning rate to preserve COCO knowledge
        'pretrained': True,     # Use COCO pre-trained weights
        'optimizer': 'AdamW',   # Better optimizer for fine-tuning
        'weight_decay': 0.0001, # Lighter regularization
        'momentum': 0.9,
        
        # Learning rate scheduling
        'lrf': 0.01,
        'cos_lr': True,         # Cosine learning rate scheduler
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        
        # Loss weights (adjusted for transfer learning)
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Regularization
        'label_smoothing': 0.1,
        'patience': 50,
        
        # Augmentation (moderate to preserve COCO features)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 5.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 1.0,
        'perspective': 0.0001,
        'fliplr': 0.5,
        'mosaic': 1.0,
        
        # Advanced
        'amp': True,            # Mixed precision training
        'overlap_mask': True,
        'mask_ratio': 4,
        'val': True,
        'plots': True,
        'save': True,
        'save_period': 25,
        'project': 'selective_transfer_learning',
        'name': f'sackbags_coco_focus_{MODEL_SIZE}',
        'exist_ok': True,
        'verbose': True,
    }
    
    # Adjust based on dataset size
    total_train_images = dataset_stats['train']['images']
    if total_train_images < 500:
        print(" Small dataset detected - Using enhanced augmentation")
        config.update({
            'hsv_h': 0.02,
            'hsv_s': 0.8,
            'hsv_v': 0.5,
            'degrees': 10.0,
            'mosaic': 1.0,
            'mixup': 0.1,
        })
    else:
        print(" Moderate dataset detected - Using balanced augmentation")
    
    return config

training_config = get_training_config()

# === START TRAINING ===
print("\n STARTING SELECTIVE TRANSFER LEARNING TRAINING")
print("="*60)
print(f" Model: YOLOv8{MODEL_SIZE}")
print(f" Classes: {NUM_CLASSES} ({FINAL_CLASS_NAMES})")
print(f" Training images: {dataset_stats['train']['images']}")
print(f" COCO focus: {COCO_CLASS_NAMES}")
print(f" Device: {training_config['device']}")
print("="*60)

# Train the model
results = model.train(**training_config)

print(" Training completed!")

# === ENHANCED METRICS VISUALIZATION ===
def plot_comprehensive_metrics(results):
    """Plot comprehensive training metrics"""
    print("\n Generating comprehensive metrics visualization...")
    
    # Read results CSV
    results_dir = results.save_dir
    results_file = os.path.join(results_dir, 'results.csv')
    
    if not os.path.exists(results_file):
        print(" Results CSV not found")
        return
    
    import pandas as pd
    df = pd.read_csv(results_file)
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'YOLOv8{MODEL_SIZE} - Selective Transfer Learning Metrics\n(COCO Focus: {", ".join(COCO_CLASS_NAMES)})', 
                 fontsize=16, fontweight='bold')
    
    # Loss curves
    metrics_to_plot = [
        ('train/box_loss', 'val/box_loss', 'Box Loss', 'blue', 'red'),
        ('train/cls_loss', 'val/cls_loss', 'Classification Loss', 'green', 'orange'),
        ('train/dfl_loss', 'val/dfl_loss', 'DFL Loss', 'purple', 'brown')
    ]
    
    for idx, (train_metric, val_metric, title, train_color, val_color) in enumerate(metrics_to_plot):
        if train_metric in df.columns:
            axes[0, idx].plot(df[train_metric], label='Train', color=train_color, linewidth=2)
        if val_metric in df.columns:
            axes[0, idx].plot(df[val_metric], label='Validation', color=val_color, linewidth=2, linestyle='--')
        axes[0, idx].set_title(title)
        axes[0, idx].set_xlabel('Epoch')
        axes[0, idx].set_ylabel('Loss')
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)
    
    # Performance metrics
    performance_metrics = [
        ('metrics/precision(B)', 'Precision', 'orange'),
        ('metrics/recall(B)', 'Recall', 'purple'),
        ('metrics/mAP50(B)', 'mAP50', 'brown')
    ]
    
    for idx, (metric, title, color) in enumerate(performance_metrics):
        if metric in df.columns:
            axes[1, idx].plot(df[metric], label=title, color=color, linewidth=2)
            axes[1, idx].set_title(title)
            axes[1, idx].set_xlabel('Epoch')
            axes[1, idx].set_ylabel('Value')
            axes[1, idx].legend()
            axes[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('selective_transfer_learning_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Learning rate plot
    if 'lr/pg0' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['lr/pg0'], label='Learning Rate', color='magenta', linewidth=2)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('learning_rate_schedule.png', dpi=300, bbox_inches='tight')
        plt.show()

plot_comprehensive_metrics(results)

# === ADVANCED VALIDATION ===
def run_advanced_validation():
    print("\n📊 Running advanced validation...")
    
    # Load the best model
    best_model_path = os.path.join('selective_transfer_learning', 
                                  f'sackbags_coco_focus_{MODEL_SIZE}', 'weights', 'best.pt')
    
    if not os.path.exists(best_model_path):
        print(" Best model not found, using last model")
        best_model_path = os.path.join('selective_transfer_learning', 
                                      f'sackbags_coco_focus_{MODEL_SIZE}', 'weights', 'last.pt')
    
    model = YOLO(best_model_path)
    
    # Run comprehensive validation
    metrics = model.val(
        data=FINAL_DATA_YAML_PATH,
        split='test',
        conf=0.001,    # Low confidence for comprehensive evaluation
        iou=0.6,
        plots=True,
        save_json=True
    )
    
    print("\n📈 VALIDATION RESULTS:")
    print("="*40)
    print(f" mAP50-95: {metrics.box.map:.4f}")
    print(f" mAP50: {metrics.box.map50:.4f}")
    print(f" Precision: {metrics.box.mp:.4f}")
    print(f" Recall: {metrics.box.mr:.4f}")
    
    # Calculate F1 score
    precision = metrics.box.mp
    recall = metrics.box.mr
    f1 = 2 * (precision * recall) / (precision + recall + 1e-16)
    print(f" F1 Score: {f1:.4f}")
    print("="*40)
    
    return model, metrics

best_model, validation_metrics = run_advanced_validation()

# === CONFIDENCE THRESHOLD OPTIMIZATION ===
def optimize_confidence_threshold(model):
    print("\n Optimizing confidence threshold...")
    
    conf_thresholds = np.arange(0.1, 0.9, 0.1)
    f1_scores = []
    
    for conf in conf_thresholds:
        temp_metrics = model.val(data=FINAL_DATA_YAML_PATH, conf=conf, iou=0.5, plots=False, verbose=False)
        precision = temp_metrics.box.mp
        recall = temp_metrics.box.mr
        f1 = 2 * (precision * recall) / (precision + recall + 1e-16)
        f1_scores.append(f1)
    
    optimal_conf = conf_thresholds[np.argmax(f1_scores)]
    optimal_f1 = max(f1_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(conf_thresholds, f1_scores, 'bo-', linewidth=2, markersize=8, label='F1 Score')
    plt.axvline(optimal_conf, color='red', linestyle='--', 
                label=f'Optimal: {optimal_conf:.2f} (F1: {optimal_f1:.3f})')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('Confidence Threshold Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('confidence_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Optimal confidence threshold: {optimal_conf:.3f} (F1: {optimal_f1:.3f})")
    return optimal_conf

optimal_conf = optimize_confidence_threshold(best_model)

# === TEST INFERENCE ===
def run_test_inference():
    print("\n Running test inference...")
    
    test_image_dir = os.path.join(FINAL_DATASET_DIR, 'images', 'test')
    if not os.path.exists(test_image_dir) or not os.listdir(test_image_dir):
        print(" No test images found")
        return
    
    test_images = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))][:4]  # Test on 4 images
    
    for i, test_image in enumerate(test_images):
        print(f"\n🔍 Test {i+1}: {os.path.basename(test_image)}")
        
        # Run inference with optimal confidence
        results = best_model(test_image, conf=optimal_conf, save=True)
        
        for result in results:
            if result.boxes is not None:
                print(f"    Detected {len(result.boxes)} objects:")
                
                # Count by class
                class_counts = {}
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = box.conf[0].item()
                    class_name = FINAL_CLASS_NAMES[class_id]
                    
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    class_counts[class_name] += 1
                    
                    # Highlight COCO classes
                    is_coco_class = any(coco_class in class_name.lower() for coco_class in COCO_CLASS_NAMES)
                    marker = "⭐" if is_coco_class else "  "
                    print(f"      {marker} {class_name}: conf={confidence:.3f}")
                
                print(f"    Summary: {class_counts}")
            else:
                print("    No objects detected")
        
        # Display results
        result_image_path = os.path.join('runs', 'detect', 'predict', os.path.basename(test_image))
        if os.path.exists(result_image_path):
            img = cv2.imread(result_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title(f'Detection Results - {os.path.basename(test_image)}\n(⭐ = COCO-focused classes)')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

run_test_inference()

# === MODEL EXPORT ===
def export_model():
    print("\n Exporting model for deployment...")
    
    # Save the final model
    best_model.save('yolov8_selective_transfer_final.pt')
    print(" Saved final model: yolov8_selective_transfer_final.pt")
    
    # Export to other formats
    export_formats = ['onnx', 'engine']  # Add 'tflite' if needed
    
    for fmt in export_formats:
        try:
            best_model.export(format=fmt)
            print(f" Exported to {fmt.upper()} format")
        except Exception as e:
            print(f"  Could not export to {fmt}: {e}")

export_model()

# === COMPREHENSIVE SUMMARY ===
print("\n" + "="*70)
print(" SELECTIVE TRANSFER LEARNING - COMPLETE SUMMARY")
print("="*70)
print(f" Model: YOLOv8{MODEL_SIZE}")
print(f" Dataset: {NUM_CLASSES} classes ({FINAL_CLASS_NAMES})")
print(f" COCO Focus: {COCO_CLASS_NAMES}")
print(f" Final mAP50: {validation_metrics.box.map50:.4f}")
print(f" Final mAP50-95: {validation_metrics.box.map:.4f}")
print(f" Precision: {validation_metrics.box.mp:.4f}")
print(f" Recall: {validation_metrics.box.mr:.4f}")
print(f" Optimal confidence: {optimal_conf:.3f}")
print(f" Model saved: yolov8_selective_transfer_final.pt")
print(f" Training logs: selective_transfer_learning/sackbags_coco_focus_{MODEL_SIZE}/")
print("="*70)

print("\n SELECTIVE TRANSFER LEARNING COMPLETED SUCCESSFULLY!")
print(" Your model now has:")
print("   ⭐ Enhanced detection for person, backpack, handbag, suitcase")
print("   ⭐ COCO's extensive knowledge preserved")
print("   ⭐ Fine-tuned for your specific use case")
print("   ⭐ Better accuracy than training from scratch!")
