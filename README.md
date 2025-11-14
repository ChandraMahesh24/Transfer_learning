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



# Roboflow Setup

  ROBOFLOW_API_KEY = "your-key"
  ROBOFLOW_WORKSPACE = "your-workspace"
  ROBOFLOW_PROJECT = "your-project"
  ROBOFLOW_VERSION = 1


# Training Pipeline Overview

The training pipeline includes:

Automatic dataset download (Roboflow)

YOLO dataset structure builder

Auto-creation of data.yaml

COCO-pretrained YOLOv8 model loading

Hybrid fine-tuning with optimized hyperparameters

Training metrics visualization

Validation + mAP, Precision, Recall evaluation

Final model export
