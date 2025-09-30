YOLOv8 Selective Transfer Learning for Luggage Detection
=========================================================

This project implements selective transfer learning using YOLOv8, focused on detecting luggage-related objects while preserving pre-trained knowledge from the COCO dataset.

Target Object Classes:
- Person (ID: 0)
- Backpack (ID: 24)
- Handbag (ID: 26)
- Suitcase (ID: 28)

---------------------------------------------------------

What is Selective Transfer Learning?
------------------------------------
Selective Transfer Learning is a fine-tuning technique that:
- Retains COCO pre-trained knowledge of specific object classes
- Fine-tunes only on a custom dataset
- Maintains general object detection capabilities
- Optimizes performance for luggage-related object categories

---------------------------------------------------------

Project Directory Structure:
----------------------------
dataset_final/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
├── labels/
│   ├── train/
│   ├── valid/
│   └── test/
└── data.yaml

---------------------------------------------------------

Setup & Installation:
---------------------
Install dependencies:
pip install roboflow ultralytics albumentations torch torchvision
pip install seaborn matplotlib opencv-python

Configure Roboflow:
ROBOFLOW_API_KEY = "your_api_key"
ROBOFLOW_WORKSPACE = "your_workspace"
ROBOFLOW_PROJECT = "your_project"
ROBOFLOW_VERSION = 1

---------------------------------------------------------

Model Initialization:
---------------------
1. Load YOLOv8 model with COCO pre-trained weights
2. Preserve knowledge of selected COCO classes
3. Fine-tune using custom dataset

---------------------------------------------------------

Selective Transfer Learning Workflow:
-------------------------------------
COCO Pre-trained Model
        ↓
Preserve: person, backpack, handbag, suitcase knowledge
        ↓
Fine-tune on custom dataset
        ↓
Optimized detection model

---------------------------------------------------------

Transfer Learning Parameters:
-----------------------------

Essential Parameters:
- pretrained: True (Use COCO weights)
- lr0: 0.001 (Lower learning rate for stability)
- weight_decay: 0.0001 (Light regularization)
- cos_lr: True (Smooth LR decay)
- warmup_epochs: 3.0 (Gradual training start)
- label_smoothing: 0.1 (Avoid overconfidence)

Loss Adjustments:
- box: 7.5
- cls: 0.5
- dfl: 1.5

Moderate Data Augmentation:
- hsv_h: 0.015
- degrees: 5.0
- mosaic: 1.0

---------------------------------------------------------

Training Command:
-----------------
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='dataset_final/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.001,
    pretrained=True,
    cos_lr=True,
    weight_decay=0.0001,
    warmup_epochs=3.0,
    label_smoothing=0.1
)

---------------------------------------------------------

Performance Monitoring:
-----------------------
- Automatic confidence thresholding based on F1 score
- Metrics:
    - mAP50-95
    - Class-wise precision and recall
    - F1-score
    - Learning rate stability

---------------------------------------------------------

Model Evaluation:
-----------------
metrics = model.val(
    data='dataset_final/data.yaml',
    conf=0.001,
    iou=0.6,
    plots=True
)

Outputs:
- mAP@50-95
- Class-wise AP
- Confusion Matrix
- Precision-Recall Curve

---------------------------------------------------------

Model Export:
-------------
Export the model to different formats:

model.export(format='onnx')    # ONNX for inference
model.export(format='engine')  # TensorRT for GPU
model.save('final_model.pt')   # PyTorch for further training

---------------------------------------------------------

Key Benefits:
-------------
1. Faster Convergence - Reduced training time (30–50%)
2. Better Accuracy - COCO pre-training improves results
3. Robust Detection - Maintains COCO class detection
4. Reduced Overfitting - Better generalization on val data

---------------------------------------------------------

Troubleshooting:
----------------
Issue: Overfitting
- Solution: Lower learning rate, increase augmentation

Issue: Low COCO class performance
- Solution: Increase classification loss weight

Issue: Slow convergence
- Solution: Adjust learning rate schedule

---------------------------------------------------------

Final Output:
-------------
The final model will:
- Accurately detect luggage-related items
- Maintain COCO class detection capabilities
- Be ready for production deployment

---------------------------------------------------------

References:
-----------
- Ultralytics YOLOv8: https://docs.ultralytics.com
- COCO Dataset: https://cocodataset.org
- Roboflow: https://roboflow.com
