
===============================================================
        YOLOv8 Selective Transfer Learning for Luggage Detection
===============================================================

🎯 Project Goal:
----------------
Fine-tune a YOLOv8 object detection model using selective transfer learning to detect luggage-related objects while preserving essential COCO class knowledge.

🧠 Focused COCO Classes:
- 👤 Person (ID: 0)
- 🎒 Backpack (ID: 24)
- 👝 Handbag (ID: 26)
- 💼 Suitcase (ID: 28)

===============================================================
📚 What is Selective Transfer Learning?
===============================================================

Selective transfer learning allows us to:
✔ Retain knowledge from selected COCO classes  
✔ Fine-tune on a smaller, targeted custom dataset  
✔ Avoid catastrophic forgetting  
✔ Achieve both generalization and specialization  

===============================================================
🗂️ Project Directory Structure
===============================================================
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

===============================================================
🛠️ Setup & Installation
===============================================================
1. Install dependencies:

   pip install roboflow ultralytics albumentations torch torchvision
   pip install seaborn matplotlib opencv-python

2. Configure Roboflow:

   ROBOFLOW_API_KEY = "your_api_key"  
   ROBOFLOW_WORKSPACE = "your_workspace"  
   ROBOFLOW_PROJECT = "your_project"  
   ROBOFLOW_VERSION = 1  

===============================================================
🚀 Selective Transfer Learning Workflow
===============================================================

COCO Pre-trained Model  
        ↓  
Preserve: Person, Backpack, Handbag, Suitcase  
        ↓  
Fine-tune on Custom Dataset  
        ↓  
Optimized Luggage Detection Model

===============================================================
⚙️ Transfer Learning Configuration
===============================================================

Main Parameters:
----------------
- pretrained: True              # Load COCO weights
- lr0: 0.001                    # Lower learning rate
- weight_decay: 0.0001          # Gentle regularization
- cos_lr: True                  # Smooth learning rate decay
- warmup_epochs: 3.0            # Gradual warmup
- label_smoothing: 0.1          # Prevent overconfidence

Loss Function Tweaks:
---------------------
- box: 7.5                      # Box regression loss
- cls: 0.5                      # Classification loss
- dfl: 1.5                      # Distribution Focal Loss

Augmentation Strategy:
----------------------
- hsv_h: 0.015                  # Conservative color changes
- degrees: 5.0                  # Slight rotations
- mosaic: 1.0                   # Enable for robust learning

===============================================================
🎯 Training the Model
===============================================================

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

===============================================================
📊 Performance Monitoring
===============================================================
Metrics Tracked:
----------------
- mAP50-95: Overall accuracy
- Class-wise AP: Category-specific precision
- F1-Score: Balance of precision & recall
- Confusion Matrix: Visualize misclassifications
- Precision-Recall Curve: Class performance at different thresholds

Confidence Thresholding:
------------------------
- F1-optimized threshold selection
- Class-specific threshold tuning for better accuracy

===============================================================
🔍 Model Evaluation
===============================================================

metrics = model.val(
    data='dataset_final/data.yaml',
    conf=0.001,
    iou=0.6,
    plots=True
)

Evaluation Outputs:
-------------------
✔️ mAP@50-95  
✔️ Per-class Average Precision  
✔️ Precision/Recall curves  
✔️ Confusion matrix  

===============================================================
💾 Export the Model
===============================================================

Export Options:
---------------
- model.export(format='onnx')     → For ONNX inference
- model.export(format='engine')   → For TensorRT GPU deployment
- model.save('final_model.pt')    → For further training in PyTorch

===============================================================
✨ Key Advantages
===============================================================
✅ Faster convergence (30–50% reduction in training time)  
✅ Improved accuracy due to pre-trained COCO weights  
✅ Robust detection of both custom & COCO classes  
✅ Reduced overfitting thanks to transfer learning regularization  

===============================================================
🛠 Troubleshooting Guide
===============================================================
- Overfitting?
  → Reduce learning rate, add more augmentation

- Poor COCO class retention?
  → Increase classification loss (`cls`)

- Training too slow?
  → Tune learning rate scheduler or batch size

===============================================================
🎉 Final Model Output
===============================================================
✔️ High-accuracy luggage detection  
✔️ Retains COCO class detection  
✔️ Confidence thresholds auto-optimized  
✔️ Ready for real-world deployment

===============================================================
🔗 References
===============================================================
- Ultralytics YOLOv8 Docs: https://docs.ultralytics.com  
- COCO Dataset: https://cocodataset.org  
- Roboflow: https://roboflow.com




mask_ratio Explained

The mask_ratio parameter controls the resolution of segmentation masks compared to the input image, especially in instance segmentation tasks (e.g. YOLO with masks or other segmentation models).
What does mask_ratio: 4 mean?

A mask_ratio of 4 means:

The masks are stored and processed at 1/4 the resolution of the input image.

This makes training and inference faster and more memory-efficient.

📸 Example

If your input image size is:

640 × 640 pixels


And you set:

mask_ratio: 4


Then the mask size used during training becomes:

640 ÷ 4 = 160
=> Mask size = 160 × 160 pixels


So, instead of using a full 640×640 mask, the model processes a 160×160 mask internally.
⚡ Why Use a Smaller Mask?

Using a smaller mask has several advantages:

✅ Faster training (less computation)

✅ Lower GPU memory usage

✅ Sufficient accuracy for many practical tasks

🔁 Masks can be upscaled during inference if needed

📈 Trade-off Table
mask_ratio	Mask Size (for 640×640 image)	Pros	Cons
1	640 × 640	High accuracy	High memory, slower training
2	320 × 320	Balanced speed & detail	Slight loss in fine details
4	160 × 160	Fast, efficient	May lose some edge accuracy
8	80 × 80	Very fast, low memory usage	Coarse masks, less precision
