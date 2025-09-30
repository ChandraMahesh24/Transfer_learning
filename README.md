# Transfer_learning

YOLOv8 Selective Transfer Learning for Luggage Detection
This project implements selective transfer learning using YOLOv8, specifically focused on detecting luggage-related objects (person, backpack, handbag, suitcase) while preserving knowledge from COCO dataset pre-training

What is Selective Transfer Learning?
Selective transfer learning is an advanced technique where we:

Preserve pre-trained knowledge on specific classes from COCO dataset

Fine-tune only on our custom dataset

Focus on related object categories to maintain detection capabilities



COCO Pre-trained Model
        ↓
Preserve: person, backpack, handbag, suitcase knowledge
        ↓
Fine-tune on custom dataset
        ↓
Optimized detection model


⚙️ Transfer Learning Parameters

  Essential Transfer Learning Parameters (Different from Normal Training):
  Parameter	Value	Purpose in Transfer Learning
  pretrained: True	True	Use COCO pre-trained weights
  lr0: 0.001	Lower than normal	Prevent overwriting pre-trained features
  weight_decay: 0.0001	Lighter	Gentle regularization
  cos_lr: True	True	Smooth learning rate decay
  warmup_epochs: 3.0	Longer warmup	Gradual adaptation
  label_smoothing: 0.1	Moderate	Prevent overconfidence
