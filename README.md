# Ship Detection from Satellite Imagery
## 4th Place – DataDriven Satellite Ship Detection Competition

This repository contains my solution that ranked 4th in the Satellite Ship Detection competition organized by DataDriven.
The challenge involved detecting ships in satellite images of varying formats, resolutions, and aspect ratios, requiring a robust and adaptive object detection pipeline.

The model achieves stable performance across heterogeneous satellite imagery thanks to custom preprocessing, scale-aware augmentations, and an optimized inference workflow.

## 📁 Project Structure
```
ship_detection/
|-- data/                     # Sample training/validation data (placeholder)
|-- models/                   # Trained model weights
|-- src/
|   |-- dataset.py            # Data loading & preprocessing
|   |-- augmenter.py          # Data augmentation pipeline
|   |-- model.py              # Detection model architecture
|   |-- utils.py              # Helper functions
|   `-- inference.py          # Inference pipeline
|-- train.ipynb               # Training workflow
|-- inference_result.ipynb    # Example predictions & evaluation
|-- requirements.txt          # Dependencies
`-- README.md                 # Documentation
```

## 🛰️ Competition Overview

Objective: Detect ships in satellite images (object detection task)

Challenges:

Highly diverse image formats

Large variations in resolution and aspect ratio

Illumination, angle, and atmospheric differences

Evaluation Metric: mAP / IoU (per competition rules)

This project focuses on normalizing heterogeneous data, optimizing detection for small objects, and building a robust multi-resolution inference strategy.

## 🧠 Approach & Model
### ✔ Data Processing

Dynamic resizing pipeline to handle different image dimensions

Scale-invariant augmentation

Random crop, flip, rotate, mosaic (if applicable)

Normalization & histogram adjustments

### ✔ Model Architecture

General features of the solution:

Custom YOLO-based backbone

Multi-scale feature extraction

Small-object oriented detection heads

Optimized anchors & IoU thresholds

### ✔ Training

Mixed precision training (AMP)

Cosine annealing LR schedule

~300 epochs

Hyperparameter tuning for anchor sizes & LR policies

## 📊 Results


| **Metric**       | **Value**                    |
| ---------------- | ---------------------------- |
| Competition Rank | 4th                          |
| mAP / IoU        | 0.8594 |

📸 Example Predictions

Available in:
👉 inference_result.ipynb
