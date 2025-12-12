# Ship Detection from Satellite Imagery
## 4th Place – Data-Driven Science Satellite Ship Detection Competition on HuggingFace

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

- Objective:

  Detect ships in satellite images (object detection task)

- Challenges:

    Highly diverse image formats
    
    Large variations in resolution and aspect ratio
    
    Illumination, angle, and atmospheric differences

- Evaluation Metric:

  mAP / IoU (per competition rules)

This project focuses on normalizing heterogeneous data, optimizing detection for small objects, and building a robust multi-resolution inference strategy.

## 🧠 Approach & Model
### ✔ Data Processing

- Image Tiling with Overlap

    Large satellite images are split into multiple smaller tiles to increase training sample count and ensure small ships are captured.
    Overlapping windows are used so ships located on patch borders are not lost.

- Aspect-Ratio–Aware Resizing

    Each tile is resized to the target model resolution using a dynamic scale factor computed from the original image width–height ratio.
    This maintains ship proportions across images of different shapes.

- Consistent Output Directory Structure

    The script automatically generates images/ and labels/ directories and writes all processed tiles into the correct structure for YOLO-style training pipelines.

- Annotation Cropping for Tiles

    Bounding boxes are recalculated for each tile.
    Only the objects that fall inside a tile’s region are kept, and their coordinates are shifted accordingly.

### ✔ Model Architecture
- MMYOLO (OpenMMLab) YOLOv8-based architecture adapted for satellite-image ship detection

- Multi-scale feature extraction to improve detection on varying ship sizes

- Small-object–oriented detection heads optimized for high-resolution aerial imagery

- Custom anchor configurations & tuned IoU thresholds to improve recall on small and dense ship clusters

### ✔ Training

- Active learning–driven iterative training to progressively refine the model using hard samples

- Cosine annealing learning-rate schedule for stable convergence

- ~300 epochs total training with periodic evaluation and checkpointing

- Hyperparameter tuning focused on anchor sizes, LR policies, and small-object sensitivity

## 📊 Results


| **Metric**       | **Value**                    |
| ---------------- | ---------------------------- |
| Competition Rank | 4th                          |
| mAP / IoU        | 0.8594 |

📸 Example Predictions

Available in:
👉 inference_result.ipynb
