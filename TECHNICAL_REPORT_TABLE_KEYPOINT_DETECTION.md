# Technical Report: Table Keypoint Detection with SegFormer++

**Project:** UpliftingTableTennis  
**Focus:** Table tennis table keypoint detection (13 keypoints) using SegFormer++  
**Date:** 2026-05-17

## Executive Summary
This report documents the table keypoint detection subsystem implemented in this repository, with primary emphasis on the SegFormer++ models (`segformerpp_b0`, `segformerpp_b2`). The subsystem predicts 13 table-referenced keypoints (corners, net posts, and center-line landmarks) from single RGB frames and supports downstream camera calibration.

The implementation uses heatmap regression with a weighted MSE objective, geometric and photometric augmentation, and subpixel keypoint extraction based on local Gaussian fitting. Evaluation on the TTHQ test split shows strong localization at practical tolerances:

- `segformerpp_b2`: PCK@2 = 0.5086, PCK@5 = 0.8793, PCK@10 = 0.9310, PCK@20 = 0.9569
- `segformerpp_b0`: PCK@2 = 0.3793, PCK@5 = 0.8793, PCK@10 = 0.9310, PCK@20 = 0.9569

At strict tolerance (2 px), `segformerpp_b2` outperforms `segformerpp_b0`, HRNet, and ViTPose in current logged results. At moderate and loose tolerances (>= 5 px), performance converges across top models, indicating that the main differentiator is subpixel precision rather than coarse detectability.

## 1. Introduction
### 1.1 Background
Accurate table keypoint detection is a core dependency for robust camera calibration and full-pipeline 3D trajectory estimation. The model must localize 13 semantically fixed landmarks under broadcast variability (viewpoint, motion blur, occlusion, compression artifacts, and lighting shifts).

### 1.2 Objective
The objective of this work is to develop and evaluate a reliable table keypoint detector using SegFormer++ that:

- Produces stable and precise 2D keypoints for real broadcast frames.
- Generalizes across TTHQ videos.
- Improves strict localization quality (PCK@2) while maintaining high practical accuracy (PCK@5/10/20).
- Integrates directly into calibration/inference utilities already present in the project.

### 1.3 Working Hypotheses
- H1: A segmentation-style transformer backbone (SegFormer++) is well suited for dense keypoint heatmap regression on table structures.
- H2: Subpixel extraction from heatmaps improves strict tolerance scores versus argmax-only decoding.
- H3: Strong geometric augmentation improves robustness to real-world camera variation.

## 2. Methodology
### 2.1 Experimental Design
#### 2.1.1 Task Definition
Given a single RGB frame, predict 13 keypoints with visibility:

- Output tensor per frame: `(13, H_out, W_out)` heatmaps.
- Decoded prediction per keypoint: `(x, y, v)` where `v in {0, 1}`.

The 13 points correspond to physically meaningful table landmarks used in reprojection and calibration.

#### 2.1.2 Dataset and Splits
Training/evaluation uses the `TTHQ` dataset loader:

- Source annotation file: `table_detection.csv` in dataset root.
- Visibility convention in CSV: `flag == 2` mapped to visible.
- Split logic by video ID:
  - Validation/Test videos: `[1, 3, 10]`
  - Train: all remaining videos
  - Validation/Test: split 50/50 after deterministic shuffle (`RandomState(0)`)

#### 2.1.3 Preprocessing and Augmentation
Transforms are resolution-aware and include:

- Perspective warp (`max_warp_factor=0.15`)
- Rotation
- Translation
- Random crop (`min_fraction=0.8`)
- Color jitter (brightness/contrast/saturation/hue)
- Resize to model-specific resolution
- ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

Flip augmentation is intentionally disabled due unresolved semantic left/right remapping.

#### 2.1.4 Model Configuration
SegFormer++ variants are instantiated from Torch Hub (`KieDani/SegformerPlusPlus`) with:

- Backbones: `b0` or `b2` (primary focus)
- Decoder head adapted from 19 channels to 13 channels
- New 13-channel head initialized from mean of original segmentation head weights/bias
- `tome_strategy='bsm_hq'`
- ImageNet-pretrained base loaded by hub
- Additional task-specific pretraining currently not enabled for table detection

Model-dependent input resolutions:

- `segformerpp_b0`: `1920 x 1088`
- `segformerpp_b2`: `1600 x 896`

#### 2.1.5 Optimization and Training Protocol
- Loss: weighted MSE on heatmaps (`weighted_mse_loss`)
- Optimizer: Adam
- Default LR: `1e-3`
- Epochs: `700`
- Batch size: default `4` (CLI overridable)
- EMA model tracking with `alpha=0.999`
- Gradient clipping: `max_norm=5.0`
- Seed control: torch, numpy, python random all set to `42`
- Validation interval (TTHQ): every `32` iterations

### 2.2 Data Analysis
#### 2.2.1 Keypoint Decoding
Predicted heatmaps are decoded with local Gaussian fitting around each channel peak:

1. Find peak activation per heatmap channel.
2. Extract local `3x3` window around peak.
3. Fit 2D Gaussian via L-BFGS-B minimization.
4. Convert from heatmap space to image space using pixel-center mapping.

A confidence threshold (`THRESHOLD = 0.1`) is used to determine visibility validity.

#### 2.2.2 Metrics
Primary metric is Percentage of Correct Keypoints (PCK) at fixed pixel tolerances:

$$
\mathrm{PCK}@\tau = \frac{\sum \mathbf{1}(\|\hat{p}_{i} - p_{i}\|_2 \le \tau \land \hat{v}_{i}=1 \land v_{i}=1)}{\sum \mathbf{1}(\hat{v}_{i}=1 \land v_{i}=1)}
$$

Reported tolerances in evaluation: 2, 5, 10, 20 pixels.  
Additional scalar diagnostics during training:

- Average Euclidean keypoint distance (pixels)
- Ratio of detected keypoints

## 3. Results
### 3.1 Quantitative Performance (TTHQ Test)
The current summary file reports:

| Model | PCK@2 | PCK@5 | PCK@10 | PCK@20 |
|---|---:|---:|---:|---:|
| segformerpp_b0 | 0.379310 | 0.879310 | 0.931034 | 0.956897 |
| segformerpp_b2 | 0.508621 | 0.879310 | 0.931034 | 0.956897 |
| hrnet | 0.387931 | 0.887931 | 0.948276 | 0.956897 |
| vitpose | 0.258621 | 0.732759 | 0.801724 | 0.827586 |

### 3.2 SegFormer++ Focused Interpretation
For the two SegFormer++ variants:

- Absolute gain at strict tolerance:  
  `segformerpp_b2 - segformerpp_b0` at PCK@2 = `+0.129311` (about +34.1% relative over b0).
- Equal performance at PCK@5/10/20 in current snapshot.

This indicates that the b2 variant improves fine localization precision while preserving already-strong moderate/loose-tolerance detectability.

## 4. Discussion
### 4.1 What Worked
- Heatmap regression with subpixel decoding produces robust practical accuracy (PCK@5 and above).
- SegFormer++ b2 improves strict localization without harming looser-tolerance metrics.
- The augmentation stack is broad and relevant to broadcast-domain variability.

### 4.2 Technical Constraints and Risks
- Evaluation file currently contains a compact benchmark snapshot; confidence intervals and run-to-run variance are not yet reported.
- Visibility handling depends on thresholded activations and annotation visibility flags; miscalibration of threshold may bias strict metrics.
- Flip augmentation remains disabled, reducing augmentation diversity.
- Table-specific pretraining for SegFormer++ is not currently implemented.

### 4.3 Relation to Alternatives
- HRNet is competitive at 5 px and 10 px in this snapshot.
- ViTPose is weaker in current table keypoint setup.
- SegFormer++ b2 is strongest for strict precision in the available evaluation summary.

## 5. Conclusions
- The SegFormer++ table keypoint detector is operational and well integrated into the project training/inference flow.
- `segformerpp_b2` is the recommended default when strict localization accuracy is important.
- Current evidence supports using SegFormer++ outputs as reliable calibration inputs, particularly when followed by optional multi-model trajectory filtering utilities already provided in the repository.

## 6. Recommendations
1. Make `segformerpp_b2` the default production model for table keypoint detection.
2. Add repeated-seed evaluation (for example, 3 to 5 runs) and report mean plus standard deviation for each PCK tolerance.
3. Add per-keypoint error analysis (13-point breakdown) to identify systematic weak landmarks.
4. Re-enable horizontal flip augmentation only after implementing correct keypoint index remapping.
5. Extend evaluation beyond TTHQ to additional camera domains for robustness validation.
6. If latency is a concern, run `compare_speed.py` with final deployment hardware and document FPS/parameter trade-offs alongside PCK.

## References
1. UpliftingTableTennis repository codebase (training, inference, and model definitions).
2. SegFormer++ implementation loaded from Torch Hub (`KieDani/SegformerPlusPlus`).
3. Kienzle et al., *Uplifting Table Tennis: A Robust, Real-World Application for 3D Trajectory and Spin Estimation*, WACV 2026.
4. TTHQ dataset processing and table annotation pipeline in this repository.

## Appendices
### Appendix A: Reproducibility Commands
Evaluate table detection models:

```bash
python -m inference.inference_tabledetection --gpu 0
```

Train SegFormer++ b2 for table keypoint detection:

```bash
python -m tabledetection.train --gpu 0 --folder results --model_name segformerpp_b2 --data tthq
```

### Appendix B: Key Implementation Components
- Model definition and SegFormer++ head adaptation: `tabledetection/models/segformer_pp.py`
- Training loop and validation metrics: `tabledetection/train.py`
- Dataset split/loading and visibility mapping: `tabledetection/dataset.py`
- Geometric and photometric transforms: `tabledetection/transforms.py`
- Evaluation summary generation: `inference/inference_tabledetection.py`
- Current metrics snapshot: `logs/evaluation/tabledetection/tabledetection_pck_summary.txt`
