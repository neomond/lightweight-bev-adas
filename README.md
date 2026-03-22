# Knowledge-Distilled BEV Perception for ADAS

**Master's Dissertation — A YOLO-Based Camera-LiDAR Fusion Framework with Fusion-Stage Distillation**

## Project Structure

```
dissertation-bev/
├── src/
│   ├── models/
│   │   ├── yolo_backbone.py    # YOLO11 image feature extractor
│   │   ├── pointpillars.py     # LiDAR point cloud encoder
│   │   ├── fusion.py           # Channel-wise attention fusion (KD target)
│   │   ├── bev_head.py         # 3D detection head
│   │   └── student.py          # Complete student pipeline
│   ├── losses/
│   │   └── distillation.py     # Feature + logit KD losses
│   ├── data/                   # Dataset loaders
│   └── utils/                  # Device detection, helpers
├── configs/
│   └── student.yaml            # Model and training configuration
├── scripts/
│   ├── verify_setup.py         # Verify everything works
│   ├── train.py                # Training script (Milestone 4)
│   └── evaluate.py             # Evaluation script (Milestone 5)
├── notebooks/                  # Jupyter notebooks for exploration
├── data/                       # Datasets (gitignored)
├── checkpoints/                # Model weights (gitignored)
├── outputs/                    # Results and figures (gitignored)
└── logs/                       # TensorBoard logs (gitignored)
```

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Verify setup
python scripts/verify_setup.py

# Test individual components
python -m src.models.yolo_backbone
python -m src.models.fusion
python -m src.losses.distillation
```

## Architecture

```
Camera Images → YOLO11 Backbone → Camera-to-BEV
                                          ↘
                                    Fusion Module (KD here) → BEV → 3D Detection
                                          ↗
LiDAR Points  → PointPillars    → LiDAR-to-BEV

Teacher: BEVFusion (frozen, provides supervision)
```

## Milestones

- M1: Environment & data loading (Wk 1-2)
- M2: YOLO backbone features (Wk 3-4)
- M3: PointPillars + fusion baseline (Wk 5-7)
- M4: Knowledge distillation (Wk 8-12)
- M5: Evaluation & ablations (Wk 13-16)
- M6: Dissertation writing (Wk 16-20)
