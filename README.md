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

<p align="center">
  <img src="outputs/student_architecture.png" width="100%" alt="Student architecture and training pipeline diagram"/>
</p>

*Camera images pass through a YOLO11 backbone and Lift-Splat-Shoot view transformer into BEV space; LiDAR points pass through PointPillars into a LiDAR BEV feature map. Both are fused via channel-wise fusion and decoded by the BEV detection head. During training, the frozen BEVFusion teacher (~100M params) supervises the fusion stage via knowledge distillation — the student network totals ~5.5M parameters, 18× smaller than the teacher.*


## Results

| Sensor Inputs & BEV Feature Maps | Fused Detection Output |
|:---:|:---:|
| ![Sensor inputs](outputs/bev_sensor_inputs.png) | ![Detection output](outputs/bev_detection_output.png) |

*Left: 6-camera surround view with BEV height/intensity/density maps from LiDAR. Right: fused BEV detections (68 objects, 10 classes) from the 5.5M-param student model.*