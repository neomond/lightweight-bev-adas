"""
Compute per-class weights for FocalLoss from known annotation counts.
Counts sourced from count_object_classes.py run on mini_train split.
"""

import torch
from pathlib import Path

CLASS_NAMES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]

# From: python scripts/count_object_classes.py --split mini_train
ANNOTATION_COUNTS = {
    "car": 5051,
    "truck": 525,
    "bus": 369,
    "trailer": 60,
    "construction_vehicle": 196,
    "pedestrian": 3657,
    "motorcycle": 212,
    "bicycle": 191,
    "traffic_cone": 1339,
    "barrier": 2323,
}

MAX_WEIGHT = 3.0  # cap to avoid instability from near-single-instance classes


def compute_class_weights():
    counts = torch.tensor([ANNOTATION_COUNTS[c] for c in CLASS_NAMES], dtype=torch.float32)
    # sqrt-dampened inverse frequency — gentler than raw inverse frequency,
    # avoids over-suppressing dominant classes (car/pedestrian) to the point
    # where the model stops learning them entirely
    inv_freq = 1.0 / torch.sqrt(counts)
    weights = inv_freq * (len(CLASS_NAMES) / inv_freq.sum())  # normalize to mean 1.0
    weights = weights.clamp(min=0.3, max=MAX_WEIGHT)  # floor AND ceiling this time

    print("Class weights (annotation-count based, capped at", MAX_WEIGHT, "):")
    for cls, w, c in zip(CLASS_NAMES, weights.tolist(), counts.tolist()):
        print(f"  {cls:22s} count={int(c):5d}  weight={w:.4f}")

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(weights, "checkpoints/class_weights.pt")
    print(f"\nSaved to checkpoints/class_weights.pt")
    return weights


if __name__ == "__main__":
    compute_class_weights()
