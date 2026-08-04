import json
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("results")
SEEDS = [42, 123, 2026]

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

all_metrics = []
for seed in SEEDS:
    metrics_path = RESULTS_DIR / f"baseline_seed{seed}_eval" / "metrics.json"
    with open(metrics_path) as f:
        all_metrics.append(json.load(f))

print("=== Per-seed results ===\n")
for seed, m in zip(SEEDS, all_metrics):
    print(
        f"Seed {seed}: mAP={m['mAP']:.4f}  NDS={m['NDS']:.4f}  "
        f"GFLOPs={m['gflops']}  FPS={m['fps']}"
    )

print("\n=== Aggregated (mean ± std) across 3 seeds ===\n")
for key in ["mAP", "NDS", "gflops", "fps", "mean_ms"]:
    vals = [m[key] for m in all_metrics]
    print(f"{key:10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

print("\n=== Per-class AP (mean ± std) ===")
for cls in CLASS_NAMES:
    vals = [m["per_class_AP"].get(cls, 0.0) for m in all_metrics]
    print(f"{cls:22s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
