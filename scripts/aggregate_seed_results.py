"""
Aggregate evaluate.py results across multiple seeds (mean ± std).

Usage:
    python scripts/aggregate_seed_results.py
"""

import json
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("results")
SEEDS = [42, 123, 2026]

CLASS_NAMES = [
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
]

# Classes worth reporting per-class error breakdowns for — i.e. classes
# with actual non-zero AP in at least one seed. Extend this list if KD
# or class weighting later brings more classes above zero.
CLASSES_WITH_SIGNAL = ["car"]


def load_all():
    all_metrics = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"baseline_seed{seed}_eval" / "metrics.json"
        with open(path) as f:
            all_metrics.append(json.load(f))
    return all_metrics


def mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def main():
    all_metrics = load_all()

    print("=== Per-seed headline results ===\n")
    for seed, m in zip(SEEDS, all_metrics):
        print(f"Seed {seed}: mAP={m['mAP']:.4f}  NDS={m['NDS']:.4f}  "
              f"GFLOPs={m['gflops']}  FPS={m['fps']}  "
              f"epoch={m['epoch']}  val_loss={m['val_loss']:.4f}")

    print("\n=== Aggregated (mean ± std) across 3 seeds — headline metrics ===\n")
    for key in ["mAP", "NDS", "gflops", "fps", "mean_ms", "peak_memory_mb"]:
        mean, std = mean_std([m[key] for m in all_metrics])
        print(f"{key:16s}: {mean:.4f} ± {std:.4f}")

    print("\n=== Aggregated (mean ± std) — official macro-averaged TP errors ===")
    print("(NOTE: these are dragged toward 1.0 by the 8-9 classes with zero")
    print(" true positives — not very informative on their own; see per-class")
    print(" breakdown below for the classes that actually have detections.)\n")
    for key in ["mATE", "mASE", "mAOE"]:
        mean, std = mean_std([m.get(key) for m in all_metrics])
        print(f"{key:16s}: {mean:.4f} ± {std:.4f}")

    print("\n=== Per-class AP (mean ± std) ===")
    for cls in CLASS_NAMES:
        vals = [m["per_class_AP"].get(cls, 0.0) for m in all_metrics]
        mean, std = mean_std(vals)
        print(f"{cls:22s}: {mean:.4f} ± {std:.4f}")

    print("\n=== Per-class error breakdown (classes with real detections only) ===")
    for cls in CLASSES_WITH_SIGNAL:
        print(f"\n{cls}:")
        for err_key in ["trans_err", "scale_err", "orient_err", "vel_err", "attr_err"]:
            vals = []
            for m in all_metrics:
                cls_err = m.get("per_class_errors", {}).get(cls, {})
                v = cls_err.get(err_key)
                vals.append(v)
            mean, std = mean_std(vals)
            print(f"  {err_key:12s}: {mean:.4f} ± {std:.4f}")

    print("\n=== Dissertation-ready summary table ===\n")
    mAP_mean, mAP_std = mean_std([m["mAP"] for m in all_metrics])
    NDS_mean, NDS_std = mean_std([m["NDS"] for m in all_metrics])
    car_ap_mean, car_ap_std = mean_std([m["per_class_AP"].get("car", 0.0) for m in all_metrics])
    car_ate_mean, car_ate_std = mean_std(
        [m.get("per_class_errors", {}).get("car", {}).get("trans_err") for m in all_metrics]
    )
    car_ase_mean, car_ase_std = mean_std(
        [m.get("per_class_errors", {}).get("car", {}).get("scale_err") for m in all_metrics]
    )
    car_aoe_mean, car_aoe_std = mean_std(
        [m.get("per_class_errors", {}).get("car", {}).get("orient_err") for m in all_metrics]
    )
    gflops_mean, _ = mean_std([m["gflops"] for m in all_metrics])

    print(f"| Metric        | Mean ± Std          |")
    print(f"|---------------|---------------------|")
    print(f"| mAP (macro)   | {mAP_mean:.4f} ± {mAP_std:.4f} |")
    print(f"| NDS           | {NDS_mean:.4f} ± {NDS_std:.4f} |")
    print(f"| Car AP        | {car_ap_mean:.4f} ± {car_ap_std:.4f} |")
    print(f"| Car ATE       | {car_ate_mean:.4f} ± {car_ate_std:.4f} |")
    print(f"| Car ASE       | {car_ase_mean:.4f} ± {car_ase_std:.4f} |")
    print(f"| Car AOE       | {car_aoe_mean:.4f} ± {car_aoe_std:.4f} |")
    print(f"| GFLOPs        | {gflops_mean:.2f}              |")


if __name__ == "__main__":
    main()