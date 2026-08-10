"""
Compute instance-weighted micro-average mAP, complementing the official
macro-average mAP (which weights all classes equally regardless of how
many real instances exist).

Usage:
    python scripts/compute_micro_map.py --metrics-json results/RUN_NAME/metrics.json
"""

import argparse
import json

# Fill in from: python scripts/count_object_classes.py --split mini_val
VAL_INSTANCE_COUNTS = {
    "car": None,           # <- fill in real counts
    "truck": None,
    "bus": None,
    "trailer": None,
    "construction_vehicle": None,
    "pedestrian": None,
    "motorcycle": None,
    "bicycle": None,
    "traffic_cone": None,
    "barrier": None,
}


def compute_micro_map(per_class_ap: dict, instance_counts: dict) -> float:
    total_weighted = 0.0
    total_count = 0
    for cls, ap in per_class_ap.items():
        count = instance_counts.get(cls, 0)
        if count is None:
            raise ValueError(f"VAL_INSTANCE_COUNTS['{cls}'] not filled in yet")
        total_weighted += ap * count
        total_count += count
    return total_weighted / total_count if total_count > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", required=True)
    args = parser.parse_args()

    with open(args.metrics_json) as f:
        m = json.load(f)

    macro_map = m["mAP"]
    micro_map = compute_micro_map(m["per_class_AP"], VAL_INSTANCE_COUNTS)

    print(f"Macro-mAP (official, equal class weighting): {macro_map:.4f}")
    print(f"Micro-mAP (instance-weighted):                 {micro_map:.4f}")
    print(f"\nPer-class contribution:")
    for cls, ap in m["per_class_AP"].items():
        count = VAL_INSTANCE_COUNTS.get(cls, 0)
        print(f"  {cls:22s}: AP={ap:.4f}  val_instances={count}")


if __name__ == "__main__":
    main()