# scripts/prune_model.py
"""
Structured pruning for StudentBEV using torch-pruning.

Prunes the model's conv layers by removing the least-important channels
(L1-norm magnitude), producing a genuinely smaller model (fewer real
parameters/FLOPs), not just zeroed weights.

Scope: prunes PointPillars encoder, ChannelWiseFusion, and BEVDetectionHead.
The YOLO camera backbone is excluded — it's a pretrained third-party model
(ultralytics YOLO11), and pruning it reliably requires YOLO-specific
tooling; mixing it into a generic torch_pruning pass risks silently
breaking its pretrained structure. CameraToBEV (LSS) is also excluded
for now, as its grid-sampling/view-transform ops are non-standard for
torch_pruning's dependency tracer.

Usage:
    python scripts/prune_model.py \
        --checkpoint checkpoints/kd_real_classweighted_seed42_best.pth \
        --config configs/student.yaml \
        --output-dir checkpoints/pruned
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.student import StudentBEV


class PrunableWrapper(nn.Module):
    """Wraps the prunable submodules (lidar encoder -> fusion -> head) into
    a single pure-tensor forward path, so torch_pruning's tracer can build
    a dependency graph without needing calibration dicts / camera inputs.

    camera_bev is treated as a fixed, non-prunable input tensor (since the
    camera branch — YOLO + LSS — is out of scope for this pruning pass).
    """
    def __init__(self, student: StudentBEV):
        super().__init__()
        self.lidar_encoder = student.lidar_encoder
        self.fusion = student.fusion
        self.detection_head = student.detection_head

    def forward(self, lidar_points: torch.Tensor, camera_bev: torch.Tensor):
        lidar_bev = self.lidar_encoder(lidar_points)
        fused_bev = self.fusion(camera_bev, lidar_bev)
        detections = self.detection_head(fused_bev)
        return detections["heatmap"], detections["regression"]


def get_ignored_layers(wrapper: PrunableWrapper) -> list:
    """Layers that must keep their exact output channel count:
    the final classification (10 classes) and regression (8 dims) heads,
    since these define the model's task interface."""
    ignored = []
    for m in wrapper.detection_head.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels in (10, 8):
            ignored.append(m)
    return ignored


def prune_at_ratio(
    checkpoint_path: str,
    config: dict,
    ratio: float,
    output_path: str,
    device: torch.device,
):
    print(f"\n{'='*60}")
    print(f"  Pruning at {ratio*100:.0f}% ratio")
    print(f"{'='*60}")

    model = StudentBEV(config["model"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    params_before = sum(p.numel() for p in model.parameters())
    wrapper_params_before = sum(
        p.numel() for p in list(model.lidar_encoder.parameters())
        + list(model.fusion.parameters())
        + list(model.detection_head.parameters())
    )
    print(f"Total model parameters:      {params_before:,}")
    print(f"Prunable-scope parameters:   {wrapper_params_before:,} "
          f"(lidar_encoder + fusion + detection_head)")

    wrapper = PrunableWrapper(model).to(device)

    # Example inputs matching the wrapper's forward signature
    example_lidar = torch.randn(1, 35000, 4).to(device)
    example_camera_bev = torch.randn(1, 256, 50, 50).to(device)

    imp = tp.importance.MagnitudeImportance(p=1)  # L1-norm channel importance
    ignored_layers = get_ignored_layers(wrapper)

    pruner = tp.pruner.MagnitudePruner(
        wrapper,
        example_inputs=(example_lidar, example_camera_bev), # type: ignore
        importance=imp,
        pruning_ratio=ratio,
        ignored_layers=ignored_layers,
    )

    # Verify the wrapper still runs before pruning (sanity check the tracer works)
    with torch.no_grad():
        _ = wrapper(example_lidar, example_camera_bev)
    print("Pre-prune forward pass OK")

    pruner.step()

    # Verify it still runs after pruning (structural consistency check)
    with torch.no_grad():
        heatmap, regression = wrapper(example_lidar, example_camera_bev)
    print(f"Post-prune forward pass OK — heatmap {heatmap.shape}, regression {regression.shape}")

    params_after = sum(p.numel() for p in model.parameters())
    print(f"Parameters after pruning:    {params_after:,}")
    print(f"Overall reduction:           {100 * (1 - params_after/params_before):.1f}%")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "epoch": 0,
        "val_loss": None,
        "config": config,
        "pruned_ratio": ratio,
        "params_before": params_before,
        "params_after": params_after,
    }, output_path)
    print(f"Saved pruned checkpoint: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/student.yaml")
    parser.add_argument("--output-dir", default="checkpoints/pruned")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ratio in args.ratios:
        pct = int(ratio * 100)
        output_path = f"{args.output_dir}/pruned_{pct}pct.pth"
        prune_at_ratio(args.checkpoint, config, ratio, output_path, device)

    print(f"\n{'='*60}")
    print(f"  All pruning ratios complete: {args.ratios}")
    print(f"  Checkpoints saved to {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()