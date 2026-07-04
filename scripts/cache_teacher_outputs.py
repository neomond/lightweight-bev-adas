"""
Cache Teacher Outputs for Offline Knowledge Distillation.

Runs the BEVFusion teacher once over the entire dataset and saves its
outputs to disk, keyed by sample_token. The KD training script then
loads these cached tensors instead of running the teacher live — this
completely decouples the teacher's environment (mmdet3d, mmcv) from
the student's training environment.

Why offline caching?
    - BEVFusion requires mmdet3d + mmcv which may conflict with Python 3.13
    - Running the teacher live doubles memory usage during training
    - Cached outputs can be reused across multiple student training runs
      (different α/β values, architectures, ablations) at zero extra cost
    - On Colab: generate cache once, download as a zip, reuse locally

Cache format:
    data/teacher_cache/{split}/{sample_token}.pt

Each .pt file contains a dict:
    {
        "fused_bev":  tensor (256, 128, 128)  — no batch dim, stored per-sample
        "heatmap":    tensor (10,  128, 128)
        "regression": tensor (8,   128, 128)
        "sample_token": str
    }

Usage:
    # With real BEVFusion weights (run on Colab):
    python scripts/cache_teacher_outputs.py \\
        --config configs/student.yaml \\
        --checkpoint checkpoints/bevfusion_pretrained.pth \\
        --split train

    # With mock teacher (for development/testing the KD pipeline):
    python scripts/cache_teacher_outputs.py \\
        --config configs/student.yaml \\
        --mock \\
        --split train

    # Cache both splits:
    python scripts/cache_teacher_outputs.py --config configs/student.yaml --mock --split train
    python scripts/cache_teacher_outputs.py --config configs/student.yaml --mock --split val
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.teacher import TeacherBEVFusion
from src.data.nuscenes_loader import NuScenesDataset, collate_fn
from src.utils.device import get_device


def cache_split(
    teacher:    TeacherBEVFusion,
    loader:     DataLoader,
    cache_dir:  Path,
    device:     torch.device,
    overwrite:  bool = False,
):
    """Cache teacher outputs for all samples in a DataLoader split."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    skipped  = 0
    computed = 0
    errors   = 0

    for batch in tqdm(loader, desc=f"Caching → {cache_dir.name}"):
        camera_images = batch["camera_images"].to(device)
        lidar_points  = batch["lidar_points"].to(device)
        calibration   = batch["calibration"]
        sample_tokens = batch["sample_tokens"]

        # Check which samples in this batch still need caching
        tokens_needed = []
        for tok in sample_tokens:
            out_path = cache_dir / f"{tok}.pt"
            if out_path.exists() and not overwrite:
                skipped += 1
            else:
                tokens_needed.append(tok)

        if not tokens_needed:
            continue

        # Run teacher forward (no_grad enforced inside teacher.forward)
        try:
            with torch.no_grad():
                teacher_out = teacher(
                    camera_images=camera_images,
                    lidar_points=lidar_points,
                    calibration=calibration,
                )
        except Exception as e:
            print(f"\n  ⚠️  Teacher forward failed for batch: {e}")
            errors += len(sample_tokens)
            continue

        # Save one file per sample — remove batch dim
        fused_bev  = teacher_out["fused_bev"].cpu()   # (B, 256, 128, 128)
        heatmap    = teacher_out["heatmap"].cpu()      # (B, 10,  128, 128)
        regression = teacher_out["regression"].cpu()   # (B, 8,   128, 128)

        for i, tok in enumerate(sample_tokens):
            if tok not in tokens_needed:
                continue
            out_path = cache_dir / f"{tok}.pt"
            torch.save({
                "fused_bev":    fused_bev[i],     # (256, 128, 128)
                "heatmap":      heatmap[i],        # (10,  128, 128)
                "regression":   regression[i],     # (8,   128, 128)
                "sample_token": tok,
            }, out_path)
            computed += 1

    return computed, skipped, errors


def main():
    parser = argparse.ArgumentParser(description="Cache BEVFusion teacher outputs")
    parser.add_argument("--config",     default="configs/student.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="BEVFusion checkpoint path (omit for mock mode)")
    parser.add_argument("--mock",       action="store_true",
                        help="Use mock teacher (for development)")
    parser.add_argument("--split",      default="train",
                        choices=["train", "val", "both"])
    parser.add_argument("--batch",      type=int, default=4)
    parser.add_argument("--workers",    type=int, default=2)
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-cache even if file already exists")
    parser.add_argument("--cache-dir",  default="data/teacher_cache")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    use_mock = args.mock or (args.checkpoint is None)

    print(f"\n{'='*60}")
    print(f"  Cache Teacher Outputs")
    print(f"  Mode:    {'MOCK' if use_mock else 'REAL'}")
    print(f"  Device:  {device}")
    print(f"  Split:   {args.split}")
    print(f"  Output:  {args.cache_dir}")
    print(f"{'='*60}\n")

    # ── Teacher ───────────────────────────────────────────────────────────
    teacher = TeacherBEVFusion(
        mock=use_mock,
        checkpoint=args.checkpoint,
        device=device,
    ).to(device)
    teacher.eval()

    # ── Dataset ───────────────────────────────────────────────────────────
    data_cfg = config["data"]
    splits = ["train", "val"] if args.split == "both" else [args.split]

    for split in splits:
        dataset = NuScenesDataset(
            dataroot=data_cfg["dataroot"],
            version=data_cfg["version"],
            split=split,
            x_range=tuple(data_cfg["x_range"]),
            y_range=tuple(data_cfg["x_range"]),
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_fn,
        )

        cache_dir = Path(args.cache_dir) / split
        print(f"Caching {split} split ({len(dataset)} samples)...")

        computed, skipped, errors = cache_split(
            teacher, loader, cache_dir, device, overwrite=args.overwrite
        )

        print(f"  ✅ {computed} computed, {skipped} skipped, {errors} errors")
        print(f"  Cache saved to: {cache_dir}/\n")

    # Verify a sample file
    sample_files = list(Path(args.cache_dir).rglob("*.pt"))
    if sample_files:
        sample = torch.load(sample_files[0], weights_only=True)
        print(f"Sample cache file verification:")
        print(f"  token:      {sample['sample_token']}")
        print(f"  fused_bev:  {sample['fused_bev'].shape}")
        print(f"  heatmap:    {sample['heatmap'].shape}")
        print(f"  regression: {sample['regression'].shape}")
        print(f"\n✅ Caching complete. Run train_with_kd.py next.")
    else:
        print("⚠️  No cache files found — something went wrong.")


if __name__ == "__main__":
    main()
