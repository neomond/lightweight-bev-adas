"""
Knowledge Distillation Training Script.

Trains the student model with combined task + distillation losses:
    L_total = α · L_feature + β · L_logit + L_task

This is the core dissertation experiment. Run AFTER:
    1. train_baseline.py   → gives "student without KD" numbers
    2. cache_teacher_outputs.py → generates teacher output cache

The delta between baseline val loss and KD val loss is the dissertation's
primary quantitative result.

Usage:
    # Development (mock teacher, local Mac):
    python scripts/train_with_kd.py --config configs/student.yaml --epochs 2 --mock-teacher

    # Full training (cached real teacher outputs, Colab):
    python scripts/train_with_kd.py --config configs/student.yaml --epochs 20

    # Resume from checkpoint:
    python scripts/train_with_kd.py --config configs/student.yaml --resume checkpoints/kd_latest.pth

Outputs:
    checkpoints/kd_best.pth      ← best val loss
    checkpoints/kd_latest.pth    ← latest epoch
    logs/kd/                     ← TensorBoard logs

TensorBoard shows all loss components separately:
    train/loss_total, train/loss_task, train/loss_kd,
    train/loss_feature, train/loss_hm_kl, train/loss_reg_l1
"""

import argparse
import os
import sys
import time
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.student import StudentBEV
from src.models.teacher import TeacherBEVFusion
from src.losses.distillation import CombinedKDLoss
from src.data.nuscenes_loader import NuScenesDataset, collate_fn
from src.utils.device import get_device

# Reuse loss functions from train_baseline
from scripts.train_baseline import (
    FocalLoss,
    build_heatmap_targets,
    detection_loss,
)


# ── Cached Teacher Dataset Wrapper ───────────────────────────────────────────

class CachedTeacherDataset(Dataset):
    """Wraps NuScenesDataset and attaches pre-cached teacher outputs.

    For each sample, loads the corresponding teacher .pt file and appends
    teacher_fused_bev, teacher_heatmap, teacher_regression to the batch dict.

    Falls back to zero tensors if a cache file is missing — this lets
    training continue gracefully if a handful of samples weren't cached,
    though those batches contribute no KD signal.
    """

    TEACHER_SHAPES = {
        "fused_bev":  (256, 128, 128),
        "heatmap":    (10,  128, 128),
        "regression": (8,   128, 128),
    }

    def __init__(self, base_dataset: NuScenesDataset, cache_dir: str):
        self.base    = base_dataset
        self.cache   = Path(cache_dir)
        self._warned = set()   # track missing tokens to avoid spam

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        token  = sample["sample_token"]
        cache_path = self.cache / f"{token}.pt"

        if cache_path.exists():
            try:
                teacher_data = torch.load(cache_path, weights_only=True)
                sample["teacher_fused_bev"]  = teacher_data["fused_bev"]
                sample["teacher_heatmap"]    = teacher_data["heatmap"]
                sample["teacher_regression"] = teacher_data["regression"]
                return sample
            except Exception as e:
                if token not in self._warned:
                    print(f"  ⚠️  Failed to load cache for {token}: {e}")
                    self._warned.add(token)

        # Fallback: zero tensors (no KD signal for this sample)
        for key, shape in self.TEACHER_SHAPES.items():
            sample[f"teacher_{key}"] = torch.zeros(shape)
        return sample


def collate_fn_with_teacher(batch):
    """Extended collate_fn that stacks teacher outputs too."""
    base = collate_fn(batch)
    base["teacher_fused_bev"]  = torch.stack([b["teacher_fused_bev"]  for b in batch])
    base["teacher_heatmap"]    = torch.stack([b["teacher_heatmap"]    for b in batch])
    base["teacher_regression"] = torch.stack([b["teacher_regression"] for b in batch])
    return base


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch_kd(
    model,
    loader,
    teacher,        # None when using cached outputs
    optimizer,
    scheduler,
    focal_loss_fn,
    kd_loss_fn,
    device,
    epoch,
    writer,
    config,
    global_step,
    use_mock_teacher: bool = False,
):
    model.train()
    x_range = tuple(config["data"]["x_range"])
    y_range = tuple(config["data"]["y_range"])
    dist_cfg = config.get("distillation", {})

    sums = {k: 0.0 for k in [
        "total", "task", "kd", "feature", "hm_kl", "reg_l1"
    ]}
    n_batches = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        camera_images = batch["camera_images"].to(device)
        lidar_points  = batch["lidar_points"].to(device)
        annotations   = batch["annotations"]
        calibration   = batch["calibration"]

        optimizer.zero_grad()

        # ── Student forward ───────────────────────────────────────────────
        student_out = model(camera_images, lidar_points, calibration)
        pred_hm     = student_out["detections"]["heatmap"]      # (B,10,50,50)
        pred_reg    = student_out["detections"]["regression"]    # (B,8,50,50)
        fused_bev   = student_out["fused_bev"]                  # (B,256,50,50)

        # ── Teacher outputs ───────────────────────────────────────────────
        if use_mock_teacher and teacher is not None:
            # Mock teacher: run live (development mode)
            teacher_out = teacher(camera_images=camera_images)
            t_fused = teacher_out["fused_bev"].to(device)
            t_hm    = teacher_out["heatmap"].to(device)
            t_reg   = teacher_out["regression"].to(device)
        else:
            # Cached teacher: load from batch dict
            t_fused = batch["teacher_fused_bev"].to(device)
            t_hm    = batch["teacher_heatmap"].to(device)
            t_reg   = batch["teacher_regression"].to(device)

        # ── Task loss (same as baseline) ──────────────────────────────────
        loss_task, loss_hm_task, _ = detection_loss(
            pred_hm, pred_reg, annotations,
            x_range, y_range, focal_loss_fn,
            reg_weight=config.get("training", {}).get("reg_weight", 2.0),
        )

        # ── Distillation loss ─────────────────────────────────────────────
        kd_losses = kd_loss_fn(
            fused_bev, pred_hm, pred_reg,
            t_fused,   t_hm,    t_reg,
        )
        loss_kd = kd_losses["loss_kd_total"]

        # ── Combined loss ─────────────────────────────────────────────────
        loss_total = loss_task + loss_kd

        if not torch.isfinite(loss_total):
            print(f"  ⚠️  Non-finite loss at batch {batch_idx}, skipping")
            continue

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────
        sums["total"]   += loss_total.item()
        sums["task"]    += loss_task.item()
        sums["kd"]      += loss_kd.item()
        sums["feature"] += kd_losses["loss_feature"].item()
        sums["hm_kl"]   += kd_losses["loss_hm_kl"].item()
        sums["reg_l1"]  += kd_losses["loss_reg_l1"].item()
        n_batches += 1

        if writer:
            writer.add_scalar("train/loss_total",   loss_total.item(),                 global_step)
            writer.add_scalar("train/loss_task",    loss_task.item(),                  global_step)
            writer.add_scalar("train/loss_kd",      loss_kd.item(),                    global_step)
            writer.add_scalar("train/loss_feature", kd_losses["loss_feature"].item(),  global_step)
            writer.add_scalar("train/loss_hm_kl",   kd_losses["loss_hm_kl"].item(),    global_step)
            writer.add_scalar("train/loss_reg_l1",  kd_losses["loss_reg_l1"].item(),   global_step)
        global_step += 1

        if batch_idx % 10 == 0:
            print(
                f"  Epoch {epoch:02d} | Batch {batch_idx:03d}/{len(loader):03d} | "
                f"Total {loss_total.item():.4f} "
                f"(task={loss_task.item():.3f} "
                f"feat={kd_losses['loss_feature'].item():.3f} "
                f"kl={kd_losses['loss_hm_kl'].item():.3f}) | "
                f"{time.time()-t0:.1f}s"
            )

    if scheduler is not None:
        scheduler.step()

    avgs = {k: v / max(n_batches, 1) for k, v in sums.items()}
    return avgs, global_step


@torch.no_grad()
def validate_kd(model, loader, focal_loss_fn, kd_loss_fn, device, config,
                use_mock_teacher=False, teacher=None):
    model.eval()
    x_range = tuple(config["data"]["x_range"])
    y_range = tuple(config["data"]["y_range"])

    total_sum = 0.0
    n_batches = 0

    for batch in loader:
        camera_images = batch["camera_images"].to(device)
        lidar_points  = batch["lidar_points"].to(device)
        annotations   = batch["annotations"]
        calibration   = batch["calibration"]

        student_out = model(camera_images, lidar_points, calibration)
        pred_hm   = student_out["detections"]["heatmap"]
        pred_reg  = student_out["detections"]["regression"]
        fused_bev = student_out["fused_bev"]

        if use_mock_teacher and teacher is not None:
            teacher_out = teacher(camera_images=camera_images)
            t_fused = teacher_out["fused_bev"].to(device)
            t_hm    = teacher_out["heatmap"].to(device)
            t_reg   = teacher_out["regression"].to(device)
        else:
            t_fused = batch["teacher_fused_bev"].to(device)
            t_hm    = batch["teacher_heatmap"].to(device)
            t_reg   = batch["teacher_regression"].to(device)

        loss_task, _, _ = detection_loss(
            pred_hm, pred_reg, annotations, x_range, y_range, focal_loss_fn,
        )
        kd_losses = kd_loss_fn(fused_bev, pred_hm, pred_reg, t_fused, t_hm, t_reg)
        loss_total = loss_task + kd_losses["loss_kd_total"]

        if torch.isfinite(loss_total):
            total_sum += loss_total.item()
            n_batches += 1

    return total_sum / max(n_batches, 1)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train student with knowledge distillation")
    parser.add_argument("--config",       default="configs/student.yaml")
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch",        type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--workers",      type=int,   default=2)
    parser.add_argument("--run-name",     default="kd")
    parser.add_argument("--cache-dir",    default="data/teacher_cache")
    parser.add_argument("--mock-teacher", action="store_true",
                        help="Use live mock teacher instead of cache (development)")
    parser.add_argument("--resume",       default=None,
                        help="Resume from checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.epochs: config["training"]["epochs"]     = args.epochs
    if args.batch:  config["training"]["batch_size"] = args.batch
    if args.lr:     config["training"]["lr"]         = args.lr

    epochs     = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr         = config["training"]["lr"]
    wd         = config["training"].get("weight_decay", 0.01)
    warmup     = config["training"].get("warmup_epochs", 1)
    dist_cfg   = config.get("distillation", {})

    device = get_device()
    pin    = device.type == "cuda"

    print(f"\n{'='*60}")
    print(f"  KD Training  —  {args.run_name}")
    print(f"  Device:  {device}")
    print(f"  Epochs:  {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"  α={dist_cfg.get('alpha',1.0)}  β={dist_cfg.get('beta',0.5)}  "
          f"T={dist_cfg.get('temperature',4.0)}")
    print(f"  Teacher: {'mock (live)' if args.mock_teacher else f'cache ({args.cache_dir})'}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────────────────────────────────
    data_cfg = config["data"]

    train_base = NuScenesDataset(
        dataroot=data_cfg["dataroot"], version=data_cfg["version"],
        split="train", x_range=tuple(data_cfg["x_range"]),
        y_range=tuple(data_cfg["x_range"]),
    )
    val_base = NuScenesDataset(
        dataroot=data_cfg["dataroot"], version=data_cfg["version"],
        split="val", x_range=tuple(data_cfg["x_range"]),
        y_range=tuple(data_cfg["x_range"]),
    )

    if args.mock_teacher:
        # Use base datasets + live mock teacher
        train_dataset = train_base
        val_dataset   = val_base
        _collate      = collate_fn
    else:
        # Wrap with cached teacher outputs
        train_dataset = CachedTeacherDataset(train_base, Path(args.cache_dir) / "train")
        val_dataset   = CachedTeacherDataset(val_base,   Path(args.cache_dir) / "val")
        _collate      = collate_fn_with_teacher

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=_collate,
        pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=_collate, pin_memory=pin,
    )
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples\n")

    # ── Models ────────────────────────────────────────────────────────────
    model = StudentBEV(config["model"]).to(device)

    teacher = None
    if args.mock_teacher:
        teacher = TeacherBEVFusion(mock=True).to(device)
        teacher.eval()

    params = model.count_parameters()
    print("Student Parameters:")
    for name, count in params.items():
        print(f"  {name:<30} {count:>10,}")
    print()

    # ── Loss functions ────────────────────────────────────────────────────
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    kd_loss_fn = CombinedKDLoss(
        alpha=dist_cfg.get("alpha", 1.0),
        beta=dist_cfg.get("beta",  0.5),
        temperature=dist_cfg.get("temperature", 4.0),
        student_channels=256,
        teacher_channels=256,
        student_h=50, student_w=50,
    )

    # ── Optimiser & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + torch.cos(
            torch.tensor((epoch - warmup) / (epochs - warmup) * 3.14159)
        ).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")
    global_step = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}  (val loss: {best_val:.4f})\n")

    # ── Logging ───────────────────────────────────────────────────────────
    log_dir  = Path("logs")  / args.run_name
    ckpt_dir = Path("checkpoints")
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir)

    history = []

    # ── Training Loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}  (lr={optimizer.param_groups[0]['lr']:.6f})")
        print("-" * 70)

        avgs, global_step = train_one_epoch_kd(
            model, train_loader, teacher, optimizer, scheduler,
            focal_loss_fn, kd_loss_fn, device, epoch, writer,
            config, global_step, args.mock_teacher,
        )

        val_loss = validate_kd(
            model, val_loader, focal_loss_fn, kd_loss_fn, device, config,
            use_mock_teacher=args.mock_teacher, teacher=teacher,
        )

        history.append({"epoch": epoch, **avgs, "val_loss": val_loss})

        print(f"\n  ▶ Epoch {epoch:02d} summary:")
        print(f"    Total:    {avgs['total']:.4f}  "
              f"(task={avgs['task']:.3f}  kd={avgs['kd']:.3f})")
        print(f"    KD:       feat={avgs['feature']:.3f}  "
              f"kl={avgs['hm_kl']:.3f}  reg={avgs['reg_l1']:.3f}")
        print(f"    Val loss: {val_loss:.4f}")

        if writer:
            writer.add_scalar("epoch/loss_total",   avgs["total"],   epoch)
            writer.add_scalar("epoch/loss_task",    avgs["task"],    epoch)
            writer.add_scalar("epoch/loss_kd",      avgs["kd"],      epoch)
            writer.add_scalar("epoch/loss_feature", avgs["feature"], epoch)
            writer.add_scalar("epoch/val_loss",     val_loss,        epoch)
            writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], epoch)

        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_loss":    val_loss,
            "config":      config,
        }, ckpt_dir / f"{args.run_name}_latest.pth")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "config":      config,
            }, ckpt_dir / f"{args.run_name}_best.pth")
            print(f"    ✅ New best: {best_val:.4f} → saved {args.run_name}_best.pth")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  KD Training complete!")
    print(f"  Best val loss: {best_val:.4f}")
    print(f"  TensorBoard:   tensorboard --logdir {log_dir}")
    print(f"{'='*60}\n")

    print("Epoch | Train Total | Task  | KD    | Val")
    print("------|-------------|-------|-------|------")
    for row in history:
        print(f"  {row['epoch']:02d}  |  {row['total']:.4f}     | "
              f"{row['task']:.3f} | {row['kd']:.3f} | {row['val_loss']:.4f}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
