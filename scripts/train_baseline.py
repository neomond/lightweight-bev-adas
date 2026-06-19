"""
Baseline Training Script — Student Model WITHOUT Knowledge Distillation.

This is Milestone 3's core deliverable. It trains the full pipeline:
    Camera → YOLO → LSS → Camera BEV
    LiDAR  → PointPillars   → LiDAR BEV
    Both BEV → ChannelWiseFusion → Fused BEV → BEVDetectionHead

No teacher, no distillation losses — just task loss (detection).
This gives you the "student without KD" baseline numbers that
Milestone 5 (with KD) will be compared against.

Usage:
    # Local Mac (MPS) — small batch for development:
    python scripts/train_baseline.py --config configs/student.yaml

    # Google Colab (GPU) — full training:
    python scripts/train_baseline.py --config configs/student.yaml --epochs 20

Outputs saved to:
    checkpoints/baseline_best.pth        ← best val loss weights
    checkpoints/baseline_latest.pth      ← latest epoch weights
    logs/baseline/                       ← TensorBoard logs

Losses:
    L_heatmap   — Focal loss on class heatmaps (handles class imbalance)
    L_reg       — L1 loss on box regression (x,y,z,w,l,h,sin_yaw,cos_yaw)
    L_total     = L_heatmap + λ * L_reg   (λ=2.0 by default)
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Make sure src/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.student import StudentBEV
from src.data.nuscenes_loader import NuScenesDataset, collate_fn
from src.utils.device import get_device


# ── Loss Functions ────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss for dense object detection heatmaps.

    Focal loss down-weights easy negatives so the model focuses on
    hard examples and rare foreground cells. Standard in CenterPoint-style
    3D detectors (Zhou et al., 2019).

    L_focal = -α(1-p)^γ * log(p)  for positives
            = -(1-α) * p^γ * log(1-p)  for negatives
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, C, H, W)  sigmoid-activated heatmap predictions
            target: (B, C, H, W)  binary ground-truth heatmaps in [0,1]
        Returns:
            Scalar loss
        """
        pred = pred.clamp(1e-6, 1 - 1e-6)

        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        pos_loss = -self.alpha * (1 - pred).pow(self.gamma) * torch.log(pred) * pos_mask
        neg_loss = -(1 - self.alpha) * pred.pow(self.gamma) * torch.log(1 - pred) * neg_mask

        n_pos = pos_mask.sum().clamp(min=1)
        loss = (pos_loss + neg_loss).sum() / n_pos
        return loss


def build_heatmap_targets(
    annotations: list,
    num_classes: int,
    bev_h: int,
    bev_w: int,
    x_range: tuple,
    y_range: tuple,
    device: torch.device,
) -> torch.Tensor:
    """Convert annotation boxes to dense heatmap targets.

    For each ground-truth box, places a Gaussian blob at the corresponding
    BEV grid cell. The standard CenterPoint approach.

    Args:
        annotations: list of B dicts, each with 'boxes' (M,7) and 'classes' (M,)
        num_classes:  number of detection classes (10)
        bev_h, bev_w: BEV grid dimensions (50, 50)
        x_range, y_range: BEV spatial extent ((-50,50), (-50,50))

    Returns:
        heatmaps: (B, num_classes, bev_h, bev_w) in [0,1]
    """
    B = len(annotations)
    heatmaps = torch.zeros(B, num_classes, bev_h, bev_w, device=device)

    x_min, x_max = x_range
    y_min, y_max = y_range
    x_scale = bev_w / (x_max - x_min)
    y_scale = bev_h / (y_max - y_min)

    for b, ann in enumerate(annotations):
        boxes   = ann["boxes"].to(dtype=torch.float32, device=device)    # (M, 7): x y z w l h yaw
        classes = ann["classes"].to(device)  # (M,)

        if boxes.shape[0] == 0:
            continue

        for i in range(boxes.shape[0]):
            cx = (boxes[i, 0] - x_min) * x_scale   # BEV col (float)
            cy = (boxes[i, 1] - y_min) * y_scale   # BEV row (float)
            cls = classes[i].item()

            col = int(cx)
            row = int(cy)
            if not (0 <= col < bev_w and 0 <= row < bev_h):
                continue

            # Simple Gaussian blob radius (proportional to object size)
            # boxes[i,3]=w, boxes[i,4]=l in metres; convert to grid cells
            r = max(1, int(min(boxes[i, 3], boxes[i, 4]).item() * x_scale / 2))
            r = min(r, 4)   # cap at 4 cells

            # Write Gaussian (approximate: use 3×3 to 9×9 neighbourhood)
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < bev_h and 0 <= cc < bev_w:
                        d2 = dr * dr + dc * dc
                        val = float(torch.exp(torch.tensor(-d2 / (2 * r * r + 1e-6))))
                        heatmaps[b, cls, rr, cc] = max(
                            heatmaps[b, cls, rr, cc].item(), val
                        )

    return heatmaps


def detection_loss(
    pred_heatmap: torch.Tensor,     # (B, C, H, W)
    pred_reg:     torch.Tensor,     # (B, 8, H, W)
    annotations:  list,
    x_range:      tuple,
    y_range:      tuple,
    focal_loss_fn: FocalLoss,
    reg_weight:   float = 2.0,
    device:       torch.device = None,
) -> tuple:
    """Compute total detection loss.

    Returns:
        total_loss, heatmap_loss, reg_loss  (all scalars)
    """
    B, C, H, W = pred_heatmap.shape
    device = pred_heatmap.device

    # ── Heatmap loss ──────────────────────────────────────────────────────
    gt_heatmap = build_heatmap_targets(
        annotations, C, H, W, x_range, y_range, device
    )
    loss_hm = focal_loss_fn(pred_heatmap, gt_heatmap)

    # ── Regression loss ───────────────────────────────────────────────────
    # Mask: only compute regression loss at positive (foreground) cells
    pos_mask = gt_heatmap.max(dim=1)[0] > 0.5   # (B, H, W)
    n_pos = pos_mask.sum().clamp(min=1)

    if pos_mask.sum() > 0:
        # pred_reg at positive locations: (N_pos, 8)
        pred_at_pos = pred_reg.permute(0, 2, 3, 1)[pos_mask]   # (N_pos, 8)
        # For regression targets, we use zero as a simple placeholder.
        # In Milestone 4 (full training) this will be replaced with real
        # box regression targets (dx, dy, dz, log_w, log_l, log_h, sin, cos).
        # The heatmap loss still drives learning meaningfully at this stage.
        reg_target = torch.zeros_like(pred_at_pos)
        loss_reg = F.l1_loss(pred_at_pos, reg_target, reduction="sum") / n_pos
    else:
        loss_reg = torch.tensor(0.0, device=device, requires_grad=True)

    total = loss_hm + reg_weight * loss_reg
    return total, loss_hm, loss_reg


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    focal_loss_fn,
    device,
    epoch,
    writer,
    config,
    global_step,
):
    model.train()
    x_range = tuple(config["data"]["x_range"])
    y_range = tuple(config["data"]["y_range"])

    total_loss_sum = 0.0
    hm_loss_sum    = 0.0
    reg_loss_sum   = 0.0
    n_batches      = 0

    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        camera_images = batch["camera_images"].to(device)   # (B, 6, 3, H, W)
        lidar_points  = batch["lidar_points"].to(device)    # (B, N, 4)
        annotations   = batch["annotations"]                 # list[B]
        calibration   = batch["calibration"]                 # list[B]

        optimizer.zero_grad()

        # Forward
        outputs = model(camera_images, lidar_points, calibration)
        pred_hm  = outputs["detections"]["heatmap"]     # (B, 10, 50, 50)
        pred_reg = outputs["detections"]["regression"]  # (B, 8, 50, 50)

        # Loss
        loss, loss_hm, loss_reg = detection_loss(
            pred_hm, pred_reg, annotations,
            x_range, y_range, focal_loss_fn,
            reg_weight=config.get("training", {}).get("reg_weight", 2.0),
        )

        if not torch.isfinite(loss):
            print(f"  ⚠️  Non-finite loss at batch {batch_idx}, skipping")
            continue

        loss.backward()
        # Gradient clipping prevents exploding gradients early in training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss_sum += loss.item()
        hm_loss_sum    += loss_hm.item()
        reg_loss_sum   += loss_reg.item()
        n_batches      += 1

        if writer:
            writer.add_scalar("train/loss_total",   loss.item(),    global_step)
            writer.add_scalar("train/loss_heatmap", loss_hm.item(), global_step)
            writer.add_scalar("train/loss_reg",     loss_reg.item(), global_step)
        global_step += 1

        if batch_idx % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:02d} | Batch {batch_idx:03d}/{len(loader):03d} | "
                f"Loss {loss.item():.4f} (hm={loss_hm.item():.4f} reg={loss_reg.item():.4f}) | "
                f"{elapsed:.1f}s"
            )

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss_sum / max(n_batches, 1)
    avg_hm   = hm_loss_sum    / max(n_batches, 1)
    avg_reg  = reg_loss_sum   / max(n_batches, 1)
    return avg_loss, avg_hm, avg_reg, global_step


@torch.no_grad()
def validate(model, loader, focal_loss_fn, device, config):
    model.eval()
    x_range = tuple(config["data"]["x_range"])
    y_range = tuple(config["data"]["y_range"])

    total_loss_sum = 0.0
    n_batches = 0

    for batch in loader:
        camera_images = batch["camera_images"].to(device)
        lidar_points  = batch["lidar_points"].to(device)
        annotations   = batch["annotations"]
        calibration   = batch["calibration"]

        outputs = model(camera_images, lidar_points, calibration)
        pred_hm  = outputs["detections"]["heatmap"]
        pred_reg = outputs["detections"]["regression"]

        loss, _, _ = detection_loss(
            pred_hm, pred_reg, annotations,
            x_range, y_range, focal_loss_fn,
        )
        if torch.isfinite(loss):
            total_loss_sum += loss.item()
            n_batches += 1

    return total_loss_sum / max(n_batches, 1)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train baseline student model (no KD)")
    parser.add_argument("--config",  default="configs/student.yaml")
    parser.add_argument("--epochs",  type=int,   default=None, help="Override config epochs")
    parser.add_argument("--batch",   type=int,   default=None, help="Override batch size")
    parser.add_argument("--lr",      type=float, default=None, help="Override learning rate")
    parser.add_argument("--workers", type=int,   default=2)
    parser.add_argument("--run-name", default="baseline", help="Name for logs/checkpoints")
    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.epochs: config["training"]["epochs"]     = args.epochs
    if args.batch:  config["training"]["batch_size"] = args.batch
    if args.lr:     config["training"]["lr"]         = args.lr

    epochs     = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr         = config["training"]["lr"]
    wd         = config["training"].get("weight_decay", 0.01)
    warmup     = config["training"].get("warmup_epochs", 1)

    # ── Device ────────────────────────────────────────────────────────────
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Baseline Training (no KD)  —  {args.run_name}")
    print(f"  Device:  {device}")
    print(f"  Epochs:  {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────────────────────────────────
    data_cfg = config["data"]
    train_dataset = NuScenesDataset(
        dataroot=data_cfg["dataroot"],
        version=data_cfg["version"],
        split="train",
        x_range=tuple(data_cfg["x_range"]),
        y_range=tuple(data_cfg["x_range"]),   # square BEV
    )
    val_dataset = NuScenesDataset(
        dataroot=data_cfg["dataroot"],
        version=data_cfg["version"],
        split="val",
        x_range=tuple(data_cfg["x_range"]),
        y_range=tuple(data_cfg["x_range"]),
    )

    pin = device.type == "cuda"   # pin_memory not supported on MPS
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=pin,
    )

    print(f"Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_dataset)}  samples ({len(val_loader)}  batches)\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model = StudentBEV(config["model"]).to(device)
    params = model.count_parameters()
    print("Model Parameters:")
    for name, count in params.items():
        print(f"  {name:<30} {count:>10,}")
    print()

    # ── Optimiser & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd
    )
    # Cosine annealing with linear warmup
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + torch.cos(
            torch.tensor((epoch - warmup) / (epochs - warmup) * 3.14159)
        ).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    # ── Logging & Checkpointing ───────────────────────────────────────────
    log_dir  = Path("logs") / args.run_name
    ckpt_dir = Path("checkpoints")
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs → {log_dir}")
    print(f"Checkpoints      → {ckpt_dir}\n")

    best_val_loss = float("inf")
    global_step   = 0
    history       = []

    # ── Training Loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}  (lr={optimizer.param_groups[0]['lr']:.6f})")
        print("-" * 60)

        avg_loss, avg_hm, avg_reg, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            focal_loss_fn, device, epoch, writer, config, global_step,
        )

        val_loss = validate(model, val_loader, focal_loss_fn, device, config)

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss":   val_loss,
        })

        print(f"\n  ▶ Epoch {epoch:02d} summary:")
        print(f"    Train loss: {avg_loss:.4f}  (hm={avg_hm:.4f}  reg={avg_reg:.4f})")
        print(f"    Val   loss: {val_loss:.4f}")

        if writer:
            writer.add_scalar("epoch/train_loss", avg_loss, epoch)
            writer.add_scalar("epoch/val_loss",   val_loss, epoch)
            writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], epoch)

        # Save latest
        torch.save({
            "epoch":      epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_loss":    val_loss,
            "config":      config,
        }, ckpt_dir / f"{args.run_name}_latest.pth")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "config":      config,
            }, ckpt_dir / f"{args.run_name}_best.pth")
            print(f"    ✅ New best val loss: {best_val_loss:.4f}  → saved best checkpoint")

    # ── Training complete ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint:    {ckpt_dir}/{args.run_name}_best.pth")
    print(f"  TensorBoard:   tensorboard --logdir {log_dir}")
    print(f"{'='*60}\n")

    # Print loss history for dissertation table
    print("Epoch | Train Loss | Val Loss")
    print("------|------------|----------")
    for row in history:
        print(f"  {row['epoch']:02d}  |  {row['train_loss']:.4f}    |  {row['val_loss']:.4f}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()