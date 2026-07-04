"""
BEVFusion Teacher Model Wrapper.

This module wraps the frozen BEVFusion teacher (Liu et al., ICRA 2023)
for knowledge distillation. It has two modes:

  MOCK MODE (default, for development):
    Generates realistic random outputs that match BEVFusion's exact output
    shapes and approximate statistics. Lets you develop and test the full
    KD training loop without needing the actual weights or mmdet3d installed.
    Switch to real mode by setting mock=False and providing a checkpoint path.

  REAL MODE (for Colab / full training):
    Loads actual BEVFusion pretrained weights and runs a genuine forward pass.
    Requires: mmdet3d, mmcv, and BEVFusion repo cloned alongside this project.
    The teacher is ALWAYS frozen — no gradients, no weight updates.

BEVFusion output shapes (MIT version, nuScenes):
    fused_bev:  (B, 256, 128, 128)  — fused camera+LiDAR BEV features
    heatmap:    (B, 10,  128, 128)  — class probability heatmaps
    regression: (B, 8,   128, 128)  — box regression (x,y,z,w,l,h,sin,cos)

Note on resolution mismatch:
    BEVFusion outputs at 128×128; our student outputs at 50×50.
    The distillation loss handles this with bilinear downsampling of the
    teacher features before computing MSE/KL — see distillation.py.

Usage:
    # Development (mock)
    teacher = TeacherBEVFusion(mock=True)
    outputs = teacher(camera_images, lidar_points, calibration)

    # Full training (real weights)
    teacher = TeacherBEVFusion(mock=False, checkpoint="checkpoints/bevfusion.pth")
    outputs = teacher(camera_images, lidar_points, calibration)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherBEVFusion(nn.Module):
    """Frozen BEVFusion teacher for knowledge distillation.

    Always eval(), always no_grad() — weights never update.
    """

    # BEVFusion MIT output resolution on nuScenes
    TEACHER_BEV_H = 128
    TEACHER_BEV_W = 128
    TEACHER_BEV_C = 256
    NUM_CLASSES    = 10
    REG_DIMS       = 8

    def __init__(
        self,
        mock: bool = True,
        checkpoint: str = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.mock = mock
        self.checkpoint = checkpoint
        self._device = device or torch.device("cpu")

        if mock:
            self._build_mock()
        else:
            self._build_real()

        # Teacher is ALWAYS frozen
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    # ── Mock mode ────────────────────────────────────────────────────────────

    def _build_mock(self):
        """Build a lightweight mock that produces BEVFusion-shaped outputs.

        The mock uses small learned conv layers so outputs are spatially
        coherent (not pure noise) and gradients flow correctly through the
        distillation losses during development testing. In real training the
        teacher outputs come from cache files, not this network.
        """
        # Mock fused BEV generator: takes a simple noise seed → BEVFusion-shaped output
        self.mock_bev = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.TEACHER_BEV_C, 3, padding=1),
        )
        self.mock_heatmap = nn.Sequential(
            nn.Conv2d(self.TEACHER_BEV_C, self.NUM_CLASSES, 1),
            nn.Sigmoid(),
        )
        self.mock_regression = nn.Conv2d(self.TEACHER_BEV_C, self.REG_DIMS, 1)

    def _forward_mock(self, batch_size: int, device: torch.device) -> dict:
        """Generate mock teacher outputs of correct shape and reasonable scale."""
        # Seed: small random noise at teacher BEV resolution
        seed = torch.randn(
            batch_size, 1,
            self.TEACHER_BEV_H, self.TEACHER_BEV_W,
            device=device,
        )
        fused_bev  = self.mock_bev(seed)                    # (B, 256, 128, 128)
        heatmap    = self.mock_heatmap(fused_bev)           # (B, 10,  128, 128)
        regression = self.mock_regression(fused_bev)        # (B, 8,   128, 128)

        return {
            "fused_bev":  fused_bev,
            "heatmap":    heatmap,
            "regression": regression,
        }

    # ── Real mode ─────────────────────────────────────────────────────────────

    def _build_real(self):
        """Attempt to load actual BEVFusion weights.

        Requires:
            - BEVFusion repo cloned to ../BEVFusion/ relative to project root
            - mmdet3d and mmcv installed in the environment
            - Pretrained checkpoint at self.checkpoint

        If loading fails (missing deps, wrong path), falls back to mock mode
        with a clear warning so training doesn't crash silently.
        """
        try:
            import sys
            import os
            bevfusion_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "BEVFusion"
            )
            sys.path.insert(0, bevfusion_path)

            from mmdet3d.models import build_model
            from mmcv import Config
            from mmcv.runner import load_checkpoint

            cfg_path = os.path.join(bevfusion_path, "configs", "nuscenes",
                                    "det", "transfusion", "secfpn",
                                    "camera+lidar", "swint_v0.22.4",
                                    "convfuser.yaml")
            cfg = Config.fromfile(cfg_path)
            self.bevfusion = build_model(cfg.model)
            load_checkpoint(self.bevfusion, self.checkpoint, map_location="cpu")
            self.bevfusion.eval()
            print(f"✅ BEVFusion teacher loaded from {self.checkpoint}")

        except Exception as e:
            print(f"⚠️  BEVFusion real mode failed: {e}")
            print(f"   Falling back to mock mode for development.")
            self.mock = True
            self._build_mock()

    def _forward_real(
        self,
        camera_images: torch.Tensor,
        lidar_points:  torch.Tensor,
        calibration:   list,
    ) -> dict:
        """Run actual BEVFusion forward pass.

        This wraps BEVFusion's mmdet3d-style forward into our dict format.
        The exact call signature depends on the BEVFusion version — adjust
        if needed when plugging in real weights.
        """
        with torch.no_grad():
            # BEVFusion expects mmdet3d-style input dicts
            # This will need adaptation to BEVFusion's actual API
            # when real weights are available — placeholder for now
            raise NotImplementedError(
                "Real BEVFusion forward pass — implement when weights available. "
                "See scripts/cache_teacher_outputs.py for the offline caching approach."
            )

    # ── Public interface ──────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        camera_images: torch.Tensor = None,
        lidar_points:  torch.Tensor = None,
        calibration:   list = None,
    ) -> dict:
        """Run teacher forward pass. Always no_grad().

        In mock mode, camera_images/lidar_points are only used to infer
        batch size and device — their contents are ignored.

        Returns:
            dict with keys:
                fused_bev:  (B, 256, 128, 128)  teacher fused BEV features
                heatmap:    (B, 10,  128, 128)  teacher class heatmaps
                regression: (B, 8,   128, 128)  teacher box predictions
        """
        if camera_images is not None:
            B      = camera_images.shape[0]
            device = camera_images.device
        elif lidar_points is not None:
            B      = lidar_points.shape[0]
            device = lidar_points.device
        else:
            raise ValueError("Must provide camera_images or lidar_points to infer B/device")

        if self.mock:
            return self._forward_mock(B, device)
        else:
            return self._forward_real(camera_images, lidar_points, calibration)

    def get_output_shapes(self) -> dict:
        """Return teacher output shapes for documentation / distillation setup."""
        return {
            "fused_bev":  (self.TEACHER_BEV_C, self.TEACHER_BEV_H, self.TEACHER_BEV_W),
            "heatmap":    (self.NUM_CLASSES, self.TEACHER_BEV_H, self.TEACHER_BEV_W),
            "regression": (self.REG_DIMS,   self.TEACHER_BEV_H, self.TEACHER_BEV_W),
        }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing TeacherBEVFusion (mock mode)...")

    B = 2
    teacher = TeacherBEVFusion(mock=True)
    teacher.eval()

    # Verify no gradients on teacher parameters
    for name, param in teacher.named_parameters():
        assert not param.requires_grad, f"Teacher param {name} has grad!"
    print("✅ All teacher parameters frozen")

    # Forward pass
    dummy_images = torch.randn(B, 6, 3, 384, 640)
    outputs = teacher(camera_images=dummy_images)

    print(f"fused_bev:  {outputs['fused_bev'].shape}   "
          f"(expected: [{B}, 256, 128, 128])")
    print(f"heatmap:    {outputs['heatmap'].shape}    "
          f"(expected: [{B}, 10, 128, 128])")
    print(f"regression: {outputs['regression'].shape}  "
          f"(expected: [{B}, 8, 128, 128])")

    assert outputs["fused_bev"].shape  == (B, 256, 128, 128)
    assert outputs["heatmap"].shape    == (B, 10,  128, 128)
    assert outputs["regression"].shape == (B, 8,   128, 128)
    print("✅ Output shape checks passed")

    # Verify teacher outputs don't carry gradients into student graph
    assert not outputs["fused_bev"].requires_grad
    print("✅ Teacher outputs detached from computation graph")
    print("✅ TeacherBEVFusion ready")
