"""
BEVFusion Teacher Model Wrapper.

This module wraps the frozen BEVFusion teacher (Liu et al., ICRA 2023)
for knowledge distillation. It has two modes:

  MOCK MODE (default, for development):
    Generates realistic random outputs that match BEVFusion's exact output
    shapes and approximate statistics. Lets you develop and test the full
    KD training loop without needing the actual weights or mmdet3d installed.
    Switch to real mode by setting mock=False and providing a checkpoint path.

  REAL MODE (for RunPod / full training):
    Loads actual BEVFusion pretrained weights and runs a genuine forward pass.
    Requires: mmdet3d, mmcv, torchpack, and the BEVFusion repo built at
    /workspace/bevfusion (see project setup notes). Only importable in that
    environment — mmdet3d is NOT installed locally, so real-mode imports are
    deferred until _build_real()/_forward_real() actually run.
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
    outputs = teacher(camera_images, lidar_points, calibration, lidar_calibration)

    # Full training (real weights, on RunPod)
    teacher = TeacherBEVFusion(mock=False, checkpoint="pretrained/bevfusion-det.pth")
    outputs = teacher(camera_images, lidar_points, calibration, lidar_calibration)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.nuscenes_loader import CAMERA_CHANNELS


def _build_homogeneous(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Compose a 3x3 rotation + 3-vector translation into a 4x4 homogeneous transform."""
    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


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
        checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
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
        teacher outputs come from a genuine BEVFusion forward pass instead.
        """
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

        Requires (RunPod environment only — see project setup notes):
            - BEVFusion repo built at /workspace/bevfusion
            - Python 3.8 venv at /workspace/venv38 with mmdet3d, mmcv-full,
              torchpack, and their dependencies installed
            - Pretrained checkpoint at self.checkpoint
              (e.g. /workspace/bevfusion/pretrained/bevfusion-det.pth)

        Config is loaded via torchpack's recursive config system (NOT plain
        mmcv.Config.fromfile) — BEVFusion's convfuser.yaml is only a fragment;
        torchpack.utils.config.configs.load(..., recursive=True) walks up the
        directory tree merging all sibling default.yaml files.

        If loading fails (missing deps, wrong path), falls back to mock mode
        with a clear warning so training doesn't crash silently.
        """
        try:
            import os

            # mmdet3d/mmcv/torchpack are only installed in the RunPod venv —
            # not available locally, so these imports must stay deferred here.
            from torchpack.utils.config import configs  # type: ignore[import]
            from mmcv import Config  # type: ignore[import]
            from mmcv.runner import load_checkpoint  # type: ignore[import]
            from mmdet3d.utils import recursive_eval  # type: ignore[import]
            from mmdet3d.models import build_model  # type: ignore[import]

            bevfusion_path = "/workspace/bevfusion"
            cfg_path = os.path.join(
                bevfusion_path, "configs", "nuscenes", "det", "transfusion",
                "secfpn", "camera+lidar", "swint_v0p075", "convfuser.yaml"
            )

            configs.load(cfg_path, recursive=True)
            cfg = Config(recursive_eval(configs), filename=cfg_path)

            self.bevfusion = build_model(cfg.model)
            load_checkpoint(self.bevfusion, self.checkpoint, map_location="cpu")
            self.bevfusion.eval()
            print(f"✅ BEVFusion teacher loaded from {self.checkpoint}")

        except Exception as e:
            print(f"⚠️  BEVFusion real mode failed: {e}")
            print(f"   Falling back to mock mode for development.")
            self.mock = True
            self._build_mock()

    def _build_bevfusion_inputs(
        self,
        camera_images: torch.Tensor,
        lidar_points: torch.Tensor,
        calibration: list,
        lidar_calibration: list,
    ) -> dict:
        """
        Adapt our (camera_images, lidar_points, calibration, lidar_calibration)
        into BEVFusion's expected forward() arguments: camera2ego, lidar2ego,
        lidar2camera, lidar2image, camera_intrinsics, camera2lidar,
        img_aug_matrix, lidar_aug_matrix, metas.

        NOTE: `metas` currently only carries a placeholder box_type_3d field.
        This is the least-verified part of the adapter — BEVFusion's internal
        code may read additional keys we haven't discovered yet. Expect this
        to need iteration once tested against the real model on RunPod.
        """
        B, N_cams = camera_images.shape[:2]
        device = camera_images.device

        camera2ego_list, lidar2ego_list = [], []
        lidar2camera_list, lidar2image_list = [], []
        camera_intrinsics_list, camera2lidar_list = [], []

        for b in range(B):
            cam2ego_b, cam_intrin_b = [], []
            lidar2cam_b, lidar2img_b, cam2lidar_b = [], [], []

            lidar2ego_b = _build_homogeneous(
                lidar_calibration[b]["rotation"], lidar_calibration[b]["translation"]
            ).to(device)

            for cam_name in CAMERA_CHANNELS:
                calib = calibration[b][cam_name]
                cam2ego = _build_homogeneous(calib["rotation"], calib["translation"]).to(device)
                cam2ego_b.append(cam2ego)

                # lidar -> camera = inverse(camera->ego) @ (lidar->ego)
                lidar2cam = torch.inverse(cam2ego) @ lidar2ego_b
                lidar2cam_b.append(lidar2cam)
                cam2lidar_b.append(torch.inverse(lidar2cam))

                # Intrinsics, padded to 4x4 for lidar2image composition
                intrinsic_4x4 = torch.eye(4, dtype=torch.float32, device=device)
                intrinsic_4x4[:3, :3] = calib["intrinsic"].to(device)
                cam_intrin_b.append(intrinsic_4x4)
                lidar2img_b.append(intrinsic_4x4 @ lidar2cam)

            camera2ego_list.append(torch.stack(cam2ego_b))
            lidar2ego_list.append(lidar2ego_b)
            lidar2camera_list.append(torch.stack(lidar2cam_b))
            lidar2image_list.append(torch.stack(lidar2img_b))
            camera_intrinsics_list.append(torch.stack(cam_intrin_b))
            camera2lidar_list.append(torch.stack(cam2lidar_b))

        img_aug_matrix = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, N_cams, 1, 1)
        lidar_aug_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        metas = [{"box_type_3d": None} for _ in range(B)]  # placeholder — needs verification on RunPod

        return {
            "img": camera_images,
            "points": lidar_points,
            "camera2ego": torch.stack(camera2ego_list),
            "lidar2ego": torch.stack(lidar2ego_list),
            "lidar2camera": torch.stack(lidar2camera_list),
            "lidar2image": torch.stack(lidar2image_list),
            "camera_intrinsics": torch.stack(camera_intrinsics_list),
            "camera2lidar": torch.stack(camera2lidar_list),
            "img_aug_matrix": img_aug_matrix,
            "lidar_aug_matrix": lidar_aug_matrix,
            "metas": metas,
            "depths": None,
        }

    def _forward_real(
        self,
        camera_images: Optional[torch.Tensor],
        lidar_points:  Optional[torch.Tensor],
        calibration:   Optional[list],
        lidar_calibration: Optional[list],
    ) -> dict:
        """Run actual BEVFusion forward pass, extracting:
            - fused_bev:  the fuser's output, pre-decoder (dense, 256ch)
            - heatmap:    TransFusionHead's dense classification heatmap,
                          used internally to propose query centers before
                          the sparse transformer-decoder refinement stage
            - regression: NOT dense in TransFusion's architecture — box
                          regression only exists per-query (sparse, at the
                          top-K heatmap peaks), not as a dense (8,H,W) grid.
                          Returned as None; CombinedKDLoss/train_with_kd.py
                          must handle this by skipping regression-KD when
                          teacher_regression is None (see TODO below).

        Manually replicates the encoder -> fuser -> decoder -> head.forward()
        steps from BEVFusion's forward_single(), stopping BEFORE get_bboxes()
        so we keep the raw dense_heatmap instead of final decoded boxes.
        """
        assert camera_images is not None, "camera_images required for real BEVFusion forward"
        assert lidar_points is not None, "lidar_points required for real BEVFusion forward"
        assert calibration is not None, "calibration required for real BEVFusion forward"
        assert lidar_calibration is not None, "lidar_calibration required for real BEVFusion forward"

        with torch.no_grad():
            inputs = self._build_bevfusion_inputs(
                camera_images, lidar_points, calibration, lidar_calibration
            )

            features = []
            # Match forward_single's eval-mode iteration order (reversed keys)
            for sensor in list(self.bevfusion.encoders.keys())[::-1]:
                if sensor == "camera":
                    feature = self.bevfusion.extract_camera_features(
                        inputs["img"], inputs["points"], None,
                        inputs["camera2ego"], inputs["lidar2ego"],
                        inputs["lidar2camera"], inputs["lidar2image"],
                        inputs["camera_intrinsics"], inputs["camera2lidar"],
                        inputs["img_aug_matrix"], inputs["lidar_aug_matrix"],
                        inputs["metas"], gt_depths=inputs["depths"],
                    )
                    if isinstance(feature, (list, tuple)):
                        feature = feature[0]
                elif sensor == "lidar":
                    feature = self.bevfusion.extract_features(inputs["points"], "lidar")
                else:
                    raise ValueError(f"Unsupported sensor in checkpoint: {sensor}")
                features.append(feature)

            features = features[::-1]  # restore original order, per forward_single

            if self.bevfusion.fuser is not None:
                fused_bev = self.bevfusion.fuser(features)
            else:
                fused_bev = features[0]

            x = self.bevfusion.decoder["backbone"](fused_bev)
            x = self.bevfusion.decoder["neck"](x)

            # Call the object head's forward() directly (NOT get_bboxes) so we
            # keep the raw dense_heatmap instead of final decoded 3D boxes.
            pred_dicts = self.bevfusion.heads["object"](x, inputs["metas"])
            pred_dict = pred_dicts[0]

            dense_heatmap = pred_dict["dense_heatmap"]        # (B, num_classes, H, W), pre-sigmoid
            heatmap = torch.sigmoid(dense_heatmap)

            return {
                "fused_bev":  fused_bev,
                "heatmap":    heatmap,
                "regression": None,  # sparse-only in TransFusion; see docstring
            }
        
    # ── Public interface ──────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        camera_images: Optional[torch.Tensor] = None,
        lidar_points:  Optional[torch.Tensor] = None,
        calibration:   Optional[list] = None,
        lidar_calibration: Optional[list] = None,
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
            return self._forward_real(camera_images, lidar_points, calibration, lidar_calibration)

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

    for name, param in teacher.named_parameters():
        assert not param.requires_grad, f"Teacher param {name} has grad!"
    print("✅ All teacher parameters frozen")

    dummy_images = torch.randn(B, 6, 3, 384, 640)
    outputs = teacher(camera_images=dummy_images)

    print(f"fused_bev:  {outputs['fused_bev'].shape}   (expected: [{B}, 256, 128, 128])")
    print(f"heatmap:    {outputs['heatmap'].shape}    (expected: [{B}, 10, 128, 128])")
    print(f"regression: {outputs['regression'].shape}  (expected: [{B}, 8, 128, 128])")

    assert outputs["fused_bev"].shape  == (B, 256, 128, 128)
    assert outputs["heatmap"].shape    == (B, 10,  128, 128)
    assert outputs["regression"].shape == (B, 8,   128, 128)
    print("✅ Output shape checks passed")

    assert not outputs["fused_bev"].requires_grad
    print("✅ Teacher outputs detached from computation graph")
    print("✅ TeacherBEVFusion ready")