"""
Complete Student BEV Model.

Assembles all components into the full pipeline:
    Camera Images -> YOLO Backbone -> CameraToBEV (LSS) -> Camera BEV
    LiDAR Points  -> PointPillars                       -> LiDAR BEV
    Camera BEV + LiDAR BEV -> ChannelWiseFusion         -> Fused BEV
    Fused BEV              -> BEVDetectionHead          -> 3D Detections

Milestone 2 change:
    Replaced torch.zeros placeholder with CameraToBEV view transformer.
    The 6-camera loop now feeds real YOLO features into the LSS module.
"""

import torch
import torch.nn as nn
from .yolo_backbone import YOLOBackbone
from .pointpillars import PointPillarsEncoder
from .fusion import ChannelWiseFusion
from .bev_head import BEVDetectionHead
from .camera_to_bev import CameraToBEV


class StudentBEV(nn.Module):
    """Complete student model for knowledge-distilled BEV perception."""

    def __init__(self, config: dict = None):
        super().__init__()
        config = config or {}

        # ── Camera branch ───────────────────────────────────────────────────
        self.camera_backbone = YOLOBackbone(
            model_size=config.get("yolo_model", "yolo11n.pt")
        )

        # View transformer: 2D camera features → BEV  (Milestone 2)
        self.camera_to_bev = CameraToBEV(config)

        # ── LiDAR branch ────────────────────────────────────────────────────
        self.lidar_encoder = PointPillarsEncoder(
            out_channels=config.get("lidar_channels", 64)
        )

        # ── Fusion module (KD target) ────────────────────────────────────────
        cam_channels = config.get("camera_bev_channels", 256)
        lid_channels = self.lidar_encoder.output_channels
        self.fusion = ChannelWiseFusion(
            camera_channels=cam_channels,
            lidar_channels=lid_channels,
            out_channels=config.get("fused_channels", 256),
        )

        # ── Detection head ───────────────────────────────────────────────────
        self.detection_head = BEVDetectionHead(
            in_channels=config.get("fused_channels", 256),
            num_classes=config.get("num_classes", 10),
        )

    def forward(
        self,
        camera_images: torch.Tensor,   # (B, N_cams, 3, H, W)
        lidar_points: torch.Tensor,    # (B, N_points, 4)
        calibration: list = None,      # list[B] of dicts keyed by camera name
    ) -> dict:
        """
        Args:
            camera_images: (B, N_cams, 3, H, W) multi-view camera images
            lidar_points:  (B, N_points, 4) LiDAR point cloud (x, y, z, intensity)
            calibration:   list of length B; each element is a dict mapping
                           camera name → {intrinsic, rotation, translation, ...}
                           Required for the view transformer. If None, the
                           camera branch falls back to zeros (debug only).

        Returns:
            Dict with:
                detections  — heatmap + regression predictions
                camera_bev  — (B, 256, 50, 50) camera BEV features
                lidar_bev   — (B, 256, 50, 50) LiDAR BEV features
                fused_bev   — (B, 256, 50, 50) fused features (KD target)
        """
        B, N_cams = camera_images.shape[:2]

        # ── Camera branch ───────────────────────────────────────────────────
        # Run each camera image through the YOLO backbone
        cam_features_list = []
        for i in range(N_cams):
            # camera_images[:, i] → (B, 3, H, W)
            features = self.camera_backbone(camera_images[:, i])
            # features is a list of 3 tensors: [P3, P4, P5]
            cam_features_list.append(features)

        # View transformation: 6 × multi-scale features → BEV
        if calibration is not None:
            camera_bev = self.camera_to_bev(cam_features_list, calibration)
        else:
            # Fallback for unit-testing without calibration data
            camera_bev = torch.zeros(B, 256, 50, 50, device=camera_images.device)

        # ── LiDAR branch ────────────────────────────────────────────────────
        lidar_bev = self.lidar_encoder(lidar_points)

        # ── Fusion (KD applied here during training) ─────────────────────
        fused_bev = self.fusion(camera_bev, lidar_bev)

        # ── Detection ────────────────────────────────────────────────────
        detections = self.detection_head(fused_bev)

        return {
            "detections": detections,
            # Intermediate features returned for knowledge distillation
            "camera_bev": camera_bev,
            "lidar_bev":  lidar_bev,
            "fused_bev":  fused_bev,
        }

    def count_parameters(self) -> dict:
        """Count parameters per component."""
        components = {
            "Camera (YOLO)":       self.camera_backbone,
            "Camera-to-BEV (LSS)": self.camera_to_bev,
            "LiDAR (PointPillars)": self.lidar_encoder,
            "Fusion Module":        self.fusion,
            "Detection Head":       self.detection_head,
        }
        counts = {}
        for name, module in components.items():
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["Total"] = sum(counts.values())
        return counts


if __name__ == "__main__":
    model = StudentBEV()
    params = model.count_parameters()
    print("Model Parameters:")
    print("-" * 45)
    for name, count in params.items():
        print(f"  {name:<30} {count:>10,}")