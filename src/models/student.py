"""
Complete Student BEV Model.

Assembles all components into the full pipeline:
    Camera Images -> YOLO Backbone -> Camera-to-BEV
    LiDAR Points  -> PointPillars  -> LiDAR BEV
    Both BEV      -> Fusion Module -> Fused BEV -> Detection Head
"""

import torch
import torch.nn as nn
from .yolo_backbone import YOLOBackbone
from .pointpillars import PointPillarsEncoder
from .fusion import ChannelWiseFusion
from .bev_head import BEVDetectionHead


class StudentBEV(nn.Module):
    """Complete student model for knowledge-distilled BEV perception."""

    def __init__(self, config: dict = None):
        super().__init__()
        config = config or {}

        # Camera branch
        self.camera_backbone = YOLOBackbone(
            model_size=config.get("yolo_model", "yolo11n.pt")
        )

        # LiDAR branch
        self.lidar_encoder = PointPillarsEncoder(
            out_channels=config.get("lidar_channels", 64)
        )

        # Fusion module (KD target)
        cam_channels = config.get("camera_bev_channels", 256)
        lid_channels = self.lidar_encoder.output_channels
        self.fusion = ChannelWiseFusion(
            camera_channels=cam_channels,
            lidar_channels=lid_channels,
            out_channels=config.get("fused_channels", 256),
        )

        # Detection head
        self.detection_head = BEVDetectionHead(
            in_channels=config.get("fused_channels", 256),
            num_classes=config.get("num_classes", 10),
        )

    def forward(self, camera_images: torch.Tensor, lidar_points: torch.Tensor) -> dict:
        """
        Args:
            camera_images: (B, N_cams, 3, H, W) multi-view camera images
            lidar_points:  (B, N_points, 4) LiDAR point cloud
            
        Returns:
            Dict with detection outputs and intermediate features for KD
        """
        # Camera branch
        B, N_cams = camera_images.shape[:2]
        # Process each camera view
        cam_features_list = []
        for i in range(N_cams):
            features = self.camera_backbone(camera_images[:, i])
            cam_features_list.append(features)

        # TODO (Milestone 2): Camera-to-BEV view transformation
        # For now, use a placeholder
        camera_bev = torch.zeros(B, 256, 50, 50, device=camera_images.device)

        # LiDAR branch
        lidar_bev = self.lidar_encoder(lidar_points)

        # Fusion (KD happens here)
        fused_bev = self.fusion(camera_bev, lidar_bev)

        # Detection
        detections = self.detection_head(fused_bev)

        return {
            "detections": detections,
            # Intermediate features for knowledge distillation
            "camera_bev": camera_bev,
            "lidar_bev": lidar_bev,
            "fused_bev": fused_bev,
        }

    def count_parameters(self) -> dict:
        """Count parameters per component."""
        components = {
            "Camera (YOLO)": self.camera_backbone,
            "LiDAR (PointPillars)": self.lidar_encoder,
            "Fusion Module": self.fusion,
            "Detection Head": self.detection_head,
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
    print("-" * 40)
    for name, count in params.items():
        print(f"  {name:<25} {count:>10,}")
