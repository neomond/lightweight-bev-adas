"""
Simplified PointPillars encoder for LiDAR feature extraction.

Converts LiDAR point clouds into BEV feature maps by:
1. Organizing points into vertical pillars on a 2D grid
2. Encoding each pillar with a small PointNet
3. Scattering encoded features back to the BEV grid
"""

import torch
import torch.nn as nn
import numpy as np


class PillarFeatureNet(nn.Module):
    """Encodes points within each pillar."""

    def __init__(self, in_channels: int = 9, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, pillar_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pillar_features: (N_pillars, max_points, in_channels)
        Returns:
            Encoded pillars: (N_pillars, out_channels)
        """
        B, N, C = pillar_features.shape
        x = pillar_features.reshape(-1, C)
        x = self.net(x)
        x = x.reshape(B, N, -1)
        # Max pooling over points in each pillar
        x = x.max(dim=1)[0]
        return x


class PointPillarsEncoder(nn.Module):
    """Full PointPillars encoder: points -> BEV feature map.
    
    In the pipeline:
        LiDAR Points -> [PointPillarsEncoder] -> LiDAR BEV Features
    """

    def __init__(
        self,
        x_range: tuple = (-50.0, 50.0),
        y_range: tuple = (-50.0, 50.0),
        z_range: tuple = (-5.0, 3.0),
        pillar_size: float = 0.5,
        max_points_per_pillar: int = 32,
        max_pillars: int = 10000,
        in_channels: int = 4,  # x, y, z, intensity
        out_channels: int = 64,
    ):
        super().__init__()
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.pillar_size = pillar_size
        self.max_points = max_points_per_pillar
        self.max_pillars = max_pillars

        self.grid_x = int((x_range[1] - x_range[0]) / pillar_size)
        self.grid_y = int((y_range[1] - y_range[0]) / pillar_size)

        # Point features: original (4) + relative to pillar center (3) + pillar center (2) = 9
        self.pillar_net = PillarFeatureNet(in_channels + 5, out_channels)

        # BEV backbone (simple CNN to process the scattered features)
        self.bev_backbone = nn.Sequential(
            nn.Conv2d(out_channels, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self._out_channels = 256

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 4) - batch of point clouds (x, y, z, intensity)
        Returns:
            BEV features: (B, C, H, W)
        """
        # For now, return a placeholder of the correct shape
        # Full pillarization will be implemented in Milestone 3
        B = points.shape[0]
        device = points.device
        bev = torch.zeros(B, 64, self.grid_y, self.grid_x, device=device)
        return self.bev_backbone(bev)

    @property
    def output_channels(self) -> int:
        return self._out_channels

    @property
    def output_size(self) -> tuple:
        return (self.grid_y // 4, self.grid_x // 4)  # After 2x stride=2


if __name__ == "__main__":
    model = PointPillarsEncoder()
    dummy_points = torch.randn(1, 30000, 4)
    bev_features = model(dummy_points)
    print(f"LiDAR BEV features: {bev_features.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
