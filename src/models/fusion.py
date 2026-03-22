"""
Channel-Wise Attention Fusion Module.

This is the CORE of the dissertation. It combines camera BEV features
and LiDAR BEV features using a lightweight channel attention mechanism,
replacing the expensive cross-attention transformers used in BEVFusion.

Knowledge distillation is applied HERE - the teacher (BEVFusion) guides
how this module learns to combine the two modalities.
"""

import torch
import torch.nn as nn


class ChannelWiseFusion(nn.Module):
    """Lightweight channel-wise attention fusion.
    
    In the pipeline:
        Camera BEV Features --|
                               |--> [ChannelWiseFusion] --> Fused BEV Features
        LiDAR BEV Features  --|
        
    This module is where knowledge distillation is applied.
    The BEVFusion teacher provides supervision signals to guide
    how the student learns to fuse camera and LiDAR features.
    """

    def __init__(self, camera_channels: int = 256, lidar_channels: int = 256, out_channels: int = 256):
        super().__init__()
        total_channels = camera_channels + lidar_channels

        # Channel attention: learn which channels matter most
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(total_channels, total_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(total_channels // 4, camera_channels),
            nn.Sigmoid(),
        )

        # Projection to output channels
        self.project = nn.Sequential(
            nn.Conv2d(camera_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Residual refinement (compensates for spatial misalignment)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, camera_bev: torch.Tensor, lidar_bev: torch.Tensor) -> torch.Tensor:
        """Fuse camera and LiDAR BEV features.
        
        Args:
            camera_bev: (B, C_cam, H, W) camera features in BEV space
            lidar_bev:  (B, C_lid, H, W) LiDAR features in BEV space
            
        Returns:
            Fused BEV features: (B, C_out, H, W)
        """
        # Concatenate for attention computation
        combined = torch.cat([camera_bev, lidar_bev], dim=1)

        # Compute channel attention weights
        weights = self.attention(combined)  # (B, C_cam)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, C_cam, 1, 1)

        # Weighted fusion
        fused = weights * camera_bev + (1 - weights) * lidar_bev

        # Project and refine
        fused = self.project(fused)
        residual = self.refine(fused)
        fused = self.relu(fused + residual)

        return fused


if __name__ == "__main__":
    fusion = ChannelWiseFusion(256, 256, 256)
    cam = torch.randn(1, 256, 50, 50)
    lid = torch.randn(1, 256, 50, 50)
    out = fusion(cam, lid)
    print(f"Camera BEV:  {cam.shape}")
    print(f"LiDAR BEV:   {lid.shape}")
    print(f"Fused BEV:   {out.shape}")
    print(f"Parameters:  {sum(p.numel() for p in fusion.parameters()):,}")
