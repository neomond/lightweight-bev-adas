"""
3D Detection Head for BEV features.

Predicts 3D bounding boxes from the fused BEV representation.
"""

import torch
import torch.nn as nn


class BEVDetectionHead(nn.Module):
    """Predict 3D bounding boxes from BEV features.
    
    In the pipeline:
        Fused BEV Features -> [BEVDetectionHead] -> 3D Bounding Boxes
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Classification head (heatmap)
        self.cls_head = nn.Conv2d(256, num_classes, 1)

        # Regression head (x, y, z, w, l, h, sin_yaw, cos_yaw)
        self.reg_head = nn.Conv2d(256, 8, 1)

    def forward(self, bev_features: torch.Tensor) -> dict:
        """
        Args:
            bev_features: (B, C, H, W) fused BEV features
        Returns:
            Dict with 'heatmap' and 'regression' predictions
        """
        shared = self.shared(bev_features)
        heatmap = torch.sigmoid(self.cls_head(shared))
        regression = self.reg_head(shared)

        return {
            "heatmap": heatmap,      # (B, num_classes, H, W)
            "regression": regression,  # (B, 8, H, W)
        }


if __name__ == "__main__":
    head = BEVDetectionHead()
    bev = torch.randn(1, 256, 50, 50)
    out = head(bev)
    print(f"Heatmap:    {out['heatmap'].shape}")
    print(f"Regression: {out['regression'].shape}")
    print(f"Parameters: {sum(p.numel() for p in head.parameters()):,}")
