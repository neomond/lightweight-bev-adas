"""
YOLO11 Backbone for image feature extraction.

This module extracts multi-scale features from camera images
using the YOLO11 backbone (not the detection head).
These features are later projected into BEV space.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLOBackbone(nn.Module):
    """Extract intermediate features from YOLO11 backbone.
    
    In the pipeline:
        Camera Images -> [YOLOBackbone] -> Image Features -> Camera-to-BEV
    """

    def __init__(self, model_size: str = "yolo11n.pt", freeze: bool = False):
        super().__init__()
        # Load pre-trained YOLO and extract only the backbone
        yolo = YOLO(model_size)
        self.backbone = yolo.model.model[:10]  # Backbone layers only

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.
        
        Args:
            x: Camera images, shape (B, 3, H, W)
            
        Returns:
            List of feature maps at different scales
        """
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # Capture features at key stages (P3, P4, P5)
            if i in [3, 5, 7]:
                features.append(x)
        return features
    
    @property
    def output_channels(self) -> list[int]:
        """Return the number of channels at each feature scale."""
        # These depend on YOLO11n - adjust if using larger variants
        return [64, 128, 256]


if __name__ == "__main__":
    # Quick test
    model = YOLOBackbone()
    dummy = torch.randn(1, 3, 640, 640)
    features = model(dummy)
    for i, f in enumerate(features):
        print(f"Feature {i}: {f.shape}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
