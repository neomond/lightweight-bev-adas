"""
tests/test_regression_targets.py

Unit tests for build_regression_targets() in scripts/train_baseline.py.
Run before any retraining to confirm the regression target fix is correct.

Usage:
    python tests/test_regression_targets.py
    # or, if you have pytest installed:
    pytest tests/test_regression_targets.py -v
"""

import sys
from pathlib import Path
import torch

# Make scripts/ importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_baseline import build_regression_targets, build_heatmap_targets


def test_basic_box_center():
    """A box at the origin should land at the center BEV cell with correct values."""
    annotations = [{
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 2.0, 4.0, 1.5, 0.0]]),  # x,y,z,w,l,h,yaw
        "classes": torch.tensor([0]),
    }]
    x_range = (-50.0, 50.0)
    y_range = (-50.0, 50.0)
    bev_h, bev_w = 50, 50

    reg_targets, pos_mask = build_regression_targets(
        annotations, bev_h, bev_w, x_range, y_range, torch.device("cpu")
    )

    # x=0 in (-50,50) with 50 cells → scale=0.5 → cx=(0-(-50))*0.5=25 → col=25 (same for row)
    assert pos_mask[0, 25, 25].item() is True, "Box should mark cell (25,25) as positive"

    vals = reg_targets[0, :, 25, 25]
    assert torch.isclose(vals[2], torch.tensor(1.0)), f"z mismatch: {vals[2]}"
    assert torch.isclose(vals[3], torch.log(torch.tensor(2.0))), f"log_w mismatch: {vals[3]}"
    assert torch.isclose(vals[4], torch.log(torch.tensor(4.0))), f"log_l mismatch: {vals[4]}"
    assert torch.isclose(vals[5], torch.log(torch.tensor(1.5))), f"log_h mismatch: {vals[5]}"
    assert torch.isclose(vals[6], torch.tensor(0.0), atol=1e-6), f"sin_yaw mismatch: {vals[6]}"
    assert torch.isclose(vals[7], torch.tensor(1.0), atol=1e-6), f"cos_yaw mismatch: {vals[7]}"
    print("✅ test_basic_box_center passed")


def test_negative_yaw():
    """Check sin/cos signs come out right for a negative yaw."""
    yaw = -1.0  # radians
    annotations = [{
        "boxes": torch.tensor([[10.0, -5.0, 0.5, 1.8, 4.5, 1.6, yaw]]),
        "classes": torch.tensor([1]),
    }]
    x_range = (-50.0, 50.0)
    y_range = (-50.0, 50.0)
    bev_h, bev_w = 50, 50

    reg_targets, pos_mask = build_regression_targets(
        annotations, bev_h, bev_w, x_range, y_range, torch.device("cpu")
    )

    # x=10 → col=int((10-(-50))*0.5)=30 ; y=-5 → row=int((-5-(-50))*0.5)=22
    assert pos_mask[0, 22, 30].item() is True, "Box should mark cell (22,30)"
    vals = reg_targets[0, :, 22, 30]
    assert torch.isclose(vals[6], torch.sin(torch.tensor(yaw)), atol=1e-6)
    assert torch.isclose(vals[7], torch.cos(torch.tensor(yaw)), atol=1e-6)
    print("✅ test_negative_yaw passed")


def test_boundary_box():
    """A box near the edge of the range shouldn't cause an out-of-bounds index."""
    annotations = [{
        "boxes": torch.tensor([[49.9, 49.9, 0.0, 2.0, 4.0, 1.5, 0.0]]),
        "classes": torch.tensor([0]),
    }]
    x_range = (-50.0, 50.0)
    y_range = (-50.0, 50.0)
    bev_h, bev_w = 50, 50

    # Should not raise, and should either mark a valid cell or (correctly) skip
    reg_targets, pos_mask = build_regression_targets(
        annotations, bev_h, bev_w, x_range, y_range, torch.device("cpu")
    )
    assert pos_mask.sum() <= 1, "At most one cell should be marked"
    print("✅ test_boundary_box passed (no out-of-bounds crash)")


def test_empty_annotations():
    """No boxes in a sample should produce an all-zero target and mask."""
    annotations = [{
        "boxes": torch.zeros(0, 7),
        "classes": torch.zeros(0, dtype=torch.long),
    }]
    x_range = (-50.0, 50.0)
    y_range = (-50.0, 50.0)
    bev_h, bev_w = 50, 50

    reg_targets, pos_mask = build_regression_targets(
        annotations, bev_h, bev_w, x_range, y_range, torch.device("cpu")
    )
    assert pos_mask.sum().item() == 0, "No boxes should mean no positive cells"
    assert reg_targets.abs().sum().item() == 0, "No boxes should mean all-zero targets"
    print("✅ test_empty_annotations passed")


def test_consistency_with_heatmap_targets():
    """build_regression_targets and build_heatmap_targets should agree on
    which cell a given box lands in — catches divergence between the two
    independent cell-assignment implementations."""
    annotations = [{
        "boxes": torch.tensor([
            [0.0, 0.0, 1.0, 2.0, 4.0, 1.5, 0.0],
            [22.3, -14.7, 0.2, 1.9, 4.3, 1.5, 0.7],
        ]),
        "classes": torch.tensor([0, 2]),
    }]
    x_range = (-50.0, 50.0)
    y_range = (-50.0, 50.0)
    bev_h, bev_w, num_classes = 50, 50, 10

    reg_targets, reg_pos_mask = build_regression_targets(
        annotations, bev_h, bev_w, x_range, y_range, torch.device("cpu")
    )
    heatmaps = build_heatmap_targets(
        annotations, num_classes, bev_h, bev_w, x_range, y_range, torch.device("cpu")
    )

    # Every cell with heatmap value == 1.0 (exact box center) should also be
    # marked positive in the regression mask, and vice versa for at least the centers.
    heatmap_peak_mask = (heatmaps.max(dim=1)[0] == 1.0)  # (B, H, W)
    # reg_pos_mask should be a subset match at the true centers
    overlap = (heatmap_peak_mask & reg_pos_mask).sum().item()
    assert overlap == 2, f"Expected both box centers to agree between the two functions, got overlap={overlap}"
    print("✅ test_consistency_with_heatmap_targets passed")


if __name__ == "__main__":
    test_basic_box_center()
    test_negative_yaw()
    test_boundary_box()
    test_empty_annotations()
    test_consistency_with_heatmap_targets()
    print("\n✅ All regression target tests passed")