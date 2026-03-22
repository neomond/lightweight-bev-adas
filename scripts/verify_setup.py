"""Verify that the entire project setup is working."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 50)
    print("  Project Setup Verification")
    print("=" * 50)
    
    # 1. Python version
    print(f"\n[1] Python: {sys.version.split()[0]}")
    
    # 2. PyTorch + device
    import torch
    print(f"[2] PyTorch: {torch.__version__}")
    
    from src.utils.device import get_device
    device = get_device()
    
    # 3. YOLO
    from ultralytics import YOLO
    print(f"[3] Ultralytics YOLO: OK")
    
    # 4. Model components
    print("\n[4] Testing model components...")
    
    from src.models.yolo_backbone import YOLOBackbone
    backbone = YOLOBackbone()
    dummy_img = torch.randn(1, 3, 640, 640)
    features = backbone(dummy_img)
    print(f"    YOLO Backbone: OK ({len(features)} feature scales)")
    
    from src.models.pointpillars import PointPillarsEncoder
    pp = PointPillarsEncoder()
    dummy_pts = torch.randn(1, 30000, 4)
    lidar_bev = pp(dummy_pts)
    print(f"    PointPillars:  OK (output: {lidar_bev.shape})")
    
    from src.models.fusion import ChannelWiseFusion
    fusion = ChannelWiseFusion()
    cam_bev = torch.randn(1, 256, 50, 50)
    lid_bev = torch.randn(1, 256, 50, 50)
    fused = fusion(cam_bev, lid_bev)
    print(f"    Fusion Module: OK (output: {fused.shape})")
    
    from src.models.bev_head import BEVDetectionHead
    head = BEVDetectionHead()
    det = head(fused)
    print(f"    Detection Head: OK (heatmap: {det['heatmap'].shape})")
    
    # 5. Losses
    print("\n[5] Testing distillation losses...")
    from src.losses.distillation import CombinedKDLoss
    kd_loss = CombinedKDLoss()
    student_out = {"fused_bev": fused, "detections": det}
    teacher_out = {
        "fused_bev": torch.randn_like(fused),
        "detections": {k: torch.randn_like(v) for k, v in det.items()},
    }
    losses = kd_loss(student_out, teacher_out)
    print(f"    Feature KD loss: {losses['loss_feature_kd'].item():.4f}")
    print(f"    Logit KD loss:   {losses['loss_logit_kd'].item():.4f}")
    
    # 6. Parameter count
    print("\n[6] Student model parameters:")
    from src.models.student import StudentBEV
    model = StudentBEV()
    for name, count in model.count_parameters().items():
        print(f"    {name:<25} {count:>10,}")
    
    print("\n" + "=" * 50)
    print("  All checks passed! Ready for Milestone 1.")
    print("=" * 50)


if __name__ == "__main__":
    main()
