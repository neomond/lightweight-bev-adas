"""
Knowledge Distillation Losses.

These losses transfer knowledge from the BEVFusion teacher
to the lightweight student model at the fusion stage.

Two types of distillation:
1. Feature-level: Aligns fused BEV feature maps
2. Logit-level: Matches detection output distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationLoss(nn.Module):
    """Align student's fused BEV features with teacher's.
    
    L_feature = (1/N) * sum(||B_student - B_teacher||^2)
    
    A projection layer handles dimension mismatch between
    teacher and student feature maps.
    """

    def __init__(self, student_channels: int = 256, teacher_channels: int = 256):
        super().__init__()
        # Projection layer if dimensions differ
        if student_channels != teacher_channels:
            self.project = nn.Conv2d(student_channels, teacher_channels, 1)
        else:
            self.project = nn.Identity()

    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_features: (B, C_s, H, W) student's fused BEV features
            teacher_features: (B, C_t, H, W) teacher's fused BEV features
        """
        student_projected = self.project(student_features)

        # Handle spatial size mismatch
        if student_projected.shape[-2:] != teacher_features.shape[-2:]:
            student_projected = F.interpolate(
                student_projected, size=teacher_features.shape[-2:], mode="bilinear", align_corners=False
            )

        return F.mse_loss(student_projected, teacher_features)


class LogitDistillationLoss(nn.Module):
    """Match student's detection outputs with teacher's.
    
    Uses KL divergence on heatmaps and L1 on regression outputs.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_output: dict, teacher_output: dict) -> torch.Tensor:
        """
        Args:
            student_output: dict with 'heatmap' and 'regression'
            teacher_output: dict with 'heatmap' and 'regression'
        """
        # Heatmap KL divergence (with temperature scaling)
        T = self.temperature
        s_heatmap = F.log_softmax(student_output["heatmap"].flatten(2) / T, dim=-1)
        t_heatmap = F.softmax(teacher_output["heatmap"].flatten(2) / T, dim=-1)
        kd_heatmap = F.kl_div(s_heatmap, t_heatmap, reduction="batchmean") * (T * T)

        # Regression L1 loss
        kd_regression = F.l1_loss(student_output["regression"], teacher_output["regression"])

        return kd_heatmap + kd_regression


class CombinedKDLoss(nn.Module):
    """Combined knowledge distillation loss.
    
    L_total = L_task + alpha * L_feature + beta * L_logit
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5,
                 student_channels: int = 256, teacher_channels: int = 256):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.feature_loss = FeatureDistillationLoss(student_channels, teacher_channels)
        self.logit_loss = LogitDistillationLoss()

    def forward(self, student_out: dict, teacher_out: dict) -> dict:
        """
        Returns dict with individual and total losses for logging.
        """
        l_feature = self.feature_loss(student_out["fused_bev"], teacher_out["fused_bev"])
        l_logit = self.logit_loss(student_out["detections"], teacher_out["detections"])

        total = self.alpha * l_feature + self.beta * l_logit

        return {
            "loss_feature_kd": l_feature,
            "loss_logit_kd": l_logit,
            "loss_kd_total": total,
        }


if __name__ == "__main__":
    # Test losses
    student_bev = torch.randn(2, 256, 50, 50)
    teacher_bev = torch.randn(2, 256, 50, 50)

    feat_loss = FeatureDistillationLoss()
    print(f"Feature KD loss: {feat_loss(student_bev, teacher_bev).item():.4f}")

    student_det = {"heatmap": torch.randn(2, 10, 50, 50), "regression": torch.randn(2, 8, 50, 50)}
    teacher_det = {"heatmap": torch.randn(2, 10, 50, 50), "regression": torch.randn(2, 8, 50, 50)}

    logit_loss = LogitDistillationLoss()
    print(f"Logit KD loss:   {logit_loss(student_det, teacher_det).item():.4f}")
