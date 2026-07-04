"""
Knowledge Distillation Losses for Fusion-Stage BEV Perception.

This is the implementation of the dissertation's core contribution:
applying knowledge distillation specifically at the fusion stage,
using BEVFusion as the frozen teacher.

Combined loss:
    L_total = α · L_feature + β · L_logit + L_task

Where:
    L_feature  — MSE between student and teacher fused BEV features
                 (spatial feature alignment at the fusion module output)

    L_logit    — KL divergence between student and teacher heatmap logits
                 with temperature scaling T (soft label distillation)
               + L1 loss between student and teacher regression outputs

    L_task     — standard detection loss from train_baseline.py
                 (focal loss on heatmaps + L1 on regression)

Resolution mismatch handling:
    BEVFusion teacher outputs at (128×128).
    Student outputs at (50×50).
    Teacher features are bilinearly downsampled to student resolution
    before computing any distillation loss — this is cleaner than
    upsampling the student, which would add parameters and change
    the student's learned representations.

Reference:
    Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
    Liu et al., "BEVFusion: Multi-Task Multi-Sensor Fusion...", ICRA 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationLoss(nn.Module):
    """MSE loss between student and teacher fused BEV features.

    Aligns the student's fusion module output with the teacher's, forcing
    the student to learn a BEV representation that spatially resembles
    what BEVFusion produces — even though the student's backbone is 18×
    smaller.

    A learnable projection layer handles the case where student and teacher
    have different channel counts (though both are 256 in our setup).

    This is the 'feature-level' distillation in the dissertation:
        L_feature = MSE(Proj(student_fused_bev), downsample(teacher_fused_bev))
    """

    def __init__(
        self,
        student_channels: int = 256,
        teacher_channels: int = 256,
        student_h: int = 50,
        student_w: int = 50,
    ):
        super().__init__()
        self.student_h = student_h
        self.student_w = student_w

        # Channel projection: only needed if channel dims differ
        if student_channels != teacher_channels:
            self.proj = nn.Conv2d(student_channels, teacher_channels, 1)
        else:
            self.proj = nn.Identity()

    def forward(
        self,
        student_fused: torch.Tensor,   # (B, 256, 50,  50)
        teacher_fused: torch.Tensor,   # (B, 256, 128, 128)
    ) -> torch.Tensor:
        """
        Args:
            student_fused: student ChannelWiseFusion output
            teacher_fused: BEVFusion fused BEV features (detached, no grad)
        Returns:
            Scalar MSE loss
        """
        # Project student channels if needed
        student = self.proj(student_fused)   # (B, C, 50, 50)

        # Downsample teacher to student resolution
        # Bilinear is correct here — we're aligning spatial feature maps,
        # not heatmaps, so smooth interpolation is appropriate
        teacher_ds = F.interpolate(
            teacher_fused.detach(),
            size=(self.student_h, self.student_w),
            mode="bilinear",
            align_corners=False,
        )   # (B, C, 50, 50)

        return F.mse_loss(student, teacher_ds)


class LogitDistillationLoss(nn.Module):
    """KL divergence + L1 loss between student and teacher detection outputs.

    Two components:

    1. Heatmap KL divergence (soft label distillation):
       Divides logits by temperature T before softmax — higher T produces
       softer probability distributions that carry more information about
       the teacher's confidence structure. This is Hinton et al.'s core
       insight: soft labels reveal inter-class relationships that hard
       one-hot labels discard.

       L_heatmap = T² · KL(softmax(student/T) || softmax(teacher/T))
       The T² factor compensates for the gradient magnitude reduction
       caused by the temperature scaling.

    2. Regression L1 (at positive BEV locations only):
       Directly aligns student box predictions with teacher predictions
       at cells where the teacher is confident there's an object.
       More informative than ground-truth regression targets for classes
       where the teacher has strong prior knowledge (e.g. cars).

       L_reg = L1(student_reg[pos], teacher_reg[pos])
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(
        self,
        student_heatmap:    torch.Tensor,   # (B, 10, 50,  50)  sigmoid activated
        teacher_heatmap:    torch.Tensor,   # (B, 10, 128, 128) sigmoid activated
        student_regression: torch.Tensor,   # (B, 8,  50,  50)
        teacher_regression: torch.Tensor,   # (B, 8,  128, 128)
    ) -> tuple:
        """
        Returns:
            loss_heatmap, loss_regression  (both scalar tensors)
        """
        B, C, H, W = student_heatmap.shape
        device = student_heatmap.device

        # ── Downsample teacher outputs to student resolution ───────────────
        teacher_hm_ds = F.interpolate(
            teacher_heatmap.detach(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )   # (B, 10, 50, 50)

        teacher_reg_ds = F.interpolate(
            teacher_regression.detach(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )   # (B, 8, 50, 50)

        # ── Heatmap KL divergence with temperature scaling ─────────────────
        # Convert sigmoid-activated heatmaps back to logit-like space
        # Clamp to avoid log(0) — sigmoid output is in (0,1) but can be
        # very close to 0 or 1 at convergence
        eps = 1e-6
        student_logits = torch.log(
            student_heatmap.clamp(eps, 1-eps) /
            (1 - student_heatmap.clamp(eps, 1-eps))
        )   # inverse sigmoid (logits)
        teacher_logits = torch.log(
            teacher_hm_ds.clamp(eps, 1-eps) /
            (1 - teacher_hm_ds.clamp(eps, 1-eps))
        )

        # Temperature-scaled softmax over spatial+class dimensions
        # Flatten (C, H, W) → single distribution per sample
        s_flat = (student_logits / self.T).reshape(B, -1)
        t_flat = (teacher_logits / self.T).reshape(B, -1)

        s_soft = F.log_softmax(s_flat, dim=-1)
        t_soft = F.softmax(t_flat, dim=-1)

        # KL(student || teacher) — note F.kl_div expects log-probabilities
        # for the first argument
        loss_hm = F.kl_div(s_soft, t_soft, reduction="batchmean")
        # T² compensation (Hinton et al. 2015)
        loss_hm = loss_hm * (self.T ** 2)

        # ── Regression L1 at teacher-positive locations ───────────────────
        # "Positive" = teacher is confident an object exists (heatmap > 0.3)
        pos_mask = teacher_hm_ds.max(dim=1)[0] > 0.3   # (B, H, W)
        n_pos = pos_mask.sum().clamp(min=1)

        if pos_mask.sum() > 0:
            # (N_pos, 8) predictions at positive cells
            s_reg = student_regression.permute(0, 2, 3, 1)[pos_mask]
            t_reg = teacher_reg_ds.permute(0, 2, 3, 1)[pos_mask]
            loss_reg = F.l1_loss(s_reg, t_reg, reduction="sum") / n_pos
        else:
            loss_reg = torch.tensor(0.0, device=device, requires_grad=True)

        return loss_hm, loss_reg


class CombinedKDLoss(nn.Module):
    """Full combined distillation loss.

    L_total = α · L_feature + β · L_logit + L_task

    Where L_logit = L_heatmap_kl + λ · L_reg_l1

    All weights are configurable from student.yaml:
        distillation.alpha  — feature distillation weight (default 1.0)
        distillation.beta   — logit distillation weight   (default 0.5)
        distillation.temperature — KL temperature          (default 4.0)

    The task loss (L_task) is computed externally in train_with_kd.py
    using the same FocalLoss + L1 from train_baseline.py, and added on
    top of the distillation losses.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta:  float = 0.5,
        temperature: float = 4.0,
        student_channels: int = 256,
        teacher_channels: int = 256,
        student_h: int = 50,
        student_w: int = 50,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

        self.feature_loss = FeatureDistillationLoss(
            student_channels, teacher_channels, student_h, student_w
        )
        self.logit_loss = LogitDistillationLoss(temperature)

    def forward(
        self,
        student_fused:      torch.Tensor,   # (B, 256, 50, 50)
        student_heatmap:    torch.Tensor,   # (B, 10,  50, 50)
        student_regression: torch.Tensor,   # (B, 8,   50, 50)
        teacher_fused:      torch.Tensor,   # (B, 256, 128, 128)
        teacher_heatmap:    torch.Tensor,   # (B, 10,  128, 128)
        teacher_regression: torch.Tensor,   # (B, 8,   128, 128)
    ) -> dict:
        """
        Returns dict with all loss components for logging:
            loss_feature, loss_hm_kl, loss_reg_l1,
            loss_kd_total  (= α·feature + β·(hm_kl + reg_l1))
        """
        # Feature-level distillation
        loss_feature = self.feature_loss(student_fused, teacher_fused)

        # Logit-level distillation
        loss_hm_kl, loss_reg_l1 = self.logit_loss(
            student_heatmap, teacher_heatmap,
            student_regression, teacher_regression,
        )
        loss_logit = loss_hm_kl + loss_reg_l1

        # Combined KD loss (task loss added externally)
        loss_kd = self.alpha * loss_feature + self.beta * loss_logit

        return {
            "loss_feature":  loss_feature,
            "loss_hm_kl":    loss_hm_kl,
            "loss_reg_l1":   loss_reg_l1,
            "loss_kd_total": loss_kd,
        }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing CombinedKDLoss...")
    B = 2

    # Student outputs (50×50)
    s_fused = torch.randn(B, 256, 50, 50, requires_grad=True)
    s_hm    = torch.sigmoid(torch.randn(B, 10, 50, 50))
    s_reg   = torch.randn(B, 8, 50, 50)

    # Teacher outputs (128×128) — detached as they would be from cache
    t_fused = torch.randn(B, 256, 128, 128).detach()
    t_hm    = torch.sigmoid(torch.randn(B, 10, 128, 128)).detach()
    t_reg   = torch.randn(B, 8, 128, 128).detach()

    kd_loss = CombinedKDLoss(alpha=1.0, beta=0.5, temperature=4.0)
    losses  = kd_loss(s_fused, s_hm, s_reg, t_fused, t_hm, t_reg)

    print("Loss components:")
    for k, v in losses.items():
        print(f"  {k:<20} {v.item():.4f}")

    # Verify gradients flow to student params only
    losses["loss_kd_total"].backward()
    print("✅ Gradients flow through KD loss")
    assert s_fused.grad is not None
    print("✅ CombinedKDLoss ready")