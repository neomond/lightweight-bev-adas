"""
Camera-to-BEV View Transformer (Simplified Lift-Splat-Shoot).

Transforms 2D camera feature maps into a unified Bird's Eye View (BEV)
representation using camera calibration (intrinsics + extrinsics).

Architecture overview:
    For each of the 6 camera views:
        1. LIFT   — predict a depth value per spatial location in the feature map
        2. SPLAT  — unproject each feature into 3D using depth + calibration
        3.         — transform 3D point to ego frame using extrinsics
        4. POOL   — scatter all 3D features onto the 2D BEV grid (max-pool)
    Aggregate all 6 cameras → single (B, C_out, H_bev, W_bev) BEV map

Design rationale (Master's level):
    Full LSS (Huang et al., 2021) uses a categorical depth distribution
    (D bins per pixel), creating a (D, H, W) frustum per camera — powerful
    but adds ~10M parameters and requires depth supervision to train.
    
    This simplified version predicts a single depth per pixel (1 value,
    not D bins), keeping the view-transformer as lightweight infrastructure
    so the dissertation's core contribution — KD at the fusion stage —
    remains the focus. The module is fully differentiable; depth is learned
    implicitly via the task loss.

Calibration tensors expected per camera (from nuscenes_loader.py):
    intrinsic:       (3, 3)  — K matrix
    rotation:        (3, 3)  — sensor-to-ego rotation (R_cs)
    translation:     (3,)    — sensor-to-ego translation (t_cs)
    ego_rotation:    (3, 3)  — ego-to-world (not needed; we work in ego frame)
    ego_translation: (3,)    — ego-to-world (not needed; we work in ego frame)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):
    """Predict a single depth value per spatial location in a feature map.
    
    Takes the richest (largest-channel) YOLO feature and produces a
    per-pixel depth estimate in metres via a small conv head + sigmoid
    scaled to [depth_min, depth_max].
    """

    def __init__(
        self,
        in_channels: int = 256,
        depth_min: float = 1.0,
        depth_max: float = 50.0,
    ):
        super().__init__()
        self.depth_min = depth_min
        self.depth_range = depth_max - depth_min

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),   # (B, 1, H_feat, W_feat)
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H_feat, W_feat)
        Returns:
            depth: (B, 1, H_feat, W_feat) in metres, range [depth_min, depth_max]
        """
        return self.head(features) * self.depth_range + self.depth_min


class CameraFeatureProjector(nn.Module):
    """Project YOLO multi-scale features to a single feature map.
    
    Takes the 3-scale output of YOLOBackbone and fuses them into
    one feature map at a fixed resolution for lifting into 3D.
    Uses the coarsest scale (highest channels, most semantic) as the
    base, and adds upsampled contributions from the other scales.
    """

    def __init__(
        self,
        in_channels_list: list = None,   # [64, 128, 256] from YOLO11n
        out_channels: int = 128,
        out_spatial: tuple = (24, 40),   # H_feat × W_feat after projection
    ):
        super().__init__()
        if in_channels_list is None:
            in_channels_list = [64, 128, 256]

        self.out_spatial = out_spatial

        # Per-scale 1×1 projections to out_channels
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for c in in_channels_list
        ])

        # Final refinement after summation
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: list of 3 tensors from YOLOBackbone at scales P3/P4/P5
        Returns:
            fused: (B, out_channels, H_feat, W_feat)
        """
        fused = None
        for proj, feat in zip(self.projections, features):
            x = proj(feat)
            # Resize to target spatial resolution
            x = F.interpolate(x, size=self.out_spatial, mode="bilinear", align_corners=False)
            fused = x if fused is None else fused + x
        return self.refine(fused)


class LiftSplatCamera(nn.Module):
    """Lift a single camera's feature map into 3D and splat onto BEV grid.
    
    Steps:
        1. Build a pixel grid at feature-map resolution
        2. Unproject pixels → normalised camera rays using K^-1
        3. Scale each ray by the predicted depth → 3D point in camera frame
        4. Transform to ego frame: P_ego = R_cs @ P_cam + t_cs
        5. Discard points outside BEV range, scatter remainder onto BEV grid
    """

    def __init__(
        self,
        feat_channels: int = 128,
        bev_channels: int = 256,
        x_range: tuple = (-50.0, 50.0),
        y_range: tuple = (-50.0, 50.0),
        bev_h: int = 50,
        bev_w: int = 50,
        depth_min: float = 1.0,
        depth_max: float = 50.0,
    ):
        super().__init__()
        self.x_range = x_range
        self.y_range = y_range
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.depth_head = DepthHead(feat_channels, depth_min, depth_max)

        # Project features to BEV channel dimension
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_channels, bev_channels, 1),
            nn.ReLU(inplace=True),
        )

    def _build_pixel_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return homogeneous pixel coordinates (3, H*W)."""
        ys = torch.arange(H, dtype=torch.float32, device=device)
        xs = torch.arange(W, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
        ones = torch.ones_like(grid_x)
        # Stack as (3, H*W): [u, v, 1]
        pixels = torch.stack([grid_x.flatten(), grid_y.flatten(), ones.flatten()], dim=0)
        return pixels

    def forward(
        self,
        features: torch.Tensor,          # (B, C_feat, H_feat, W_feat)
        intrinsic: torch.Tensor,          # (B, 3, 3)
        rotation: torch.Tensor,           # (B, 3, 3)  R_cs  sensor→ego
        translation: torch.Tensor,        # (B, 3)     t_cs
        bev_canvas: torch.Tensor,         # (B, C_bev, bev_h, bev_w)  accumulator
        bev_count: torch.Tensor,          # (B, 1,     bev_h, bev_w)  hit counter
    ) -> tuple:
        """Project this camera's features onto the BEV canvas.
        
        Returns updated (bev_canvas, bev_count).
        """
        B, C, H, W = features.shape
        device = features.device

        # ── 1. Predict depth  ──────────────────────────────────────────────
        depth = self.depth_head(features)   # (B, 1, H, W)

        # ── 2. Project features to BEV channel dim ────────────────────────
        proj_feats = self.feat_proj(features)   # (B, C_bev, H, W)

        # ── 3. Build pixel grid & unproject through K^-1 ──────────────────
        pixels = self._build_pixel_grid(H, W, device)   # (3, H*W)

        # K_inv: (B, 3, 3) — invert intrinsic for each sample
        # We need to scale the intrinsic to match the feature-map resolution.
        # The intrinsic is calibrated for the original image (640×384).
        # Feature maps are at (W_feat, H_feat) = (W, H).
        # Correction: scale fx, fy, cx, cy accordingly.
        # Original image size used by loader: width=640, height=384
        scale_x = W / 640.0
        scale_y = H / 384.0

        K = intrinsic.clone()   # (B, 3, 3)
        K[:, 0, :] *= scale_x   # fx, cx row
        K[:, 1, :] *= scale_y   # fy, cy row

        K_inv = torch.linalg.inv(K)   # (B, 3, 3)

        # Unproject: rays in camera frame, (B, 3, H*W)
        # pixels: (3, H*W) → expand to (B, 3, H*W)
        pixels_b = pixels.unsqueeze(0).expand(B, -1, -1)  # (B, 3, H*W)
        rays = K_inv @ pixels_b                             # (B, 3, H*W)

        # ── 4. Scale by depth → 3D points in camera frame ─────────────────
        depth_flat = depth.view(B, 1, H * W)    # (B, 1, H*W)
        points_cam = rays * depth_flat           # (B, 3, H*W)

        # ── 5. Transform to ego frame ──────────────────────────────────────
        # P_ego = R_cs @ P_cam + t_cs
        # rotation: (B, 3, 3), translation: (B, 3)
        points_ego = rotation @ points_cam              # (B, 3, H*W)
        points_ego = points_ego + translation.unsqueeze(-1)  # (B, 3, H*W)

        x_ego = points_ego[:, 0, :]   # (B, H*W)
        y_ego = points_ego[:, 1, :]   # (B, H*W)

        # ── 6. Map ego-frame (x, y) to BEV grid indices ───────────────────
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range

        # Normalise to [0, 1] then scale to grid size
        u = (x_ego - x_min) / (x_max - x_min)   # (B, H*W)
        v = (y_ego - y_min) / (y_max - y_min)   # (B, H*W)

        col = (u * self.bev_w).long()   # (B, H*W)
        row = (v * self.bev_h).long()   # (B, H*W)

        valid = (col >= 0) & (col < self.bev_w) & (row >= 0) & (row < self.bev_h)
        # valid: (B, H*W)

        # ── 7. Scatter features onto BEV canvas ───────────────────────────
        proj_flat = proj_feats.view(B, -1, H * W)   # (B, C_bev, H*W)

        for b in range(B):
            v_mask = valid[b]                  # (H*W,)
            if v_mask.sum() == 0:
                continue
            rows_b = row[b][v_mask]            # (N_valid,)
            cols_b = col[b][v_mask]            # (N_valid,)
            feats_b = proj_flat[b][:, v_mask]  # (C_bev, N_valid)
            idx = rows_b * self.bev_w + cols_b  # linear index (N_valid,)

            # Accumulate: add features; we'll average later via bev_count
            bev_flat = bev_canvas[b].view(-1, self.bev_h * self.bev_w)   # (C_bev, H*W_bev)
            bev_flat.scatter_add_(1, idx.unsqueeze(0).expand(bev_flat.shape[0], -1), feats_b)
            bev_canvas[b] = bev_flat.view(-1, self.bev_h, self.bev_w)

            cnt_flat = bev_count[b].view(1, -1)   # (1, H*W_bev)
            ones = torch.ones(1, v_mask.sum(), device=device)
            cnt_flat.scatter_add_(1, idx.unsqueeze(0), ones)
            bev_count[b] = cnt_flat.view(1, self.bev_h, self.bev_w)

        return bev_canvas, bev_count


class CameraToBEV(nn.Module):
    """Full multi-camera view transformer.
    
    Processes all 6 camera views and aggregates their contributions
    into a single BEV feature map.
    
    Usage in student.py:
        camera_to_bev = CameraToBEV(config)
        camera_bev = camera_to_bev(cam_features_list, calibration)
        # camera_bev: (B, 256, 50, 50)
    """

    CAMERAS = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(self, config: dict = None):
        super().__init__()
        config = config or {}

        feat_channels = config.get("feat_channels", 128)     # intermediate feat dim
        bev_channels  = config.get("camera_bev_channels", 256)
        bev_h = bev_w = config.get("bev_grid_size", 50)

        x_range = tuple(config.get("x_range", [-50.0, 50.0]))
        y_range = tuple(config.get("y_range", [-50.0, 50.0]))

        # Feature scale fusion: YOLO11n outputs [64, 128, 256] at (P3, P4, P5)
        yolo_channels = config.get("yolo_output_channels", [64, 128, 256])

        # Shared feature projector (same weights for all cameras → parameter efficient)
        self.feature_projector = CameraFeatureProjector(
            in_channels_list=yolo_channels,
            out_channels=feat_channels,
            out_spatial=(24, 40),   # ~1/16 of 384×640
        )

        # Shared lift-splat module (same weights for all cameras)
        self.lift_splat = LiftSplatCamera(
            feat_channels=feat_channels,
            bev_channels=bev_channels,
            x_range=x_range,
            y_range=y_range,
            bev_h=bev_h,
            bev_w=bev_w,
            depth_min=config.get("depth_min", 1.0),
            depth_max=config.get("depth_max", 50.0),
        )

        # Final BEV refinement after aggregating all cameras
        self.bev_refine = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

        self.bev_channels = bev_channels
        self.bev_h = bev_h
        self.bev_w = bev_w

    def forward(
        self,
        cam_features_list: list,    # list of 6 × list-of-3-tensors from YOLOBackbone
        calibration: list,          # list of B dicts, each keyed by camera name
    ) -> torch.Tensor:
        """
        Args:
            cam_features_list: Outer list is per-camera (6), inner list is per-scale (3).
                               Each scale tensor: (B, C, H, W)
            calibration: list of length B. Each element is a dict:
                            {cam_name: {intrinsic, rotation, translation, ...}}
        Returns:
            camera_bev: (B, C_bev, bev_h, bev_w)
        """
        # Infer B from first feature
        B = cam_features_list[0][0].shape[0]
        device = cam_features_list[0][0].device

        # Accumulators
        bev_canvas = torch.zeros(B, self.bev_channels, self.bev_h, self.bev_w, device=device)
        bev_count  = torch.zeros(B, 1,                 self.bev_h, self.bev_w, device=device)

        for cam_idx, cam_name in enumerate(self.CAMERAS):
            # ── Project YOLO multi-scale features to single map ───────────
            cam_feats = cam_features_list[cam_idx]   # list of 3 tensors (B, C_i, H_i, W_i)
            projected = self.feature_projector(cam_feats)  # (B, feat_channels, 24, 40)

            # ── Gather calibration tensors for this camera ─────────────────
            # calibration is a list of B dicts
            intrinsics   = torch.stack([calibration[b][cam_name]["intrinsic"]   for b in range(B)]).to(device)
            rotations    = torch.stack([calibration[b][cam_name]["rotation"]    for b in range(B)]).to(device)
            translations = torch.stack([calibration[b][cam_name]["translation"] for b in range(B)]).to(device)

            # ── Lift-Splat this camera onto the shared BEV canvas ─────────
            bev_canvas, bev_count = self.lift_splat(
                projected, intrinsics, rotations, translations,
                bev_canvas, bev_count,
            )

        # ── Average over overlapping camera contributions ──────────────────
        # Add small epsilon to avoid division by zero in unseen BEV cells
        bev_canvas = bev_canvas / (bev_count + 1e-6)

        # ── Final spatial refinement ───────────────────────────────────────
        camera_bev = self.bev_refine(bev_canvas)

        return camera_bev


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("Testing CameraToBEV module...")

    B, N_cams = 2, 6
    device = torch.device("cpu")

    # Simulate YOLOBackbone outputs for 6 cameras
    # YOLO11n feature shapes at layers 3/5/7 for 640×384 input:
    #   P3: (B, 64,  48, 80)
    #   P4: (B, 128, 24, 40)
    #   P5: (B, 256, 12, 20)
    fake_features = [
        [
            torch.randn(B, 64, 48, 80),
            torch.randn(B, 128, 24, 40),
            torch.randn(B, 256, 12, 20),
        ]
        for _ in range(N_cams)
    ]

    # Simulate calibration dicts
    fake_calib = []
    for b in range(B):
        sample_calib = {}
        for cam in CameraToBEV.CAMERAS:
            sample_calib[cam] = {
                "intrinsic":   torch.eye(3) * 500,   # rough focal length
                "rotation":    torch.eye(3),
                "translation": torch.tensor([0.0, 0.0, 1.5]),  # 1.5m height
            }
        fake_calib.append(sample_calib)

    config = {
        "feat_channels": 128,
        "camera_bev_channels": 256,
        "bev_grid_size": 50,
        "x_range": [-50.0, 50.0],
        "y_range": [-50.0, 50.0],
        "yolo_output_channels": [64, 128, 256],
        "depth_min": 1.0,
        "depth_max": 50.0,
    }

    model = CameraToBEV(config).to(device)
    output = model(fake_features, fake_calib)

    print(f"Input:  6 cameras × 3 feature scales")
    print(f"Output: {output.shape}  (expected: [{B}, 256, 50, 50])")
    assert output.shape == (B, 256, 50, 50), f"Shape mismatch: {output.shape}"
    print("✅ Shape check passed")

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
    print("✅ CameraToBEV module ready")
