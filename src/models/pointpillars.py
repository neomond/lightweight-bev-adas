"""
PointPillars encoder for LiDAR feature extraction.

Converts LiDAR point clouds into BEV feature maps by:
  1. Pillarize  — assign each point to a 2D grid cell (pillar)
  2. Augment    — add per-point features relative to the pillar centre
  3. Encode     — PillarFeatureNet (PointNet-style) encodes each pillar → vector
  4. Scatter    — place encoded vectors back onto the 2D BEV grid
  5. Backbone   — lightweight CNN refines the scattered pseudo-image

Reference: Lang et al., "PointPillars: Fast Encoders for Object Detection
           from Point Clouds", CVPR 2019.

Padding note (nuScenes loader):
  The loader pads short point clouds with zero rows so all batches have the
  same shape (B, 35000, 4).  Zero-padded points sit at (0,0,0) which is
  inside the BEV range, so they MUST be masked out before pillarization.
  We detect them as rows where x==0 AND y==0 AND z==0 AND intensity==0.
"""

import torch
import torch.nn as nn


# ── Pillar Feature Network ────────────────────────────────────────────────────

class PillarFeatureNet(nn.Module):
    """Encode all points in a pillar into a single feature vector.

    Input features per point (9 total):
        [x, y, z, intensity,          ← raw point attributes
         Δx, Δy, Δz,                  ← offset from pillar centre (xc, yc, mean-z)
         xc, yc]                      ← pillar centre coords in ego frame

    The PointNet-style max-pool over points makes the network
    permutation-invariant and handles variable point counts.
    """

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
            (N_pillars, out_channels)  — one vector per pillar
        """
        N, P, C = pillar_features.shape
        # Flatten pillars×points for BatchNorm1d, then unflatten
        x = pillar_features.reshape(N * P, C)
        x = self.net(x)
        x = x.reshape(N, P, -1)
        # Max-pool over points: permutation invariant
        x = x.max(dim=1)[0]   # (N_pillars, out_channels)
        return x


# ── Pillarization (pure-PyTorch, batched) ────────────────────────────────────

def pillarize(
    points: torch.Tensor,          # (B, N, 4)  x y z intensity
    x_range: tuple,
    y_range: tuple,
    pillar_size: float,
    grid_x: int,
    grid_y: int,
    max_points: int,
    max_pillars: int,
) -> tuple:
    """Assign LiDAR points to pillars and build augmented feature tensors.

    Returns:
        pillar_feats:  (B, max_pillars, max_points, 9)
        pillar_coords: (B, max_pillars, 2)  — (row, col) in BEV grid
        num_pillars:   (B,)                 — actual non-empty pillars per sample
    """
    B, N, _ = points.shape
    device = points.device

    # Output tensors (zero-initialised → masked entries stay zero)
    pillar_feats  = torch.zeros(B, max_pillars, max_points, 9, device=device)
    pillar_coords = torch.zeros(B, max_pillars, 2, dtype=torch.long, device=device)
    num_pillars   = torch.zeros(B, dtype=torch.long, device=device)

    x_min, x_max = x_range
    y_min, y_max = y_range

    for b in range(B):
        pts = points[b]   # (N, 4)

        # ── 1. Mask out zero-padded rows ──────────────────────────────────
        # Padding rows inserted by the loader are exactly (0,0,0,0)
        valid_mask = ~((pts[:, 0] == 0) & (pts[:, 1] == 0) &
                       (pts[:, 2] == 0) & (pts[:, 3] == 0))

        # Also clip to BEV range (loader already does this, belt-and-braces)
        in_range = (
            (pts[:, 0] >= x_min) & (pts[:, 0] < x_max) &
            (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
        )
        valid_mask = valid_mask & in_range
        pts = pts[valid_mask]   # (N_valid, 4)

        if pts.shape[0] == 0:
            continue

        # ── 2. Compute pillar (col, row) index for each point ─────────────
        col = ((pts[:, 0] - x_min) / pillar_size).long().clamp(0, grid_x - 1)
        row = ((pts[:, 1] - y_min) / pillar_size).long().clamp(0, grid_y - 1)
        pillar_idx = row * grid_x + col   # unique 1-D pillar id  (N_valid,)

        # ── 3. Find unique occupied pillars, cap at max_pillars ───────────
        unique_ids, inverse = torch.unique(pillar_idx, return_inverse=True)
        n_occ = unique_ids.shape[0]
        n_keep = min(n_occ, max_pillars)

        # Keep the first n_keep unique pillars (deterministic)
        kept_ids   = unique_ids[:n_keep]                    # (n_keep,)
        # Map point → kept pillar slot (-1 = dropped)
        slot = torch.full((n_occ,), -1, dtype=torch.long, device=device)
        slot[:n_keep] = torch.arange(n_keep, device=device)
        point_slot = slot[inverse]   # (N_valid,) slot index per point

        # ── 4. Compute pillar centres ─────────────────────────────────────
        kept_col = kept_ids % grid_x
        kept_row = kept_ids // grid_x
        xc = kept_col.float() * pillar_size + x_min + pillar_size / 2   # (n_keep,)
        yc = kept_row.float() * pillar_size + y_min + pillar_size / 2   # (n_keep,)

        # ── 5. Build per-pillar point tensors ─────────────────────────────
        # point_count[s] = how many valid points went to slot s
        point_count = torch.zeros(n_keep, dtype=torch.long, device=device)

        for s in range(n_keep):
            mask_s = (point_slot == s)
            pts_s  = pts[mask_s]                        # (n_pts_s, 4)
            n_s    = min(pts_s.shape[0], max_points)
            pts_s  = pts_s[:n_s]                        # trim to max_points

            # Augment: append Δx, Δy, Δz from pillar mean + pillar centre
            cx, cy = xc[s], yc[s]
            cz = pts_s[:, 2].mean()
            dx = pts_s[:, 0] - cx
            dy = pts_s[:, 1] - cy
            dz = pts_s[:, 2] - cz
            cx_rep = cx.expand(n_s)
            cy_rep = cy.expand(n_s)

            # Final 9-dim feature: [x, y, z, i, dx, dy, dz, xc, yc]
            aug = torch.stack([
                pts_s[:, 0], pts_s[:, 1], pts_s[:, 2], pts_s[:, 3],
                dx, dy, dz, cx_rep, cy_rep
            ], dim=1)   # (n_s, 9)

            pillar_feats[b, s, :n_s] = aug
            pillar_coords[b, s, 0] = kept_row[s]
            pillar_coords[b, s, 1] = kept_col[s]
            point_count[s] = n_s

        num_pillars[b] = n_keep

    return pillar_feats, pillar_coords, num_pillars


# ── Full PointPillars Encoder ─────────────────────────────────────────────────

class PointPillarsEncoder(nn.Module):
    """Full PointPillars encoder: point cloud → BEV feature map.

    Pipeline:
        LiDAR Points (B, N, 4)
            → pillarize()              group points into grid cells
            → PillarFeatureNet         encode each pillar → vector
            → scatter onto BEV grid    (B, out_channels, grid_y, grid_x)
            → bev_backbone (CNN)       (B, 256, grid_y/4, grid_x/4)
    """

    def __init__(
        self,
        x_range: tuple = (-50.0, 50.0),
        y_range: tuple = (-50.0, 50.0),
        z_range: tuple = (-5.0, 3.0),
        pillar_size: float = 0.5,
        max_points_per_pillar: int = 32,
        max_pillars: int = 10000,
        in_channels: int = 4,    # x, y, z, intensity from loader
        out_channels: int = 64,
    ):
        super().__init__()
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.pillar_size = pillar_size
        self.max_points = max_points_per_pillar
        self.max_pillars = max_pillars

        self.grid_x = int((x_range[1] - x_range[0]) / pillar_size)   # 200
        self.grid_y = int((y_range[1] - y_range[0]) / pillar_size)   # 200

        # 9 = 4 raw + 3 relative offsets + 2 pillar centre coords
        self.pillar_net = PillarFeatureNet(in_channels + 5, out_channels)

        # Pseudo-image backbone: 200×200 → 50×50
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
            points: (B, N, 4)  x, y, z, intensity  (may contain zero-padded rows)
        Returns:
            BEV feature map: (B, 256, grid_y//4, grid_x//4)  →  (B, 256, 50, 50)
        """
        B = points.shape[0]
        device = points.device

        # ── Step 1: Pillarize ──────────────────────────────────────────────
        pillar_feats, pillar_coords, num_pillars = pillarize(
            points,
            self.x_range, self.y_range,
            self.pillar_size,
            self.grid_x, self.grid_y,
            self.max_points, self.max_pillars,
        )
        # pillar_feats:  (B, max_pillars, max_points, 9)
        # pillar_coords: (B, max_pillars, 2)  [row, col]

        # ── Step 2: Encode pillars ─────────────────────────────────────────
        # Reshape so PillarFeatureNet sees (N_pillars_total, max_points, 9)
        pillar_feats_flat = pillar_feats.view(B * self.max_pillars, self.max_points, 9)
        encoded = self.pillar_net(pillar_feats_flat)   # (B*max_pillars, out_channels)
        encoded = encoded.view(B, self.max_pillars, -1)  # (B, max_pillars, C)

        # ── Step 3: Scatter onto BEV pseudo-image ─────────────────────────
        C = encoded.shape[-1]
        bev = torch.zeros(B, C, self.grid_y, self.grid_x, device=device)

        for b in range(B):
            n = num_pillars[b].item()
            if n == 0:
                continue
            rows = pillar_coords[b, :n, 0]   # (n,)
            cols = pillar_coords[b, :n, 1]   # (n,)
            feats = encoded[b, :n]            # (n, C)
            # Scatter: bev[b, :, row, col] = encoded feature
            bev[b, :, rows, cols] = feats.t()   # (C, n) assigned to grid positions

        # ── Step 4: BEV backbone ───────────────────────────────────────────
        bev_out = self.bev_backbone(bev)   # (B, 256, 50, 50)
        return bev_out

    @property
    def output_channels(self) -> int:
        return self._out_channels

    @property
    def output_size(self) -> tuple:
        return (self.grid_y // 4, self.grid_x // 4)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing PointPillarsEncoder with realistic data...")

    B, N = 2, 35000
    device = torch.device("cpu")

    # Simulate a batch like the nuScenes loader produces:
    #   real points in [-50,50] range + zero-padding at the end
    real_n = 20000
    pts = torch.zeros(B, N, 4)
    pts[:, :real_n, 0] = torch.empty(B, real_n).uniform_(-49, 49)  # x
    pts[:, :real_n, 1] = torch.empty(B, real_n).uniform_(-49, 49)  # y
    pts[:, :real_n, 2] = torch.empty(B, real_n).uniform_(-4, 2)    # z
    pts[:, :real_n, 3] = torch.empty(B, real_n).uniform_(0, 1)     # intensity
    # rows [real_n:] are zero — as the loader produces

    model = PointPillarsEncoder().to(device)
    out = model(pts)

    print(f"Input points:  {pts.shape}")
    print(f"Output BEV:    {out.shape}  (expected: [{B}, 256, 50, 50])")
    assert out.shape == (B, 256, 50, 50), f"Shape mismatch: {out.shape}"
    print("✅ Shape check passed")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:    {params:,}")

    # Verify gradients flow (important for training)
    loss = out.sum()
    loss.backward()
    print("✅ Gradient check passed")
    print("✅ PointPillarsEncoder ready")