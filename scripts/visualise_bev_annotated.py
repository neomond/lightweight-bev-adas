"""
Milestone 1 — Improved BEV Annotation Visualisation.

Produces a cleaner, more clearly labelled version of the BEV annotation
plot for the dissertation. Improvements over the original:
  - Object labels printed next to each detection box
  - Distance from ego vehicle annotated
  - Class counts shown in legend
  - Cleaner colour scheme with better contrast
  - Ego vehicle clearly labelled
  - Grid with distance rings for spatial context

Usage:
    python scripts/visualise_bev_annotated.py \
        --config configs/student.yaml \
        --sample-idx 0 \
        --output results/milestone1/bev_annotated.png
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion


# ── Colour palette ─────────────────────────────────────────────────────────────
COLOURS = {
    'car':                  '#4A90D9',   # blue
    'truck':                '#2ECC71',   # green
    'bus':                  '#27AE60',   # dark green
    'trailer':              '#1ABC9C',   # teal
    'construction_vehicle': '#E67E22',   # orange
    'pedestrian':           '#E74C3C',   # red
    'motorcycle':           '#9B59B6',   # purple
    'bicycle':              '#F39C12',   # amber
    'traffic_cone':         '#F1C40F',   # yellow
    'barrier':              '#FF6B6B',   # bright coral/red — clearly visible against grey LiDAR
}

CLASS_ABBREV = {
    'car':                  'Car',
    'truck':                'Truck',
    'bus':                  'Bus',
    'trailer':              'Trailer',
    'construction_vehicle': 'Constr.',
    'pedestrian':           'Ped.',
    'motorcycle':           'Moto.',
    'bicycle':              'Bike',
    'traffic_cone':         'Cone',
    'barrier':              'Barrier',
}

CAT_MAP = {
    'vehicle.car':                           'car',
    'vehicle.truck':                         'truck',
    'vehicle.bus.bendy':                     'bus',
    'vehicle.bus.rigid':                     'bus',
    'vehicle.trailer':                       'trailer',
    'vehicle.construction':                  'construction_vehicle',
    'human.pedestrian.adult':                'pedestrian',
    'human.pedestrian.child':                'pedestrian',
    'human.pedestrian.construction_worker':  'pedestrian',
    'human.pedestrian.police_officer':       'pedestrian',
    'vehicle.motorcycle':                    'motorcycle',
    'vehicle.bicycle':                       'bicycle',
    'movable_object.trafficcone':            'traffic_cone',
    'movable_object.barrier':                'barrier',
}


def load_lidar_bev(nusc, sample, x_range=(-50, 50), y_range=(-50, 50)):
    """Load LiDAR and return BEV density map."""
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pc = LidarPointCloud.from_file(
        os.path.join(nusc.dataroot, lidar_data['filename'])
    )
    points = pc.points.T.astype(np.float32)

    calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    rot   = Quaternion(calib['rotation']).rotation_matrix.astype(np.float32)
    trans = np.array(calib['translation'], dtype=np.float32)

    # Filter artifact points
    valid = (
        np.isfinite(points[:, :3]).all(axis=1) &
        (np.abs(points[:, 0]) < 200) &
        (np.abs(points[:, 1]) < 200)
    )
    points = points[valid]
    points[:, :3] = points[:, :3] @ rot.T + trans

    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] < x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] < y_range[1])
    )
    points = points[mask]
    return points


def get_annotations(nusc, sample):
    """Return list of annotation dicts in ego frame."""
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose   = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    ego_trans  = np.array(ego_pose['translation'])
    ego_rot    = Quaternion(ego_pose['rotation'])

    annotations = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        cls = CAT_MAP.get(ann['category_name'])
        if cls is None:
            continue

        global_pos = np.array(ann['translation'])
        ego_pos    = ego_rot.inverse.rotate(global_pos - ego_trans)
        x, y, z    = ego_pos

        w, l, h   = ann['size']
        global_yaw = Quaternion(ann['rotation']).yaw_pitch_roll[0]
        ego_yaw    = ego_rot.yaw_pitch_roll[0]
        yaw        = global_yaw - ego_yaw

        dist = np.sqrt(x**2 + y**2)

        annotations.append({
            'cls':  cls,
            'x':    x,
            'y':    y,
            'z':    z,
            'w':    w,
            'l':    l,
            'h':    h,
            'yaw':  yaw,
            'dist': dist,
        })

    return annotations


def draw_rotated_box(ax, cx, cy, w, l, yaw, colour, alpha=0.85, lw=1.8):
    """Draw a rotated 2D bounding box with a triangle heading indicator."""
    hw, hl = w / 2, l / 2
    corners = np.array([
        [-hl, -hw],
        [ hl, -hw],
        [ hl,  hw],
        [-hl,  hw],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rot_corners = (R @ corners.T).T.copy()
    rot_corners[:, 0] += cx
    rot_corners[:, 1] += cy

    poly = plt.Polygon(
        rot_corners, closed=True,
        edgecolor=colour, facecolor='none',
        alpha=1.0, linewidth=lw,
    )
    ax.add_patch(poly)

    # Triangle on front edge to show heading direction clearly
    front_mid  = np.array([cx + hl * c,       cy + hl * s])
    front_left = np.array([cx + hl * c - hw * 0.5 * (-s),
                            cy + hl * s - hw * 0.5 * c])
    front_right= np.array([cx + hl * c + hw * 0.5 * (-s),
                            cy + hl * s + hw * 0.5 * c])
    tip        = np.array([cx + (hl + min(hl * 0.6, 2.0)) * c,
                            cy + (hl + min(hl * 0.6, 2.0)) * s])

    triangle = plt.Polygon(
        [front_left, front_right, tip],
        closed=True, edgecolor=colour,
        facecolor=colour, alpha=1.0, linewidth=0,
    )
    ax.add_patch(triangle)


def visualise(nusc, sample_token, output_path, x_range=(-50, 50), y_range=(-50, 50)):
    """Main visualisation function."""
    sample = nusc.get('sample', sample_token)

    # Load data
    points      = load_lidar_bev(nusc, sample, x_range, y_range)
    annotations = get_annotations(nusc, sample)

    # Count per class
    class_counts = {}
    for ann in annotations:
        class_counts[ann['cls']] = class_counts.get(ann['cls'], 0) + 1

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor='#0D1117')
    ax.set_facecolor('#0D1117')

    # ── LiDAR density background ──────────────────────────────────────────────
    H, W = 500, 500
    bev = np.zeros((H, W), dtype=np.float32)
    px = ((points[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * W).astype(int)
    py = ((points[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * H).astype(int)
    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    np.add.at(bev, (py[valid], px[valid]), 1)
    bev = np.log1p(bev)
    bev = bev / bev.max() if bev.max() > 0 else bev

    ax.imshow(
        bev, origin='lower',
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        cmap='Greys', alpha=0.35, vmin=0, vmax=1,
    )

    # ── Distance rings ────────────────────────────────────────────────────────
    for r in [10, 20, 30, 40, 50]:
        circle = plt.Circle(
            (0, 0), r, color='#FFFFFF', fill=False,
            linestyle='--', linewidth=0.4, alpha=0.2,
        )
        ax.add_patch(circle)
        ax.text(
            r * 0.707, r * 0.707, f'{r}m',
            color='#FFFFFF', fontsize=7, alpha=0.4,
            ha='center', va='center',
        )

    # ── Grid ──────────────────────────────────────────────────────────────────
    ax.grid(color='#FFFFFF', linestyle=':', linewidth=0.3, alpha=0.15)

    # ── Annotations ───────────────────────────────────────────────────────────
    label_counts = {}   # track how many labels per class to avoid clutter

    # Label distance thresholds per class for readability
    LABEL_DIST = {
        'barrier':   20,      # barriers: label only if very close
        'pedestrian': 30,     # pedestrians: label if within 30m
        'car':        50,     # vehicles: label if within 50m
        'truck':      50,
        'bus':        50,
        'trailer':    50,
        'construction_vehicle': 40,
        'motorcycle': 40,
        'bicycle':    40,
        'traffic_cone': 25,
    }

    for ann in sorted(annotations, key=lambda a: a['dist']):
        cls    = ann['cls']
        colour = COLOURS.get(cls, '#FFFFFF')
        abbrev = CLASS_ABBREV.get(cls, cls)

        # Draw box
        lw = 2.5 if cls == 'barrier' else 2.5
        draw_rotated_box(
            ax, ann['x'], ann['y'],
            ann['w'], ann['l'], ann['yaw'],
            colour=colour, lw=lw,
        )

        # Draw label only within distance threshold
        max_dist = LABEL_DIST.get(cls, 40)
        if max_dist is None or ann['dist'] > max_dist:
            label_counts[cls] = label_counts.get(cls, 0) + 1
            continue

        dist_str  = f'{ann["dist"]:.0f}m'
        label_str = f'{abbrev} {dist_str}'

        label_x = ann['x'] + ann['l'] * 0.5 + 1.2
        label_y = ann['y'] + ann['w'] * 0.5 + 0.8

        ax.text(
            label_x, label_y, label_str,
            color=colour, fontsize=8.0, fontweight='bold',
            ha='left', va='bottom',
            path_effects=[
                pe.withStroke(linewidth=2.5, foreground='#0D1117')
            ],
        )

        label_counts[cls] = label_counts.get(cls, 0) + 1

    # ── Ego vehicle ───────────────────────────────────────────────────────────
    ego_box = mpatches.FancyBboxPatch(
        (-1.0, -2.2), 2.0, 4.4,
        boxstyle='round,pad=0.1',
        edgecolor='#F1C40F', facecolor='#F1C40F',
        alpha=0.9, linewidth=2, zorder=10,
    )
    ax.add_patch(ego_box)
    ax.text(
        0, -3.5, 'EGO\nVEHICLE',
        color='#F1C40F', fontsize=8, fontweight='bold',
        ha='center', va='top',
        path_effects=[pe.withStroke(linewidth=2, foreground='#0D1117')],
    )

    # Forward direction arrow
    ax.annotate(
        '', xy=(6, 0), xytext=(2.5, 0),
        arrowprops=dict(
            arrowstyle='->', color='#F1C40F',
            lw=2.0, mutation_scale=15,
        ),
        zorder=11,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = []
    for cls, count in sorted(class_counts.items()):
        colour = COLOURS.get(cls, '#FFFFFF')
        abbrev = CLASS_ABBREV.get(cls, cls)
        handle = mpatches.Patch(
            facecolor=colour, edgecolor=colour,
            alpha=0.8,
            label=f'{abbrev} ({count})',
        )
        legend_handles.append(handle)

    # Add ego to legend
    legend_handles.insert(0, mpatches.Patch(
        facecolor='#F1C40F', edgecolor='#F1C40F',
        alpha=0.9, label='Ego vehicle',
    ))

    legend = ax.legend(
        handles=legend_handles,
        loc='upper right',
        framealpha=0.85,
        facecolor='#1A1F2E',
        edgecolor='#4A5568',
        labelcolor='white',
        fontsize=9,
        title='Detected Objects',
        title_fontsize=10,
    )
    legend.get_title().set_color('white')

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('X — Forward (m)', color='#AAAAAA', fontsize=11)
    ax.set_ylabel('Y — Left (m)', color='#AAAAAA', fontsize=11)
    ax.tick_params(colors='#AAAAAA', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#4A5568')

    total = len(annotations)
    ax.set_title(
        f'Bird\'s Eye View — {total} Annotated Objects\n'
        f'nuScenes Mini Dataset · 100m × 100m Scene · LiDAR + Ground Truth',
        color='white', fontsize=13, fontweight='bold', pad=15,
    )

    # ── Stats box ─────────────────────────────────────────────────────────────
    stats_lines = [f'Total objects: {total}']
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        stats_lines.append(f'  {CLASS_ABBREV[cls]}: {count}')

    stats_text = '\n'.join(stats_lines)
    ax.text(
        x_range[0] + 1, y_range[1] - 1,
        stats_text,
        color='#AAAAAA', fontsize=8,
        va='top', ha='left',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='#1A1F2E',
            edgecolor='#4A5568',
            alpha=0.85,
        ),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0D1117')
    plt.close()
    print(f'Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',        default='configs/student.yaml')
    parser.add_argument('--sample-idx',    type=int, default=0)
    parser.add_argument('--output',        default='results/milestone1/bev_annotated_improved.png')
    parser.add_argument('--list-samples',  action='store_true',
                        help='List all val samples with annotation counts')
    parser.add_argument('--split',         default='val',
                        choices=['train', 'val'],
                        help='Dataset split to use')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    nusc = NuScenes(
        version=data_cfg['version'],
        dataroot=data_cfg['dataroot'],
        verbose=False,
    )

    # Get a val sample with lots of objects
    from src.data.nuscenes_loader import NuScenesDataset
    dataset = NuScenesDataset(
        dataroot=data_cfg['dataroot'],
        version=data_cfg['version'],
        split=args.split,
        x_range=tuple(data_cfg['x_range']),
        y_range=tuple(data_cfg['x_range']),
    )

    # List samples mode
    if args.list_samples:
        print(f"{'Idx':<6} {'Annotations':<14} {'Token'}")
        print('-' * 60)
        for i, tok in enumerate(dataset.samples):
            s = nusc.get('sample', tok)
            print(f"{i:<6} {len(s['anns']):<14} {tok}")
        return

    # Use the specified sample index directly
    # Default idx=0 uses first val sample (matches original M1 visualisation)
    sample_token = dataset.samples[args.sample_idx]
    sample = nusc.get('sample', sample_token)
    n_ann = len(sample['anns'])
    print(f'Using sample idx={args.sample_idx}: {sample_token} ({n_ann} annotations)')
    visualise(nusc, sample_token, args.output)


if __name__ == '__main__':
    main()