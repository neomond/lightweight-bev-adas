"""
Render a full nuScenes scene (all keyframes) as a video combining:
  - Top: 6-camera mosaic (front-left/front/front-right/back-left/back/back-right)
  - Bottom: annotated BEV (LiDAR density + ground-truth boxes), same style as
    visualise_bev_annotated.py

Produces one .mp4 per scene. Designed to run locally where your nuScenes
mini data already lives (VSCode / Colab) — not something I can execute here
since I don't have the dataset.

Usage:
    python scripts/render_scene_video.py \
        --config configs/student.yaml \
        --scene-name scene-0061 \
        --output outputs/scene_video.mp4 \
        --fps 2

    # List available scenes first if you don't know the name:
    python scripts/render_scene_video.py --config configs/student.yaml --list-scenes

Requirements beyond your existing repo deps:
    pip install imageio imageio-ffmpeg
"""

import argparse
import os
import sys
from pathlib import Path
from typing import cast

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion


# ── Reused verbatim from visualise_bev_annotated.py, so the BEV panel looks
#    identical to your existing dissertation figures ──────────────────────────
COLOURS = {
    'car':                  '#4A90D9',
    'truck':                '#2ECC71',
    'bus':                  '#27AE60',
    'trailer':              '#1ABC9C',
    'construction_vehicle': '#E67E22',
    'pedestrian':           '#E74C3C',
    'motorcycle':           '#9B59B6',
    'bicycle':              '#F39C12',
    'traffic_cone':         '#F1C40F',
    'barrier':              '#FF6B6B',
}
CLASS_ABBREV = {
    'car': 'Car', 'truck': 'Truck', 'bus': 'Bus', 'trailer': 'Trailer',
    'construction_vehicle': 'Constr.', 'pedestrian': 'Ped.',
    'motorcycle': 'Moto.', 'bicycle': 'Bike', 'traffic_cone': 'Cone',
    'barrier': 'Barrier',
}
CAT_MAP = {
    'vehicle.car': 'car', 'vehicle.truck': 'truck',
    'vehicle.bus.bendy': 'bus', 'vehicle.bus.rigid': 'bus',
    'vehicle.trailer': 'trailer', 'vehicle.construction': 'construction_vehicle',
    'human.pedestrian.adult': 'pedestrian', 'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'vehicle.motorcycle': 'motorcycle', 'vehicle.bicycle': 'bicycle',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.barrier': 'barrier',
}

CAMERA_LAYOUT = [
    ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
    ['CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT'],
]


def load_lidar_bev(nusc, sample, x_range=(-50, 50), y_range=(-50, 50)):
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_data['filename']))
    points = pc.points.T.astype(np.float32)

    calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    rot = Quaternion(calib['rotation']).rotation_matrix.astype(np.float32)
    trans = np.array(calib['translation'], dtype=np.float32)

    # Filter sensor-artifact points BEFORE the rotation matmul — a small
    # number of raw .pcd.bin points have extreme/NaN/Inf values (including
    # on the z-axis) that otherwise overflow float32 during @ rot.T.
    valid = (
        np.isfinite(points[:, :3]).all(axis=1)
        & (np.abs(points[:, 0]) < 200)
        & (np.abs(points[:, 1]) < 200)
        & (np.abs(points[:, 2]) < 200)
    )
    points = points[valid]
    points[:, :3] = points[:, :3] @ rot.T + trans

    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] < x_range[1])
        & (points[:, 1] >= y_range[0]) & (points[:, 1] < y_range[1])
    )
    return points[mask]


def get_annotations(nusc, sample):
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    ego_trans = np.array(ego_pose['translation'])
    ego_rot = Quaternion(ego_pose['rotation'])

    annotations = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        cls = CAT_MAP.get(ann['category_name'])
        if cls is None:
            continue
        global_pos = np.array(ann['translation'])
        x, y, z = ego_rot.inverse.rotate(global_pos - ego_trans)
        w, l, h = ann['size']
        global_yaw = Quaternion(ann['rotation']).yaw_pitch_roll[0]
        yaw = global_yaw - ego_rot.yaw_pitch_roll[0]
        annotations.append({
            'cls': cls, 'x': x, 'y': y, 'w': w, 'l': l,
            'yaw': yaw, 'dist': float(np.hypot(x, y)),
        })
    return annotations


def draw_rotated_box(ax, cx, cy, w, l, yaw, colour, lw=2.0):
    hw, hl = w / 2, l / 2
    corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rc = (R @ corners.T).T.copy()
    rc[:, 0] += cx
    rc[:, 1] += cy
    ax.add_patch(Polygon(rc, closed=True, edgecolor=colour, facecolor='none', linewidth=lw))

    front_mid = np.array([cx + hl * c, cy + hl * s])
    front_left = np.array([cx + hl * c - hw * 0.5 * (-s), cy + hl * s - hw * 0.5 * c])
    front_right = np.array([cx + hl * c + hw * 0.5 * (-s), cy + hl * s + hw * 0.5 * c])
    tip = np.array([cx + (hl + min(hl * 0.6, 2.0)) * c, cy + (hl + min(hl * 0.6, 2.0)) * s])
    ax.add_patch(Polygon([front_left, front_right, tip], closed=True,
                          edgecolor=colour, facecolor=colour, linewidth=0))


def compute_bev_maps(points, x_range=(-50, 50), y_range=(-50, 50), grid=250):
    """Bin a point cloud into height / intensity / density BEV grids.
    points: (N, 4) array of x, y, z, intensity (ego frame).
    Mirrors the style of your Milestone 1 bev_maps.png figure.
    """
    H = W = grid
    height_map = np.full((H, W), np.nan, dtype=np.float32)
    intensity_sum = np.zeros((H, W), dtype=np.float32)
    intensity_count = np.zeros((H, W), dtype=np.float32)
    density_map = np.zeros((H, W), dtype=np.float32)

    if points.shape[0] == 0:
        return height_map, np.zeros((H, W), dtype=np.float32), density_map

    px = ((points[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * W).astype(int)
    py = ((points[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * H).astype(int)
    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px, py = px[valid], py[valid]
    z = points[valid, 2]
    intensity = points[valid, 3] if points.shape[1] > 3 else np.zeros_like(z)

    np.add.at(density_map, (py, px), 1)

    # max height per cell
    order = np.argsort(z)
    height_map[py[order], px[order]] = z[order]

    np.add.at(intensity_sum, (py, px), intensity)
    np.add.at(intensity_count, (py, px), 1)
    intensity_map = np.divide(
        intensity_sum, intensity_count,
        out=np.zeros_like(intensity_sum), where=intensity_count > 0,
    )

    density_map = np.log1p(density_map)
    return height_map, intensity_map, density_map


def render_frame(nusc, sample_token, x_range=(-50, 50), y_range=(-50, 50),
                  show_annotated_bev=True):
    """Render one combined camera-mosaic (+ optional annotated BEV) + LiDAR
    height/intensity/density frame, return as an RGB array.

    show_annotated_bev=False drops the ground-truth-boxes BEV panel and its
    legend entirely, leaving only raw sensor input: cameras + LiDAR maps.
    """
    sample = nusc.get('sample', sample_token)

    if show_annotated_bev:
        fig = plt.figure(figsize=(14, 16), facecolor='#0D1117', dpi=150)
        gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 2.0, 1.15],
                      hspace=0.18, wspace=0.05)
        maps_row = 3
    else:
        fig = plt.figure(figsize=(14, 11), facecolor='#0D1117', dpi=150)
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.15],
                      hspace=0.18, wspace=0.05)
        maps_row = 2

    # ── camera mosaic (top 2 rows) ──────────────────────────────────────────
    for r, row in enumerate(CAMERA_LAYOUT):
        for c, cam in enumerate(row):
            ax = fig.add_subplot(gs[r, c])
            cam_data = nusc.get('sample_data', sample['data'][cam])
            img = Image.open(os.path.join(nusc.dataroot, cam_data['filename'])).convert('RGB')
            ax.imshow(img)
            ax.set_title(cam, color='#AAAAAA', fontsize=11)
            ax.axis('off')

    # LiDAR points are needed either way (for the maps row); annotations are
    # only needed for the optional boxes-on-BEV panel.
    points = load_lidar_bev(nusc, sample, x_range, y_range)

    if show_annotated_bev:
        # ── BEV panel (main plot, 2 cols) + dedicated legend panel (1 col) ──
        ax = fig.add_subplot(gs[2, 0:2])
        ax.set_facecolor('#0D1117')

        annotations = get_annotations(nusc, sample)

        H, W = 500, 500
        bev = np.zeros((H, W), dtype=np.float32)
        px = ((points[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * W).astype(int)
        py = ((points[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * H).astype(int)
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        np.add.at(bev, (py[valid], px[valid]), 1)
        bev = np.log1p(bev)
        bev = bev / bev.max() if bev.max() > 0 else bev
        ax.imshow(bev, origin='lower', extent=(*x_range, *y_range), cmap='Greys', alpha=0.35)

        for r in [10, 20, 30, 40, 50]:
            ax.add_patch(Circle((0, 0), r, color='#FFFFFF', fill=False,
                                 linestyle='--', linewidth=0.4, alpha=0.2))

        for ann in annotations:
            colour = COLOURS.get(ann['cls'], '#FFFFFF')
            draw_rotated_box(ax, ann['x'], ann['y'], ann['w'], ann['l'], ann['yaw'], colour)

        ego_box = mpatches.FancyBboxPatch((-1.0, -2.2), 2.0, 4.4, boxstyle='round,pad=0.1',
                                           edgecolor='#F1C40F', facecolor='#F1C40F',
                                           alpha=0.9, linewidth=2, zorder=10)
        ax.add_patch(ego_box)
        ax.annotate('', xy=(6, 0), xytext=(2.5, 0),
                    arrowprops=dict(arrowstyle='->', color='#F1C40F', lw=2.0, mutation_scale=15))

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_aspect('equal')
        ax.tick_params(colors='#AAAAAA', labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor('#4A5568')
        ax.set_title(f'BEV — {len(annotations)} objects  ·  token {sample_token[:8]}',
                     color='white', fontsize=13)

        # ── legend panel — fixed, all 10 classes, so color mapping stays
        #    consistent across every frame regardless of what's present ────
        lax = fig.add_subplot(gs[2, 2])
        lax.set_facecolor('#0D1117')
        lax.axis('off')
        legend_handles = [mpatches.Patch(facecolor='#F1C40F', edgecolor='#F1C40F',
                                          alpha=0.9, label='Ego vehicle')]
        for cls in CLASS_ABBREV:
            legend_handles.append(mpatches.Patch(
                facecolor=COLOURS[cls], edgecolor=COLOURS[cls],
                alpha=0.85, label=CLASS_ABBREV[cls],
            ))
        legend = lax.legend(
            handles=legend_handles, loc='center left', frameon=True,
            framealpha=0.9, facecolor='#1A1F2E', edgecolor='#4A5568',
            labelcolor='white', fontsize=13, title='Detected Objects',
            title_fontsize=15, borderaxespad=0, handlelength=2.0,
            handleheight=1.4, labelspacing=0.9,
        )
        legend.get_title().set_color('white')

    # ── height / intensity / density maps ────────────────────────────────────
    height_map, intensity_map, density_map = compute_bev_maps(points, x_range, y_range)

    map_specs = [
        (height_map, 'viridis', 'BEV height map', 'Height (m)'),
        (intensity_map, 'hot', 'BEV intensity map', 'Intensity'),
        (density_map, 'magma', 'BEV density map', 'log(count+1)'),
    ]
    for col, (data, cmap, title, cbar_label) in enumerate(map_specs):
        mx = fig.add_subplot(gs[maps_row, col])
        mx.set_facecolor('#0D1117')
        im = mx.imshow(
            np.ma.masked_invalid(data), origin='lower',
            extent=(*x_range, *y_range), cmap=cmap,
        )
        mx.scatter([0], [0], marker='*', color='red', s=140, zorder=10, label='Ego')
        mx.legend(loc='upper right', fontsize=10, framealpha=0.85,
                  facecolor='#1A1F2E', edgecolor='#4A5568', labelcolor='white')
        mx.set_title(title, color='white', fontsize=12, pad=8)
        mx.tick_params(colors='#AAAAAA', labelsize=9)
        for spine in mx.spines.values():
            spine.set_edgecolor('#4A5568')
        cbar = fig.colorbar(im, ax=mx, fraction=0.046, pad=0.03)
        cbar.set_label(cbar_label, color='#AAAAAA', fontsize=10)
        cbar.ax.tick_params(colors='#AAAAAA', labelsize=9)

    fig.canvas.draw()
    canvas = cast(FigureCanvasAgg, fig.canvas)
    frame = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def get_scene_tokens(nusc, scene):
    tokens = []
    token = scene['first_sample_token']
    while token:
        tokens.append(token)
        token = nusc.get('sample', token)['next']
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/student.yaml')
    parser.add_argument('--scene-name', default=None, help='e.g. scene-0061')
    parser.add_argument('--scene-idx', type=int, default=0,
                         help='Used if --scene-name not given')
    parser.add_argument('--output', default='outputs/scene_video.mp4')
    parser.add_argument('--fps', type=float, default=2.0,
                         help='nuScenes keyframes are natively ~2Hz; raise this '
                              'to speed up playback, not add real frames')
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--list-scenes', action='store_true')
    parser.add_argument('--hide-annotated-bev', action='store_true',
                         help='Drop the ground-truth-boxes BEV panel and its legend, '
                              'leaving only raw sensor input: cameras + LiDAR height/'
                              'intensity/density maps.')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_cfg = config['data']
    nusc = NuScenes(version=data_cfg['version'], dataroot=data_cfg['dataroot'], verbose=False)

    if args.list_scenes:
        print(f"{'Idx':<5} {'Name':<16} {'#Samples'}")
        for i, sc in enumerate(nusc.scene):
            print(f"{i:<5} {sc['name']:<16} {sc['nbr_samples']}")
        return

    scene = None
    if args.scene_name:
        scene = next((s for s in nusc.scene if s['name'] == args.scene_name), None)
        if scene is None:
            raise ValueError(f'Scene {args.scene_name} not found. Use --list-scenes.')
    else:
        scene = nusc.scene[args.scene_idx]

    tokens = get_scene_tokens(nusc, scene)
    if args.max_frames:
        tokens = tokens[: args.max_frames]
    print(f"Rendering scene '{scene['name']}' — {len(tokens)} keyframes"
          f"{' (no annotations — camera + LiDAR only)' if args.hide_annotated_bev else ''}")

    import imageio
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.output, fps=args.fps, macro_block_size=None)
    for i, token in enumerate(tokens):
        frame = render_frame(nusc, token, show_annotated_bev=not args.hide_annotated_bev)
        writer.append_data(frame)
        print(f'  frame {i + 1}/{len(tokens)}')
    writer.close()
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()