"""
Count detected/annotated objects per class in nuScenes mini.

Reports two distinct numbers per class, since they answer different questions:
  - unique_instances : how many distinct physical objects (via instance_token)
  - total_annotations: how many per-frame annotation boxes exist in total
                        (the same object counted once per keyframe it appears in)

Usage:
    # Single scene (matches what render_scene_video.py renders)
    python scripts/count_object_classes.py \
        --config configs/student.yaml \
        --scene-name scene-0061

    # Whole official split (feeds directly into the class-imbalance discussion
    # in your dissertation's Limitations section)
    python scripts/count_object_classes.py \
        --config configs/student.yaml \
        --split mini_train \
        --output-csv outputs/class_counts_train.csv \
        --output-chart outputs/class_counts_train.png

    # All 10 mini scenes regardless of split assignment
    python scripts/count_object_classes.py --config configs/student.yaml --split all
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

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
CLASS_ORDER = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier',
]


def get_scene_tokens(nusc, scene):
    tokens = []
    token = scene['first_sample_token']
    while token:
        tokens.append(token)
        token = nusc.get('sample', token)['next']
    return tokens


def resolve_scenes(nusc, args):
    """Return the list of scene dicts to count over, based on the chosen scope."""
    if args.scene_name:
        scene = next((s for s in nusc.scene if s['name'] == args.scene_name), None)
        if scene is None:
            raise ValueError(f'Scene {args.scene_name} not found.')
        return [scene]

    if args.split == 'all':
        return list(nusc.scene)

    split_scenes = set(create_splits_scenes().get(args.split, []))
    if not split_scenes:
        raise ValueError(
            f"Split '{args.split}' has no scenes for this nuScenes version — "
            f"for the mini dataset use 'mini_train' or 'mini_val'."
        )
    return [s for s in nusc.scene if s['name'] in split_scenes]


def count_classes(nusc, scenes):
    unique_instances = defaultdict(set)
    total_annotations = defaultdict(int)
    n_samples = 0

    for scene in scenes:
        for sample_token in get_scene_tokens(nusc, scene):
            sample = nusc.get('sample', sample_token)
            n_samples += 1
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                cls = CAT_MAP.get(ann['category_name'])
                if cls is None:
                    continue
                total_annotations[cls] += 1
                unique_instances[cls].add(ann['instance_token'])

    return unique_instances, total_annotations, n_samples


def print_table(unique_instances, total_annotations):
    print(f"\n{'Class':<22} {'Unique objects':<16} {'Total annotations':<18}")
    print('-' * 58)
    for cls in CLASS_ORDER:
        u = len(unique_instances.get(cls, set()))
        t = total_annotations.get(cls, 0)
        print(f'{cls:<22} {u:<16} {t:<18}')
    print('-' * 58)
    u_total = sum(len(v) for v in unique_instances.values())
    t_total = sum(total_annotations.values())
    print(f"{'TOTAL':<22} {u_total:<16} {t_total:<18}\n")


def save_csv(path, unique_instances, total_annotations):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('class,unique_objects,total_annotations\n')
        for cls in CLASS_ORDER:
            u = len(unique_instances.get(cls, set()))
            t = total_annotations.get(cls, 0)
            f.write(f'{cls},{u},{t}\n')
    print(f'Saved CSV: {path}')


def save_chart(path, unique_instances, total_annotations, title_suffix=''):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    classes = CLASS_ORDER
    unique_counts = [len(unique_instances.get(c, set())) for c in classes]
    total_counts = [total_annotations.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    w = 0.38

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    b1 = ax.bar(x - w / 2, unique_counts, w, label='Unique objects', color='#4A90D9')
    b2 = ax.bar(x + w / 2, total_counts, w, label='Total annotations (all frames)', color='#E67E22')

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{int(h)}', (bar.get_x() + bar.get_width() / 2, h),
                            ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.set_ylabel('Count')
    ax.set_title(f'Per-Class Object Counts{title_suffix}', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.4)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved chart: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/student.yaml')
    parser.add_argument('--scene-name', default=None,
                         help='Count a single scene, e.g. scene-0061')
    parser.add_argument('--split', default='mini_train',
                         choices=['mini_train', 'mini_val', 'all'],
                         help='Used if --scene-name not given')
    parser.add_argument('--output-csv', default=None)
    parser.add_argument('--output-chart', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_cfg = config['data']
    nusc = NuScenes(version=data_cfg['version'], dataroot=data_cfg['dataroot'], verbose=False)

    scenes = resolve_scenes(nusc, args)
    scope_label = args.scene_name if args.scene_name else args.split
    print(f"Counting objects over: {scope_label}  ({len(scenes)} scene(s))")

    unique_instances, total_annotations, n_samples = count_classes(nusc, scenes)
    print(f'Scanned {n_samples} keyframes across {len(scenes)} scene(s)')
    print_table(unique_instances, total_annotations)

    if args.output_csv:
        save_csv(args.output_csv, unique_instances, total_annotations)
    if args.output_chart:
        save_chart(args.output_chart, unique_instances, total_annotations,
                   title_suffix=f' — {scope_label}')


if __name__ == '__main__':
    main()
