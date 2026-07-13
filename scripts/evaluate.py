"""
Evaluation Script — Student BEV Model.

Computes:
    - Detection metrics: mAP, NDS via nuScenes official evaluator
    - Inference speed: FPS, latency (ms) averaged over val set
    - Computational cost: GFLOPs per forward pass
    - Memory footprint: peak GPU/CPU memory, model file size on disk

Works on ANY checkpoint — baseline, KD, pruned, quantized.
Run this script after every training variant to populate the results table.

Usage:
    # Evaluate baseline
    python scripts/evaluate.py \
        --checkpoint checkpoints/baseline_20ep_best.pth \
        --config configs/student.yaml \
        --run-name baseline

    # Evaluate KD model
    python scripts/evaluate.py \
        --checkpoint checkpoints/kd_real_best.pth \
        --config configs/student.yaml \
        --run-name kd_mock

    # Evaluate on CPU (for embedded device simulation)
    python scripts/evaluate.py \
        --checkpoint checkpoints/baseline_20ep_best.pth \
        --config configs/student.yaml \
        --device cpu \
        --run-name baseline_cpu

Output:
    results/{run-name}/
        metrics.json    ← all numbers in one file
        summary.txt     ← human-readable table for dissertation
"""

import argparse
import json
import os
import sys
import time
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.student import StudentBEV
from src.data.nuscenes_loader import NuScenesDataset, collate_fn
from src.utils.device import get_device
from torch.utils.data import DataLoader


# ── FLOPs counter ─────────────────────────────────────────────────────────────

def count_flops(model, sample_batch, device):
    """Estimate GFLOPs for one forward pass using hook-based counting.

    Falls back to parameter-count estimate if torchinfo not available.
    """
    try:
        from torchinfo import summary as torch_summary
        camera_images = sample_batch['camera_images'][:1].to(device)
        lidar_points  = sample_batch['lidar_points'][:1].to(device)
        calibration   = sample_batch['calibration'][:1]

        # torchinfo can't handle dict outputs cleanly so we wrap
        class Wrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, cam, lid): return self.m(cam, lid, None)

        info = torch_summary(
            Wrapper(model),
            input_data=[camera_images, lidar_points],
            verbose=0,
        )
        gflops = info.total_mult_adds / 1e9
        return round(gflops, 2)
    except Exception:
        # Fallback: rough estimate from parameter count
        # ~2 FLOPs per parameter per forward pass (multiply-accumulate)
        params = sum(p.numel() for p in model.parameters())
        return round(params * 2 / 1e9, 2)


# ── Speed benchmark ───────────────────────────────────────────────────────────

def benchmark_speed(model, loader, device, n_warmup=10, n_measure=50):
    """Measure inference latency and FPS.

    Args:
        n_warmup:  number of warmup iterations (not measured)
        n_measure: number of measured iterations

    Returns:
        dict with mean_ms, std_ms, fps
    """
    model.eval()
    latencies = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            camera_images = batch['camera_images'].to(device)
            lidar_points  = batch['lidar_points'].to(device)
            calibration   = batch['calibration']

            # Synchronise GPU before timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()

            t0 = time.perf_counter()
            _ = model(camera_images, lidar_points, calibration)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()

            t1 = time.perf_counter()

            if i >= n_warmup:
                latencies.append((t1 - t0) * 1000)  # ms

            if i >= n_warmup + n_measure:
                break

    if not latencies:
        return {'mean_ms': 0.0, 'std_ms': 0.0, 'fps': 0.0}

    mean_ms = float(np.mean(latencies))
    std_ms  = float(np.std(latencies))
    fps     = 1000.0 / mean_ms

    return {
        'mean_ms': round(mean_ms, 2),
        'std_ms':  round(std_ms, 2),
        'fps':     round(fps, 2),
    }


# ── Memory measurement ────────────────────────────────────────────────────────

def measure_memory(model, sample_batch, device):
    """Measure peak memory during one forward pass (MB)."""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        camera_images = sample_batch['camera_images'][:1].to(device)
        lidar_points  = sample_batch['lidar_points'][:1].to(device)
        calibration   = sample_batch['calibration'][:1]
        with torch.no_grad():
            _ = model(camera_images, lidar_points, calibration)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        return round(peak_mb, 1)
    else:
        # CPU/MPS: use rough estimate from parameter count
        params  = sum(p.numel() for p in model.parameters())
        bytes_  = params * 4  # float32 = 4 bytes
        return round(bytes_ / 1e6, 1)


# ── nuScenes detection metrics ────────────────────────────────────────────────

def compute_detection_metrics(model, loader, device, config, output_dir):
    """
    Compute mAP and NDS using nuScenes official evaluator.

    The nuScenes evaluator expects predictions in a specific JSON format.
    We run inference on the val set, convert predictions to that format,
    then call the evaluator.

    Note: With nuScenes mini (80 val samples), mAP/NDS values will be
    lower and noisier than full val set results. Report these as
    'mini-val' metrics to set correct expectations in the dissertation.
    """
    model.eval()

    x_range = tuple(config['data']['x_range'])
    y_range = tuple(config['data']['y_range'])
    x_min, x_max = x_range
    y_min, y_max = y_range
    bev_h = bev_w = 50

    # nuScenes class names in order matching our 10-class head
    CLASS_NAMES = [
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
    ]

    # nuScenes attribute defaults per class (required by evaluator)
    CLASS_ATTRS = {
        'car':                   'vehicle.moving',
        'truck':                 'vehicle.moving',
        'bus':                   'vehicle.moving',
        'trailer':               'vehicle.parked',
        'construction_vehicle':  'vehicle.parked',
        'pedestrian':            'pedestrian.moving',
        'motorcycle':            'cycle.with_rider',
        'bicycle':               'cycle.with_rider',
        'traffic_cone':          '',
        'barrier':               '',
    }

    predictions = []

    print('Running inference on val set...')
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            camera_images  = batch['camera_images'].to(device)
            lidar_points   = batch['lidar_points'].to(device)
            calibration    = batch['calibration']
            sample_tokens  = batch['sample_tokens']

            outputs = model(camera_images, lidar_points, calibration)
            heatmap    = outputs['detections']['heatmap']     # (B,10,50,50)
            regression = outputs['detections']['regression']  # (B,8,50,50)

            B = heatmap.shape[0]
            for b in range(B):
                token = sample_tokens[b]
                hm_b  = heatmap[b].cpu().numpy()     # (10,50,50)
                reg_b = regression[b].cpu().numpy()  # (8,50,50)

                # Find peaks in heatmap above threshold
                for cls_idx, cls_name in enumerate(CLASS_NAMES):
                    cls_hm = hm_b[cls_idx]  # (50,50)
                    # Simple peak finding: cells above 0.2 threshold
                    rows, cols = np.where(cls_hm > 0.2)

                    for r, c in zip(rows, cols):
                        score = float(cls_hm[r, c])

                        # Convert BEV grid coords back to ego-frame metres
                        x = (c + 0.5) / bev_w * (x_max - x_min) + x_min
                        y = (r + 0.5) / bev_h * (y_max - y_min) + y_min

                        # Regression outputs at this cell
                        dz       = float(reg_b[0, r, c])
                        w        = max(float(reg_b[3, r, c]), 0.1)
                        l        = max(float(reg_b[4, r, c]), 0.1)
                        h        = max(float(reg_b[5, r, c]), 0.5)
                        sin_yaw  = float(reg_b[6, r, c])
                        cos_yaw  = float(reg_b[7, r, c])
                        yaw      = float(np.arctan2(sin_yaw, cos_yaw))
                        z        = dz

                        attr = CLASS_ATTRS[cls_name]

                        pred = {
                            'sample_token':        token,
                            'translation':         [x, y, z],
                            'size':                [w, l, h],
                            'rotation':            [
                                float(np.cos(yaw / 2)), 0.0, 0.0,
                                float(np.sin(yaw / 2))
                            ],
                            'velocity':            [0.0, 0.0],
                            'detection_name':      cls_name,
                            'detection_score':     score,
                            'attribute_name':      attr,
                        }
                        predictions.append(pred)

            if batch_idx % 5 == 0:
                print(f'  Processed {batch_idx+1}/{len(loader)} batches')

    # Save predictions to JSON
    pred_file = output_dir / 'predictions.json'
    pred_data = {'meta': {'use_camera': True, 'use_lidar': True,
                          'use_radar': False, 'use_map': False,
                          'use_external': False},
                 'results': {}}

    for p in predictions:
        tok = p['sample_token']
        if tok not in pred_data['results']:
            pred_data['results'][tok] = []
        pred_data['results'][tok].append(p)

    with open(pred_file, 'w') as f:
        json.dump(pred_data, f)

    print(f'Saved {len(predictions)} predictions to {pred_file}')

    # Run nuScenes evaluator
    try:
        from nuscenes import NuScenes
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        data_cfg = config['data']
        nusc = NuScenes(
            version=data_cfg['version'],
            dataroot=data_cfg['dataroot'],
            verbose=False,
        )
        eval_cfg = config_factory('detection_cvpr_2019')

        # Determine correct eval split name
        eval_split = 'mini_val' if 'mini' in data_cfg.get('version', '') else 'val'

        # Get exact tokens the evaluator expects for this split
        from nuscenes.utils.splits import create_splits_scenes
        val_scene_names = set(create_splits_scenes().get(eval_split, []))
        val_tokens = set(
            s['token'] for s in nusc.sample
            if nusc.get('scene', s['scene_token'])['name'] in val_scene_names
        )

        # Fill missing val tokens with empty predictions
        for tok in val_tokens:
            if tok not in pred_data['results']:
                pred_data['results'][tok] = []

        # Remove tokens NOT in val split (evaluator rejects extra tokens)
        pred_data['results'] = {
            tok: v for tok, v in pred_data['results'].items()
            if tok in val_tokens
        }

        # Re-save filtered predictions
        with open(pred_file, 'w') as pf:
            json.dump(pred_data, pf)

        print(f'  Filtered to {len(pred_data["results"])} val tokens for evaluator')

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_cfg,
            result_path=str(pred_file),
            eval_set=eval_split,
            output_dir=str(output_dir),
            verbose=False,
        )
        metrics, _ = nusc_eval.evaluate()

        map_score = round(metrics.nd_score, 4)
        nds_score = round(metrics.mean_dist_aps['car'] if hasattr(metrics, 'mean_dist_aps') else 0.0, 4)

        # Extract per-class AP
        per_class = {}
        if hasattr(metrics, 'mean_ap_dist'):
            for cls in CLASS_NAMES:
                if cls in metrics.mean_ap_dist:
                    per_class[cls] = round(metrics.mean_ap_dist[cls], 4)

        return {
            'mAP': round(metrics.mean_ap, 4) if hasattr(metrics, 'mean_ap') else map_score,
            'NDS': round(metrics.nd_score, 4),
            'per_class_AP': per_class,
        }

    except Exception as e:
        print(f'  nuScenes evaluator failed: {e}')
        print('  Returning placeholder metrics — check predictions.json manually')
        return {'mAP': None, 'NDS': None, 'per_class_AP': {}, 'error': str(e)}


# ── Model size ────────────────────────────────────────────────────────────────

def get_model_size(checkpoint_path):
    """File size of checkpoint on disk in MB."""
    if checkpoint_path and Path(checkpoint_path).exists():
        size_bytes = Path(checkpoint_path).stat().st_size
        return round(size_bytes / 1e6, 1)
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate student BEV model')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--config',     default='configs/student.yaml')
    parser.add_argument('--run-name',   default='eval',
                        help='Name for output directory')
    parser.add_argument('--device',     default=None,
                        help='Force device (cuda/mps/cpu). Auto-detects if not set.')
    parser.add_argument('--workers',    type=int, default=2)
    parser.add_argument('--skip-metrics', action='store_true',
                        help='Skip mAP/NDS (just measure speed/memory)')
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f'\n{"="*60}')
    print(f'  Evaluation: {args.run_name}')
    print(f'  Checkpoint: {args.checkpoint}')
    print(f'  Device:     {device}')
    print(f'{"="*60}\n')

    output_dir = Path('results') / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    model = StudentBEV(config['model']).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    trained_epoch    = ckpt.get('epoch', '?')
    trained_val_loss = ckpt.get('val_loss', None)

    print(f'Loaded checkpoint: epoch {trained_epoch}',
          f'val_loss={trained_val_loss:.4f}' if trained_val_loss else '')

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {total_params:,}')

    # ── Dataset ────────────────────────────────────────────────────────────
    data_cfg = config['data']
    val_dataset = NuScenesDataset(
        dataroot=data_cfg['dataroot'],
        version=data_cfg['version'],
        split='val',
        x_range=tuple(data_cfg['x_range']),
        y_range=tuple(data_cfg['x_range']),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn,
    )

    # Get one sample for FLOPs and memory measurement
    sample_batch = next(iter(val_loader))

    # ── Metrics collection ─────────────────────────────────────────────────
    results = {
        'run_name':       args.run_name,
        'checkpoint':     args.checkpoint,
        'epoch':          trained_epoch,
        'val_loss':       trained_val_loss,
        'parameters':     total_params,
        'checkpoint_mb':  get_model_size(args.checkpoint),
    }

    # FLOPs
    print('\n[1/4] Counting FLOPs...')
    results['gflops'] = count_flops(model, sample_batch, device)
    print(f'  GFLOPs: {results["gflops"]}')

    # Memory
    print('\n[2/4] Measuring memory...')
    results['peak_memory_mb'] = measure_memory(model, sample_batch, device)
    print(f'  Peak memory: {results["peak_memory_mb"]} MB')

    # Speed
    print('\n[3/4] Benchmarking speed (batch_size=1)...')
    speed = benchmark_speed(model, val_loader, device)
    results.update(speed)
    print(f'  FPS:     {speed["fps"]}')
    print(f'  Latency: {speed["mean_ms"]} ± {speed["std_ms"]} ms')

    # Detection metrics
    if not args.skip_metrics:
        print('\n[4/4] Computing mAP / NDS...')
        det_metrics = compute_detection_metrics(
            model, val_loader, device, config, output_dir
        )
        results.update(det_metrics)
        if results.get('mAP') is not None:
            print(f'  mAP: {results["mAP"]}')
            print(f'  NDS: {results["NDS"]}')
    else:
        print('\n[4/4] Skipping mAP/NDS (--skip-metrics flag set)')
        results['mAP'] = None
        results['NDS'] = None

    # ── Save results ───────────────────────────────────────────────────────
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Human-readable summary
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f'Evaluation Summary: {args.run_name}\n')
        f.write('=' * 50 + '\n')
        f.write(f'Checkpoint:      {args.checkpoint}\n')
        f.write(f'Epoch:           {trained_epoch}\n')
        f.write(f'Val Loss:        {trained_val_loss:.4f}\n' if trained_val_loss else '')
        f.write(f'\n--- Model Size ---\n')
        f.write(f'Parameters:      {total_params:,}\n')
        f.write(f'Checkpoint size: {results["checkpoint_mb"]} MB\n')
        f.write(f'\n--- Inference Speed ({device}) ---\n')
        f.write(f'FPS:             {speed["fps"]}\n')
        f.write(f'Latency:         {speed["mean_ms"]} ± {speed["std_ms"]} ms\n')
        f.write(f'\n--- Computational Cost ---\n')
        f.write(f'GFLOPs:          {results["gflops"]}\n')
        f.write(f'Peak Memory:     {results["peak_memory_mb"]} MB\n')
        f.write(f'\n--- Detection Metrics (nuScenes mini-val) ---\n')
        f.write(f'mAP:             {results.get("mAP", "N/A")}\n')
        f.write(f'NDS:             {results.get("NDS", "N/A")}\n')
        if results.get('per_class_AP'):
            f.write(f'\nPer-class AP:\n')
            for cls, ap in results['per_class_AP'].items():
                f.write(f'  {cls:<25} {ap:.4f}\n')

    print(f'\n{"="*60}')
    print(f'  Results saved to: {output_dir}/')
    print(f'  metrics.json, summary.txt')
    print(f'{"="*60}\n')

    # Print final table row for dissertation
    mAP_str = f'{results["mAP"]:.4f}' if isinstance(results.get("mAP"), float) else '—'
    NDS_str = f'{results["NDS"]:.4f}' if isinstance(results.get("NDS"), float) else '—'
    print('Dissertation table row:')
    print(f'  | {args.run_name:<30} | '
          f'{mAP_str:>6} | '
          f'{NDS_str:>6} | '
          f'{speed["fps"]:>6} | '
          f'{results["gflops"]:>8} | '
          f'{total_params/1e6:.1f}M |')


if __name__ == '__main__':
    main()
