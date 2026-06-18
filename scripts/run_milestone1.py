"""Milestone 1: Environment & Data Loading Verification."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.nuscenes_loader import NuScenesDataset, collate_fn
from src.utils.visualise import plot_full_sample, lidar_to_bev
from src.utils.device import get_device

def main():
    print("=" * 60)
    print("  Milestone 1: Environment & Data Loading")
    print("=" * 60)
    device = get_device()
    dataroot = "data/nuscenes"

    if not os.path.exists(dataroot):
        print(f"\n  Dataset not found at: {os.path.abspath(dataroot)}")
        print(f"\n  To set up the dataset:")
        print(f"    1. Go to https://www.nuscenes.org/nuscenes#download")
        print(f"    2. Create a free account")
        print(f"    3. Download 'Mini' split (metadata + file blobs)")
        print(f"    4. mkdir -p data/nuscenes")
        print(f"    5. Extract both archives into data/nuscenes/")
        print(f"\n  Expected structure:")
        print(f"    data/nuscenes/v1.0-mini/")
        print(f"    data/nuscenes/samples/")
        print(f"    data/nuscenes/sweeps/")
        print(f"    data/nuscenes/maps/")
        sys.exit(1)
    dataset = NuScenesDataset(dataroot=dataroot, version="v1.0-mini", split="train")
    
    print(f"\n[1] Loading sample 0...")
    sample = dataset[0]
    print(f"  Camera images: {sample['camera_images'].shape}")
    print(f"  LiDAR points:  {sample['lidar_points'].shape}")
    print(f"  Annotations:   {sample['annotations']['boxes'].shape[0]} objects")
    
    if sample['annotations']['names']:
        print(f"  Classes: {sorted(set(sample['annotations']['names']))}")
    
    print(f"\n[2] LiDAR statistics:")
    pts = sample['lidar_points']
    valid = (pts[:,0] != 0) | (pts[:,1] != 0) | (pts[:,2] != 0)
    vp = pts[valid]
    print(f"  Valid points: {valid.sum().item():,}")
    print(f"  X: [{vp[:,0].min():.1f}, {vp[:,0].max():.1f}] m")
    print(f"  Y: [{vp[:,1].min():.1f}, {vp[:,1].max():.1f}] m")
    print(f"  Z: [{vp[:,2].min():.1f}, {vp[:,2].max():.1f}] m")
    
    print(f"\n[3] BEV projection...")
    bev = lidar_to_bev(sample['lidar_points'])
    print(f"  Grid: {bev['height'].shape}, Coverage: {(bev['density']>0).sum()/bev['density'].size*100:.1f}%")
    
    print(f"\n[4] DataLoader test...")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"  Batch cameras: {batch['camera_images'].shape}")
    print(f"  Batch LiDAR:   {batch['lidar_points'].shape}")
    print(f"\n[5] Generating visualisations...")
    os.makedirs("outputs", exist_ok=True)
    plot_full_sample(sample, save_dir="outputs/milestone1_sample0")
    
    print(f"\n[6] Model pipeline test...")
    from src.models.fusion import ChannelWiseFusion
    fusion = ChannelWiseFusion()
    fused = fusion(torch.randn(1,256,50,50), torch.randn(1,256,50,50))
    print(f"  Fusion output: {fused.shape}")
    print(f"\n{'='*60}")
    print(f"  Milestone 1 Complete!")
    print(f"{'='*60}")
    print(f"  Saved to: outputs/milestone1_sample0/")
    print(f"  Next: Milestone 2 - YOLO backbone features")

if __name__ == "__main__":
    main()
