"""BEV Visualisation Utilities."""
import numpy as np, torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASS_COLOURS = {"car":"#3498db","truck":"#2980b9","bus":"#1abc9c","trailer":"#16a085",
    "construction_vehicle":"#f39c12","pedestrian":"#e74c3c","motorcycle":"#9b59b6",
    "bicycle":"#8e44ad","traffic_cone":"#e67e22","barrier":"#95a5a6"}

def lidar_to_bev(points, x_range=(-50,50), y_range=(-50,50), resolution=0.25):
    if isinstance(points, torch.Tensor): points = points.numpy()
    x, y, z, intensity = points[:,0], points[:,1], points[:,2], points[:,3]
    valid = (x != 0) | (y != 0) | (z != 0)
    x, y, z, intensity = x[valid], y[valid], z[valid], intensity[valid]
    mask = (x >= x_range[0]) & (x < x_range[1]) & (y >= y_range[0]) & (y < y_range[1])
    x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]
    gw = int((x_range[1]-x_range[0])/resolution); gh = int((y_range[1]-y_range[0])/resolution)
    xi = np.clip(((x-x_range[0])/resolution).astype(int), 0, gw-1)
    yi = np.clip(((y-y_range[0])/resolution).astype(int), 0, gh-1)
    bev_h, bev_i, bev_d = np.full((gh,gw),-10.0), np.zeros((gh,gw)), np.zeros((gh,gw))
    for i in range(len(x)):
        bev_h[yi[i],xi[i]] = max(bev_h[yi[i],xi[i]], z[i])
        bev_i[yi[i],xi[i]] = max(bev_i[yi[i],xi[i]], intensity[i])
        bev_d[yi[i],xi[i]] += 1
    bev_h[bev_h == -10.0] = 0
    return {"height": bev_h, "intensity": bev_i, "density": bev_d}

def plot_bev_maps(bev, x_range=(-50,50), y_range=(-50,50), save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("LiDAR Bird's Eye View (BEV)", fontsize=16, fontweight="bold")
    ext = [x_range[0], x_range[1], y_range[0], y_range[1]]
    for ax, (key, cmap, label) in zip(axes, [("height","viridis","Height (m)"),("intensity","hot","Intensity"),("density","magma","log(count+1)")]):
        data = np.log1p(bev[key]) if key == "density" else bev[key]
        im = ax.imshow(data, cmap=cmap, origin="lower", extent=ext)
        ax.set_title(f"BEV {key} map"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.plot(0, 0, "r*", markersize=12, label="Ego"); ax.legend()
        plt.colorbar(im, ax=ax, label=label, shrink=0.7)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"Saved: {save_path}")
    plt.show()

def plot_bev_with_boxes(bev_density, annotations, x_range=(-50,50), y_range=(-50,50), title="BEV with annotations", save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(np.log1p(bev_density), cmap="Greys", origin="lower", extent=[x_range[0],x_range[1],y_range[0],y_range[1]], alpha=0.7)
    boxes = annotations["boxes"].numpy() if isinstance(annotations["boxes"], torch.Tensor) else annotations["boxes"]
    for i, name in enumerate(annotations["names"]):
        x, y, z, w, l, h, yaw = boxes[i]; colour = CLASS_COLOURS.get(name, "#95a5a6")
        corners = np.array([[-l/2,-w/2],[l/2,-w/2],[l/2,w/2],[-l/2,w/2],[-l/2,-w/2]])
        rot = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        corners = corners @ rot.T; corners[:,0] += x; corners[:,1] += y
        ax.plot(corners[:,0], corners[:,1], color=colour, linewidth=2)
    ax.add_patch(plt.Rectangle((-1,-2), 2, 4, fill=True, facecolor="yellow", edgecolor="black", linewidth=2, zorder=5))
    legend = [mpatches.Patch(facecolor="yellow", edgecolor="black", label="Ego")]
    for cls in sorted(set(annotations["names"])): legend.append(mpatches.Patch(facecolor=CLASS_COLOURS.get(cls,"#95a5a6"), label=cls))
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title(title, fontsize=14)
    ax.set_xlim(x_range); ax.set_ylim(y_range); ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"Saved: {save_path}")
    plt.show()

def plot_camera_views(camera_images, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    fig.suptitle("Multi-view Camera Images", fontsize=16, fontweight="bold")
    layout = [["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT"],["CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT"]]
    from src.data.nuscenes_loader import CAMERA_CHANNELS
    for r, row in enumerate(layout):
        for c, cam in enumerate(row):
            img = camera_images[CAMERA_CHANNELS.index(cam)].permute(1,2,0).numpy()
            axes[r][c].imshow(np.clip(img,0,1)); axes[r][c].set_title(cam.replace("CAM_","")); axes[r][c].axis("off")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"Saved: {save_path}")
    plt.show()

def plot_full_sample(sample, x_range=(-50,50), y_range=(-50,50), save_dir=None):
    import os
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    print("Plotting camera views...")
    plot_camera_views(sample["camera_images"], save_path=os.path.join(save_dir,"cameras.png") if save_dir else None)
    print("Generating BEV from LiDAR...")
    bev = lidar_to_bev(sample["lidar_points"], x_range=x_range, y_range=y_range)
    plot_bev_maps(bev, x_range=x_range, y_range=y_range, save_path=os.path.join(save_dir,"bev_maps.png") if save_dir else None)
    print("Plotting BEV with annotations...")
    n = sample["annotations"]["boxes"].shape[0]
    plot_bev_with_boxes(bev["density"], sample["annotations"], x_range=x_range, y_range=y_range,
        title=f"BEV with {n} annotated objects", save_path=os.path.join(save_dir,"bev_annotations.png") if save_dir else None)
    print("Done!")
