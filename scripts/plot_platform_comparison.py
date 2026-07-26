"""
Platform Comparison Plots -- MPS (Apple Silicon) vs RTX 3090 (CUDA).

Generates two dissertation figures from evaluate.py results:
    1. fig_platform_speed_slope.png    - FPS + latency slope charts
    2. fig_full_metrics_slope.png      - FPS, GFLOPs, mAP/NDS comparison

Usage:
    python scripts/plot_platform_comparison.py

Edit the mps, gpu, gflops, mAP_gpu, NDS_gpu values below to match
your latest results/{run_name}/metrics.json values before regenerating.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linewidth"] = 0.6
plt.rcParams["axes.edgecolor"] = "#444444"
plt.rcParams["axes.linewidth"] = 0.8

# Colors (muted, IEEE/Nature-style palette)
C_MPS  = "#8c9bab"
C_GPU  = "#2c3e50"
C_LINE = "#999999"
C_MAP  = "#4a7ba6"
C_NDS  = "#b5654f"

# Data (update from results/{run_name}/metrics.json)
mps = {"fps": 0.45, "latency": 2204.57}
gpu = {"fps": 1.32, "latency": 759.04}
gflops = 14.29
mAP_gpu, NDS_gpu = 0.0377, 0.0436

x = [0, 1]
labels = ["Apple Silicon\n(MPS)", "RTX 3090\n(CUDA)"]

legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_MPS, markersize=9, label="Apple Silicon (MPS)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_GPU, markersize=9, label="RTX 3090 (CUDA)"),
]


def plot_speed_comparison(output_path="results/fig_platform_speed_slope.png"):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))

    y_fps = [mps["fps"], gpu["fps"]]
    axes[0].plot(x, y_fps, color=C_LINE, linewidth=1.4, zorder=2)
    axes[0].scatter(x, y_fps, color=[C_MPS, C_GPU], s=90, zorder=3, edgecolor="white", linewidth=1.2)
    axes[0].axhline(y=10, color="#a33", linestyle=(0, (5, 3)), linewidth=1.1, zorder=1)
    axes[0].text(0.5, 10.4, "Min. ADAS target - 10 FPS", color="#a33", fontsize=9, ha="center")
    for xi, yi in zip(x, y_fps):
        axes[0].annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                          xytext=(0, 12), ha="center", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_xlim(-0.3, 1.3)
    axes[0].set_ylim(0, 12)
    axes[0].set_ylabel("Frames per second (FPS)")
    axes[0].set_title("Inference Speed", fontsize=12, pad=10)
    axes[0].legend(handles=legend_handles, fontsize=8.5, frameon=False, loc="upper left")

    y_lat = [mps["latency"], gpu["latency"]]
    axes[1].plot(x, y_lat, color=C_LINE, linewidth=1.4, zorder=2)
    axes[1].scatter(x, y_lat, color=[C_MPS, C_GPU], s=90, zorder=3, edgecolor="white", linewidth=1.2)
    for xi, yi in zip(x, y_lat):
        axes[1].annotate(f"{yi:.0f} ms", (xi, yi), textcoords="offset points",
                          xytext=(0, 12), ha="center", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_xlim(-0.3, 1.3)
    axes[1].set_ylim(0, 2600)
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Inference Latency", fontsize=12, pad=10)
    axes[1].legend(handles=legend_handles, fontsize=8.5, frameon=False, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {output_path}")


def plot_full_metrics_comparison(output_path="results/fig_full_metrics_slope.png"):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5))

    y_fps = [mps["fps"], gpu["fps"]]
    axes[0].plot(x, y_fps, color=C_LINE, linewidth=1.4, zorder=2)
    axes[0].scatter(x, y_fps, color=[C_MPS, C_GPU], s=80, zorder=3, edgecolor="white", linewidth=1.2)
    for xi, yi in zip(x, y_fps):
        axes[0].annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                          xytext=(0, 10), ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["MPS", "RTX 3090"], fontsize=9)
    axes[0].set_xlim(-0.3, 1.3)
    axes[0].set_ylim(0, 1.8)
    axes[0].set_ylabel("FPS")
    axes[0].set_title("Inference Speed", fontsize=11, pad=10)
    axes[0].legend(handles=legend_handles, fontsize=8, frameon=False, loc="upper left")

    axes[1].scatter([0], [gflops], color="#3d5266", s=90, zorder=3, edgecolor="white", linewidth=1.2)
    axes[1].annotate(f"{gflops:.2f}", (0, gflops), textcoords="offset points",
                      xytext=(0, 12), ha="center", fontsize=10)
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["Student model"], fontsize=9)
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(0, 17)
    axes[1].set_ylabel("GFLOPs")
    axes[1].set_title("Computational Cost\n(architecture-level)", fontsize=11, pad=10)

    map_nds_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_MAP, markersize=9, label="mAP"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NDS, markersize=9, label="NDS"),
    ]
    axes[2].scatter([0], [mAP_gpu], color=C_MAP, s=90, zorder=3, edgecolor="white", linewidth=1.2)
    axes[2].scatter([0], [NDS_gpu], color=C_NDS, s=90, zorder=3, edgecolor="white", linewidth=1.2)
    axes[2].annotate(f"{mAP_gpu:.4f}", (0, mAP_gpu), textcoords="offset points",
                      xytext=(15, -3), ha="left", fontsize=9)
    axes[2].annotate(f"{NDS_gpu:.4f}", (0, NDS_gpu), textcoords="offset points",
                      xytext=(15, -3), ha="left", fontsize=9)
    axes[2].set_xticks([0])
    axes[2].set_xticklabels(["RTX 3090\n(mini-val)"], fontsize=9)
    axes[2].set_xlim(-1, 1.3)
    axes[2].set_ylim(0, 0.06)
    axes[2].set_ylabel("Score")
    axes[2].set_title("Detection Metrics\n(MPS: not computed)", fontsize=11, pad=10)
    axes[2].legend(handles=map_nds_legend, fontsize=9, frameon=False, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    plot_speed_comparison()
    plot_full_metrics_comparison()
