"""
Dissertation Plotting Utilities
================================
Self-contained functions to generate all figure types used throughout the
dissertation, styled consistently (serif fonts, 300 DPI, print-friendly).

Usage examples are at the bottom of this file (`if __name__ == "__main__"`).
Each function saves directly to OUTPUT_DIR and returns the file path.

Run standalone:
    python plot_results.py

Or import individual functions in a notebook / Colab cell:
    from plot_results import plot_training_curve, plot_bar_comparison, ...
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Global style (matches all previous dissertation figures) ───────────────

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'grid.color': '#dddddd',
    'grid.linewidth': 0.6,
})

BLUE   = '#2C5F8A'
ORANGE = '#D97B29'
GREEN  = '#4C8C6A'
GREY   = '#8A8A8A'
RED    = '#B34A4A'

OUTPUT_DIR = Path('figures')
OUTPUT_DIR.mkdir(exist_ok=True)


# ── 1. Dual-axis training curve (e.g. KD train/val loss over epochs) ───────

def plot_training_curve(
    epochs, train_series: dict, val_series: dict,
    filename='fig_training_curve.png',
    title='Training Progress',
    xlabel='Epoch',
    best_epoch=None,
):
    """
    epochs:      list of epoch numbers, e.g. [13,14,...,20]
    train_series: dict of {label: values} plotted on left axis (training losses)
    val_series:   dict of {label: values} plotted on right axis (validation loss)
    best_epoch:  optional epoch to highlight as 'best' (annotated with arrow)
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    markers = ['o', 's', '^', 'D', 'v']
    colors_train = [BLUE, GREEN, ORANGE, GREY]
    for i, (label, values) in enumerate(train_series.items()):
        style = '-' if i == 0 else '--'
        lw = 2 if i == 0 else 1.6
        ax1.plot(epochs, values, color=colors_train[i % len(colors_train)],
                  linewidth=lw, linestyle=style,
                  marker=markers[i % len(markers)], markersize=4 if i else 5,
                  label=label)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Training Loss')
    ax1.set_xticks(epochs)
    ax1.grid(True, alpha=0.5)
    ax1.spines['top'].set_visible(False)

    ax2 = ax1.twinx()
    val_label, val_values = list(val_series.items())[0]
    ax2.plot(epochs, val_values, color=RED, linewidth=2.2, marker='D',
              markersize=5, label=val_label)
    ax2.set_ylabel(val_label, color=RED)
    ax2.tick_params(axis='y', labelcolor=RED)
    ax2.spines['top'].set_visible(False)

    if best_epoch is not None and best_epoch in epochs:
        idx = epochs.index(best_epoch)
        best_val = val_values[idx]
        ax2.scatter([best_epoch], [best_val], s=110, facecolors='none',
                     edgecolors=RED, linewidths=2, zorder=5)
        ax2.annotate(f'Best: {best_val:.4f}\n(epoch {best_epoch})',
                      xy=(best_epoch, best_val),
                      xytext=(best_epoch - epochs[-1]*0.25, best_val - (max(val_values)-min(val_values))*0.15),
                      fontsize=9, color=RED, ha='left',
                      arrowprops=dict(arrowstyle='->', color=RED, lw=1))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                frameon=True, framealpha=0.9, edgecolor='#cccccc')

    ax1.set_title(title)
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


# ── 2. Simple bar comparison (e.g. baseline vs KD, N configs) ──────────────

def plot_bar_comparison(
    labels, values, filename='fig_bar_comparison.png',
    title='Comparison', ylabel='Value',
    colors=None, annotate_fmt='{:.4f}',
    reference_line=None, reference_label=None,
):
    """
    labels: list of category names (x-axis)
    values: list of numeric values (y-axis)
    colors: optional list of colors, defaults to BLUE/ORANGE/GREEN cycling
    reference_line: optional horizontal dashed line value (e.g. baseline)
    """
    if colors is None:
        palette = [BLUE, ORANGE, GREEN, RED, GREY]
        colors = [palette[i % len(palette)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(max(6.5, 1.3*len(labels)), 5))
    bars = ax.bar(labels, values, color=colors, width=0.5,
                    edgecolor='#333333', linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(values)*0.015,
                  annotate_fmt.format(val), ha='center', va='bottom',
                  fontsize=10.5, fontweight='bold')

    if reference_line is not None:
        ax.axhline(reference_line, color=GREY, linestyle='--', linewidth=1.2)
        ax.text(len(labels)-0.4, reference_line + max(values)*0.02,
                  reference_label or f'{reference_line:.4f}',
                  fontsize=8.5, color=GREY, ha='right', style='italic')

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


# ── 3. Line/trade-off plot (e.g. pruning sparsity vs val loss) ─────────────

def plot_tradeoff(
    x_values, y_values, point_labels=None,
    filename='fig_tradeoff.png', title='Trade-off',
    xlabel='X', ylabel='Y',
    reference_line=None, reference_label=None,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_values, y_values, color=BLUE, linewidth=2, marker='o',
             markersize=8, zorder=3)

    for i, (x, y) in enumerate(zip(x_values, y_values)):
        label = point_labels[i] if point_labels else None
        if label:
            ax.annotate(label, xy=(x, y), xytext=(x, y + (max(y_values)-min(y_values))*0.06),
                          ha='center', fontsize=8.5)
        ax.text(x, y - (max(y_values)-min(y_values))*0.05, f'{y:.3f}',
                  ha='center', fontsize=9.5, fontweight='bold', color=BLUE)

    if reference_line is not None:
        ax.axhline(reference_line, color=GREY, linestyle='--', linewidth=1.2)
        ax.text(max(x_values)*1.02, reference_line - (max(y_values)-min(y_values))*0.05,
                  reference_label or f'{reference_line:.4f}',
                  fontsize=8.5, color=GREY, ha='right', style='italic')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.5)

    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


# ── 4. Per-class / categorical bar chart (e.g. per-class AP) ───────────────

def plot_categorical_bars(
    categories, values, filename='fig_categorical.png',
    title='Per-Category Results', ylabel='Value',
    highlight_threshold=0.0, mean_line=None, mean_label=None,
):
    """Bars above highlight_threshold get GREEN, others GREY (e.g. per-class AP)."""
    colors = [GREEN if v > highlight_threshold else GREY for v in values]

    fig, ax = plt.subplots(figsize=(max(7, 0.9*len(categories)), 5))
    bars = ax.bar(categories, values, color=colors, edgecolor='#333333',
                    linewidth=0.7, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(values)*0.02,
                  f'{val:.3f}', ha='center', fontsize=9.5,
                  fontweight='bold' if val > highlight_threshold else 'normal',
                  color='#222222' if val > highlight_threshold else '#888888')

    if mean_line is not None:
        ax.axhline(mean_line, color=BLUE, linestyle='--', linewidth=1.3, zorder=1)
        ax.text(len(categories)-0.6, mean_line + max(values)*0.02,
                  mean_label or f'{mean_line:.4f}', fontsize=9, color=BLUE,
                  ha='right', style='italic')

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


# ── 5. Horizontal breakdown + side comparison (e.g. parameter breakdown) ───

def plot_breakdown_with_comparison(
    component_labels, component_values,
    compare_labels, compare_values,
    filename='fig_breakdown.png',
    breakdown_title='Breakdown', comparison_title='Comparison',
    breakdown_xlabel='Value', comparison_ylabel='Value',
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                     gridspec_kw={'width_ratios': [1.3, 1]})

    palette = [BLUE, '#5B8DB8', GREEN, ORANGE, GREY]
    colors1 = [palette[i % len(palette)] for i in range(len(component_labels))]
    bars = ax1.barh(component_labels, component_values, color=colors1,
                      edgecolor='#333333', linewidth=0.7)
    for bar, val in zip(bars, component_values):
        ax1.text(val + max(component_values)*0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=9.5)
    ax1.set_xlabel(breakdown_xlabel)
    ax1.set_title(breakdown_title)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, axis='x', alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.invert_yaxis()

    colors2 = [BLUE, RED][:len(compare_labels)]
    bars2 = ax2.bar(compare_labels, compare_values, color=colors2, width=0.5,
                      edgecolor='#333333', linewidth=0.8)
    for bar, val in zip(bars2, compare_values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + max(compare_values)*0.02,
                   f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
    ax2.set_ylabel(comparison_ylabel)
    ax2.set_title(comparison_title)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, axis='y', alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, max(compare_values) * 1.15)

    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Example usage (matches all figures already produced) ───────────────────

if __name__ == '__main__':

    # Example 1: KD training curve
    epochs = list(range(13, 21))
    plot_training_curve(
        epochs=epochs,
        train_series={
            'Train Total Loss': [0.9869,0.9691,0.9512,0.9217,0.9159,0.9102,0.8653,0.8428],
            'Task Loss':        [0.581,0.566,0.545,0.522,0.511,0.505,0.469,0.448],
            'KD Loss':          [0.405,0.403,0.406,0.400,0.405,0.406,0.397,0.395],
        },
        val_series={'Validation Loss': [1.1922,1.1464,1.1877,1.1735,1.1269,1.1862,1.1687,1.1681]},
        title='Knowledge Distillation Training Progress (Epochs 13\u201320)',
        best_epoch=17,
        filename='fig_kd_training_curve.png',
    )

    # Example 2: Baseline vs KD
    plot_bar_comparison(
        labels=['LiDAR-only\nBaseline', 'Camera-LiDAR\nFusion + KD'],
        values=[0.6612, 1.1269],
        title='Baseline vs. Knowledge-Distilled Model\n(Validation Loss, Lower is Better)',
        ylabel='Best Validation Loss',
        colors=[BLUE, ORANGE],
        filename='fig_baseline_vs_kd.png',
    )

    # Example 3: Pruning trade-off
    plot_tradeoff(
        x_values=[8.8, 17.5, 26.3],
        y_values=[0.828, 1.116, 1.172],
        point_labels=['20% target', '40% target', '60% target'],
        title='Structured Pruning: Validation Loss vs. Effective Sparsity',
        xlabel='Effective Sparsity (%)',
        ylabel='Validation Loss',
        reference_line=0.6622,
        reference_label='Unpruned baseline (0.6622)',
        filename='fig_pruning_tradeoff.png',
    )

    # Example 4: Per-class AP
    plot_categorical_bars(
        categories=['car','pedestrian','truck','bus','trailer',
                    'construction\nvehicle','motorcycle','bicycle',
                    'traffic\ncone','barrier'],
        values=[0.244,0.131,0,0,0,0,0,0,0,0],
        title='Per-Class Detection AP \u2014 Baseline Model (nuScenes mini-val)',
        ylabel='Average Precision (AP)',
        mean_line=0.0375,
        mean_label='mAP = 0.0375',
        filename='fig_map_per_class.png',
    )

    # Example 5: Parameter breakdown + scale comparison
    plot_breakdown_with_comparison(
        component_labels=['YOLO11\nBackbone','Camera-to-BEV\n(LSS)','PointPillars',
                           'Fusion\nModule','Detection\nHead'],
        component_values=[1.115744,1.494401,0.374848,1.346176,1.185810],
        compare_labels=['Student\n(this work)','BEVFusion\nTeacher'],
        compare_values=[5.5, 100],
        breakdown_title='Student Architecture\nParameter Breakdown (Total: 5.5M)',
        comparison_title='Scale Comparison\n(18\u00d7 reduction)',
        breakdown_xlabel='Parameters (Millions)',
        comparison_ylabel='Parameters (Millions)',
        filename='fig_parameter_breakdown.png',
    )

    print(f"All example figures saved to {OUTPUT_DIR.resolve()}/")
