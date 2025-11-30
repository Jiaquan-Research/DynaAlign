"""
utils_plot.py - Unified Visualization Style for Chaos Scans.

Ensures all heatmaps (Open/Closed Loop) share the same:
- Colormap (Viridis)
- Scale (Vmin/Vmax) for fair comparison
- Axis labels and fonts
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def save_heatmap(matrix: np.ndarray,
                 L: int,
                 lam_values: list,
                 title: str,
                 filename: str,
                 output_dir: str):
    """
    Standardized Heatmap Plotter using Viridis colormap.

    Args:
        matrix: 2D numpy array [Lambda, Time]
        L: Latency parameter (for labeling)
        lam_values: List of lambda values used (Y-axis)
        title: Chart title
        filename: Output filename
        output_dir: Directory to save the image
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(10, 5), dpi=150)

    # Use fixed vmin/vmax to ensure visual comparability across L=1 and L=50.
    # vmin=0, vmax=1.8 covers the typical reward range of GoodhartEnv.
    # This prevents auto-scaling from hiding the deterioration in L=50.
    im = plt.imshow(
        matrix,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmin=0,
        vmax=1.8,
        interpolation='nearest'
    )

    plt.colorbar(im, label="Reward (G_sum)")
    plt.xlabel("Optimization Steps (Time/Pressure)")
    plt.ylabel("Regulation Strength (Lambda)")

    # Format Y-ticks to show Lambda values clearly
    # Subsample ticks if too many
    step = max(1, len(lam_values) // 5)
    plt.yticks(
        ticks=np.arange(len(lam_values))[::step],
        labels=[str(l) for l in lam_values][::step]
    )

    plt.title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    print(f"✅ [Plot] Saved: {out_path}")