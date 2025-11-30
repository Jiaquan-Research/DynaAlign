"""
make_narrative_chart.py

Assembles the "Cybernetic Trinity" narrative chart for README/Whitepaper.
Combines:
1. Open-Loop Baseline (Chaos)
2. Closed-Loop L=50 (Failure)
3. Closed-Loop L=1 (Success)

Output: narrative_trinity_chart.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def make_chart():
    # Fix paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    img_dir = os.path.join(root_dir, "experiments", "results", "chaos_scan")

    # Files to assemble (The Narrative Arc)
    files = [
        ("heatmap_open_L1.png", "1. Baseline: Uncontrolled Chaos"),
        ("heatmap_closed_L50.png", "2. High Latency (L=50): Control Failure"),
        ("heatmap_closed_L1.png", "3. DynaAlign (L=1): Stabilization")
    ]

    images = []
    print(f"🖼️  Assembling Narrative Chart from {img_dir}...")

    for fname, title in files:
        path = os.path.join(img_dir, fname)
        if os.path.exists(path):
            images.append((mpimg.imread(path), title))
        else:
            print(f"⚠️ Missing file: {fname}. Please run open/closed loop scans first.")
            return

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (img, title) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(img_dir, "narrative_trinity_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Narrative Chart Created: {out_path}")


if __name__ == "__main__":
    make_chart()