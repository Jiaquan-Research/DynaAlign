# experiments/make_latency_strip.py

import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT_DIR = get_project_root()

# TODO: 如果你的 heatmap 不在这个目录，就改这里
IMAGE_DIR = os.path.join(ROOT_DIR, "experiments", "results", "chaos_scan")

# 文件名按你的上传习惯设置；如果不同就改这张表
HEATMAP_FILES: List[Tuple[str, str]] = [
    ("L = 1", "heatmap_L1.png"),
    ("L = 2", "heatmap_L2.png"),
    ("L = 5", "heatmap_L5.png"),
    ("L = 10", "heatmap_L10.png"),
    ("L = 50", "heatmap_L50.png"),
]


def make_latency_strip():
    images = []
    labels = []

    for label, fname in HEATMAP_FILES:
        path = os.path.join(IMAGE_DIR, fname)
        if not os.path.exists(path):
            print(f"⚠️ Missing heatmap file: {path}")
            continue
        img = mpimg.imread(path)
        images.append(img)
        labels.append(label)

    if not images:
        print("❌ No heatmap images found. 请确认 IMAGE_DIR 和文件名配置正确。")
        return

    n = len(images)
    # 宽图像：每个 heatmap 给 4 宽度单位（可按审美调整）
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(4 * n, 3),
        squeeze=False,
    )
    axes = axes[0]

    for i, (img, label) in enumerate(zip(images, labels)):
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        "Latency Robustness Strip (Heatmaps across Delay Window L)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_dir = os.path.join(IMAGE_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "latency_robustness_strip.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Saved latency strip to: {out_path}")


if __name__ == "__main__":
    make_latency_strip()
