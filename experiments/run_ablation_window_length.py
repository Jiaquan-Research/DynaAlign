"""
Ablation Study: Window Length Robustness
----------------------------------------
Tests Phoenix's performance across different feedback delays (Window Lengths).
Proves the "Delay is a Filter" hypothesis:
- Too small L: Noise amplifies (Low survival)
- Optimal L: Filter works (High survival)
- Too large L: Lag kills (Low survival)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# [Path Fix]
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.env.goodhart_env import GoodhartEnv
    from src.controllers.phoenix_controller_v9_3 import PhoenixControllerV9_3
except ImportError:
    sys.exit(1)

# --- Config ---
OUTPUT_DIR = os.path.join(current_dir, "results", "ablation")
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_PATH = os.path.join(OUTPUT_DIR, "ablation_window_length.png")

LAM = 5.5  # Moderate penalty
GAIN = 0.5
SEEDS = 10

# Scanning Window Lengths
WINDOWS = [5, 10, 20, 30, 50, 70, 100]


def run_ablation():
    print(f"🚀 Starting Window Length Ablation (Real Physics)")
    survival_rates = []

    for w in tqdm(WINDOWS, desc="Scanning Windows"):
        alive_count = 0
        for s in range(SEEDS):
            # Dynamic L for Environment
            env = GoodhartEnv(L=w, lam=LAM, seed=2000 + s)
            ctrl = PhoenixControllerV9_3(ctrl_scale=GAIN)

            obs = env.reset()
            rewards = []
            for _ in range(200):
                g, m_e, m_a = env.compute_step_metrics()
                a = ctrl.update(g, m_e, m_a)
                _, r, d, _ = env.step(a)
                rewards.append(r)
                if d: break

            if len(rewards) > 20 and np.mean(rewards[-20:]) > 0.1:
                alive_count += 1

        survival_rates.append(alive_count / SEEDS)

    # --- Plotting ---
    plt.figure(figsize=(8, 5))
    plt.plot(WINDOWS, survival_rates, marker='s', linewidth=2, color='orange')
    plt.title(f"Phoenix Robustness: Window Length Sensitivity ($\lambda$={LAM})")
    plt.xlabel("Window Length L (Delay)")
    plt.ylabel("Survival Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)

    for x, y in zip(WINDOWS, survival_rates):
        plt.text(x, y + 0.03, f"{y:.0%}", ha='center')

    plt.tight_layout()
    plt.savefig(IMG_PATH, dpi=300)
    print(f"✅ Ablation Plot saved: {IMG_PATH}")


if __name__ == "__main__":
    run_ablation()