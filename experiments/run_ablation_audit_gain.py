"""
Ablation Study: Audit Gain Sensitivity
--------------------------------------
Tests Phoenix's robustness across different control strengths (Gains).
Demonstrates the "Inverted-U" or "Saturation" curve of control efficacy.

Physics: Real GoodhartEnv (L=30, Lambda=5.5) - A challenging but survivable zone.
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
except ImportError as e:
    print(f"❌ Dependency Error: {e}")
    sys.exit(1)

# --- Config ---
OUTPUT_DIR = os.path.join(current_dir, "results", "ablation")
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_PATH = os.path.join(OUTPUT_DIR, "ablation_audit_gain.png")

# Use the "Green Peninsula" coordinates where control matters
L = 30
LAM = 6.0
SEEDS = 10
EPISODES = 50

# Scanning Gain from 0.0 (No Control) to 2.0 (Aggressive)
GAINS = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]


def run_ablation():
    print(f"🚀 Starting Audit Gain Ablation (Real Physics)")
    print(f"   L={L}, Lambda={LAM}")

    survival_rates = []

    for g in tqdm(GAINS, desc="Scanning Gains"):
        alive_count = 0

        for s in range(SEEDS):
            # Using seeds 100+ to avoid overlap with other experiments
            env = GoodhartEnv(L=L, lam=LAM, seed=1000 + s)
            ctrl = PhoenixControllerV9_3(ctrl_scale=g)

            obs = env.reset()
            rewards = []

            for _ in range(EPISODES * 5):  # Run slightly longer
                g_t, m_e, m_a = env.compute_step_metrics()
                a = ctrl.update(g_t, m_e, m_a)

                # Baseline action = 0, purely testing controller capability
                _, r, done, _ = env.step(a)
                rewards.append(r)

                if done: break

            # Survival Check
            # If average of last 20 steps > 0.1, it survived
            if len(rewards) > 20 and np.mean(rewards[-20:]) > 0.1:
                alive_count += 1

        survival_rates.append(alive_count / SEEDS)

    # --- Plotting ---
    plt.figure(figsize=(8, 5))
    plt.plot(GAINS, survival_rates, marker='o', linewidth=2, color='purple')
    plt.title(f"Phoenix Robustness: Audit Gain Sensitivity (L={L}, $\lambda$={LAM})")
    plt.xlabel("Control Gain (ctrl_scale)")
    plt.ylabel("Survival Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)

    # Annotate
    for x, y in zip(GAINS, survival_rates):
        plt.text(x, y + 0.03, f"{y:.0%}", ha='center')

    plt.tight_layout()
    plt.savefig(IMG_PATH, dpi=300)
    print(f"✅ Ablation Plot saved: {IMG_PATH}")


if __name__ == "__main__":
    run_ablation()