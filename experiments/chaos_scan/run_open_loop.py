"""
run_open_loop.py (BASELINE) - FIXED

[Open-Loop Scan]
Tests the Goodhart Physics Engine without any feedback controller.
This acts as the "Control Group" for the experiment.

Fixes: Added 'done' check to prevent IndexError when steps > EPISODE_LEN.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- Path Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from env.goodhart_env import GoodhartEnv
from experiments.chaos_scan.utils_plot import save_heatmap

def run_baseline(L_values=[1, 50], steps=300):
    # Scan Lambda from 0 to 10
    lam_values = list(range(11))
    out_dir = os.path.join(ROOT_DIR, "experiments", "results", "chaos_scan")

    # Ramp Action: 0.1 -> 1.5
    action_seq = np.linspace(0.1, 1.5, steps)

    print(f"📉 Starting Open-Loop Baseline Scan...")

    for L in L_values:
        print(f"   -> Running Baseline L={L}...")
        matrix = []
        for lam in lam_values:
            # Re-init env
            env = GoodhartEnv(L=L, lam=lam)
            obs = env.reset()
            if isinstance(obs, tuple): obs = obs[0]

            row = []
            for t in range(steps):
                # Blind Action (Open Loop)
                action = np.array([action_seq[t]], dtype=np.float32)

                # --- FIX START: Handle Environment Reset ---
                # Check done to prevent IndexError when t > EPISODE_LEN (200)
                _, reward, done, _ = env.step(action)
                row.append(float(reward))

                if done:
                    # Reset internal state but KEEP iterating 't'
                    # to maintain the high-pressure ramp simulation
                    env.reset()
                # --- FIX END ---

            matrix.append(row)

        # Plot
        save_heatmap(
            np.array(matrix),
            L,
            lam_values,
            title=f"Baseline: Uncontrolled Collapse (L={L})",
            filename=f"heatmap_open_L{L}.png",
            output_dir=out_dir
        )

if __name__ == "__main__":
    run_baseline()