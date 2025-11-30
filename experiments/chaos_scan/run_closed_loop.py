"""
run_closed_loop.py - The DynaAlign Validation Experiment.

[Scientific Goal]
Uses 'MiniPhoenixController' to stabilize the system.
Demonstrates the 'Latency Trap':
1. Low Latency (L=1) -> High Stability (Green)
2. High Latency (L=50) -> Control Failure (Purple)

This proves that Alignment is a control problem sensitive to feedback delays.
"""

import os
import sys
import numpy as np

# --- Path Setup (Fixed for ModuleNotFoundError) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from env.goodhart_env import GoodhartEnv
from experiments.chaos_scan.core import MiniPhoenixController
from experiments.chaos_scan.utils_plot import save_heatmap

def run_dynaalign_scan(L_values=[1, 15, 50], steps=300):
    lam_values = list(range(11))
    out_dir = os.path.join(ROOT_DIR, "experiments", "results", "chaos_scan")

    print(f"🔄 Starting Closed-Loop Mini-Phoenix Scan...")

    for L in L_values:
        print(f"   -> Testing Latency L={L}...")
        matrix = []

        for lam in lam_values:
            # 1. Environment Setup
            # We fix internal environment smoothing to 5 to isolate the
            # Controller Transport Delay effect defined by 'L'.
            env = GoodhartEnv(L=5, lam=lam)
            obs = env.reset()
            if isinstance(obs, tuple): obs = obs[0]

            # 2. Controller Setup (The Mini-Phoenix)
            # We pass 'L' here to simulate the Transport Delay in perception.
            # Gain=3.0 is aggressive enough to show oscillation at high L.
            controller = MiniPhoenixController(L=L, gain=3.0)

            row = []
            base_ramp = np.linspace(0.1, 1.5, steps) # User Intent (Pressure)

            for t in range(steps):
                # A. Get Real-time Metrics from Environment
                # aar = Average Absolute Roll (Instability Metric)
                _, aar, _ = env.compute_step_metrics()

                # B. Controller Decision (With internal delay simulation)
                # Calculates penalty based on delayed perception of instability
                correction = controller.get_correction(aar)

                # C. Final Action Calculation
                # Action = User_Intent - Controller_Correction
                intent = base_ramp[t]
                action_val = np.clip(intent - correction, 0.0, 2.0)

                # D. Execute Env Step
                action = np.array([action_val], dtype=np.float32)
                _, reward, done, _ = env.step(action)
                row.append(float(reward))

                if done:
                    env.reset()
                    controller.reset() # Important: Reset controller state

            matrix.append(row)

        # Narrative Titles for the Plots
        title_suffix = ""
        if L == 1: title_suffix = "(Ideal: Low Latency)"
        elif L == 15: title_suffix = "(Robust: Sweet Spot)"
        elif L == 50: title_suffix = "(Failure: High Latency)"

        save_heatmap(
            np.array(matrix),
            L,
            lam_values,
            title=f"DynaAlign Closed-Loop L={L}\n{title_suffix}",
            filename=f"heatmap_closed_L{L}.png",
            output_dir=out_dir
        )

if __name__ == "__main__":
    run_dynaalign_scan()