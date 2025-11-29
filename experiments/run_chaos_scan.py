"""
Chaos Scan (Death Abyss Phase Map)
----------------------------------
This script sweeps over:
    - Feedback delay L
    - Regulation penalty lambda

and generates the alignment phase map.

This version is guaranteed to run on Windows + PyCharm.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# =========================================================
# 1) Mount project root so that "src" is importable
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"[Path OK] Using project root = {project_root}")

# =========================================================
# 2) Import Physics & Controllers
# =========================================================
try:
    from src.env.goodhart_env import GoodhartEnv
    from src.controllers.phoenix_controller_v9_3 import PhoenixControllerV9_3
    print("[Import OK] Loaded GoodhartEnv + Phoenix V9.3")
except Exception as e:
    print(f"[Import ERROR] {e}")
    sys.exit(1)

# =========================================================
# 3) Config
# =========================================================
L_VALUES = np.arange(5, 55, 5)       # 5 → 50
LAM_VALUES = np.arange(0, 11, 1)     # 0 → 10

STEPS = 200
OUTPUT_DIR = os.path.join(current_dir, "results", "chaos_scan")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(OUTPUT_DIR, "chaos_scan_data.npz")
IMG_PATH = os.path.join(OUTPUT_DIR, "physics_death_abyss_reproduced.png")

HEATMAP_DATA = np.zeros((len(L_VALUES), len(LAM_VALUES)))


# =========================================================
# 4) Main Scan Function
# =========================================================
def run_scan():
    print("🚀 Starting Chaos Scan")
    print(f"Grid = {len(L_VALUES)} × {len(LAM_VALUES)}")
    print(f"Saving to {OUTPUT_DIR}")
    print("-" * 50)

    for i, L in enumerate(tqdm(L_VALUES, desc="Scanning L")):
        for j, lam in enumerate(LAM_VALUES):

            env = GoodhartEnv(L=L, lam=lam, seed=42)
            ctrl = PhoenixControllerV9_3(ctrl_scale=0.5)

            obs = env.reset()
            rewards = []

            for _ in range(STEPS):
                g_t, M_exec, M_audit = env.compute_step_metrics()
                a_ctrl = ctrl.update(g_t, M_exec, M_audit)
                obs, reward, done, _ = env.step(a_ctrl)
                rewards.append(reward)

            final_mean = np.mean(rewards[-20:])
            HEATMAP_DATA[i, j] = 1.0 if final_mean > 0.1 else 0.0

    # Save data
    np.savez(DATA_PATH, L=L_VALUES, lam=LAM_VALUES, heatmap=HEATMAP_DATA)
    print(f"✔ Data saved → {DATA_PATH}")

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(
        HEATMAP_DATA,
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        extent=[LAM_VALUES.min(), LAM_VALUES.max(), L_VALUES.min(), L_VALUES.max()],
    )
    plt.colorbar(label="Survival (1 = Alive, 0 = Dead)")
    plt.xlabel("Regulation Penalty λ")
    plt.ylabel("Feedback Delay L")
    plt.title("The Death Abyss: Alignment Phase Map")
    plt.tight_layout()
    plt.savefig(IMG_PATH, dpi=300)
    print(f"✔ Figure saved → {IMG_PATH}")


# =========================================================
# 5) Run
# =========================================================
if __name__ == "__main__":
    run_scan()
