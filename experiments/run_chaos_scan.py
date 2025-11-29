"""
run_chaos_scan.py
------------------------------------------
Generates the Phase Map / Death Abyss Heatmap
Scanning:
    - Feedback Delay L (5 → 60)
    - Regulation Penalty lambda (0 → 10)

Output:
    - physics_death_abyss.png
    - chaos_scan_data.npz
------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.env.goodhart_env import GoodhartEnv
from src.controllers.phoenix_controller_v9_3 import PhoenixControllerV9_3


# ============================================================
# Configuration
# ============================================================

L_values = np.arange(5, 65, 5)          # Delay steps
lam_values = np.arange(0, 11, 1)        # Regulation penalty

EPISODES = 1                             # Deterministic environment scan
STEPS = 200                              # Environment steps per episode

RESULTS = np.zeros((len(L_values), len(lam_values)))


# ============================================================
# Run Phase Scan
# ============================================================

for i, L in enumerate(tqdm(L_values, desc="Scanning L")):
    for j, lam in enumerate(lam_values):

        # Create env with scanned parameters
        env = GoodhartEnv(L=L, lam=lam, seed=0)

        # Controller does not directly intervene in scan
        ctrl = PhoenixControllerV9_3()

        obs = env.reset()
        mean_trace = []

        for _ in range(STEPS):
            # No PPO policy needed: scanning pure dynamics
            g_t, Mexec, Maud = env.compute_step_metrics()

            a_ctrl = ctrl.update(g_t, Mexec, Maud)     # Controller tracks stability
            a_total = a_ctrl                           # No agent action

            obs, reward, done, info = env.step(a_total)
            mean_trace.append(reward)

        # Average final 10% of rewards to classify collapse
        tail_mean = np.mean(mean_trace[-20:])
        RESULTS[i, j] = tail_mean


# ============================================================
# Save Raw Data
# ============================================================
np.savez("experiments/results/chaos_scan/chaos_scan_data.npz",
         L_values=L_values,
         lam_values=lam_values,
         results=RESULTS)


# ============================================================
# Plot Heatmap
# ============================================================
plt.figure(figsize=(10, 7))
plt.imshow(RESULTS,
           origin="lower",
           aspect="auto",
           cmap="inferno",
           extent=[lam_values.min(), lam_values.max(),
                   L_values.min(), L_values.max()])

plt.colorbar(label="Final Mean Reward")
plt.title("DynaAlign Phase Map (Death Abyss)")
plt.xlabel("Regulation Penalty λ")
plt.ylabel("Delay L")

plt.savefig("experiments/results/chaos_scan/physics_death_abyss_reproduced.png",
            dpi=300)
plt.close()

print("Chaos scan complete.")
print("Saved: chaos_scan_data.npz and physics_death_abyss_reproduced.png.")
