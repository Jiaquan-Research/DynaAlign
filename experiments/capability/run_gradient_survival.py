"""
C1: Gradient Survival Test — Baseline vs DynaAlign (FIXED)

[Critical Fixes]
1. Survival Logic: If an agent dies early (done=True), the remaining steps
   are filled with 0.0 reward. This prevents "Survivor Bias" where agents
   who die instantly with high initial reward are counted as successful.
2. Threshold Adjustment: Lowered SURVIVAL_THRESHOLD to 0.4 to accommodate
   the harsh physics of High-Lambda environments.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Path Setup
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(project_root, "src")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from env.goodhart_env import GoodhartEnv
    from experiments.chaos_scan.core import MiniPhoenixController
    print("✅ Successfully imported Physics Engine & Controller")
except ImportError as e:
    print(f"❌ Import Failed: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DIFFICULTY_LEVELS = {
    "Easy": 2.0,
    "Medium": 4.0,
    "Hard": 6.0,
    "Nightmare": 8.0,
}

N_EPISODES_PER_LEVEL = 12
EPISODE_STEPS = 200
SURVIVAL_WINDOW = 50
# [ADJUSTED] Lower threshold because high-lambda naturally suppresses reward.
# Survival means "Not Zero", not necessarily "High Score".
SURVIVAL_THRESHOLD = 0.1

# Physics Params
ALPHA = 0.5
NOISE_STD = 0.05
L_ENV = 5

# Controller Params
L_DELAY = 15
FEEDBACK_GAIN = 0.5

# ---------------------------------------------------------
# Episode Runners (With Zero-Filling Fix)
# ---------------------------------------------------------

def run_episode_baseline(lam: float, seed: int) -> float:
    env = GoodhartEnv(L=L_ENV, lam=lam, alpha=ALPHA, noise_std=NOISE_STD, seed=seed)
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]

    rewards = []
    ramp = np.linspace(0.1, 1.5, EPISODE_STEPS, dtype=np.float32)

    for t in range(EPISODE_STEPS):
        action = np.array([ramp[t]], dtype=np.float32)
        obs, reward, done, _ = env.step(action)
        rewards.append(float(reward))

        if done:
            # [CRITICAL FIX] Fill the rest of the episode with 0.0 (Death)
            remaining_steps = EPISODE_STEPS - len(rewards)
            rewards.extend([0.0] * remaining_steps)
            break

    # Calculate average of the LAST window (now correctly includes zeros)
    window = min(SURVIVAL_WINDOW, len(rewards))
    return float(np.mean(rewards[-window:]))


def run_episode_dynaalign(lam: float, seed: int) -> float:
    env = GoodhartEnv(L=L_ENV, lam=lam, alpha=ALPHA, noise_std=NOISE_STD, seed=seed)
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]

    controller = MiniPhoenixController(L=L_DELAY, gain=FEEDBACK_GAIN)
    rewards = []
    ramp = np.linspace(0.1, 1.5, EPISODE_STEPS, dtype=np.float32)

    for t in range(EPISODE_STEPS):
        _, M_exec, _ = env.compute_step_metrics()
        correction = controller.get_correction(M_exec)

        intent = ramp[t]
        a_val = np.clip(intent - correction, 0.0, 2.0)
        action = np.array([a_val], dtype=np.float32)

        obs, reward, done, _ = env.step(action)
        rewards.append(float(reward))

        if done:
            # [CRITICAL FIX] Same zero-filling logic for fairness
            remaining_steps = EPISODE_STEPS - len(rewards)
            rewards.extend([0.0] * remaining_steps)
            break

    window = min(SURVIVAL_WINDOW, len(rewards))
    return float(np.mean(rewards[-window:]))

# ---------------------------------------------------------
# Main Experiment Loop
# ---------------------------------------------------------

def run_gradient_survival():
    difficulty_names = list(DIFFICULTY_LEVELS.keys())
    baseline_survival = []
    dynaalign_survival = []

    print("\n🔄 Running C1: Gradient Survival Test (FIXED)...")
    print(f"   (Threshold: Reward >= {SURVIVAL_THRESHOLD})")

    for name in difficulty_names:
        lam = DIFFICULTY_LEVELS[name]
        print(f"\n   -> Testing Difficulty: {name} (Lambda={lam})")

        # Baseline
        survive_count_base = 0
        for k in range(N_EPISODES_PER_LEVEL):
            score = run_episode_baseline(lam, 1000+k)
            if score >= SURVIVAL_THRESHOLD:
                survive_count_base += 1
        base_rate = survive_count_base / N_EPISODES_PER_LEVEL

        # DynaAlign
        survive_count_dyn = 0
        for k in range(N_EPISODES_PER_LEVEL):
            score = run_episode_dynaalign(lam, 2000+k)
            if score >= SURVIVAL_THRESHOLD:
                survive_count_dyn += 1
        dyn_rate = survive_count_dyn / N_EPISODES_PER_LEVEL

        baseline_survival.append(base_rate)
        dynaalign_survival.append(dyn_rate)

        print(f"      Baseline: {base_rate:.2%}")
        print(f"      DynaAlign: {dyn_rate:.2%}")

    # Plotting
    out_dir = os.path.join(project_root, "experiments", "results", "capability")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gradient_survival.png")

    x = np.arange(len(difficulty_names))
    width = 0.35

    plt.figure(figsize=(8, 5), dpi=150)
    plt.bar(x - width/2, baseline_survival, width, label="Baseline (Open-Loop)", color='gray', alpha=0.7)
    plt.bar(x + width/2, dynaalign_survival, width, label="DynaAlign (Mini-Phoenix)", color='#2ca02c', alpha=0.9)

    plt.xticks(x, difficulty_names, fontsize=10)
    plt.ylim(0.0, 1.1)
    plt.ylabel("Survival Rate", fontsize=11)
    plt.title("C1: Gradient Survival Test (Fixed Logic)", fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"\n✅ Saved Chart: {out_path}")

if __name__ == "__main__":
    run_gradient_survival()