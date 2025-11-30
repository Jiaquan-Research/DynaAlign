"""
C2: Legion Stability Test — Multi-Seed Robustness

Goal:
    Under a fixed high-pressure setting (e.g. "Nightmare" difficulty),
    run many random seeds and compare the distribution of
    steady-state rewards between:

        - Baseline (Open-Loop)
        - DynaAlign (Mini-Phoenix)

Key metrics:
    - For each seed, compute the average reward over the last W steps.
    - Plot the distribution (boxplot) for both agents.

Notes:
    - Survival threshold is kept at 0.1 (same as C1), but in most cases
      both agents will "survive". C2 focuses on distributional stability,
      not survival rate per se.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Path setup
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from env.goodhart_env import GoodhartEnv
    from experiments.chaos_scan.core import MiniPhoenixController
    print("✅ Successfully imported Physics Engine & Controller")
except ImportError as e:
    print(f"❌ Import Failed: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

# Fixed high-pressure setting (Nightmare-style)
LAMBDA_NIGHTMARE = 8.0

# Physics params (keep与 C1 一致，保证可比)
ALPHA = 0.5
NOISE_STD = 0.05
L_ENV = 5

# Controller params (DynaAlign)
L_DELAY = 15
FEEDBACK_GAIN = 0.5

# Legion settings
N_SEEDS = 48          # 军团规模
EPISODE_STEPS = 200
WINDOW = 50           # 末尾窗口

SURVIVAL_THRESHOLD = 0.1   # 与 C1 一致，用于打印统计信息

# ---------------------------------------------------------
# Episode runners (沿用 C1 的“死亡补 0”逻辑)
# ---------------------------------------------------------

def run_episode_baseline(lam: float, seed: int) -> float:
    env = GoodhartEnv(L=L_ENV, lam=lam, alpha=ALPHA,
                      noise_std=NOISE_STD, seed=seed)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    rewards = []
    ramp = np.linspace(0.1, 1.5, EPISODE_STEPS, dtype=np.float32)

    for t in range(EPISODE_STEPS):
        action = np.array([ramp[t]], dtype=np.float32)
        obs, reward, done, _ = env.step(action)
        rewards.append(float(reward))

        if done:
            remaining = EPISODE_STEPS - len(rewards)
            rewards.extend([0.0] * remaining)
            break

    window = min(WINDOW, len(rewards))
    return float(np.mean(rewards[-window:]))


def run_episode_dynaalign(lam: float, seed: int) -> float:
    env = GoodhartEnv(L=L_ENV, lam=lam, alpha=ALPHA,
                      noise_std=NOISE_STD, seed=seed)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

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
            remaining = EPISODE_STEPS - len(rewards)
            rewards.extend([0.0] * remaining)
            controller.reset()
            break

    window = min(WINDOW, len(rewards))
    return float(np.mean(rewards[-window:]))


# ---------------------------------------------------------
# Main Legion Experiment
# ---------------------------------------------------------

def run_legion_test():
    lam = LAMBDA_NIGHTMARE
    print(f"\n🔄 Running C2: Legion Stability Test at Lambda={lam} ...")

    baseline_scores = []
    dynaalign_scores = []

    survive_base = 0
    survive_dyn = 0

    for i in range(N_SEEDS):
        seed_base = 3000 + i
        seed_dyn = 4000 + i

        s_base = run_episode_baseline(lam, seed_base)
        s_dyn = run_episode_dynaalign(lam, seed_dyn)

        baseline_scores.append(s_base)
        dynaalign_scores.append(s_dyn)

        if s_base >= SURVIVAL_THRESHOLD:
            survive_base += 1
        if s_dyn >= SURVIVAL_THRESHOLD:
            survive_dyn += 1

    baseline_scores = np.array(baseline_scores)
    dynaalign_scores = np.array(dynaalign_scores)

    # 打印统计信息
    print("\n📊 Legion Results (Nightmare):")
    print(f"   Baseline  mean={baseline_scores.mean():.3f}, "
          f"std={baseline_scores.std():.3f}, "
          f"survival={survive_base}/{N_SEEDS}")
    print(f"   DynaAlign mean={dynaalign_scores.mean():.3f}, "
          f"std={dynaalign_scores.std():.3f}, "
          f"survival={survive_dyn}/{N_SEEDS}")

    # -------------------------------------------------
    # Plot: Boxplot / Distribution
    # -------------------------------------------------
    out_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "capability")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "legion_stability.png")

    plt.figure(figsize=(7, 5), dpi=150)

    data = [baseline_scores, dynaalign_scores]
    labels = ["Baseline (Open-Loop)", "DynaAlign (Mini-Phoenix)"]

    box = plt.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
    )

    # 简单配色
    colors = ["lightgray", "#98df8a"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    plt.ylabel(f"Avg Reward over last {WINDOW} steps")
    plt.title("C2: Legion Stability Test (Nightmare Difficulty)", fontsize=13, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"\n✅ Saved Legion Chart: {out_path}")


if __name__ == "__main__":
    run_legion_test()
