"""
Run PPO Baseline Agent on GoodhartEnv (V9.3 Physics)
-----------------------------------------------------
This script reproduces the official PPO baseline curve
for the whitepaper and Phoenix comparison experiments.
"""

import numpy as np
import torch
import os
import sys
from tqdm import trange

import matplotlib.pyplot as plt

# ============================================================
# Auto-Mount Project Root for Imports
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================================
# Imports
# ============================================================
from src.env.goodhart_env import GoodhartEnv
from src.agents.ppo_agent import PPOAgent


# ============================================================
# Config
# ============================================================
EPISODES = 50
STEPS = 200
L = 30         # Default latency used in V9.3 baseline
LAM = 2.0      # Mild penalty so PPO can survive
SEED = 42

SAVE_DIR = os.path.join(current_dir, "results", "ppo_baseline")
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# MAIN LOGIC
# ============================================================
def run_baseline():
    env = GoodhartEnv(L=L, lam=LAM, seed=SEED)
    agent = PPOAgent()

    all_rewards = []

    for ep in trange(EPISODES, desc="Training PPO Baseline"):
        obs = env.reset()

        rollouts = {
            "obs": [],
            "actions": [],
            "logp": [],
            "values": [],
            "rewards": [],
            "dones": [],
        }

        ep_rewards = []

        for t in range(STEPS):
            action, logp, value = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)

            rollouts["obs"].append(obs)
            rollouts["actions"].append([action])
            rollouts["logp"].append(logp)
            rollouts["values"].append(value)
            rollouts["rewards"].append(reward)
            rollouts["dones"].append(float(done))

            obs = next_obs
            ep_rewards.append(reward)

            if done:
                break

        all_rewards.append(np.mean(ep_rewards))
        agent.update(rollouts)

    # ============================================================
    # Save and Plot
    # ============================================================
    np.save(os.path.join(SAVE_DIR, "baseline_rewards.npy"), np.array(all_rewards))

    plt.figure(figsize=(8, 5))
    plt.plot(all_rewards, label="PPO Baseline Reward")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("PPO Baseline Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "baseline_curve.png"), dpi=300)

    print("Done!")
    print(f"Rewards saved to: {SAVE_DIR}")
    print(f"Plot saved to: {SAVE_DIR}/baseline_curve.png")


if __name__ == "__main__":
    run_baseline()
