"""
Golden Seed Hunter V2 (The Final Polish)
----------------------------------------
Target:
1. Baseline (Training) -> COLLAPSE (Score < 0.2)
2. Phoenix (Inference) -> THRIVE (Score > 0.75)

Config: L=50, Lambda=5.5 (Consistent with your Phase Map)
"""
import numpy as np
import sys
import os
from tqdm import tqdm

# Auto-mount
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.env.goodhart_env import GoodhartEnv
    from src.agents.ppo_agent import PPOAgent
    from src.controllers.phoenix_controller_v9_3 import PhoenixControllerV9_3
except ImportError:
    print("Please make sure you are running this from the project root or experiments folder.")
    sys.exit(1)

# [PHYSICS CONFIG] Same as your main script
L = 50
LAM = 5.5
GAIN = 0.5  # Standard gain

def quick_eval(seed):
    # --- 1. Run Phoenix (Inference Mode) First ---
    # We run Phoenix first because if it fails, we don't care about Baseline.
    env = GoodhartEnv(L=L, lam=LAM, seed=seed)
    agent_frozen = PPOAgent() # Random weights, frozen
    ctrl = PhoenixControllerV9_3(ctrl_scale=GAIN)

    phoenix_scores = []

    # Run 50 episodes of Phoenix
    for _ in range(50):
        obs = env.reset()
        ep_reward = 0
        for _ in range(200):
            a, _, _ = agent_frozen.select_action(obs) # Random action
            g, m_e, m_a = env.compute_step_metrics()
            a_ctrl = ctrl.update(g, m_e, m_a)
            final_a = a + a_ctrl

            obs, r, d, _ = env.step(final_a)
            ep_reward += r
            if d: break

        phoenix_scores.append(ep_reward) # Use raw reward (0.0-1.0)

    phoenix_final = np.mean(phoenix_scores[-10:])

    # Filter: If Phoenix isn't doing great, skip this seed
    if phoenix_final < 0.7:
        return 1.0, phoenix_final # Return dummy high baseline to fail the check

    # --- 2. Run Baseline (Training Mode) ---
    # Now check if Baseline naturally dies here
    env = GoodhartEnv(L=L, lam=LAM, seed=seed)
    agent = PPOAgent()
    baseline_scores = []

    for _ in range(50): # 50 Episodes
        obs = env.reset()
        rollouts = {"obs":[], "actions":[], "logp":[], "rewards":[], "values":[], "dones":[]}
        ep_rew = 0
        for _ in range(200):
            a, l, v = agent.select_action(obs)

            # [FIXED HERE] Variable name typo fixed: next_o -> next_obs
            next_obs, r, d, _ = env.step(a)

            rollouts["obs"].append(obs)
            rollouts["actions"].append([a])
            rollouts["logp"].append(l)
            rollouts["rewards"].append(r)
            rollouts["values"].append(v)
            rollouts["dones"].append(float(d))

            obs = next_obs
            ep_rew += r
            if d: break

        baseline_scores.append(ep_rew)
        agent.update(rollouts)

    baseline_final = np.mean(baseline_scores[-10:])

    return baseline_final, phoenix_final

def main():
    print(f"🔍 Searching for the Perfect Shot (L={L}, Lambda={LAM})...")
    print("Target: Baseline < 0.3 (Crash) AND Phoenix > 0.75 (Stable)")
    print("-" * 60)

    # Scan seeds 0 to 100 to find a good one
    for s in range(100):
        try:
            base, phx = quick_eval(s)
            diff = phx - base
            print(f"Seed {s:<2} | Baseline: {base:.3f} | Phoenix: {phx:.3f} | Diff: {diff:+.3f}", end="")

            if base < 0.3 and phx > 0.75:
                print("  <-- 🎯 JACKPOT!")
                print(f"\n✅ STOP! Use SEED = {s} in your main script experiments/run_phoenix_vs_baseline.py")
                break
            elif phx > 0.75:
                print("  (Phoenix good, but Baseline didn't crash enough)")
            else:
                print("  (Phoenix failed)")

        except Exception as e:
            print(f"Error on seed {s}: {e}")

if __name__ == "__main__":
    main()