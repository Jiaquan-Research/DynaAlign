import numpy as np
import matplotlib.pyplot as plt
from src.env.goodhart_env import GoodhartEnv
from src.controllers.phoenix_controller_v9_3 import PhoenixController


def run_episode(env, controller, max_steps=200):
    """Runs a single episode and returns final 'alive' state (1=alive, 0=collapse)."""
    obs = env.reset()
    controller.reset()

    for _ in range(max_steps):
        action = 0  # PPO baseline removed; we only care about environment stability
        corrected_action = controller.step(obs, action)
        obs, reward, done, info = env.step(corrected_action)

        if done:
            return 0
    return 1


def chaos_scan(latency_values, seeds_per_point=5):
    """Scans L across a range and measures collapse probability."""
    results = []

    for L in latency_values:
        env = GoodhartEnv(latency=L)
        controller = PhoenixController()

        alive_count = 0
        for seed in range(seeds_per_point):
            np.random.seed(seed)
            alive_count += run_episode(env, controller)

        survival_rate = alive_count / seeds_per_point
        results.append(survival_rate)
        print(f"[L={L}] Survival Rate: {survival_rate:.2f}")

    return np.array(results)


if __name__ == "__main__":
    latency_range = np.arange(1, 40, 1)
    results = chaos_scan(latency_range, seeds_per_point=5)

    plt.figure(figsize=(10, 5))
    plt.plot(latency_range, results, label="Phoenix Survival", color="green")
    plt.axvline(20, linestyle="--", color="red", label="Bifurcation (~20)")
    plt.xlabel("Feedback Latency L")
    plt.ylabel("Survival Probability")
    plt.title("Chaos Scan (Death Abyss) - Phoenix Stability")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("chaos_scan_output.png", dpi=300)
    plt.show()
