"""
A1: Collapse Mechanism Curve
----------------------------------

This script reproduces the core mechanism figure of the V9.3 Goodhart engine:
    - r_mean(t)
    - M_exec(t)
    - M_audit(t)

It uses the latest GoodhartEnv (uploaded by user), which includes:
    • 50-dim proxy metric vector r
    • sliding window L
    • nonlinear collapse penalty for lambda > 5
    • noisy sampling + proxy subsampling
    • ensemble median statistics

Output:
    - mechanism_curve.png
Saved to:
    experiments/results/physics/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Path setup
# ----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))  # go up to project root
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from env.goodhart_env import GoodhartEnv


# -------------------------------------------------------------
# Run a single episode and record r_mean, M_exec, M_audit
# -------------------------------------------------------------
def run_mechanism_episode(
        L=50,
        lam=6.0,
        alpha=0.05,
        noise_std=0.05,
        steps=500
):
    """
    Run one full trajectory under rising optimization pressure.
    We slowly ramp the action from 0.1 to 1.5 to induce collapse.
    """
    env = GoodhartEnv(L=L, lam=lam, alpha=alpha, noise_std=noise_std)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    # Storage
    r_means = []
    exec_list = []
    audit_list = []
    actions = []

    # Ramp action from 0.1 → 1.5
    action_seq = np.linspace(0.1, 1.5, steps)

    for t in range(steps):
        a = np.array([action_seq[t]], dtype=np.float32)

        # Step environment
        obs, reward, done, info = env.step(a)

        # Metrics from GoodhartEnv
        g_t, aar, skew = env.compute_step_metrics()

        r_means.append(env.r_current.mean())   # r_mean(t)
        exec_list.append(aar)                  # M_exec
        audit_list.append(skew)                # M_audit
        actions.append(a[0])

        if done:
            break

    return {
        "r_mean": np.array(r_means),
        "exec": np.array(exec_list),
        "audit": np.array(audit_list),
        "action": np.array(actions)
    }


# -------------------------------------------------------------
# Plot mechanism curve
# -------------------------------------------------------------
def plot_mechanism_curve(data, out_path):

    t = np.arange(len(data["r_mean"]))

    plt.figure(figsize=(10, 6), dpi=150)

    # r_mean(t)
    plt.plot(t, data["r_mean"], label="r_mean (proxy value)", linewidth=2)

    # M_exec(t)
    plt.plot(t, data["exec"], label="M_exec (instability AAR)", linewidth=2)

    # M_audit(t)
    plt.plot(t, data["audit"], label="M_audit (skew / bias)", linewidth=2)

    plt.xlabel("Time Step")
    plt.ylabel("Magnitude")
    plt.title("Collapse Mechanism – V9.3 Goodhart Engine", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path)
    plt.close()

    print(f"✅ Saved mechanism plot: {out_path}")


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    OUT_DIR = os.path.join(ROOT_DIR, "experiments", "results", "physics")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Running Collapse Mechanism Simulation...")

    data = run_mechanism_episode(
        L=50,       # default consistent with env
        lam=6.0,    # in the collapse zone
        alpha=0.05,
        noise_std=0.05,
        steps=600
    )

    out_path = os.path.join(OUT_DIR, "mechanism_curve.png")
    plot_mechanism_curve(data, out_path)


if __name__ == "__main__":
    main()
