"""
B2: Alpha Nonlinearity Scan — V9.3 Goodhart Engine
--------------------------------------------------
This experiment sweeps across different α values to
analyze how reward nonlinearity induces Goodhart
instability and collapse.

Physics Logic:
- Low α  -> smooth reward -> stable dynamics
- Mid α  -> nonlinear drift -> rising instability
- High α -> explosive reward -> strong collapse behavior

This script matches GoodhartEnv (V9.3) API 100% correctly.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------- Path Setup ----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = CURRENT_DIR
while "src" not in os.listdir(ROOT_DIR):
    parent = os.path.dirname(ROOT_DIR)
    if parent == ROOT_DIR:
        break
    ROOT_DIR = parent

SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from env.goodhart_env import GoodhartEnv

# ---------- Core Experiment ----------
def run_alpha_scan(
    alpha_values=None,
    lam: float = 2.0,
    L: int = 15,
    n_steps: int = 200,
    seed: int = 42,
):
    """
    Sweep α, hold λ constant.
    """

    if alpha_values is None:
        alpha_values = np.linspace(0.5, 4.0, 15)

    alpha_values = np.asarray(alpha_values)

    r_ss_list     = []
    mexec_ss_list = []
    maudit_ss_list = []

    # Ramp pressure schedule (same as B1)
    action_ramp = np.linspace(0.1, 1.5, n_steps, dtype=np.float32)

    print("🔄 Running B2 Alpha Scan ...")

    for alpha in alpha_values:
        env = GoodhartEnv(L=L, lam=lam, seed=seed, alpha=alpha)
        obs = env.reset()
        if isinstance(obs, tuple): obs = obs[0]

        r_hist = []
        mexec_hist = []
        maudit_hist = []

        for t in range(n_steps):
            a = np.array([action_ramp[t]], dtype=np.float32)
            obs, reward, done, _ = env.step(a)

            # Correct metric unpacking
            g_t, M_exec, M_audit = env.compute_step_metrics()

            r_hist.append(float(reward))
            mexec_hist.append(float(M_exec))
            maudit_hist.append(float(M_Audit := M_audit))

            if done:
                break

        # Steady state (last 50 steps)
        w = min(50, len(r_hist))
        r_ss_list.append(np.mean(r_hist[-w:]))
        mexec_ss_list.append(np.mean(mexec_hist[-w:]))
        maudit_ss_list.append(np.mean(maudit_hist[-w:]))

    return (
        alpha_values,
        np.array(r_ss_list),
        np.array(mexec_ss_list),
        np.array(maudit_ss_list),
    )


# ---------- Plotting ----------
def plot_alpha_phase_curve(
    alpha_values,
    r_ss,
    mexec_ss,
    maudit_ss,
    out_path: str,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=150)

    # Blue: Reward
    plt.plot(alpha_values, r_ss, label="Reward Magnitude (r_mean)",
             color='#1f77b4', linewidth=2.5, marker='o')

    # Orange: Instability
    plt.plot(alpha_values, mexec_ss, label="Instability (M_exec)",
             color='#ff7f0e', linewidth=2, linestyle='--')

    # Green: Skew
    plt.plot(alpha_values, maudit_ss, label="Audit Skew (M_audit)",
             color='#2ca02c', linewidth=2, linestyle=':')

    # Phase transition annotation
    plt.axvspan(1.8, 2.2, color='gray', alpha=0.1,
                label="Phase Transition (α≈2)")

    plt.xlabel("Reward Nonlinearity (Alpha)")
    plt.ylabel("Steady-State Metric Value")
    plt.title("B2: Alpha Nonlinearity Phase Curve (Physics Scan)",
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path)
    plt.close()

    print(f"✅ Saved B2 Curve to: {out_path}")


# ---------- Entry Point ----------
if __name__ == "__main__":
    out_dir = os.path.join(ROOT_DIR, "experiments", "results", "physics")
    out_path = os.path.join(out_dir, "alpha_phase_curve.png")

    alpha_vals, r_ss, mexec_ss, maudit_ss = run_alpha_scan()
    plot_alpha_phase_curve(alpha_vals, r_ss, mexec_ss, maudit_ss, out_path)
