"""
B3: Epsilon Noise Phase Curve — V9.3 Goodhart Engine (FIXED)

This experiment scans different sensor noise levels (epsilon)
to test system robustness.

Physics Logic:
- Low Noise: System is confident but rigid.
- Mid Noise: Stochastic Resonance? (Potential stability gain).
- High Noise: Signal drowning, control failure.

Fixes applied:
- Corrected variable name M_Audit -> M_audit.
- Robust path setup.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# [CRITICAL] Robust Path Setup
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # Fallback
# Smarter root finder
temp_root = current_dir
while "src" not in os.listdir(temp_root):
    parent = os.path.dirname(temp_root)
    if parent == temp_root: break
    temp_root = parent
project_root = temp_root

src_dir = os.path.join(project_root, "src")

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from env.goodhart_env import GoodhartEnv
except ImportError:
    print("❌ Failed to import GoodhartEnv. Check paths.")
    sys.exit(1)


# ---------------------------------------------------------
# Core Experiment
# ---------------------------------------------------------
def run_epsilon_scan(
        eps_values=None,
        lam: float = 5.0,  # Near Phase Transition
        alpha: float = 2.0,  # Greedy Limit
        L: int = 15,  # Sweet Spot
        n_steps: int = 200,
        seed: int = 42,
):
    if eps_values is None:
        eps_values = np.linspace(0.0, 0.30, 15)

    eps_values = np.asarray(eps_values, dtype=float)

    r_ss_list = []
    mexec_ss_list = []
    maudit_ss_list = []

    # Ramp pressure
    base_ramp = np.linspace(0.1, 1.5, n_steps, dtype=np.float32)

    print(f"\n🔄 Running B3 Epsilon Scan...")

    for eps in eps_values:
        env = GoodhartEnv(
            L=L,
            lam=lam,
            alpha=alpha,
            noise_std=float(eps),
            seed=seed,
        )
        obs = env.reset()
        if isinstance(obs, tuple): obs = obs[0]

        r_hist = []
        mexec_hist = []
        maudit_hist = []

        for t in range(n_steps):
            a = np.array([base_ramp[t]], dtype=np.float32)
            obs, reward, done, _ = env.step(a)

            # Metrics
            g_t, M_exec, M_audit = env.compute_step_metrics()

            r_hist.append(float(reward))
            mexec_hist.append(float(M_exec))

            # [FIXED TYPO HERE]
            maudit_hist.append(float(M_audit))

            if done:
                break

        # Steady State (Last 50)
        window = min(50, len(r_hist))
        r_ss = np.mean(r_hist[-window:])
        mexec_ss = np.mean(mexec_hist[-window:])
        maudit_ss = np.mean(maudit_hist[-window:])

        r_ss_list.append(r_ss)
        mexec_ss_list.append(mexec_ss)
        maudit_ss_list.append(maudit_ss)

        print(f"   -> Eps={eps:.2f} | R={r_ss:.2f} | M_exec={mexec_ss:.2f}")

    return (
        eps_values,
        np.array(r_ss_list),
        np.array(mexec_ss_list),
        np.array(maudit_ss_list),
    )


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
def plot_epsilon_phase_curve(
        eps_values,
        r_ss,
        mexec_ss,
        maudit_ss,
        out_path: str,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=150)

    # Reward (Gold)
    plt.plot(eps_values, r_ss, label="Reward Magnitude (r_mean)",
             color='#ffbf00', linewidth=2.5, marker='o')

    # Instability (Blue)
    plt.plot(eps_values, mexec_ss, label="Instability (M_exec)",
             color='#1f77b4', linewidth=2, linestyle='--')

    # Skew (Green)
    plt.plot(eps_values, maudit_ss, label="Audit Skew (M_audit)",
             color='#2ca02c', linewidth=2, linestyle=':')

    plt.xlabel("Sensor Noise Std (Epsilon)")
    plt.ylabel("Steady-State Metric Value")
    plt.title("B3: Epsilon Noise Phase Curve (Robustness Check)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path)
    plt.close()

    print(f"✅ Saved B3 Curve to: {out_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    out_dir = os.path.join(project_root, "experiments", "results", "physics")
    out_path = os.path.join(out_dir, "epsilon_phase_curve.png")

    eps_vals, r_ss, mexec_ss, maudit_ss = run_epsilon_scan()
    plot_epsilon_phase_curve(eps_vals, r_ss, mexec_ss, maudit_ss, out_path)