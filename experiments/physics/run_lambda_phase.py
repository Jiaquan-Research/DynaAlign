"""
B1: Lambda Phase Curve — V9.3 Goodhart Engine (FIXED PATHS)

This experiment scans different regulation strengths (lambda)
to show the "Phase Transition" of the system.

Path Fix: Explicitly adds project root and src to sys.path
to ensure 'env' module is found.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# [CRITICAL FIX] Robust Path Setup
# ---------------------------------------------------------
# 1. Get the directory of this script: .../DynaAlign/experiments/physics
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up TWO levels to get Project Root: .../DynaAlign
project_root = os.path.dirname(os.path.dirname(current_dir))

# 3. Define Source Directory: .../DynaAlign/src
src_dir = os.path.join(project_root, "src")

# 4. Add to sys.path (Insert at 0 to prioritize)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"🔧 Path Debug:")
print(f"   Root: {project_root}")
print(f"   Src:  {src_dir}")

# ---------------------------------------------------------
# Imports (Now safe)
# ---------------------------------------------------------
try:
    from env.goodhart_env import GoodhartEnv
    print("✅ Successfully imported GoodhartEnv")
except ImportError as e:
    print(f"❌ Import Failed: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# Core Experiment
# ---------------------------------------------------------
def run_lambda_scan(
    lam_values=None,
    L: int = 15, # Engineering Sweet Spot
    n_steps: int = 200,
    seed: int = 42,
):
    if lam_values is None:
        lam_values = np.arange(0, 11, 1)

    lam_values = np.asarray(lam_values, dtype=float)

    r_ss_list = []     # Steady-state Reward
    mexec_ss_list = [] # Steady-state Instability
    maudit_ss_list = [] # Steady-state Skew

    # Ramp schedule: 0.1 -> 1.5
    action_ramp = np.linspace(0.1, 1.5, n_steps, dtype=np.float32)

    print(f"\n🔄 Running B1 Lambda Scan (L={L})...")

    for lam in lam_values:
        env = GoodhartEnv(L=L, lam=float(lam), seed=seed)
        obs = env.reset()
        if isinstance(obs, tuple): obs = obs[0]

        r_hist = []
        mexec_hist = []
        maudit_hist = []

        for t in range(n_steps):
            a = np.array([action_ramp[t]], dtype=np.float32)
            obs, reward, done, _ = env.step(a)

            # [CRITICAL API] Correctly unpack metrics
            # g_t is drift, M_exec is instability, M_audit is skew
            g_t, M_exec, M_audit = env.compute_step_metrics()

            # Record metrics
            r_hist.append(float(reward))
            mexec_hist.append(float(M_exec))
            maudit_hist.append(float(M_audit))

            if done:
                break

        # Calculate Steady State (Average of last 50 steps)
        window = min(50, len(r_hist))
        r_ss = float(np.mean(r_hist[-window:]))
        mexec_ss = float(np.mean(mexec_hist[-window:]))
        maudit_ss = float(np.mean(maudit_hist[-window:]))

        r_ss_list.append(r_ss)
        mexec_ss_list.append(mexec_ss)
        maudit_ss_list.append(maudit_ss)

        print(f"   -> Lambda={lam:.1f} | Reward={r_ss:.2f} | Instability={mexec_ss:.2f}")

    return (
        lam_values,
        np.array(r_ss_list),
        np.array(mexec_ss_list),
        np.array(maudit_ss_list),
    )

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
def plot_lambda_phase_curve(
    lam_values,
    r_ss,
    mexec_ss,
    maudit_ss,
    out_path: str,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=150)

    # 1. Reward Curve (Blue) - Primary Axis
    plt.plot(lam_values, r_ss, label="Reward Magnitude (r_mean)",
             color='#1f77b4', linewidth=3, marker='o')

    # 2. Instability Curve (Orange)
    plt.plot(lam_values, mexec_ss, label="Instability (M_exec)",
             color='#ff7f0e', linewidth=2, linestyle='--')

    # 3. Skew Curve (Green)
    plt.plot(lam_values, maudit_ss, label="Audit Skew (M_audit)",
             color='#2ca02c', linewidth=2, linestyle=':')

    # Annotate Phase Transition Zone (Based on lam > 5 logic)
    plt.axvspan(4.5, 5.5, color='gray', alpha=0.1, label="Phase Transition (λ≈5)")
    plt.text(5.0, max(r_ss)*0.8, "Collapse\nThreshold", ha='center', color='gray')

    plt.xlabel("Regulation Strength (Lambda)", fontsize=12)
    plt.ylabel("Steady-State Metric Value", fontsize=12)
    plt.title("B1: Lambda Phase Curve (Physics Scan)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(out_path)
    plt.close()

    print(f"✅ Saved B1 Curve to: {out_path}")

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Define output directory
    out_dir = os.path.join(project_root, "experiments", "results", "physics")
    out_path = os.path.join(out_dir, "lambda_phase_curve.png")

    # Run
    lam_values, r_ss, mexec_ss, maudit_ss = run_lambda_scan()

    # Plot
    plot_lambda_phase_curve(lam_values, r_ss, mexec_ss, maudit_ss, out_path)