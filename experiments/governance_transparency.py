import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Governance Transparency Experiment
# ============================================================
# Question:
# Does opacity in governance boundaries destabilize
# long-term system health under partial observability?
# ============================================================

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

def run_simulation(policy, T=400, seed=42):
    rng = np.random.default_rng(seed)

    H = np.zeros(T)
    H[0] = 0.95

    info_quality = np.clip(
        0.55 + rng.normal(0, 0.15, T),
        0.0, 1.0
    )

    base_threshold = 0.55

    for t in range(T - 1):
        U = 1.0 - info_quality[t]

        if policy == "transparent":
            threshold = base_threshold
        elif policy == "opaque":
            threshold = base_threshold + rng.normal(0.0, 0.10)
        else:
            raise ValueError("Unknown policy")

        act = 1 if info_quality[t] >= threshold else 0

        if act == 1:
            H[t + 1] = H[t] - 0.03 * U
        else:
            H[t + 1] = H[t] + 0.015 * (1.0 - U)

        H[t + 1] = np.clip(H[t + 1], 0.0, 1.0)

    return H


# ---------------- Run ----------------
plt.figure(figsize=(10, 6))
results = {}

for policy, color in [("transparent", "blue"), ("opaque", "red")]:
    H = run_simulation(policy)
    results[policy] = H

    plt.plot(
        H,
        label=f"{policy.upper()} | Final H={H[-1]:.2f}",
        color=color,
        alpha=0.85
    )

plt.title("Effect of Governance Transparency on System Stability")
plt.xlabel("Time")
plt.ylabel("Structural Health (H)")
plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

out_path = FIG_DIR / "governance_transparency.png"
plt.savefig(out_path, dpi=150)
plt.close()

print("=== Governance Transparency Summary ===")
for k, v in results.items():
    print(f"[{k}] H_final={v[-1]:.3f} | H_mean={np.mean(v):.3f} | H_std={np.std(v):.3f}")

print(f"Figure saved to: {out_path}")
