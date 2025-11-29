# ============================================================
# DynaAlign Environment Core Logic (Physics Engine)
# Implements: Multi-Agent Goodhart Dynamics + Whitepaper Physics (L, Lambda)
# ============================================================

import numpy as np
from scipy import stats

# -------------------------------
# Default Config
# -------------------------------
N_B = 50
EPISODE_LEN = 200
ALPHA_B = 0.1
NOISE_STD = 0.05
DEFAULT_WINDOW = 50
N_PROXIES_EXEC = 10
N_PROXIES_JUDICIAL = 3
PROXY_SUBSAMPLE_RATIO = 0.7


def sample_indices(n_b, n_proxies, ratio, mode="bootstrap", rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    n_sample = max(1, int(n_b * ratio))
    idx_list = []
    if mode == "bootstrap":
        for _ in range(n_proxies):
            idx = rng.choice(n_b, size=n_sample, replace=False)
            idx_list.append(np.sort(idx))
    elif mode == "disjoint":
        order = rng.permutation(n_b)
        cursor = 0
        for _ in range(n_proxies):
            start = cursor
            end = start + n_sample
            if end > n_b:
                order = rng.permutation(n_b)
                start = 0
                end = n_sample
                cursor = end
            else:
                cursor = end
            idx_list.append(np.sort(order[start:end]))
    return idx_list


class GoodhartEnv:
    """
    The Real Physics Engine of DynaAlign.
    Simulates misalignment dynamics with Feedback Delay (L) and Regulation Penalty (Lambda).
    """

    def __init__(
            self,
            L=DEFAULT_WINDOW,  # Feedback Delay (Lag)
            lam=2.0,  # Regulation Penalty (Lambda)
            n_b=N_B,
            alpha=ALPHA_B,
            noise_std=NOISE_STD,
            seed=0,
    ):
        self.L = int(L)
        self.lam = float(lam)
        self.n_b = n_b
        self.alpha = alpha
        self.noise_std = noise_std
        self.seed = seed

        # History buffer
        self.max_history = max(self.L + 1, EPISODE_LEN + 1)
        self.r_history = np.zeros((self.max_history, n_b), dtype=np.float32)
        self.r_current = np.zeros(n_b, dtype=np.float32)

        self.t = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        # Start at 1.0 (Aligned State) to observe collapse
        self.r_current = np.random.normal(1.0, 0.05, size=(self.n_b,)).astype(np.float32)
        self.r_history.fill(0.0)
        self.r_history[0] = self.r_current
        self.t = 0
        return self._get_obs()

    def step(self, a):
        # 1. Regulation Penalty (The Death Abyss Force)
        # Stronger penalty logic matched to V9.3 Whitepaper
        penalty = 0.0
        if self.lam > 5.0:
            # Sharper collapse: -0.08 * (lam - 5)^1.5
            penalty = -0.08 * ((self.lam - 5.0) ** 1.5)

        # 2. Agent Dynamics
        noise = np.random.normal(0, self.noise_std, size=(self.n_b,))
        perceived = a + penalty + noise
        self.r_current += self.alpha * (perceived - self.r_current)

        # Hard clip at 0.0 (Death)
        self.r_current = np.maximum(self.r_current, 0.0)

        # 3. Update History
        self.t += 1
        if self.t < self.max_history:
            self.r_history[self.t] = self.r_current

        done = (self.t >= EPISODE_LEN)
        reward = float(self.r_current.mean())

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        mean = self.r_current.mean()
        trend = 0.0
        if self.t > 0:
            trend = self.r_history[self.t].mean() - self.r_history[self.t - 1].mean()
        std = self.r_current.std()
        return np.array([mean, trend, std], dtype=np.float32)

    def compute_step_metrics(self):
        """Calculates observable metrics based on the sliding window L."""
        g_t = self.r_history[self.t].mean() - self.r_history[self.t - 1].mean()
        s = max(0, self.t - self.L)
        hist = self.r_history[s:self.t + 1]

        def aar(seq):
            if len(seq) < 2: return 0.0
            diffs = np.abs(np.diff(seq))
            if diffs.size == 0: return 0.0
            std_seq = np.std(seq)
            return float(diffs.mean() / (std_seq + 1e-8)) if std_seq > 1e-8 else float(diffs.mean())

        idx_exec = sample_indices(self.n_b, N_PROXIES_EXEC, PROXY_SUBSAMPLE_RATIO, rng_seed=self.seed + self.t)
        idx_jud = sample_indices(self.n_b, N_PROXIES_JUDICIAL, PROXY_SUBSAMPLE_RATIO, mode="disjoint",
                                 rng_seed=self.seed + self.t + 1)

        exec_vals = [aar(hist[:, group].mean(axis=1)) for group in idx_exec]
        M_exec = float(np.median(exec_vals))

        jud_vals = [stats.skew(hist[:, group].mean(axis=1), bias=False) if hist.shape[0] >= 3 else 0.0 for group in
                    idx_jud]
        M_audit = float(np.median(jud_vals))

        if np.isnan(g_t): g_t = 0.0
        if np.isnan(M_exec): M_exec = 0.0
        if np.isnan(M_audit): M_audit = 0.0

        return g_t, M_exec, M_audit