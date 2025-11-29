# ============================================================
# DynaAlign Environment Core Logic
# Full implementation of Goodhart Dynamics (V9.3 Physics)
# ============================================================

import numpy as np
from scipy import stats

# -------------------------------
# Default Parameters (V9.3)
# -------------------------------
N_B = 50
EPISODE_LEN = 200
ALPHA_B = 0.1
NOISE_STD = 0.05

DEFAULT_L = 50
DEFAULT_LAMBDA = 2.0

N_PROXIES_EXEC = 10
N_PROXIES_JUDICIAL = 3
PROXY_SUBSAMPLE_RATIO = 0.7


# ============================================================
# Sampling function
# ============================================================
def sample_indices(n_b, n_proxies, ratio, mode="bootstrap", rng_seed=0):
    """
    Sample proxy groups from n_b agents.
    Returns a list of sorted numpy index arrays.
    """
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
                # restart if exhausted
                order = rng.permutation(n_b)
                start = 0
                end = n_sample
                cursor = end
            else:
                cursor = end
            idx_list.append(np.sort(order[start:end]))

    return idx_list


# ============================================================
# GoodhartEnv V9.3 — with Dynamic L and Lambda
# ============================================================
class GoodhartEnv:
    def __init__(
        self,
        L: int = DEFAULT_L,               # Feedback Delay Window
        lam: float = DEFAULT_LAMBDA,     # Regulation Penalty λ
        n_b: int = N_B,
        alpha: float = ALPHA_B,
        noise_std: float = NOISE_STD,
        seed: int = 0,
    ):
        self.L = int(L)
        self.lam = float(lam)

        self.n_b = n_b
        self.alpha = alpha
        self.noise_std = noise_std
        self.seed = seed

        # History buffer for AAR/Skew calculations
        self.max_history = max(self.L + 1, EPISODE_LEN + 1)
        self.r_history = np.zeros((self.max_history, n_b), dtype=np.float32)
        self.r_current = np.zeros(n_b, dtype=np.float32)

        self.exec_trace = []
        self.audit_trace = []

        self.t = 0
        self.reset()

    # --------------------------------------------------------
    def reset(self):
        np.random.seed(self.seed)

        # Start aligned at 1.0 to observe collapse behavior
        self.r_current = np.random.normal(1.0, 0.05, size=(self.n_b,)).astype(np.float32)

        self.r_history.fill(0.0)
        self.r_history[0] = self.r_current

        self.exec_trace = []
        self.audit_trace = []

        self.t = 0
        return self._get_obs()

    # --------------------------------------------------------
    def step(self, a):
        """
        a : global control signal applied to all agents
        """

        # Regulation Penalty — Laffer Curve effect
        penalty = 0.0
        if self.lam > 5.0:
            penalty = -0.08 * ((self.lam - 5.0) ** 1.5)

        # Agent Dynamics
        noise = np.random.normal(0, self.noise_std, size=(self.n_b,))
        perceived = a + penalty + noise

        self.r_current += self.alpha * (perceived - self.r_current)
        self.r_current = np.maximum(self.r_current, 0.0)  # Hard floor (Death)

        # Update history
        self.t += 1
        if self.t < self.max_history:
            self.r_history[self.t] = self.r_current

        done = (self.t >= EPISODE_LEN)
        reward = float(self.r_current.mean())

        return self._get_obs(), reward, done, {}

    # --------------------------------------------------------
    def _get_obs(self):
        mean = self.r_current.mean()
        trend = 0.0 if self.t == 0 else (self.r_history[self.t].mean() - self.r_history[self.t - 1].mean())
        std = self.r_current.std()
        return np.array([mean, trend, std], dtype=np.float32)

    # ============================================================
    # Metrics: g_t, M_exec, M_audit
    # ============================================================
    def compute_step_metrics(self):
        """
        Returns:
            g_t : recent real reward change
            Mexec : Execution AAR
            Maud : Judicial Skew
        """

        # Goal metric
        g_t = self.r_history[self.t].mean() - self.r_history[self.t - 1].mean()

        # Window-L history
        s = max(0, self.t - self.L)
        hist = self.r_history[s:self.t + 1]

        # AAR helper
        def aar(seq):
            if len(seq) < 2:
                return 0.0
            diffs = np.abs(np.diff(seq))
            std_seq = np.std(seq)
            return float(diffs.mean() / (std_seq + 1e-8)) if std_seq > 1e-8 else float(diffs.mean())

        # Execution proxies
        idx_exec = sample_indices(self.n_b, N_PROXIES_EXEC, PROXY_SUBSAMPLE_RATIO, rng_seed=self.seed + self.t)
        exec_vals = [
            aar(hist[:, group].mean(axis=1))
            for group in idx_exec
        ]
        M_exec = float(np.median(exec_vals))

        # Judicial proxies (Skew)
        idx_jud = sample_indices(self.n_b, N_PROXIES_JUDICIAL, PROXY_SUBSAMPLE_RATIO,
                                 mode="disjoint", rng_seed=self.seed + self.t + 1)

        jud_vals = [
            stats.skew(hist[:, group].mean(axis=1), bias=False)
            if hist.shape[0] >= 3 else 0.0
            for group in idx_jud
        ]
        M_audit = float(np.median(jud_vals))

        # Safety
        if np.isnan(g_t): g_t = 0.0
        if np.isnan(M_exec): M_exec = 0.0
        if np.isnan(M_audit): M_audit = 0.0

        return g_t, M_exec, M_audit
