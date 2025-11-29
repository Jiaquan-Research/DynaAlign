# ============================================================
# DynaAlign Environment Core Logic
# Full implementation of AAR / Skew / Mex / Mau / Window-L
# and proxy grouping behavior.
# ============================================================

import numpy as np
from scipy import stats

# -------------------------------
# Config (aligned with V8.x / V9.3 PPO runs)
# -------------------------------
N_B = 50
EPISODE_LEN = 200
ALPHA_B = 0.1
NOISE_STD = 0.05
WINDOW_STEPS = 50
N_PROXIES_EXEC = 10
N_PROXIES_JUDICIAL = 3
PROXY_SUBSAMPLE_RATIO = 0.7


# ============================================================
# Sampling function
# ============================================================
def sample_indices(n_b, n_proxies, ratio, mode="bootstrap", rng_seed=0):
    """
    Sample proxy groups from n_b agents.

    Returns a list of numpy arrays, each being a sorted index group.
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
                # restart if we run out of indices
                order = rng.permutation(n_b)
                start = 0
                end = n_sample
                cursor = end
            else:
                cursor = end

            idx_list.append(np.sort(order[start:end]))

    return idx_list


# ============================================================
# GoodhartEnv: Core Dynamics Environment
# ============================================================
class GoodhartEnv:
    def __init__(
        self,
        n_b: int = N_B,
        alpha: float = ALPHA_B,
        noise_std: float = NOISE_STD,
        seed: int = 0,
    ):
        self.n_b = n_b
        self.alpha = alpha
        self.noise_std = noise_std
        self.seed = seed

        # History of each agent's reward (for AAR/Skew)
        self.r_history = np.zeros((EPISODE_LEN + 1, n_b), dtype=np.float32)
        self.r_current = np.zeros(n_b, dtype=np.float32)

        # Exec / audit traces (for Mex / Mau, if needed)
        self.exec_trace = []
        self.audit_trace = []

        self.t = 0
        self.reset()

    # --------------------------------------------------------
    # reset()
    # --------------------------------------------------------
    def reset(self):
        """
        Reset episode. Initial reward scale is kept consistent with
        earlier D5 runs (Normal(0, 0.1)).
        """
        np.random.seed(self.seed)
        self.r_current = np.random.normal(
            0.0, 0.1, size=(self.n_b,)
        ).astype(np.float32)
        self.r_history.fill(0.0)
        self.r_history[0] = self.r_current
        self.exec_trace = []
        self.audit_trace = []
        self.t = 0
        return self._get_obs()

    # --------------------------------------------------------
    # step(action)
    # --------------------------------------------------------
    def step(self, a: float):
        """
        One environment step.

        a: scalar control signal broadcast to all agents.

        Returns:
            obs : np.ndarray of shape (3,)
            done: bool
        (Note: reward is derived from state; PPO scripts can define it.)
        """
        noise = np.random.normal(0, self.noise_std, size=(self.n_b,))
        perceived = a + noise  # perceived regulatory pressure
        self.r_current += self.alpha * (perceived - self.r_current)

        self.t += 1
        if self.t < EPISODE_LEN + 1:
            self.r_history[self.t] = self.r_current

        done = (self.t >= EPISODE_LEN)
        return self._get_obs(), done

    # --------------------------------------------------------
    # get observation
    # --------------------------------------------------------
    def _get_obs(self):
        mean = self.r_current.mean()
        trend = (
            self.r_history[self.t].mean() - self.r_history[self.t - 1].mean()
            if self.t > 0
            else 0.0
        )
        std = self.r_current.std()
        return np.array([mean, trend, std], dtype=np.float32)

    # ============================================================
    # Core metrics for Phoenix:
    # compute_step_metrics()
    # ============================================================
    def compute_step_metrics(self):
        """
        Returns:
            g_t   : real mean reward change
            M_exec: AAR of execution proxies (proxy of Goodhart distortion)
            M_aud : Skew of judicial proxies (audit divergence)
        """

        # 1) Real reward change
        g_t = self.r_history[self.t].mean() - self.r_history[self.t - 1].mean()

        # 2) Window-L history sequence
        s = max(0, self.t - WINDOW_STEPS)
        hist = self.r_history[s:self.t + 1]  # shape = (L, n_b)
        L = hist.shape[0]  # noqa: F841 (kept for clarity)

        # ------- AAR Calculation -------
        def aar(seq):
            """
            Average Absolute Return (AAR) with safe handling.

            If seq length < 2, returns 0.0 to avoid NaN from np.diff([]).
            """
            if len(seq) < 2:
                return 0.0

            diffs = np.abs(np.diff(seq))
            if diffs.size == 0:
                return 0.0

            std_seq = np.std(seq)
            if std_seq > 1e-8:
                return float(diffs.mean() / (std_seq + 1e-8))
            else:
                return float(diffs.mean())

        # 3) Sampling proxies
        idx_exec = sample_indices(
            self.n_b,
            N_PROXIES_EXEC,
            PROXY_SUBSAMPLE_RATIO,
            rng_seed=self.seed + self.t,
        )
        idx_jud = sample_indices(
            self.n_b,
            N_PROXIES_JUDICIAL,
            PROXY_SUBSAMPLE_RATIO,
            mode="disjoint",
            rng_seed=self.seed + self.t + 1,
        )

        # 4) Exec proxies: group mean → AAR
        exec_vals = []
        for group in idx_exec:
            seq_group = hist[:, group]          # (L, k)
            seq_mean = seq_group.mean(axis=1)   # (L,)
            exec_vals.append(aar(seq_mean))
        M_exec = float(np.median(exec_vals))

        # 5) Judicial proxies: group mean → Skew
        jud_vals = []
        for group in idx_jud:
            seq_group = hist[:, group]
            seq_mean = seq_group.mean(axis=1)
            if len(seq_mean) >= 3:
                jud_vals.append(stats.skew(seq_mean, bias=False))
            else:
                jud_vals.append(0.0)
        M_audit = float(np.median(jud_vals))

        # Numerical safety
        if np.isnan(g_t):
            g_t = 0.0
        if np.isnan(M_exec):
            M_exec = 0.0
        if np.isnan(M_audit):
            M_audit = 0.0

        return g_t, M_exec, M_audit
