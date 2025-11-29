"""
Phoenix Controller V9.3 (Compatible with GoodhartEnv V8.x/V9.3)
---------------------------------------------------------------

This controller does NOT modify env parameters (no lambda/L inside env).
Instead, it produces a control-action term that is ADDED to PPO's action:

    a_total = a_agent + a_ctrl

Metrics come from:
    g_t, M_exec, M_audit = env.compute_step_metrics()

Mechanisms:
    - Robust Z-score stability detection (based on MAD)
    - Debounce (avoid reacting to single-step spikes)
    - Cooldown window
    - Bounded control signal
"""

import numpy as np


class PhoenixControllerV9_3:
    def __init__(
        self,
        ctrl_scale: float = 0.5,   # Control strength
        z_threshold: float = 2.5,  # Robust Z-score threshold
        debounce_steps: int = 4,
        cooldown_steps: int = 8,
    ):
        # Controller output: a_ctrl
        self.ctrl = 0.0

        # Metric histories
        self.g_history = []
        self.exec_history = []
        self.audit_history = []

        self.z_threshold = z_threshold
        self.debounce_steps = debounce_steps
        self.cooldown_steps = cooldown_steps

        self.debounce_counter = 0
        self.cooldown_counter = 0

        self.ctrl_scale = ctrl_scale

    # ------------------------------------------------------------
    # Robust Z-score
    # ------------------------------------------------------------
    def _robust_z(self, arr):
        """
        Median-based Robust Z-score using MAD.
        If not enough history, returns 0.0 (neutral).
        """
        if len(arr) < 6:
            return 0.0
        arr = np.array(arr)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) + 1e-8
        return (arr[-1] - med) / (1.4826 * mad)

    # ------------------------------------------------------------
    # Update controller (called once per environment step)
    # ------------------------------------------------------------
    def update(self, g_t, M_exec, M_audit):
        """
        Update internal state given current metrics and
        return current control signal (a_ctrl).
        """
        # Record metrics
        self.g_history.append(g_t)
        self.exec_history.append(M_exec)
        self.audit_history.append(M_audit)

        # Compute robust Z-scores
        z_g = self._robust_z(self.g_history)
        z_exec = self._robust_z(self.exec_history)

        # Instability detection:
        # - g_t strongly negative (collapse)
        # - execution metric too high (oscillation)
        unstable = (z_g < -self.z_threshold) or (z_exec > self.z_threshold)

        # Debounce: require instability to persist several steps
        if unstable:
            self.debounce_counter += 1
        else:
            self.debounce_counter = 0

        # Cooldown: after strong actuation, wait a few steps
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        # -------------------- Control Policy ---------------------
        if self.debounce_counter < self.debounce_steps:
            # Stable zone → decay toward 0
            self.ctrl *= 0.85
        else:
            # Unstable zone → apply intervention
            self.ctrl += self.ctrl_scale
            self.cooldown_counter = self.cooldown_steps

        # Safety saturation
        self.ctrl = float(np.clip(self.ctrl, -3.0, 3.0))

        return self.ctrl
