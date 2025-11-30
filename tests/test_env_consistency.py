import numpy as np
import pytest
import sys
import os

# --- Path Fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.env.goodhart_env import GoodhartEnv


def test_env_deterministic_reset():
    """Env reset must be deterministic under same seed."""
    env1 = GoodhartEnv(L=20, lam=5.5, seed=123)
    env2 = GoodhartEnv(L=20, lam=5.5, seed=123)

    obs1 = env1.reset()
    obs2 = env2.reset()

    assert np.allclose(obs1, obs2), "Reset should be deterministic under same seed."


def test_env_metrics_consistency():
    """compute_step_metrics() should return finite valid numbers."""
    env = GoodhartEnv(L=30, lam=5.5, seed=42)
    env.reset()

    g, mex, mau = env.compute_step_metrics()

    assert np.isfinite(g)
    assert np.isfinite(mex)
    assert np.isfinite(mau)

    assert 0.0 <= g <= 1.0, "g_t must be normalized to [0,1] range."


def test_step_returns_valid_transition():
    """Env.step() should return (obs, reward, done, info) with valid ranges."""
    env = GoodhartEnv(L=20, lam=5.5, seed=321)
    obs = env.reset()

    action = 0.0
    next_obs, reward, done, info = env.step(action)

    assert next_obs is not None
    assert isinstance(done, bool)
    assert np.isfinite(reward)
