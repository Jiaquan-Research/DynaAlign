import pytest
import numpy as np
from src.controllers.phoenix_controller_v9_3 import PhoenixControllerV9_3


def test_controller_basic_output_range():
    ctrl = PhoenixControllerV9_3(ctrl_scale=1.0)
    u = ctrl.update(0.5, 1.0, 1.0)
    assert np.isfinite(u)


def test_controller_negative_feedback():
    ctrl = PhoenixControllerV9_3(ctrl_scale=1.0)
    # Warm up history to avoid Z-score returning 0
    for _ in range(10):
        ctrl.update(0.2, 0.2, 0.2)

    u_low = ctrl.update(0.2, 0.5, 0.5)

    # Trigger a spike
    # We need to persist the spike to pass the debounce check (4 steps)
    u_high = 0.0
    for _ in range(6):
        u_high = ctrl.update(0.9, 5.0, 5.0)

    assert abs(u_high) >= abs(u_low)


def test_controller_hard_stop_on_spike():
    """
    V9.3 triggers HARD STOP only on sudden spikes, AFTER debounce.
    """
    # Use higher scale to reach HARD STOP faster
    ctrl = PhoenixControllerV9_3(ctrl_scale=1.0, debounce_steps=3)

    # 1. Warm-up normal values (establish a baseline)
    for _ in range(10):
        ctrl.update(0.5, 0.1, 0.1)

    # 2. Sudden abnormal spike (must persist > debounce_steps)
    hard_stop_triggered = False
    for _ in range(10):
        # Massive spike in Execution Instability (Mex)
        u = ctrl.update(0.5, 10.0, 10.0)

        # Check if control signal ramps up
        if abs(u) > 1.5:
            hard_stop_triggered = True
            break

    assert hard_stop_triggered, f"Controller did not trigger HARD STOP. Final u={u}"


def test_controller_cooldown():
    """
    After reaching HARD STOP, controller should eventually cool down.
    """
    ctrl = PhoenixControllerV9_3(ctrl_scale=1.0, debounce_steps=2, cooldown_steps=5)

    # 1. Warm-up
    for _ in range(10):
        ctrl.update(0.5, 0.1, 0.1)

    # 2. Trigger HARD STOP (Force it)
    # We manually force the controller state to simulate a post-spike state
    # or just run a loop until it triggers.
    triggered = False
    for _ in range(10):
        u = ctrl.update(0.5, 20.0, 20.0)
        if abs(u) > 1.5:
            triggered = True
            break

    assert triggered, "Failed to trigger HARD STOP setup for cooldown test."

    # 3. Cooldown Phase
    # Metrics return to normal, controller should decay
    final_u = 10.0
    for _ in range(15):  # Run enough steps to clear cooldown + decay
        final_u = ctrl.update(0.5, 0.1, 0.1)

    # It should have decayed significantly from the max (3.0)
    assert abs(final_u) < 1.0, f"Cooldown failed. Signal stayed high: {final_u}"