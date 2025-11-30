"""
core.py - The "Mini-Phoenix" Controller (Mathematical Twin).

This module implements a lightweight version of the Phoenix V9.3 engine.
It strips away the LLM/PyTorch dependencies but preserves the
Control Theory logic (AAR monitoring, Laffer Penalty) and
Transport Delay simulation using a deque buffer.

Goal: Provide a reproducible, lightweight controller for Chaos Scans
that mirrors the dynamics of the full system.
"""

import numpy as np
from collections import deque

class MiniPhoenixController:
    def __init__(self, L: int, gain: float = 3.0):
        """
        Initialize the Mini-Phoenix Controller.

        Args:
            L (int): The Transport Delay (Latency) in steps.
                     This represents how old the data is when the controller sees it.
            gain (float): The feedback gain (Aggressiveness of control).
        """
        self.L = L
        self.gain = gain

        # Simulates the "Real-World" transport delay pipeline.
        # The controller only 'sees' data after it passes through this queue.
        # maxlen=L enforces the delay window.
        self.delay_queue = deque(maxlen=L)

    def get_correction(self, current_aar: float) -> float:
        """
        Compute the control correction (penalty) based on INSTABILITY.

        Logic:
        1. Push current real-time metric into the delay queue.
        2. Read the OLDEST metric from the queue (simulating lag).
           - If L=1, we read immediate history.
           - If L=50, we read history from 50 steps ago.
        3. Apply PID-like proportional correction based on that perceived state.
        """
        # 1. Ingest real-time signal into the pipeline
        self.delay_queue.append(current_aar)

        # 2. Perceive signal (Subject to Latency L)
        # If queue is not full (warm-up phase), assume stability (0.0).
        # Otherwise, read the metric from L steps ago.
        if len(self.delay_queue) < self.L:
            perceived_instability = 0.0
        else:
            perceived_instability = self.delay_queue[0]

        # 3. Compute Correction (Laffer Penalty Logic)
        # Correction = Gain * Instability
        # Higher instability -> Stronger braking (Penalty)
        correction = self.gain * perceived_instability

        return correction

    def reset(self):
        """Clear the delay buffer to prevent state leakage between episodes."""
        self.delay_queue.clear()