import numpy as np


class PhoenixControllerV9_3:
    """
    Phoenix Controller (Official Public Release - V9.3)

    A lightweight inference-time controller that stabilizes PPO
    by monitoring alignment vital signs (gap, Mexec) and applying
    gentle corrections only when drift patterns appear.
    """

    def __init__(self, window=10, gain=0.05, panic_gap=0.8, panic_mexec=1.5):
        self.window = window
        self.gain = gain

        self.panic_gap = panic_gap
        self.panic_mexec = panic_mexec

        self.gap_hist = []
        self.mexec_hist = []

    def observe(self, gap, mexec):
        """ Update internal metric traces. """
        self.gap_hist.append(gap)
        self.mexec_hist.append(mexec)

        if len(self.gap_hist) > self.window:
            self.gap_hist = self.gap_hist[-self.window:]
            self.mexec_hist = self.mexec_hist[-self.window:]

    def is_panic(self):
        """ Panic trigger based on last-window maximums. """
        if len(self.gap_hist) < self.window:
            return False

        if max(self.gap_hist) > self.panic_gap:
            return True
        if max(self.mexec_hist) > self.panic_mexec:
            return True

        return False

    def correct(self, action):
        """
        Apply gentle correction to PPO action under drift/panic.
        """
        if not self.is_panic():
            return action  # no intervention

        # Phoenix correction = soft damping
        corrected = action * (1 - self.gain)
        return corrected
