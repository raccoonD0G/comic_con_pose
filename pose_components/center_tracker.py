from __future__ import annotations
from typing import Tuple
import numpy as np

class CenterTracker:
    """EMA-based center smoothing with deadzone."""

    def __init__(self, w: int, h: int, smooth: float, deadzone: int):
        self.cx = w * 0.5
        self.cy = h * 0.5
        self.have = False
        self.smooth = float(np.clip(smooth, 0.0, 1.0))
        self.deadzone = int(deadzone)
        self.W = w
        self.H = h

    def update(self, cx: float, cy: float) -> None:
        if not (np.isfinite(cx) and np.isfinite(cy)):
            return
        if not self.have:
            self.cx, self.cy = cx, cy
            self.have = True
        else:
            a = self.smooth
            self.cx = a * self.cx + (1.0 - a) * cx
            self.cy = a * self.cy + (1.0 - a) * cy

    def offset(self) -> Tuple[float, float]:
        if not self.have:
            return 0.0, 0.0
        ndx = (self.W * 0.5) - self.cx
        ndy = (self.H * 0.5) - self.cy
        if abs(ndx) < self.deadzone:
            ndx = 0.0
        if abs(ndy) < self.deadzone:
            ndy = 0.0
        return ndx, ndy