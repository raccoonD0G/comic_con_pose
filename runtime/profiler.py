"""Simple profiling helper used during frame processing."""
from __future__ import annotations

from time import perf_counter
from typing import Dict


class Profiler:
    __slots__ = ("t0", "marks", "frame_ms_ema", "fps_ema", "enabled")

    def __init__(self, enabled: bool = True) -> None:
        self.t0 = perf_counter()
        self.marks: Dict[str, float] = {}
        self.enabled = enabled
        self.frame_ms_ema = None
        self.fps_ema = None

    def tick(self) -> None:
        self.t0 = perf_counter()

    def mark(self, key: str) -> None:
        if not self.enabled:
            return
        now = perf_counter()
        ms = (now - self.t0) * 1000.0
        self.marks[key] = self.marks.get(key, 0.0) + ms
        self.t0 = now

    def end_frame(self, frame_idx: int, alpha: float = 0.1, every: int = 60) -> None:
        total_ms = sum(self.marks.values())
        self.frame_ms_ema = total_ms if self.frame_ms_ema is None else (1 - alpha) * self.frame_ms_ema + alpha * total_ms
        fps = 1000.0 / max(total_ms, 1e-6)
        self.fps_ema = fps if self.fps_ema is None else (1 - alpha) * self.fps_ema + alpha * fps
        if frame_idx % every == 0:
            parts = " | ".join(f"{k}:{v:.1f}ms" for k, v in self.marks.items())
            print(
                f"[PROF] frame={frame_idx} total={total_ms:.1f}ms | {parts} | ema={self.frame_ms_ema:.1f}ms | fps≈{self.fps_ema:.1f}"
            )
            self.marks.clear()
