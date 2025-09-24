from __future__ import annotations
import time, threading, argparse
from collections import deque
from typing import Optional
import numpy as np
import cv2

class CameraCapture:
    """Webcam/file capture with optional thread + pacing."""

    def __init__(self, args: argparse.Namespace, cap: cv2.VideoCapture):
        self.args = args
        self.cap = cap
        self.use_webcam = (not args.src.strip())
        self.q: Optional[deque] = deque(maxlen=max(1, args.queue_len)) if self.use_webcam else None
        self.stop_flag = False
        self.thread: Optional[threading.Thread] = None

        # pacing
        self.target_fps = max(1e-3, float(args.target_fps))
        self.interval = 1.0 / self.target_fps
        self.next_t = time.perf_counter()

    def start(self) -> None:
        if not self.use_webcam:
            return

        def loop() -> None:
            while not self.stop_flag:
                if self.cap.grab():
                    ok, f = self.cap.retrieve()
                    if ok:
                        assert self.q is not None
                        self.q.append(f)
                else:
                    time.sleep(0.001)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def _pace_tick(self) -> bool:
        now = time.perf_counter()
        if self.args.pace == "sleep":
            if now < self.next_t:
                time.sleep(self.next_t - now)
            self.next_t += self.interval
            return True
        else:  # drop
            if now >= self.next_t:
                behind = int((now - self.next_t) / self.interval)
                self.next_t += (behind + 1) * self.interval
                return True
            else:
                time.sleep(min(0.001, self.next_t - now))
                return False

    def read(self) -> Optional[np.ndarray]:
        if self.use_webcam:
            if not self._pace_tick():
                return None
            if not self.q:
                return None
            frame = self.q.pop()  # newest
            self.q.clear()
            return frame
        else:
            ok, f = self.cap.read()
            return f if ok else None

    def stop(self) -> None:
        self.stop_flag = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
