"""Background worker for asynchronous pose inference."""
from __future__ import annotations

import queue
import threading
from typing import Optional, Tuple


class PoseWorker:
    """Runs pose estimation on a background thread."""

    def __init__(self, pose_model, scale: float, max_queue: int = 1) -> None:
        self._pose = pose_model
        self._scale = float(scale)
        self._queue: "queue.Queue[Optional[object]]" = queue.Queue(maxsize=max(1, max_queue))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

        self._result_lock = threading.Lock()
        self._result_id = 0
        self._result_xy = None
        self._result_conf = None

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame) -> None:
        """Queue a frame for pose inference, dropping stale frames if needed."""

        if self._stop_event.is_set():
            return

        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def latest(self) -> Optional[Tuple[int, object, object]]:
        """Return the most recent inference result, if any."""

        with self._result_lock:
            if self._result_id == 0:
                return None
            return self._result_id, self._result_xy, self._result_conf

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is None:
                self._queue.task_done()
                continue

            xy_all_src, conf_all = self._pose.predict_downscaled(frame, scale=self._scale)

            with self._result_lock:
                self._result_id += 1
                self._result_xy = xy_all_src
                self._result_conf = conf_all

            self._queue.task_done()


__all__ = ["PoseWorker"]
