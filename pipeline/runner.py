"""Threaded runner for launching the capture pipeline from the GUI."""
from __future__ import annotations

import argparse
import threading
import sys
from typing import Optional

from .loop import build_context, cleanup, run_loop


class PipelineRunner:
    def __init__(self, app: "ControlPanelCallbacks") -> None:
        self.app = app
        self.thread: Optional[threading.Thread] = None
        self.ctx = None
        self.stop_event: Optional[threading.Event] = None
        self._lock = threading.Lock()

    def start(self, args: argparse.Namespace) -> None:
        with self._lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("Pipeline already running")
            stop_event = threading.Event()
            self.stop_event = stop_event

            def worker() -> None:
                ctx = None
                try:
                    ctx = build_context(args, stop_event=stop_event)
                    with self._lock:
                        self.ctx = ctx
                    self.app.on_pipeline_running_threadsafe()
                    run_loop(ctx)
                except Exception as exc:  # pragma: no cover - runtime failures
                    self.app.on_pipeline_error_threadsafe(exc)
                finally:
                    if ctx is not None:
                        try:
                            cleanup(ctx)
                        except Exception as exc:  # pragma: no cover - cleanup failures
                            print("[ERROR] Cleanup failed:", exc, file=sys.stderr)
                    self.app.on_pipeline_finished_threadsafe()
                    with self._lock:
                        self.thread = None
                        self.ctx = None
                        self.stop_event = None

            thread = threading.Thread(target=worker, name="pipeline-thread", daemon=True)
            self.thread = thread

        thread.start()

    def stop(self) -> None:
        with self._lock:
            event = self.stop_event
            ctx = self.ctx
        if event:
            event.set()
        if ctx:
            ctx.cam_mgr.stop()

    def is_running(self) -> bool:
        with self._lock:
            return bool(self.thread and self.thread.is_alive())


class ControlPanelCallbacks:
    """Protocol for callbacks used by PipelineRunner."""

    def on_pipeline_running_threadsafe(self) -> None: ...

    def on_pipeline_error_threadsafe(self, exc: Exception) -> None: ...

    def on_pipeline_finished_threadsafe(self) -> None: ...
