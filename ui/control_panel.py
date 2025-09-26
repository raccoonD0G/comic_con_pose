"""Tkinter control panel for launching the pipeline."""
from __future__ import annotations

import sys
import traceback

from pipeline.runner import PipelineRunner
from ui.gui_startup import SettingsForm, defaults_from_schema, namespace_from_dict


class ControlPanelApp:
    def __init__(self) -> None:
        try:
            import tkinter as tk
            from tkinter import messagebox, ttk
        except Exception as exc:  # pragma: no cover - environment without Tk
            raise RuntimeError("Tkinter GUI is not available") from exc

        self.messagebox = messagebox

        self.root = tk.Tk()
        self.root.title("WebcamCutout — Controller")
        self.root.geometry("960x760")

        defaults = defaults_from_schema()
        self.form = SettingsForm(self.root, defaults)
        self.form.focus_first()

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=8, pady=8)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(controls, textvariable=self.status_var).pack(side="left")

        self.stop_btn = ttk.Button(controls, text="Stop", command=self.on_stop, state="disabled")
        self.stop_btn.pack(side="right")
        self.start_btn = ttk.Button(controls, text="Start", command=self.on_start)
        self.start_btn.pack(side="right", padx=6)

        self.runner = PipelineRunner(self)
        self._requested_stop = False
        self._error_reported = False
        self._closing = False

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def run(self) -> None:
        self.root.mainloop()

    def on_start(self) -> None:
        if self.runner.is_running():
            self.messagebox.showinfo("Info", "Pipeline is already running.")
            return
        try:
            values = self.form.collect_values()
        except Exception as exc:
            self.messagebox.showerror("Invalid input", str(exc))
            return

        args = namespace_from_dict(values)

        self._requested_stop = False
        self._error_reported = False
        self.form.set_running(True)
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.queue_status("Initializing…")
        try:
            self.runner.start(args)
        except RuntimeError as exc:
            self._error_reported = True
            self.queue_status("Error")
            self.messagebox.showerror("Error", str(exc))
            self._reset_controls()

    def on_stop(self) -> None:
        if not self.runner.is_running():
            return
        self._requested_stop = True
        self.queue_status("Stopping…")
        self.stop_btn.configure(state="disabled")
        self.runner.stop()

    def on_close(self) -> None:
        if self.runner.is_running():
            self._closing = True
            if not self._requested_stop:
                self.on_stop()
            self.root.after(200, self._wait_close)
            return
        self.root.destroy()

    def _wait_close(self) -> None:
        if self.runner.is_running():
            self.root.after(200, self._wait_close)
        else:
            self.root.destroy()

    def queue_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))

    def on_pipeline_running_threadsafe(self) -> None:
        self.root.after(0, lambda: self.queue_status("Running"))

    def on_pipeline_error_threadsafe(self, exc: Exception) -> None:
        tb = "".join(traceback.format_exception(exc.__class__, exc, exc.__traceback__))

        def show_error() -> None:
            self._error_reported = True
            self.queue_status("Error")
            print(tb, file=sys.stderr)
            self.messagebox.showerror("Pipeline error", str(exc))

        self.root.after(0, show_error)

    def on_pipeline_finished_threadsafe(self) -> None:
        self.root.after(0, self._on_pipeline_finished)

    def _on_pipeline_finished(self) -> None:
        self._reset_controls()
        if self._error_reported:
            pass
        elif self._requested_stop:
            self.queue_status("Stopped")
        else:
            self.queue_status("Idle")
        self._requested_stop = False
        if self._closing:
            self.root.after(50, self.root.destroy)

    def _reset_controls(self) -> None:
        self.form.set_running(False)
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
