"""Dataclasses shared across the capture pipeline."""
from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class RunContext:
    args: argparse.Namespace
    OUT_W: int
    OUT_H: int
    KP_THR: float
    use_nearest: bool

    cap: Any
    cam_mgr: Any
    center: Any
    device: Any
    pose: Any
    rvm: Optional[Any]
    hands: Any
    sender: Any
    udp: Any
    pose_worker: Optional[Any]

    stats: Optional[Dict[str, Any]]
    lut_gamma_dark: Optional[Any]
    stop_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class Caches:
    last_xy_all_src: Optional[Any] = None
    last_conf_all: Optional[Any] = None
    last_bbox_src_debug: Optional[Any] = None
    last_hands_src: Optional[List[Dict[str, Any]]] = None
    last_alpha_src: Optional[Any] = None
    last_rvm_mode: Optional[str] = None
    last_rvm_input_hw: Optional[Tuple[int, int]] = None
    frames: int = 0
    last_log: float = 0.0
    last_pose_result_id: int = -1


__all__ = ["RunContext", "Caches"]
