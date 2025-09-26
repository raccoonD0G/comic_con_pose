"""Eager imports for runtime dependencies and shared helpers."""
from __future__ import annotations

import cv2
import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "PyTorch is required but not installed. Install it with "
        "`pip install torch --index-url https://download.pytorch.org/whl/cu121` "
        "or the appropriate wheel for your platform."
    ) from exc

if torch.cuda.is_available():  # pragma: no cover - CUDA optional
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

try:
    import NDIlib as ndi
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "NDIlib is required but not installed. Install the NewTek NDI runtime "
        "or SDK to enable NDI output."
    ) from exc

try:  # pragma: no cover - optional dependency
    import mediapipe as mp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    mp = None  # type: ignore[assignment]

from ultralytics import YOLO

from pose_components import (
    CameraCapture,
    CenterTracker,
    HandsDetector,
    NDISender,
    PoseDetector,
    RVM,
    UDPPoseSender,
    build_gamma_dark_lut,
    compute_bg_stats,
)
from utils import (
    HANDS_EDGES,
    HAND_HEAD_FMT,
    HEADER_V2_FMT,
    K,
    MAGIC,
    ONLY_NEAREST,
    SKELETON_EDGES,
    UDP_IP,
    UDP_PORT,
    VERSION,
    apply_affine_xy,
    bbox_apply_affine,
    clip_int,
    crop_safe,
    cutout_alpha_inplace,
    draw_hands,
    draw_pose,
    ema_rect,
    fourcc_to_str,
    kps_to_bbox,
    make_bbox_mask,
    make_letterbox_affine,
    padded_bbox_src,
    person_center,
    warp_mask_to_canvas,
)


__all__ = [
    "cv2",
    "np",
    "torch",
    "ndi",
    "mp",
    "YOLO",
    "ONLY_NEAREST",
    "K",
    "SKELETON_EDGES",
    "HANDS_EDGES",
    "UDP_IP",
    "UDP_PORT",
    "HEADER_V2_FMT",
    "MAGIC",
    "VERSION",
    "HAND_HEAD_FMT",
    "fourcc_to_str",
    "kps_to_bbox",
    "make_letterbox_affine",
    "apply_affine_xy",
    "bbox_apply_affine",
    "person_center",
    "clip_int",
    "draw_pose",
    "draw_hands",
    "make_bbox_mask",
    "warp_mask_to_canvas",
    "cutout_alpha_inplace",
    "padded_bbox_src",
    "ema_rect",
    "crop_safe",
    "CameraCapture",
    "CenterTracker",
    "UDPPoseSender",
    "NDISender",
    "RVM",
    "PoseDetector",
    "HandsDetector",
    "compute_bg_stats",
    "build_gamma_dark_lut",
]
