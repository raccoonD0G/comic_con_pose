"""Lazy loading for heavy runtime dependencies and shared helpers."""
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

cv2 = None  # type: ignore[assignment]
np = None  # type: ignore[assignment]
torch = None  # type: ignore[assignment]
ndi = None  # type: ignore[assignment]
mp = None  # type: ignore[assignment]
YOLO = None  # type: ignore[assignment]

_runtime_modules_loaded = False

ONLY_NEAREST: bool = True
K: Dict[str, int] = {}
SKELETON_EDGES: List[Tuple[int, int]] = []
HANDS_EDGES: List[Tuple[int, int]] = []
UDP_IP: str = "127.0.0.1"
UDP_PORT: int = 7777
HEADER_V2_FMT: str = "<4sBBHHQ"
MAGIC: bytes = b"POSE"
VERSION: int = 2
HAND_HEAD_FMT: str = "<HBf"

fourcc_to_str: Any = None
kps_to_bbox: Any = None
make_letterbox_affine: Any = None
apply_affine_xy: Any = None
bbox_apply_affine: Any = None
person_center: Any = None
clip_int: Any = None
draw_pose: Any = None
draw_hands: Any = None
make_bbox_mask: Any = None
warp_mask_to_canvas: Any = None
cutout_alpha_inplace: Any = None
padded_bbox_src: Any = None
ema_rect: Any = None
crop_safe: Any = None

_utils_loaded = False

CameraCapture = None  # type: ignore[assignment]
CenterTracker = None  # type: ignore[assignment]
UDPPoseSender = None  # type: ignore[assignment]
NDISender = None  # type: ignore[assignment]
RVM = None  # type: ignore[assignment]
PoseDetector = None  # type: ignore[assignment]
HandsDetector = None  # type: ignore[assignment]
compute_bg_stats: Any = None
build_gamma_dark_lut: Any = None

_components_loaded = False


def ensure_runtime_modules() -> None:
    """Import heavy dependencies on demand."""

    global _runtime_modules_loaded, cv2, np, torch, ndi, mp, YOLO

    if _runtime_modules_loaded:
        return

    if np is None:
        globals()["np"] = importlib.import_module("numpy")
    if cv2 is None:
        globals()["cv2"] = importlib.import_module("cv2")
    if torch is None:
        try:
            globals()["torch"] = importlib.import_module("torch")
        except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
            raise ModuleNotFoundError(
                "PyTorch is required but not installed. Install it with "
                "`pip install torch --index-url https://download.pytorch.org/whl/cu121` "
                "or the appropriate wheel for your platform."
            ) from exc
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("medium")
            except Exception:  # pragma: no cover - guard for old torch
                pass
    if ndi is None:
        globals()["ndi"] = importlib.import_module("NDIlib")
    if mp is None:
        try:
            globals()["mp"] = importlib.import_module("mediapipe")
        except Exception:  # pragma: no cover - optional dependency
            globals()["mp"] = None
    if YOLO is None:
        globals()["YOLO"] = importlib.import_module("ultralytics").YOLO

    _runtime_modules_loaded = True


def _require_utils_loaded() -> None:
    if not _utils_loaded:
        raise RuntimeError("Utility helpers accessed before ensure_utils_loaded()")


def ensure_utils_loaded() -> None:
    """Hydrate shared utility helpers from the utils module."""

    global _utils_loaded, ONLY_NEAREST, K, SKELETON_EDGES, HANDS_EDGES
    global UDP_IP, UDP_PORT, HEADER_V2_FMT, MAGIC, VERSION, HAND_HEAD_FMT
    global fourcc_to_str, kps_to_bbox, make_letterbox_affine, apply_affine_xy
    global bbox_apply_affine, person_center, clip_int, draw_pose, draw_hands
    global make_bbox_mask, warp_mask_to_canvas, cutout_alpha_inplace
    global padded_bbox_src, ema_rect, crop_safe

    if _utils_loaded:
        return

    utils = importlib.import_module("utils")
    ONLY_NEAREST = getattr(utils, "ONLY_NEAREST")
    K = getattr(utils, "K")
    SKELETON_EDGES = getattr(utils, "SKELETON_EDGES")
    HANDS_EDGES = getattr(utils, "HANDS_EDGES")
    UDP_IP = getattr(utils, "UDP_IP")
    UDP_PORT = getattr(utils, "UDP_PORT")
    HEADER_V2_FMT = getattr(utils, "HEADER_V2_FMT")
    MAGIC = getattr(utils, "MAGIC")
    VERSION = getattr(utils, "VERSION")
    HAND_HEAD_FMT = getattr(utils, "HAND_HEAD_FMT")
    fourcc_to_str = getattr(utils, "fourcc_to_str")
    kps_to_bbox = getattr(utils, "kps_to_bbox")
    make_letterbox_affine = getattr(utils, "make_letterbox_affine")
    apply_affine_xy = getattr(utils, "apply_affine_xy")
    bbox_apply_affine = getattr(utils, "bbox_apply_affine")
    person_center = getattr(utils, "person_center")
    clip_int = getattr(utils, "clip_int")
    draw_pose = getattr(utils, "draw_pose")
    draw_hands = getattr(utils, "draw_hands")
    make_bbox_mask = getattr(utils, "make_bbox_mask")
    warp_mask_to_canvas = getattr(utils, "warp_mask_to_canvas")
    cutout_alpha_inplace = getattr(utils, "cutout_alpha_inplace")
    padded_bbox_src = getattr(utils, "padded_bbox_src")
    ema_rect = getattr(utils, "ema_rect")
    crop_safe = getattr(utils, "crop_safe")

    globals()["np"] = globals()["np"] or getattr(utils, "np")
    globals()["cv2"] = globals()["cv2"] or getattr(utils, "cv2")

    _utils_loaded = True


def ensure_components_loaded() -> None:
    """Import pipeline components when required."""

    global _components_loaded
    global CameraCapture, CenterTracker, UDPPoseSender, NDISender
    global RVM, PoseDetector, HandsDetector, compute_bg_stats, build_gamma_dark_lut

    if _components_loaded:
        return

    mod = importlib.import_module("pose_components")
    CameraCapture = getattr(mod, "CameraCapture")
    CenterTracker = getattr(mod, "CenterTracker")
    UDPPoseSender = getattr(mod, "UDPPoseSender")
    NDISender = getattr(mod, "NDISender")
    RVM = getattr(mod, "RVM")
    PoseDetector = getattr(mod, "PoseDetector")
    HandsDetector = getattr(mod, "HandsDetector")
    compute_bg_stats = getattr(mod, "compute_bg_stats")
    build_gamma_dark_lut = getattr(mod, "build_gamma_dark_lut")

    _components_loaded = True


__all__ = [
    "ensure_runtime_modules",
    "ensure_utils_loaded",
    "ensure_components_loaded",
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
