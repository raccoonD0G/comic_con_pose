"""Initialization helpers for building the runtime context."""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional, Tuple

from runtime import lazy_modules as lazy
from runtime import resource_path


def init_ndi_or_exit() -> None:
    if not lazy.ndi.initialize():
        print("[ERROR] NDI initialize failed. Install NDI Runtime/Tools and retry.")
        sys.exit(1)


def open_capture(args: argparse.Namespace, width: int, height: int) -> Any:
    cv2 = lazy.cv2
    if args.src.strip():
        cap = cv2.VideoCapture(args.src, cv2.CAP_FFMPEG)
    else:
        backend = cv2.CAP_DSHOW if args.backend == "ds" else (cv2.CAP_MSMF if args.backend == "ms" else 0)
        cap = cv2.VideoCapture(args.cam, backend)
        if not getattr(args, "ndi_follow_camera", False):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if args.mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if args.cap_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, args.cap_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERROR] Cannot open source.")
        lazy.ndi.destroy()
        sys.exit(1)
    return cap


def probe_source_size_or_exit(cap: Any) -> Tuple[int, int]:
    cv2 = lazy.cv2
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if cam_w > 0 and cam_h > 0:
        return cam_w, cam_h

    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Cannot probe source size.")
        cap.release()
        lazy.ndi.destroy()
        sys.exit(1)
    cam_h, cam_w = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cam_w, cam_h


def build_models(args: argparse.Namespace) -> Tuple[Any, Any, Optional[Any], Any]:
    device = lazy.torch.device("cuda:0")
    model_path = resource_path("yolo11n-pose.pt")
    use_half = bool(args.half)
    pose = lazy.PoseDetector(model_path, device, use_half=use_half, conf=args.conf, imgsz=args.imgsz)
    rvm = lazy.RVM(device, half=use_half) if args.matte else None
    hands = lazy.HandsDetector(args.hands, args.hands_max, args.hands_det_conf, args.hands_track_conf, args.hands_complexity)
    return device, pose, rvm, hands


def warmup_cuda_kernels(
    args: argparse.Namespace,
    device: Any,
    pose: Any,
    rvm: Optional[Any],
    cam_w: int,
    cam_h: int,
) -> None:
    torch = lazy.torch
    np = lazy.np

    if getattr(args, "no_warmup", False):
        return
    if getattr(device, "type", None) != "cuda":
        return

    try:
        stream = torch.cuda.Stream(device=device)
        dummy = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        with torch.cuda.stream(stream):
            pose.predict_downscaled(dummy, scale=0.0)
            if rvm is not None:
                rvm.alpha(
                    dummy,
                    downsample=float(args.rvm_down),
                    enforce_stride=True,
                    stride=8,
                    reset_on_resize=True,
                )
        torch.cuda.current_stream(device).wait_stream(stream)
        if rvm is not None:
            rvm.reset_states()
    except Exception as exc:  # pragma: no cover - warm-up is best effort
        print(f"[WARN] CUDA warm-up skipped: {exc}")


def build_sender_or_exit(args: argparse.Namespace, width: int, height: int) -> Any:
    try:
        return lazy.NDISender(args.ndi_name, width, height, args.fpsN, args.fpsD)
    except RuntimeError as exc:
        print("[ERROR]", exc)
        lazy.ndi.destroy()
        sys.exit(1)


def prepare_preview(args: argparse.Namespace, width: int, height: int) -> None:
    if args.no_preview:
        return
    cv2 = lazy.cv2
    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Preview", 450, int(450 * height / width))


def prepare_grading(args: argparse.Namespace, width: int, height: int) -> Tuple[Optional[Dict[str, Any]], Optional[Any]]:
    if not args.grade:
        return None, None
    if args.grade_mode == "auto":
        stats = lazy.compute_bg_stats(args.bg_ref, width, height)
        if not stats.get("ok", False):
            stats = {
                "dark": args.grade_dark,
                "gamma": args.grade_gamma,
                "contrast": args.grade_contrast,
                "tint_bgr": (args.tint_b, args.tint_g, args.tint_r),
                "tint_strength": args.tint_strength,
            }
    else:
        stats = {
            "dark": args.grade_dark,
            "gamma": args.grade_gamma,
            "contrast": args.grade_contrast,
            "tint_bgr": (args.tint_b, args.tint_g, args.tint_r),
            "tint_strength": args.tint_strength,
        }
    lut = lazy.build_gamma_dark_lut(stats["gamma"], stats["dark"])
    stats["contrast"] = float(stats.get("contrast", 1.0))
    return stats, lut
