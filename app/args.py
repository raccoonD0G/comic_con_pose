"""Argument parsing with GUI fallback."""
from __future__ import annotations

import argparse

from ui.gui_startup import get_args_with_gui_fallback


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="", help="Video path (empty for webcam)")
    ap.add_argument("--cam", type=int, default=0, help="Camera index if --src empty")
    ap.add_argument("--w", type=int, default=1080, help="NDI canvas width (if not following camera)")
    ap.add_argument("--h", type=int, default=1920, help="NDI canvas height (if not following camera)")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO image size")
    ap.add_argument("--kp_thr", type=float, default=0.25, help="Keypoint conf threshold")
    ap.add_argument("--seg_thr", type=float, default=0.5, help="Segmentation threshold (reserved)")
    ap.add_argument("--ndi_name", type=str, default="WebcamCutout", help="NDI sender name")
    ap.add_argument("--fpsN", type=int, default=60000, help="NDI frame_rate numerator")
    ap.add_argument("--fpsD", type=int, default=1001, help="NDI frame_rate denominator")
    ap.add_argument("--backend", choices=["ds", "ms", "auto"], default="ds", help="OpenCV backend")
    ap.add_argument("--no-preview", dest="no_preview", action="store_true", default=True, help="Disable local preview")
    ap.add_argument("--show-skel", action="store_true", help="Draw skeleton on preview")
    ap.add_argument("--nearest-only", dest="nearest_only", action="store_true", help="Force ONLY_NEAREST=True")

    ap.add_argument("--center", dest="center", action="store_true", default=False, help="Enable centering")
    ap.add_argument("--no-center", dest="center", action="store_false", help="Disable centering")
    ap.add_argument("--center-method", choices=["bbox", "hips", "nose", "shoulders"], default="shoulders")
    ap.add_argument("--center-smooth", type=float, default=0.8)
    ap.add_argument("--center-deadzone", type=int, default=12)

    ap.add_argument("--target-fps", type=float, default=60)
    ap.add_argument("--pace", choices=["sleep", "drop"], default="drop")
    ap.add_argument("--queue-len", type=int, default=1)
    ap.add_argument("--cap-fps", type=float, default=0.0)
    ap.add_argument("--mjpg", action="store_true")
    ap.add_argument("--horizontal-flip", dest="horizontal_flip", default=True, action="store_true",
                    help="Flip the output horizontally")
    ap.add_argument("--flip-vertical", dest="flip_vertical", action="store_true", default=False,
                    help="Flip the output vertically")

    ap.add_argument("--hands", action="store_true", default=False)
    ap.add_argument("--hands-max", type=int, default=2)
    ap.add_argument("--hands-det-conf", type=float, default=0.5)
    ap.add_argument("--hands-track-conf", type=float, default=0.5)
    ap.add_argument("--hands-complexity", type=int, default=0, choices=[0, 1])

    ap.add_argument("--ndi-follow-camera", dest="ndi_follow_camera", action="store_true", default=False)
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Rotate input CW")
    ap.add_argument("--bbox-clip", dest="bbox_clip", action="store_true", default=False, help="Mask alpha outside expanded bbox")

    ap.add_argument("--bg-ref", type=str, default="Background.png", help="Background image for auto grade")
    ap.add_argument("--grade-mode", choices=["auto", "manual"], default="auto")
    ap.add_argument("--grade-dark", type=float, default=0.82, help="brightness scale")
    ap.add_argument("--grade-gamma", type=float, default=0.95, help="gamma (<1 darker)")
    ap.add_argument("--grade-contrast", type=float, default=1.08, help="contrast strength")
    ap.add_argument("--tint-b", type=int, default=18, help="B channel offset (manual)")
    ap.add_argument("--tint-g", type=int, default=5, help="G channel offset (manual)")
    ap.add_argument("--tint-r", type=int, default=12, help="R channel offset (manual)")
    ap.add_argument("--tint-strength", type=float, default=0.45, help="tint blend (0~1)")

    ap.add_argument("--grade", dest="grade", action="store_true", default=False, help="Enable grading")
    ap.add_argument("--no-grade", dest="grade", action="store_false", help="Disable grading")

    ap.add_argument("--matte", dest="matte", action="store_true", default=False, help="Enable RVM matting")
    ap.add_argument("--no-matte", dest="matte", action="store_false", help="Disable matting (opaque)")

    ap.add_argument("--pose-scale", type=float, default=0.6, help="Pose inference scale (0.3~1.0)")
    ap.add_argument("--pose-every", type=int, default=1, help="Run pose every N frames")
    ap.add_argument("--hands-every", type=int, default=3, help="Run mediapipe hands every N frames")
    ap.add_argument("--rvm-every", type=int, default=1, help="Run RVM alpha every N frames")
    ap.add_argument("--rvm-down", type=float, default=0.25, help="RVM downsample_ratio (smaller=faster)")
    ap.add_argument("--half", action="store_true", default=True, help="Use FP16 on CUDA for pose/RVM")
    ap.add_argument("--hands-roi", dest="hands_roi", action="store_true", default=True, help="Crop hands by expanded person bbox")
    ap.add_argument("--roi-pad-x", type=float, default=0.10, help="Hands ROI pad ratio X")
    ap.add_argument("--roi-pad-y", type=float, default=0.15, help="Hands ROI pad ratio Y")

    return ap


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    return get_args_with_gui_fallback(parser)
