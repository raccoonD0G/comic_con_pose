# -*- coding: utf-8 -*-
"""
NDI RGBA sender with Pose+Hands UDP and optional RVM matting.

Refactor + Optimize
- Keep original features & CLI, but organize into small, testable units
- Strong typing, docstrings, clear resource lifecycles
- Minimal per-frame allocations, fast paths
- Pose/Hands/RVM frame sampling + downscaled inference + ROI cropping + FP16 + caching

Requirements
- OpenCV, NumPy, Torch, ultralytics, NDIlib, MediaPipe

Author: refactor+opt by ChatGPT
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import threading
import traceback
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch

import NDIlib as ndi
import mediapipe as mp
from ultralytics import YOLO

import os

from time import perf_counter

class Profiler:
    __slots__ = ("t0","marks","frame_ms_ema","fps_ema","enabled")
    def __init__(self, enabled=True):
        self.t0 = perf_counter()
        self.marks = {}
        self.enabled = enabled
        self.frame_ms_ema = None
        self.fps_ema = None

    def tick(self):
        """현재 시각 기준으로 초기화"""
        self.t0 = perf_counter()

    def mark(self, key):
        """구간 경과(ms)를 기록"""
        if not self.enabled: return
        now = perf_counter()
        ms = (now - self.t0) * 1000.0
        self.marks[key] = self.marks.get(key, 0.0) + ms
        self.t0 = now

    def end_frame(self, frame_idx, alpha=0.1, n=60):
        """프레임 종료 시 평균 갱신 & 로그 출력"""
        total_ms = sum(self.marks.values())
        # EMA 업데이트
        self.frame_ms_ema = total_ms if self.frame_ms_ema is None else (1-alpha)*self.frame_ms_ema + alpha*total_ms
        fps = 1000.0 / max(total_ms, 1e-6)
        self.fps_ema = fps if self.fps_ema is None else (1-alpha)*self.fps_ema + alpha*fps
        # 로그 주기
        if frame_idx % n == 0:
            parts = " | ".join(f"{k}:{v:.1f}ms" for k,v in self.marks.items())
            print(f"[PROF] frame={frame_idx} total={total_ms:.1f}ms | {parts} | ema={self.frame_ms_ema:.1f}ms | fps≈{self.fps_ema:.1f}")
            self.marks.clear()

# ---------------------------
# Runtime perf hints (CUDA)
# ---------------------------
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass


def resource_path(relative_path):
    """ PyInstaller 실행 또는 개발 환경에서 리소스 파일 경로 얻기 """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 임시 폴더에서 실행 중
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# =========================
# Utils
# =========================

from utils import (
    ONLY_NEAREST, K, SKELETON_EDGES, HANDS_EDGES,
    UDP_IP, UDP_PORT, HEADER_V2_FMT, MAGIC, VERSION, HAND_HEAD_FMT,
    fourcc_to_str, kps_to_bbox, make_letterbox_affine, apply_affine_xy,
    bbox_apply_affine, person_center, clip_int,
    draw_pose, draw_hands, make_bbox_mask, warp_mask_to_canvas, cutout_alpha_inplace, padded_bbox_src, ema_rect, crop_safe, padded_bbox_src
)
from ui.gui_startup import (
    SettingsForm,
    defaults_from_schema,
    get_args_with_gui_fallback,
    namespace_from_dict,
)

# =========================
# Components
# =========================

from pose_components import (
    CameraCapture, CenterTracker, UDPPoseSender, NDISender,
    RVM, PoseDetector, HandsDetector,
    compute_bg_stats, build_gamma_dark_lut,
)

# =========================
# Argument Parsing (GUI fallback)
# =========================
def parse_args() -> argparse.Namespace:
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
    ap.add_argument("--mirror", default=True, action="store_true")

    ap.add_argument("--hands", action="store_true", default=False)
    ap.add_argument("--hands-max", type=int, default=2)
    ap.add_argument("--hands-det-conf", type=float, default=0.5)
    ap.add_argument("--hands-track-conf", type=float, default=0.5)
    ap.add_argument("--hands-complexity", type=int, default=0, choices=[0, 1])

    ap.add_argument("--ndi-follow-camera", dest="ndi_follow_camera", action="store_true", default=False)
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Rotate input CW")
    ap.add_argument("--bbox-clip", dest="bbox_clip", action="store_true", default=False,
                    help="Mask alpha outside expanded bbox")

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
    ap.add_argument("--hands-roi", dest="hands_roi", action="store_true", default=True,
                    help="Crop hands by expanded person bbox")
    ap.add_argument("--roi-pad-x", type=float, default=0.10, help="Hands ROI pad ratio X")
    ap.add_argument("--roi-pad-y", type=float, default=0.15, help="Hands ROI pad ratio Y")

    # 핵심: GUI fallback 호출
    return get_args_with_gui_fallback(ap)

# =========================
# Main processing
# =========================

@dataclass
class RunContext:
    args: argparse.Namespace
    OUT_W: int
    OUT_H: int
    KP_THR: float
    use_nearest: bool

    # runtime objects
    cap: cv2.VideoCapture
    cam_mgr: CameraCapture
    center: CenterTracker
    device: torch.device
    pose: PoseDetector
    rvm: Optional[RVM]
    hands: HandsDetector
    sender: NDISender
    udp: UDPPoseSender

    # grading
    stats: Optional[Dict]
    lut_gamma_dark: Optional[np.ndarray]
    stop_event: threading.Event = field(default_factory=threading.Event)

@dataclass
class Caches:
    last_xy_all_src: Optional[np.ndarray] = None
    last_conf_all: Optional[np.ndarray] = None
    last_bbox_src_debug: Optional[np.ndarray] = None
    last_hands_src: List[Dict] = None
    last_alpha_src: Optional[np.ndarray] = None
    frames: int = 0
    last_log: float = time.time()

# =========================
# Setup helpers
# =========================
def init_ndi_or_exit() -> None:
    if not ndi.initialize():
        print("[ERROR] NDI initialize failed. Install NDI Runtime/Tools and retry.")
        sys.exit(1)

def open_capture(args: argparse.Namespace, W: int, H: int) -> cv2.VideoCapture:
    if args.src.strip():
        cap = cv2.VideoCapture(args.src, cv2.CAP_FFMPEG)
    else:
        be = cv2.CAP_DSHOW if args.backend == "ds" else (cv2.CAP_MSMF if args.backend == "ms" else 0)
        cap = cv2.VideoCapture(args.cam, be)
        if not getattr(args, "ndi_follow_camera", False):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        if args.mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if args.cap_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, args.cap_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERROR] Cannot open source.")
        ndi.destroy()
        sys.exit(1)
    return cap

def probe_source_size_or_exit(cap: cv2.VideoCapture) -> Tuple[int, int]:
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if cam_w <= 0 or cam_h <= 0:
        ok, tmp = cap.read()
        if not ok:
            print("[ERROR] Cannot probe source size.")
            cap.release()
            ndi.destroy()
            sys.exit(1)
        cam_h, cam_w = tmp.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cam_w, cam_h

def build_models(args: argparse.Namespace) -> Tuple[torch.device, PoseDetector, Optional[RVM], HandsDetector]:
    device = torch.device("cuda:0")
    model_path = resource_path("yolo11n-pose.pt")
    use_half = bool(args.half)
    pose = PoseDetector(model_path, device, use_half=use_half, conf=args.conf, imgsz=args.imgsz)
    rvm = RVM(device, half=use_half) if args.matte else None
    hands = HandsDetector(args.hands, args.hands_max, args.hands_det_conf, args.hands_track_conf,
                          args.hands_complexity)
    return device, pose, rvm, hands

def build_sender_or_exit(args: argparse.Namespace, OUT_W: int, OUT_H: int) -> NDISender:
    try:
        return NDISender(args.ndi_name, OUT_W, OUT_H, args.fpsN, args.fpsD)
    except RuntimeError as e:
        print("[ERROR]", e)
        ndi.destroy()
        sys.exit(1)

def prepare_preview(args: argparse.Namespace, OUT_W: int, OUT_H: int) -> None:
    if not args.no_preview:
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preview", 450, int(450 * OUT_H / OUT_W))

def prepare_grading(args: argparse.Namespace, OUT_W: int, OUT_H: int) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
    if not args.grade:
        return None, None
    if args.grade_mode == "auto":
        stats = compute_bg_stats(args.bg_ref, OUT_W, OUT_H)
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
    lut = build_gamma_dark_lut(stats["gamma"], stats["dark"])
    return stats, lut

def build_context(args: argparse.Namespace, stop_event: Optional[threading.Event] = None) -> RunContext:
    W, H = args.w, args.h
    KP_THR = float(np.clip(args.kp_thr, 0.0, 1.0))
    use_nearest = args.nearest_only or ONLY_NEAREST

    init_ndi_or_exit()
    cap = open_capture(args, W, H)
    cam_w, cam_h = probe_source_size_or_exit(cap)

    OUT_W, OUT_H = (cam_w, cam_h) if args.ndi_follow_camera else (W, H)
    center = CenterTracker(OUT_W, OUT_H, args.center_smooth, args.center_deadzone)

    device, pose, rvm, hands = build_models(args)
    sender = build_sender_or_exit(args, OUT_W, OUT_H)
    prepare_preview(args, OUT_W, OUT_H)

    cam_mgr = CameraCapture(args, cap); cam_mgr.start()
    udp = UDPPoseSender(UDP_IP, UDP_PORT)
    stats, lut = prepare_grading(args, OUT_W, OUT_H)

    return RunContext(
        args=args, OUT_W=OUT_W, OUT_H=OUT_H, KP_THR=KP_THR, use_nearest=use_nearest,
        cap=cap, cam_mgr=cam_mgr, center=center, device=device, pose=pose, rvm=rvm,
        hands=hands, sender=sender, udp=udp, stats=stats, lut_gamma_dark=lut,
        stop_event=stop_event or threading.Event()
    )

# =========================
# Per-frame helpers
# =========================
def rotate_if_needed(frame_bgr: np.ndarray, rot: int) -> np.ndarray:
    rot = (rot or 0) % 360
    if rot == 90:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame_bgr

def run_pose_every(ctx: RunContext, caches: Caches, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    xy_all_src = conf_all = None
    if (caches.frames % max(1, ctx.args.pose_every)) == 0:
        xy_all_src, conf_all = ctx.pose.predict_downscaled(frame_bgr, scale=float(ctx.args.pose_scale))
        if xy_all_src is not None:
            caches.last_xy_all_src = xy_all_src
            caches.last_conf_all = conf_all
    else:
        xy_all_src = caches.last_xy_all_src
        conf_all = caches.last_conf_all

    bbox_src_debug = None
    if xy_all_src is not None:
        if (ctx.use_nearest and xy_all_src.shape[0] > 1):
            bboxes = [kps_to_bbox(xy_all_src[i], conf_all[i] if conf_all is not None else None, thr=ctx.KP_THR)
                      for i in range(xy_all_src.shape[0])]
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
            idx = int(np.argmax(areas))
            xy_use_src = xy_all_src[idx:idx + 1]
            conf_use = conf_all[idx:idx + 1] if conf_all is not None else None
            bbox_src_debug = bboxes[idx].astype(np.float32)
        else:
            xy_use_src = xy_all_src
            conf_use = conf_all
            if xy_all_src.shape[0] == 1:
                bbox_src_debug = kps_to_bbox(
                    xy_all_src[0], conf_all[0] if conf_all is not None else None, thr=ctx.KP_THR
                ).astype(np.float32)
    else:
        xy_use_src = None
        conf_use = None

    if bbox_src_debug is not None:
        caches.last_bbox_src_debug = bbox_src_debug
    else:
        bbox_src_debug = caches.last_bbox_src_debug

    return xy_use_src, conf_use, bbox_src_debug

def run_hands_every(ctx: RunContext, caches: Caches, frame_bgr: np.ndarray, bbox_src_debug: Optional[np.ndarray]) -> List[Dict]:
    if not ctx.args.hands:
        return []
    if ((caches.frames % max(1, ctx.args.hands_every)) == 0) or (caches.last_hands_src is None):
        roi_xyxy = None
        if ctx.args.hands_roi and (bbox_src_debug is not None):
            x1, y1, x2, y2 = bbox_src_debug.tolist()
            pad_x = (x2 - x1) * float(ctx.args.roi_pad_x)
            pad_y = (y2 - y1) * float(ctx.args.roi_pad_y)
            roi_xyxy = (int(x1 - pad_x), int(y1 - pad_y), int(x2 + pad_x), int(y2 + pad_y))
        hands_src = ctx.hands.detect(frame_bgr, roi_xyxy=roi_xyxy)
        caches.last_hands_src = hands_src
        return hands_src
    return caches.last_hands_src

def run_matting_every(ctx: RunContext, caches: Caches, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    if not (ctx.args.matte and (ctx.rvm is not None)):
        return None
    if ((caches.frames % max(1, ctx.args.rvm_every)) == 0) or (caches.last_alpha_src is None):
        alpha_src = ctx.rvm.alpha(frame_bgr, downsample=float(ctx.args.rvm_down))
        caches.last_alpha_src = alpha_src
        return alpha_src
    return caches.last_alpha_src

# 고정 입력 크기(8의 배수): 세로 기준 384~512, 가로는 종횡비 맞춰서 택
RVM_IN_H, RVM_IN_W = 384, 256  # (예시) 8의 배수

def run_rvm_on_roi_or_full(ctx, caches, frame_bgr, bbox_src, fw, fh):
    use_cutout = bool(getattr(ctx.args, "bbox_cutout", False))
    matte_on   = bool(getattr(ctx.args, "matte", False))
    if not matte_on:
        return None

    # ROI가 없으면 전체 프레임(이 경우만 state/size 고정 이점이 적음)
    if bbox_src is None or not use_cutout:
        return run_matting_every(ctx, caches, frame_bgr)

    # 1) 패딩+EMA로 ROI 안정화
    rect = padded_bbox_src(
        bbox_src, fw, fh,
        pad_x=getattr(ctx.args, "bbox_pad_x", 0.15),  # 조금 여유
        pad_y=getattr(ctx.args, "bbox_pad_y", 0.20),
    )
    if rect is None:
        return run_matting_every(ctx, caches, frame_bgr)

    last_rect = getattr(caches, "last_roi_rect", None)
    rect = ema_rect(last_rect, rect, alpha=0.7)
    rect = snap_rect_to_grid(rect, grid=8, fw=fw, fh=fh)  # 좌표 8px 격자 정렬(선택)
    caches.last_roi_rect = rect

    # 2) 크롭 후 '고정 크기'로 리사이즈해서 RVM
    x1,y1,x2,y2 = rect
    roi_bgr = frame_bgr[y1:y2, x1:x2]
    roi_h, roi_w = roi_bgr.shape[:2]
    if roi_h <= 0 or roi_w <= 0:
        return np.zeros((fh, fw), np.uint8)

    roi_bgr_in = cv2.resize(roi_bgr, (RVM_IN_W, RVM_IN_H), interpolation=cv2.INTER_LINEAR)

    # 고정 크기 입력이므로 wrapper는 reset_on_resize=False, enforce_stride=False 권장
    roi_alpha_in = ctx.rvm.alpha(
        roi_bgr_in, downsample=float(ctx.args.rvm_down),
        enforce_stride=False, reset_on_resize=False
    )  # (RVM_IN_H, RVM_IN_W)

    # 3) 알파를 ROI 원크기로 되돌리고 소스 크기에 붙여넣기
    roi_alpha = cv2.resize(roi_alpha_in, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
    alpha_src = np.zeros((fh, fw), dtype=np.uint8)
    alpha_src[y1:y2, x1:x2] = roi_alpha
    return alpha_src

def snap_rect_to_grid(rect, grid, fw, fh):
    """(x1,y1,x2,y2)를 grid 배수로 스냅(프레임 경계 클램프)."""
    x1,y1,x2,y2 = rect
    x1 = max(0, (x1 // grid) * grid)
    y1 = max(0, (y1 // grid) * grid)
    x2 = min(fw, ((x2 + grid - 1) // grid) * grid)
    y2 = min(fh, ((y2 + grid - 1) // grid) * grid)
    if x2 <= x1: x2 = min(fw, x1 + grid)
    if y2 <= y1: y2 = min(fh, y1 + grid)
    return (x1,y1,x2,y2)

def compute_center_and_transform(ctx: RunContext, xy_use_src: Optional[np.ndarray], conf_use: Optional[np.ndarray],
                                 fw: int, fh: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    M_base = make_letterbox_affine(fw, fh, ctx.OUT_W, ctx.OUT_H)
    if ctx.args.center and (xy_use_src is not None) and (xy_use_src.shape[0] >= 1):
        xy0_canvas = apply_affine_xy(xy_use_src[0], M_base)
        conf0 = (conf_use[0] if (conf_use is not None) else None)
        cx, cy = person_center(xy0_canvas, conf0, kp_thr=ctx.KP_THR, method=ctx.args.center_method)
        ctx.center.update(cx, cy)
    ndx, ndy = ctx.center.offset() if ctx.args.center else (0.0, 0.0)
    M = M_base.copy()
    M[0, 2] += ndx
    M[1, 2] += ndy
    return M_base, M, ndx, ndy

def apply_matte_and_grade(ctx: RunContext, frame_bgr: np.ndarray, M: np.ndarray,
                          bbox_canvas: Optional[np.ndarray], alpha_src: Optional[np.ndarray]) -> np.ndarray:
    frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    frame_rgba = cv2.warpAffine(
        frame_rgba, M, (ctx.OUT_W, ctx.OUT_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0 if ctx.args.matte else 255),
    )

    if ctx.args.matte and alpha_src is not None:
        alpha_canvas = cv2.warpAffine(
            alpha_src, M, (ctx.OUT_W, ctx.OUT_H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        if ctx.args.bbox_clip and (bbox_canvas is not None):
            x1, y1, x2, y2 = bbox_canvas.astype(int)
            mx = int(0.10 * (x2 - x1 + 1)); my = int(0.15 * (y2 - y1 + 1))
            x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
            x2 = min(ctx.OUT_W - 1, x2 + mx); y2 = min(ctx.OUT_H - 1, y2 + my)
            clip = np.zeros((ctx.OUT_H, ctx.OUT_W), dtype=np.uint8)
            clip[y1: y2 + 1, x1: x2 + 1] = 255
            alpha_canvas = cv2.min(alpha_canvas, clip)
        kernel = np.ones((3, 3), dtype=np.uint8)
        alpha_canvas = cv2.morphologyEx(alpha_canvas, cv2.MORPH_OPEN, kernel, iterations=1)
        alpha_canvas = cv2.morphologyEx(alpha_canvas, cv2.MORPH_DILATE, kernel, iterations=1)
        frame_rgba[..., 3] = alpha_canvas
    else:
        frame_rgba[..., 3] = 255

    if ctx.args.grade and (ctx.stats is not None) and (ctx.lut_gamma_dark is not None):
        alpha = frame_rgba[..., 3]
        if np.any(alpha > 8):
            rgb = frame_rgba[..., :3]
            graded = cv2.LUT(rgb, ctx.lut_gamma_dark)
            contrast = float(ctx.stats["contrast"])
            if abs(contrast - 1.0) > 1e-3:
                graded = cv2.addWeighted(graded, contrast, np.full_like(graded, 128), 1.0 - contrast,
                                         128 * (1.0 - contrast))
            tb, tg, tr = [int(x) for x in ctx.stats["tint_bgr"]]
            tint_strength = float(ctx.stats["tint_strength"])
            if tint_strength > 1e-4:
                tinted = cv2.add(graded, np.array([tb, tg, tr], dtype=np.uint8))
                graded = cv2.addWeighted(graded, 1.0 - tint_strength, tinted, tint_strength, 0.0)
            mask = alpha.astype(np.uint8)
            fg = cv2.bitwise_and(graded, graded, mask=mask)
            inv = cv2.bitwise_not(mask)
            bg = cv2.bitwise_and(rgb, rgb, mask=inv)
            rgb[:] = cv2.add(bg, fg)
    return frame_rgba

def prepare_coordinates_and_hands(ctx: RunContext, M: np.ndarray, xy_use_src, conf_use,
                                  hands_src, bbox_canvas) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Dict]]:
    xy_send = None
    conf_send = None
    if xy_use_src is not None:
        xy_send = apply_affine_xy(xy_use_src, M)
        conf_send = conf_use

    hands_send: List[Dict] = []
    if hands_src:
        def inside_expanded_bbox_canvas(centroid_canvas: Iterable[float], bbox: np.ndarray, mx_ratio=0.10, my_ratio=0.15) -> bool:
            if bbox is None:
                return True
            x1, y1, x2, y2 = bbox
            mx = int(mx_ratio * (x2 - x1 + 1)); my = int(my_ratio * (y2 - y1 + 1))
            x1 = max(0, int(x1) - mx); y1 = max(0, int(y1) - my)
            x2 = min(ctx.OUT_W - 1, int(x2) + mx); y2 = min(ctx.OUT_H - 1, int(y2) + my)
            cx, cy = centroid_canvas
            return (x1 <= cx <= x2) and (y1 <= cy <= y2)

        for Hs in hands_src:
            xy_src = Hs["xy_src"]
            centroid_src = np.array(Hs["centroid_src"], np.float32)
            xy_canvas = apply_affine_xy(xy_src, M)
            centroid_canvas = apply_affine_xy(centroid_src, M)
            if ctx.use_nearest and (bbox_canvas is not None):
                if not inside_expanded_bbox_canvas(centroid_canvas, bbox_canvas):
                    continue
            hands_send.append({"xy": xy_canvas, "score": float(Hs["score"]), "handed": int(Hs["handed"])})
    return xy_send, conf_send, hands_send

def mirror_if_needed(ctx: RunContext, frame_rgba: np.ndarray, xy_send, hands_send, bbox_canvas):
    if not ctx.args.mirror:
        return frame_rgba, xy_send, hands_send, bbox_canvas
    frame_rgba = cv2.flip(frame_rgba, 1)
    if xy_send is not None:
        xy_send[..., 0] = (ctx.OUT_W - 1) - xy_send[..., 0]
    for Hs in hands_send:
        Hs["xy"][..., 0] = (ctx.OUT_W - 1) - Hs["xy"][..., 0]
        if Hs.get("handed") in (0, 1):
            Hs["handed"] = 1 - int(Hs["handed"])  # swap L/R
    if bbox_canvas is not None:
        x1, y1, x2, y2 = bbox_canvas
        bbox_canvas = np.array([(ctx.OUT_W - 1) - x2, y1, (ctx.OUT_W - 1) - x1, y2], np.float32)
    return frame_rgba, xy_send, hands_send, bbox_canvas

def show_preview(ctx: RunContext, frame_rgba: np.ndarray, xy_send, conf_send, bbox_canvas) -> bool:
    if ctx.args.no_preview:
        return False
    preview_bgra = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGRA)
    alpha_ch = preview_bgra[..., 3]
    fg_bgr = preview_bgra[..., 0:3]
    bg = np.full((ctx.OUT_H, ctx.OUT_W, 3), (0, 255, 0), dtype=np.uint8)
    composite = bg.copy()
    cv2.copyTo(fg_bgr, alpha_ch, composite)

    if ctx.args.show_skel:
        if xy_send is not None:
            draw_pose(composite, xy_send, conf_send, ctx.KP_THR, draw_names=False)
            if bbox_canvas is not None:
                x1, y1, x2, y2 = bbox_canvas.astype(int)
                x1 = np.clip(x1, 0, ctx.OUT_W - 1); x2 = np.clip(x2, 0, ctx.OUT_W - 1)
                y1 = np.clip(y1, 0, ctx.OUT_H - 1); y2 = np.clip(y2, 0, ctx.OUT_H - 1)
                cv2.rectangle(composite, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(composite, "NEAREST", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
        # hands overlay
        # (draw_hands는 hands_send를 직접 받지 않으므로 필요 시 위에서 변환)
    cv2.imshow("Preview", composite)
    return (cv2.waitKey(1) & 0xFF) == 27  # ESC pressed

def log_fps(caches: Caches) -> None:
    now = time.time()
    if now - caches.last_log >= 1.0:
        print(f"[FPS] {caches.frames} fps")
        caches.frames = 0
        caches.last_log = now

# =========================
# Main loop
# =========================
def run_loop(ctx: RunContext) -> None:
    caches = Caches(last_hands_src=[])
    prof = Profiler(enabled=True)

    print("[INFO] Running... (ESC to quit)")
    try:
        while not ctx.stop_event.is_set():
            prof.tick()  # 프레임 시작

            if ctx.stop_event.is_set():
                break

            frame_bgr = ctx.cam_mgr.read()
            if frame_bgr is None:
                if ctx.stop_event.is_set():
                    break
                if not ctx.args.no_preview:
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                if not ctx.cam_mgr.use_webcam:
                    break
                continue

            frame_bgr = rotate_if_needed(frame_bgr, ctx.args.rotate)
            fh, fw = frame_bgr.shape[:2]
            caches.frames += 1

            # 1) Pose/bbox
            xy_use_src, conf_use, bbox_src_debug = run_pose_every(ctx, caches, frame_bgr)
            prof.mark("pose")

            # 1.5) Hands
            hands_src = run_hands_every(ctx, caches, frame_bgr, bbox_src_debug)
            prof.mark("hands")

            # 2) Matting
            alpha_src = run_rvm_on_roi_or_full(ctx, caches, frame_bgr, bbox_src_debug, fw, fh)
            prof.mark("rvm")

            # 3) Transform
            M_base, M, _, _ = compute_center_and_transform(ctx, xy_use_src, conf_use, fw, fh)
            prof.mark("affine")

            # 4) 박스
            use_cutout = bool(getattr(ctx.args, "bbox_cutout", False))
            rect_src_for_cut = None
            rect_canvas_for_cut = None
            if use_cutout:
                rect_src_for_cut = padded_bbox_src(
                    bbox_src_debug, fw, fh,
                    pad_x=getattr(ctx.args, "bbox_pad_x", 0.10),
                    pad_y=getattr(ctx.args, "bbox_pad_y", 0.10),
                )
                rect_canvas_for_cut = (
                    bbox_apply_affine(rect_src_for_cut, M) if rect_src_for_cut else None
                )
            bbox_canvas = (rect_canvas_for_cut if use_cutout
                           else (bbox_apply_affine(bbox_src_debug, M) if bbox_src_debug is not None else None))
            prof.mark("bbox")

            # 5) RGBA
            frame_rgba = apply_matte_and_grade(ctx, frame_bgr, M, bbox_canvas, alpha_src)
            prof.mark("matte")

            # 6) 컷아웃
            if use_cutout:
                cutout_alpha_inplace(frame_rgba[..., 3], rect_canvas_for_cut)
            prof.mark("cutout")

            # 7) 좌표/손 데이터
            xy_send, conf_send, hands_send = prepare_coordinates_and_hands(
                ctx, M, xy_use_src, conf_use, hands_src, bbox_canvas
            )
            prof.mark("prep")

            # 8) Mirror
            frame_rgba, xy_send, hands_send, bbox_canvas = mirror_if_needed(
                ctx, frame_rgba, xy_send, hands_send, bbox_canvas
            )
            prof.mark("mirror")

            # 9) UDP + NDI
            fr = frame_rgba if frame_rgba.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame_rgba)
            ctx.udp.send(xy_send, conf_send, hands_send)
            ctx.sender.send_rgba(fr)
            prof.mark("io")

            # 10) Preview
            if ctx.stop_event.is_set():
                break

            if show_preview(ctx, frame_rgba, xy_send, conf_send, bbox_canvas):
                break
            prof.mark("preview")

            # 11) FPS log
            log_fps(caches)
            prof.end_frame(caches.frames)  # 주기적으로 출력


    except KeyboardInterrupt:
        pass


def cleanup(ctx: RunContext) -> None:
    ctx.stop_event.set()
    ctx.cam_mgr.stop()
    ctx.cap.release()
    if not ctx.args.no_preview:
        cv2.destroyAllWindows()
    ctx.sender.close()
    ndi.destroy()
    ctx.hands.close()
    print("[INFO] Finished.")


class PipelineRunner:
    """Background thread runner for the capture/render pipeline."""

    def __init__(self, app: "ControlPanelApp") -> None:
        self.app = app
        self.thread: Optional[threading.Thread] = None
        self.ctx: Optional[RunContext] = None
        self.stop_event: Optional[threading.Event] = None
        self._lock = threading.Lock()

    def start(self, args: argparse.Namespace) -> None:
        with self._lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("Pipeline already running")
            stop_event = threading.Event()
            self.stop_event = stop_event

            def worker() -> None:
                ctx: Optional[RunContext] = None
                try:
                    ctx = build_context(args, stop_event=stop_event)
                    with self._lock:
                        self.ctx = ctx
                    self.app.on_pipeline_running_threadsafe()
                    run_loop(ctx)
                except Exception as exc:  # pragma: no cover - runtime failure path
                    self.app.on_pipeline_error_threadsafe(exc)
                finally:
                    if ctx is not None:
                        try:
                            cleanup(ctx)
                        except Exception as exc:  # pragma: no cover - cleanup failure
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

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Thread-safe notifications from PipelineRunner
    # ------------------------------------------------------------------
    def queue_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))

    def on_pipeline_running_threadsafe(self) -> None:
        self.root.after(0, lambda: self.queue_status("Running"))

    def on_pipeline_error_threadsafe(self, exc: Exception) -> None:
        tb = ''.join(traceback.format_exception(exc.__class__, exc, exc.__traceback__))

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


# =========================
# Public entry
# =========================

def run_pipeline(args: argparse.Namespace) -> None:
    stop_event = threading.Event()
    ctx = build_context(args, stop_event=stop_event)
    try:
        run_loop(ctx)
    finally:
        cleanup(ctx)


def main() -> None:
    if len(sys.argv) > 1:
        args = parse_args()
        run_pipeline(args)
        return

    try:
        app = ControlPanelApp()
    except RuntimeError as exc:
        print(f"[WARN] {exc}. Falling back to CLI mode.", file=sys.stderr)
        args = parse_args()
        run_pipeline(args)
        return

    app.run()


if __name__ == "__main__":
    main()
