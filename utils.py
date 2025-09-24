# utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

# =========================
# Constants & Mappings
# =========================

# 메인이 CLI로 바꿀 수도 있으니 기본값만 둠 (필요시 main에서 override)
ONLY_NEAREST: bool = True

K: Dict[str, int] = {
    "nose": 0, "leye": 1, "reye": 2, "lear": 3, "rear": 4,
    "lsh": 5, "rsh": 6, "lel": 7, "rel": 8, "lwr": 9, "rwr": 10,
    "lhip": 11, "rhip": 12, "lkn": 13, "rkn": 14, "lank": 15, "rank": 16,
}

SKELETON_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

HANDS_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# UDP Header (필요시 메인에서 import해 재사용)
UDP_IP = "127.0.0.1"
UDP_PORT = 7777
HEADER_V2_FMT = "<4sBBHHQ"  # magic(4s), version(B), flags(B), persons(H), hands(H), ts_ms(Q)
MAGIC = b"POSE"
VERSION = 2
HAND_HEAD_FMT = "<HBf"  # hand_id(H), handed(B:0=R,1=L,2=U), score(f)


# =========================
# Cutout
# =========================

def padded_bbox_src(bbox, fw, fh, pad_x=0.10, pad_y=0.10):
    """소스 좌표 bbox에 비율 패딩을 주고 프레임 경계로 클램프."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = map(float, bbox)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = pad_x * bw
    py = pad_y * bh
    ix1 = max(0, int(np.floor(x1 - px)))
    iy1 = max(0, int(np.floor(y1 - py)))
    ix2 = min(fw, int(np.ceil (x2 + px)))
    iy2 = min(fh, int(np.ceil (y2 + py)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return (ix1, iy1, ix2, iy2)

def ema_rect(prev, curr, alpha=0.7):
    """사각형 EMA(박스 점프 줄이기). prev/curr = (x1,y1,x2,y2) or None"""
    if prev is None: return curr
    return tuple(int(round(alpha*p + (1-alpha)*c)) for p, c in zip(prev, curr))

def crop_safe(img, rect):
    x1,y1,x2,y2 = rect
    return img[y1:y2, x1:x2]

def cutout_alpha_inplace(alpha: np.ndarray, bbox_canvas) -> None:
    """
    frame_rgba[...,3] 같은 알파 채널 배열을 사각형(bbox_canvas) 밖 0으로 in-place 처리.
    bbox_canvas: (x1, y1, x2, y2) or None
    """
    h, w = alpha.shape[:2]
    if bbox_canvas is None:
        alpha[:] = 0
        return

    x1f, y1f, x2f, y2f = map(float, bbox_canvas)
    x1 = max(0, int(np.floor(x1f)))
    y1 = max(0, int(np.floor(y1f)))
    x2 = min(w, int(np.ceil (x2f)))
    y2 = min(h, int(np.ceil (y2f)))

    if x2 <= x1 or y2 <= y1:
        alpha[:] = 0
        return

    # 위/아래 띠
    if y1 > 0: alpha[:y1, :] = 0
    if y2 < h: alpha[y2:, :] = 0
    # 좌/우 띠
    if x1 > 0: alpha[y1:y2, :x1] = 0
    if x2 < w: alpha[y1:y2, x2:] = 0

def make_bbox_mask(h: int, w: int, bbox, pad_ratio_x: float = 0.10, pad_ratio_y: float = 0.10) -> np.ndarray:
    if bbox is None:
        return np.full((h, w), 255, dtype=np.uint8)
    x1, y1, x2, y2 = map(float, bbox)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = pad_ratio_x * bw
    py = pad_ratio_y * bh
    ix1 = max(0, int(np.floor(x1 - px)))
    iy1 = max(0, int(np.floor(y1 - py)))
    ix2 = min(w, int(np.ceil (x2 + px)))
    iy2 = min(h, int(np.ceil (y2 + py)))
    m = np.zeros((h, w), dtype=np.uint8)
    if ix2 > ix1 and iy2 > iy1:
        m[iy1:iy2, ix1:ix2] = 255
    return m

def warp_mask_to_canvas(mask_src: np.ndarray, M: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    # 마스크는 최근접 보간 + 단일 채널 유지
    return cv2.warpAffine(mask_src, M[:2, :], (out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=0)

def padded_bbox_src(bbox, fw, fh, pad_x, pad_y):
    """
    YOLO bbox (x1,y1,x2,y2)에 비율 패딩만 적용한 단순 버전.
    pad_x=0.1 이면 가로 폭의 10%, pad_y=0.1 이면 세로 높이의 10% 확장.
    """
    if bbox is None:
        return None

    x1, y1, x2, y2 = map(float, bbox)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    # 비율 패딩
    x1 -= pad_x * bw
    x2 += pad_x * bw
    y1 -= pad_y * bh
    y2 += pad_y * bh

    # 프레임 경계로 클램프
    ix1 = max(0, int(np.floor(x1)))
    iy1 = max(0, int(np.floor(y1)))
    ix2 = min(fw, int(np.ceil(x2)))
    iy2 = min(fh, int(np.ceil(y2)))

    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return (ix1, iy1, ix2, iy2)

# =========================
# Helper / Geometry
# =========================

def fourcc_to_str(v: int) -> str:
    try:
        return "".join([chr((v >> (8 * i)) & 0xFF) for i in range(4)])
    except Exception:
        return "----"

def kps_to_bbox(xy: np.ndarray, conf: Optional[np.ndarray] = None, thr: float = 0.25) -> np.ndarray:
    """keypoints -> tight bbox [x1,y1,x2,y2]"""
    if conf is not None:
        m = (conf >= thr)
        m = m & np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        pts = xy[m] if np.any(m) else xy
    else:
        m = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        pts = xy[m] if np.any(m) else xy
    if pts.size == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
    x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def make_letterbox_affine(src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    s = min(dst_w / float(src_w), dst_h / float(src_h))
    ox = (dst_w - src_w * s) * 0.5
    oy = (dst_h - src_h * s) * 0.5
    return np.array([[s, 0, ox], [0, s, oy]], np.float32)

def apply_affine_xy(xy: np.ndarray, M: np.ndarray) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float32).copy()
    x = arr[..., 0]
    y = arr[..., 1]
    arr[..., 0] = x * M[0, 0] + y * M[0, 1] + M[0, 2]
    arr[..., 1] = x * M[1, 0] + y * M[1, 1] + M[1, 2]
    return arr

def bbox_apply_affine(b: np.ndarray, M: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = b
    pts = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], np.float32)
    pts2 = apply_affine_xy(pts, M)
    x_min = float(np.min(pts2[:, 0])); y_min = float(np.min(pts2[:, 1]))
    x_max = float(np.max(pts2[:, 0])); y_max = float(np.max(pts2[:, 1]))
    return np.array([x_min, y_min, x_max, y_max], np.float32)

def person_center(xy: np.ndarray, conf: Optional[np.ndarray], kp_thr: float, method: str) -> Tuple[float, float]:
    if method == "hips":
        li, ri = K["lhip"], K["rhip"]
        ok_l = np.all(np.isfinite(xy[li])) and (conf is None or conf[li] >= kp_thr)
        ok_r = np.all(np.isfinite(xy[ri])) and (conf is None or conf[ri] >= kp_thr)
        if ok_l and ok_r:
            return float((xy[li, 0] + xy[ri, 0]) * 0.5), float((xy[li, 1] + xy[ri, 1]) * 0.5)

    if method == "nose":
        ni = K["nose"]
        ok_n = np.all(np.isfinite(xy[ni])) and (conf is None or conf[ni] >= kp_thr)
        if ok_n:
            return float(xy[ni, 0]), float(xy[ni, 1])

    if method == "shoulders":
        li, ri = K["lsh"], K["rsh"]
        ok_l = np.all(np.isfinite(xy[li])) and (conf is None or conf[li] >= kp_thr)
        ok_r = np.all(np.isfinite(xy[ri])) and (conf is None or conf[ri] >= kp_thr)
        if ok_l and ok_r:
            return float((xy[li, 0] + xy[ri, 0]) * 0.5), float((xy[li, 1] + xy[ri, 1]) * 0.5)

    # fallback to bbox center
    b = kps_to_bbox(xy, conf, thr=kp_thr)
    return float((b[0] + b[2]) * 0.5), float((b[1] + b[3]) * 0.5)

def clip_int(v: float | int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


# =========================
# Drawing (preview-only)
# =========================

def draw_pose(frame: np.ndarray,
              xy_all: Optional[np.ndarray],
              conf_all: Optional[np.ndarray],
              kp_thr: float,
              draw_names: bool = False) -> None:
    if xy_all is None or len(xy_all) == 0:
        return
    N = xy_all.shape[0]
    # edges
    for i in range(N):
        xy = xy_all[i]
        conf = None if conf_all is None else conf_all[i].reshape(-1)
        for a, b in SKELETON_EDGES:
            if np.any(np.isnan(xy[[a, b]])):
                continue
            if conf is not None:
                if (a < len(conf) and conf[a] < kp_thr) or (b < len(conf) and conf[b] < kp_thr):
                    continue
            p1 = tuple(np.round(xy[a]).astype(int))
            p2 = tuple(np.round(xy[b]).astype(int))
            cv2.line(frame, p1, p2, (0, 255, 255), 2)
    # points
    for i in range(N):
        xy = xy_all[i]
        conf = None if conf_all is None else conf_all[i].reshape(-1)
        for j in range(xy.shape[0]):
            if np.any(np.isnan(xy[j])):
                continue
            if conf is not None and j < len(conf) and conf[j] < kp_thr:
                continue
            p = tuple(np.round(xy[j]).astype(int))
            cv2.circle(frame, p, 4, (255, 255, 255), -1)
            if draw_names:
                cv2.putText(frame, str(j), (p[0] + 5, p[1] - 2),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hands(frame: np.ndarray,
               hands_xy_list: List[np.ndarray],
               hands_conf_list: Optional[List[np.ndarray]] = None,
               thr: float = 0.25) -> None:
    if not hands_xy_list:
        return
    for i, xy in enumerate(hands_xy_list):
        conf = hands_conf_list[i] if (hands_conf_list and i < len(hands_conf_list)) else None
        for a, b in HANDS_EDGES:
            if np.any(np.isnan(xy[[a, b]])):
                continue
            if conf is not None and ((a < len(conf) and conf[a] < thr) or (b < len(conf) and conf[b] < thr)):
                continue
            p1 = tuple(np.round(xy[a]).astype(int))
            p2 = tuple(np.round(xy[b]).astype(int))
            cv2.line(frame, p1, p2, (255, 255, 0), 2)
        for j in range(xy.shape[0]):
            if np.any(np.isnan(xy[j])):
                continue
            if conf is not None and j < len(conf) and conf[j] < thr:
                continue
            p = tuple(np.round(xy[j]).astype(int))
            cv2.circle(frame, p, 3, (255, 255, 255), -1)


__all__ = [
    "ONLY_NEAREST", "K", "SKELETON_EDGES", "HANDS_EDGES",
    "UDP_IP", "UDP_PORT", "HEADER_V2_FMT", "MAGIC", "VERSION", "HAND_HEAD_FMT",
    "fourcc_to_str", "kps_to_bbox", "make_letterbox_affine", "apply_affine_xy",
    "bbox_apply_affine", "person_center", "clip_int",
    "draw_pose", "draw_hands",
]
