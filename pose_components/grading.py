from __future__ import annotations
from typing import Dict
import cv2, numpy as np

def compute_bg_stats(path: str, out_w: int, out_h: int) -> Dict:
    """Extract rough HSV stats from a background reference for auto grading."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False}
    img = cv2.resize(img, (out_w, out_h))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w = hsv.shape[:2]
    ys, ye = int(h * 0.25), int(h * 0.75)
    xs, xe = int(w * 0.25), int(w * 0.75)
    crop = hsv[ys:ye, xs:xe]

    mean_v = float(np.mean(crop[..., 2])) / 255.0
    mean_h = float(np.mean(crop[..., 0])) / 180.0
    mean_s = float(np.mean(crop[..., 1])) / 255.0

    neon_center = 0.64
    neon_dist = abs(mean_h - neon_center)
    tint_strength = float(np.clip(0.65 - neon_dist, 0.0, 0.65)) * (0.6 + 0.4 * mean_s)

    dark_scale = np.interp(mean_v, [0.2, 0.6], [0.76, 0.9])
    gamma = np.interp(mean_v, [0.2, 0.6], [0.90, 1.0])
    contrast = np.interp(mean_v, [0.2, 0.6], [1.10, 1.03])

    base_b = int(np.clip(25 + 40 * (0.5 - abs(mean_h - 0.66)) * mean_s, 10, 40))
    base_r = int(np.clip(15 + 30 * (0.5 - abs(mean_h - 0.66)) * mean_s, 5, 35))
    base_g = int(np.clip(8 + 10 * (1.0 - mean_s), 0, 16))

    return {
        "ok": True,
        "dark": float(dark_scale),
        "gamma": float(gamma),
        "contrast": float(contrast),
        "tint_bgr": (base_b, base_g, base_r),
        "tint_strength": float(tint_strength),
    }

def build_gamma_dark_lut(gamma: float, dark: float) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    return np.clip(((x / 255.0) ** max(1e-6, gamma)) * 255.0 * dark, 0, 255).astype(np.uint8)

