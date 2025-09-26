from __future__ import annotations
from typing import Optional, Tuple
import contextlib
import cv2, torch
import numpy as np
from ultralytics import YOLO

class PoseDetector:
    def __init__(self, weights: str, device: torch.device, use_half: bool = False, conf: float = 0.35,
                 imgsz: int = 640):
        self.model = YOLO(weights).to(device)
        self.device = device
        try:
            self.model.fuse()
        except Exception:
            pass
        self.use_half = (use_half and device.type == "cuda")
        if self.use_half:
            try:
                self.model.model.half()
            except Exception:
                self.use_half = False
        self.conf = conf
        self.imgsz = imgsz
        self.dev_arg = 0 if device.type == "cuda" else "cpu"
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def predict_downscaled(self, frame_bgr: np.ndarray, scale: float
                           ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if scale <= 0.0 or scale >= 1.0:
            return self._predict_raw(frame_bgr)

        h, w = frame_bgr.shape[:2]
        dw = max(16, int(w * scale))
        dh = max(16, int(h * scale))
        small = cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_AREA)

        xy_small, conf_small = self._predict_raw(small)
        if xy_small is None:
            return None, None
        sx = w / float(dw)
        sy = h / float(dh)
        xy = xy_small.copy()
        xy[..., 0] *= sx
        xy[..., 1] *= sy
        return xy, conf_small

    def _predict_raw(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        stream = self.stream
        stream_ctx = (
            torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        )
        with stream_ctx:
            res = self.model.predict(
                source=frame_bgr,
                device=self.dev_arg,
                stream=False,
                show=False,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
                half=self.use_half,
            )
        if not res:
            return None, None
        kps = res[0].keypoints
        if kps is None or len(kps) == 0:
            return None, None

        xy_tensor = kps.xy.detach()
        if stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)
        xy_all = xy_tensor.to("cpu", non_blocking=bool(stream)).numpy()  # (N,17,2)
        conf_all = None
        if hasattr(kps, "confidence") and kps.confidence is not None:
            conf_tensor = kps.confidence.detach()
            conf_all = conf_tensor.to("cpu", non_blocking=bool(stream)).numpy()
            if conf_all.ndim == 3 and conf_all.shape[-1] == 1:
                conf_all = conf_all[..., 0]
        return xy_all, conf_all
