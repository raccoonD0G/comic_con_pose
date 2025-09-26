from __future__ import annotations

import contextlib
import cv2
import torch
import numpy as np


def pad_to_multiple(img: np.ndarray, m: int = 8) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """오른쪽/아래로만 패딩해서 (H,W)를 m의 배수로 맞춘다. pad=(top,bottom,left,right)"""
    h, w = img.shape[:2]
    H = ((h + m - 1) // m) * m
    W = ((w + m - 1) // m) * m
    if H == h and W == w:
        return img, (0, 0, 0, 0)
    pad_b, pad_r = H - h, W - w
    img2 = cv2.copyMakeBorder(img, 0, pad_b, 0, pad_r, cv2.BORDER_REPLICATE)
    return img2, (0, pad_b, 0, pad_r)


class RVM:
    """Robust Video Matting wrapper (torch hub) with FP16 + safe state handling."""

    def __init__(self, device: torch.device, half: bool = False):
        self.model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").to(device).eval()
        self.device = device
        self.r1 = self.r2 = self.r3 = self.r4 = None
        self.use_half = (half and device.type == "cuda")
        if self.use_half:
            self.model.half()
        # 마지막 입력 텐서 크기(패딩 적용 후 기준)
        self._last_hw: tuple[int, int] | None = None
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def reset_states(self) -> None:
        self.r1 = self.r2 = self.r3 = self.r4 = None
        self._last_hw = None

    def _ensure_state_for(self, h: int, w: int, reset_on_resize: bool) -> None:
        if reset_on_resize and self._last_hw != (h, w):
            self.reset_states()
            self._last_hw = (h, w)

    @torch.no_grad()
    def alpha(
        self,
        bgr: np.ndarray,
        downsample: float = 0.25,
        *,
        enforce_stride: bool = True,  # True면 8배수 패딩
        stride: int = 8,
        reset_on_resize: bool = True,  # 크기 변하면 state 리셋
    ) -> np.ndarray:
        """
        반환: uint8 알파 (H,W), [0..255]
        ROI 크기가 프레임마다 달라져도 안전하게 동작.
        """
        # 0) (선택) stride 배수로 패딩
        src = bgr
        pad = (0, 0, 0, 0)
        if enforce_stride:
            src, pad = pad_to_multiple(bgr, m=stride)

        H, W = src.shape[:2]
        self._ensure_state_for(H, W, reset_on_resize)

        stream = self.stream
        stream_ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        with stream_ctx:
            # 1) RGB float[0,1] 텐서
            src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(src_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
            t = t.to(torch.float16 if self.use_half else torch.float32).div_(255.0)

            # 2) 추론
            _, pha, self.r1, self.r2, self.r3, self.r4 = self.model(
                t, self.r1, self.r2, self.r3, self.r4, downsample_ratio=float(downsample)
            )

            alpha_pad_tensor = pha[0, 0].clamp_(0, 1).mul_(255).byte()

        if stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)
        alpha_pad = alpha_pad_tensor.to("cpu", non_blocking=bool(stream)).numpy()

        # 4) 패딩을 되돌려 원래 크기로 크롭
        if pad != (0, 0, 0, 0):
            h, w = bgr.shape[:2]
            alpha = alpha_pad[:h, :w]
        else:
            alpha = alpha_pad
        return alpha
