from __future__ import annotations
import NDIlib as ndi
import numpy as np

class NDISender:
    """Thin wrapper over NDI VideoFrameV2 lifecycle (async if available)."""
    def __init__(self, name: str, w: int, h: int, fpsN: int, fpsD: int):
        self.name = name
        self.vf = ndi.VideoFrameV2()
        self.vf.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA
        self.vf.xres = w
        self.vf.yres = h
        self.vf.frame_rate_N = fpsN
        self.vf.frame_rate_D = fpsD
        self.vf.line_stride_in_bytes = w * 4

        cfg = ndi.SendCreate()
        cfg.ndi_name = name
        cfg.clock_video = False
        cfg.clock_audio = False

        self.sender = ndi.send_create(cfg)
        if self.sender is None:
            raise RuntimeError("NDI sender create failed.")
        print(f"[OK] NDI sender ready: {name}")

        self._async_fn = None
        for fname in ("send_send_video_async_v2", "send_send_video_scatter_async"):
            self._async_fn = getattr(ndi, fname, None)
            if self._async_fn is not None:
                print(f"[OK] Using NDI async: ndi.{fname}()")
                break
        if self._async_fn is None:
            print("[WARN] No NDI async sender found. Using sync send_send_video_v2().")

        self._last_frame_ref = None

    def send_rgba(self, rgba: np.ndarray) -> None:
        self.vf.data = rgba
        if self._async_fn:
            self._last_frame_ref = rgba  # keep alive until next call
            self._async_fn(self.sender, self.vf)
        else:
            ndi.send_send_video_v2(self.sender, self.vf)

    def close(self) -> None:
        ndi.send_destroy(self.sender)
