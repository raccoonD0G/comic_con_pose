from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import cv2, mediapipe as mp, numpy as np
from utils import clip_int as _clip_int

class HandsDetector:
    """MediaPipe Hands wrapper (optional) + ROI crop."""

    def __init__(self, enable: bool, max_hands: int, det_conf: float, track_conf: float, complexity: int):
        self.enabled = enable
        self.mph = None
        if enable:
            self.mph = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=int(max_hands),
                model_complexity=int(complexity),
                min_detection_confidence=float(det_conf),
                min_tracking_confidence=float(track_conf),
            )

    def detect(self, frame_bgr: np.ndarray,
               roi_xyxy: Optional[Tuple[int, int, int, int]] = None) -> List[Dict]:
        if not self.enabled or self.mph is None:
            return []
        h, w = frame_bgr.shape[:2]

        if roi_xyxy is not None:
            x1, y1, x2, y2 = roi_xyxy
            x1 = _clip_int(x1, 0, w - 1); x2 = _clip_int(x2, 0, w - 1)
            y1 = _clip_int(y1, 0, h - 1); y2 = _clip_int(y2, 0, h - 1)
            if x2 <= x1 or y2 <= y1:
                roi = frame_bgr
                rx, ry = 0, 0
            else:
                roi = frame_bgr[y1:y2, x1:x2]
                rx, ry = x1, y1
        else:
            roi = frame_bgr
            rx, ry = 0, 0

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res_h = self.mph.process(rgb)
        out: List[Dict] = []
        if res_h.multi_hand_landmarks:
            R_h, R_w = roi.shape[:2]
            for i, hand_landmarks in enumerate(res_h.multi_hand_landmarks):
                handed = 2
                score = 1.0
                if res_h.multi_handedness and i < len(res_h.multi_handedness):
                    cls = res_h.multi_handedness[i].classification[0]
                    if cls and cls.label:
                        label = cls.label.lower()
                        handed = 0 if label.startswith("right") else (1 if label.startswith("left") else 2)
                    if cls and cls.score is not None:
                        score = float(cls.score)

                xy21 = np.zeros((21, 2), np.float32)
                for j, lm in enumerate(hand_landmarks.landmark):
                    xy21[j, 0] = float(lm.x * R_w + rx)
                    xy21[j, 1] = float(lm.y * R_h + ry)

                centroid = (float(np.nanmean(xy21[:, 0])), float(np.nanmean(xy21[:, 1])))
                out.append({"xy_src": xy21, "score": score, "handed": handed, "centroid_src": centroid})
        return out

    def close(self) -> None:
        if self.mph:
            self.mph.close()
