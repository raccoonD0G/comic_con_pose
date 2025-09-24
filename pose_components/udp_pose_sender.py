from __future__ import annotations
import socket, struct, time
from typing import Dict, List, Optional
import numpy as np
from utils import HEADER_V2_FMT, MAGIC, VERSION, HAND_HEAD_FMT

class UDPPoseSender:
    def __init__(self, ip: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (ip, port)

    def send(self, xy_all: Optional[np.ndarray], conf_all: Optional[np.ndarray], hands_pack: List[Dict]) -> None:
        n_person = int(xy_all.shape[0]) if (xy_all is not None) else 0
        n_hands = int(len(hands_pack)) if hands_pack else 0
        ts_ms = int(time.time() * 1000)

        pkt = bytearray(struct.pack(HEADER_V2_FMT, MAGIC, VERSION, 0, n_person, n_hands, ts_ms))

        # persons
        for pid in range(n_person):
            pkt += struct.pack("<H", pid)
            pts = xy_all[pid].astype(np.float32).reshape(-1)  # 34 floats
            pkt += pts.tobytes(order="C")
            if conf_all is not None:
                conf = conf_all[pid].astype(np.float32).reshape(-1)  # 17 floats
                pkt += conf.tobytes(order="C")
            else:
                pkt += (np.zeros((17,), np.float32)).tobytes(order="C")

        # hands
        for hid, H in enumerate(hands_pack):
            xy21 = H["xy"].astype(np.float32).reshape(-1)
            handed = int(H.get("handed", 2))
            score = float(H.get("score", 1.0))
            pkt += struct.pack(HAND_HEAD_FMT, hid, handed, score)
            pkt += xy21.tobytes(order="C")
            confh = np.full((21,), score, dtype=np.float32)
            pkt += confh.tobytes(order="C")

        self.sock.sendto(pkt, self.dest)
