from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from PIL import Image

import mediapipe as mp

@dataclass(frozen=True)
class FaceDet:
    score: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h in pixels

_detector = None

def _get_detector():
    global _detector
    if _detector is None:
        _detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
    return _detector

def detect_faces(pil_img: Image.Image) -> List[FaceDet]:
    rgb = np.asarray(pil_img)
    det = _get_detector()
    result = det.process(rgb)

    dets: List[FaceDet] = []
    if not result.detections:
        return dets

    h, w = rgb.shape[:2]
    for d in result.detections:
        score = float(d.score[0]) if d.score else 0.0
        rb = d.location_data.relative_bounding_box
        x = max(0, int(rb.xmin * w))
        y = max(0, int(rb.ymin * h))
        bw = max(1, int(rb.width * w))
        bh = max(1, int(rb.height * h))
        dets.append(FaceDet(score=score, bbox=(x, y, bw, bh)))
    return dets

def face_count(pil_img: Image.Image, min_score: float = 0.6) -> int:
    return sum(1 for d in detect_faces(pil_img) if d.score >= min_score)
