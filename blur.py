import cv2
import numpy as np
from PIL import Image

def blur_score_laplacian(pil_img: Image.Image, crop: float = 0.6) -> float:
    """
    Returns a blur score where higher = sharper using variance of Laplacian.
    Works best on downsampled previews.
    Made it crop=0.6 to keep the central 60% in both width/height- better for landscapes    """
    # PIL -> OpenCV grayscale
    img = np.array(pil_img)  # RGB
    h, w = img.shape[:2]
    cw, ch = int(w * crop), int(h * crop)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    center = img[y0:y0+ch, x0:x0+cw]
    gray = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def normalize_blur(v: float, v_min: float = 100.0, v_max: float = 2000.0) -> float:
    """
    Convert raw Laplacian variance to a 0..1 score.
    These bounds are heuristic—tune later.
    """
    x = (v - v_min) / (v_max - v_min)
    return float(max(0.0, min(1.0, x)))
