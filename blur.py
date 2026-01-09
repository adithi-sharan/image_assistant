import cv2
import numpy as np
from PIL import Image

def blur_score_laplacian(pil_img: Image.Image) -> float:
    """
    Returns a blur score (higher = sharper) using variance of Laplacian.
    Works best on downsampled previews (like yours).
    """
    # PIL -> OpenCV grayscale
    img = np.array(pil_img)                  # RGB
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(v)

def normalize_blur(v: float, v_min: float = 100.0, v_max: float = 2000.0) -> float:
    """
    Convert raw Laplacian variance to a 0..1 score.
    These bounds are heuristic—tune later.
    """
    x = (v - v_min) / (v_max - v_min)
    return float(max(0.0, min(1.0, x)))
