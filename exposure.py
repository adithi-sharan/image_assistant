import numpy as np
from PIL import Image

def exposure_metrics(pil_img: Image.Image):
    """
    Returns (too_dark, too_bright, clip_black, clip_white) in 0..1
    Computed on luminance.
    """
    arr = np.asarray(pil_img).astype(np.float32) / 255.0  # RGB 0..1

    # luminance (Rec. 709)
    y = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]

    clip_black = float((y <= 0.01).mean())
    clip_white = float((y >= 0.99).mean())

    too_dark = float((y < 0.20).mean())
    too_bright = float((y > 0.90).mean())

    return too_dark, too_bright, clip_black, clip_white


def exposure_score(pil_img: Image.Image) -> tuple[float, list[str]]:
    """
    Returns (0..1 score, tags)
    """
    too_dark, too_bright, clip_black, clip_white = exposure_metrics(pil_img)
    tags: list[str] = []

    if clip_white > 0.02:
        tags.append("CLIPPED_HIGHLIGHTS")
    if clip_black > 0.02:
        tags.append("CRUSHED_SHADOWS")
    if too_dark > 0.70:
        tags.append("VERY_DARK")
    if too_bright > 0.70:
        tags.append("VERY_BRIGHT")

    score = 1.0
    score -= min(0.6, clip_white * 8.0)
    score -= min(0.6, clip_black * 8.0)
    score -= min(0.4, max(0.0, too_dark - 0.5))
    score -= min(0.4, max(0.0, too_bright - 0.5))

    score = float(max(0.0, min(1.0, score)))
    return score, tags
