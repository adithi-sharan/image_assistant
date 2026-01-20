import cv2
import numpy as np
from PIL import Image

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)

def face_count(pil_img: Image.Image) -> int:
    """
    Returns number of detected faces using Haar cascades.
    Works best on downsampled previews (like 768px long edge).
    """
    arr = np.array(pil_img)  # RGB
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return int(len(faces))
