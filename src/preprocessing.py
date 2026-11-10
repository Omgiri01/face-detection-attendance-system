import cv2
import numpy as np


def preprocess_face(gray_face: np.ndarray, size: int = 200) -> np.ndarray:
    """Resize, denoise lightly, and apply CLAHE for better contrast."""
    # Ensure grayscale
    if len(gray_face.shape) == 3:
        gray_face = cv2.cvtColor(gray_face, cv2.COLOR_BGR2GRAY)

    # Resize to a fixed size
    face = cv2.resize(gray_face, (size, size))

    # Slight blur to reduce sensor noise
    face = cv2.GaussianBlur(face, (3, 3), 0)

    # Adaptive histogram equalization (CLAHE) for lighting robustness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face = clahe.apply(face)

    return face

