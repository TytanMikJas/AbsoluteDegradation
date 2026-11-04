"""
Handles casting operations between different image formats.
Includes conversions between PIL and OpenCV formats.
"""

import cv2
import numpy as np


def rgb2gray(img_np: np.ndarray) -> np.ndarray:
    """Converts a RGB numpy image to Grayscale."""
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)


def gray2rgb(img_np: np.ndarray) -> np.ndarray:
    """Converts a Grayscale numpy image to RGB."""
    if img_np.ndim != 2:
        raise ValueError("Input image must be 2-dimensional (Grayscale).")
    return cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
