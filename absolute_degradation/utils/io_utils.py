"""
Utils for loading, saving of images
"""

from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import numpy.typing as npt


def read_image(path: Path) -> np.ndarray:
    """
    Reads an image from a path, converts to float32, and normalizes to [0, 1].
    Reads image as RGB.
    Raises IOError if the image cannot be read.
    """
    img = cv2.imread(str(path))

    if img is None:
        raise IOError(f"Failed to decode image: {path}. File may be corrupt.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def read_texture(path: Path) -> np.ndarray:
    """
    Reads an image from a path, converts to float32, and normalizes to [0, 1].
    Reads image as RGBA.
    Raises IOError if the image cannot be read.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise IOError(f"Failed to decode image: {path}. File may be corrupt.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img.astype(np.float32) / 255.0


def read_image_grayscale(path: Path) -> np.ndarray:
    """
    Reads an image as Grayscale from a path, converts to float32, and normalizes to [0, 1].
    Raises IOError if the image cannot be read.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Failed to decode image: {path}. File may be corrupt.")
    return img.astype(np.float32) / 255.0


def save_frame(path: Path, img: np.ndarray) -> None:
    """
    Saves a frame to disk.
    Safely handles float-to-int conversion for the 0-255 range.
    Args:
        path (Path): Path to save the image. Creates parent directories if needed.
        img (np.ndarray): Image array RGB (H, W, 3) to save expects float32 or uint8 both in [0..255].
    Raises:
        IOError: If the image cannot be saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(path), img_bgr)
    if not success:
        raise IOError(f"Failed to write image to {path}")


def get_image_generator(
    frame_files: list[Path],
    grayscale: bool = True,
) -> Generator[npt.NDArray[np.float32], None, None]:
    """
    Yields images one by one to save input memory.
    relies on read_image to raise IOError if a file is corrupt.
    """
    read_fn = read_image_grayscale if grayscale else read_image
    for f in frame_files:
        yield read_fn(f)


def getfilelist(path: Path, extensions=("jpg", "jpeg", "png")) -> list[Path]:
    """Recursively returns a list of files in a directory."""
    files_list: list[Path] = []
    for ext in extensions:
        ext = ext.removeprefix(".")
        files_list.extend(path.rglob(f"*.{ext}"))

    files_list.sort()
    return files_list
