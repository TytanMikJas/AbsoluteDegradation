from pathlib import Path

import cv2
import numpy as np


# CURRENTLY UNUSED
def create_alpha_mask_from_grayscale(
    image_path: Path,
    output_path: Path,
    intensity=2.0,
    color: tuple[int, int, int] | int = 0,
) -> None:
    """
    Treats black and white texture as alpha mask
    and replaces black background with given color.
    Args:
        image_path (Path): Path to the input texture image.
        output_path (Path): Path to save the output texture image.
        intensity (float, optional): Intensity of the alpha mask effect. Higher values make the mask more pronounced.
                                     Defaults to 2.0.
        color (tuple[int, int, int] | int, optional): Color of the degradation
    """
    texture_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    alpha_mask = (1 - texture_gray / 255.0) ** (1 / intensity) * 255  # type: ignore

    texture_bgra = np.zeros((*texture_gray.shape, 4), dtype=np.uint8)  # type: ignore
    texture_bgra[:, :, :3] = color
    texture_bgra[:, :, 3] = alpha_mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), texture_bgra)


def remove_background_with_adaptive_threshold(
    image_path: Path, output_path: Path, block_size: int = 35, C: int = 10
) -> None:
    texture_gray_og = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if texture_gray_og is None:
        raise IOError(f"Failed to decode image: {image_path}. File may be corrupt.")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    texture_gray = cv2.GaussianBlur(texture_gray_og, ksize=(5, 5), sigmaX=1, sigmaY=1)  # type: ignore
    texture_gray_og = cv2.erode(texture_gray_og, kernel=kernel, iterations=1)  # type: ignore

    binary_image = cv2.adaptiveThreshold(
        texture_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        C,
    )
    texture_mask = binary_image.astype(np.uint8)

    texture_bgra = cv2.cvtColor(texture_gray_og, cv2.COLOR_GRAY2BGRA)  # type: ignore
    texture_bgra[:, :, 3] = texture_mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), texture_bgra)
