import math
import random
from logging import getLogger
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

from absolute_degradation.textures.texture import Texture

logger = getLogger(__name__)


def rotated_rect_with_max_area(w: float, h: float, angle: float) -> tuple[float, float]:
    """
    Computes the width and height of the largest possible axis-aligned rectangle
    within a rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # Visualization of the geometry calculation
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))

    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # Half constrained case
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # Fully constrained case
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return max(wr, 1.0), max(hr, 1.0)


def center_crop(img_np: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """
    Crops the center of a numpy image.
    Args:
        img_np: np.ndarray (H, W) or (H, W, C)
        new_width: int target width
        new_height: int target height
    """
    height, width = img_np.shape[:2]
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return img_np[top:bottom, left:right].copy()


def resize(img_np: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resizes a numpy image to the target dimensions."""
    h, w = img_np.shape[:2]
    upscaling = target_h > h or target_w > w
    interpolation = cv2.INTER_LINEAR if upscaling else cv2.INTER_NEAREST
    return cv2.resize(
        img_np,
        (target_w, target_h),
        interpolation=interpolation,
    ).clip(0, 1)


def rotate(img, angle):
    """
    Rotate image like PIL.Image.rotate(angle, expand=True, BILINEAR)
    img: np.ndarray (H, W) or (H, W, C)
    angle: degrees (positive = counter-clockwise)
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)


def ensure_min_size(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Up-scale *img* so that it is at least (target_h, target_w)."""
    h, w = img.shape[:2]
    if h < target_h or w < target_w:
        scale = max(target_h / h, target_w / w)
        new_h = int(math.ceil(h * scale))
        new_w = int(math.ceil(w * scale))
        img = resize(img, new_h, new_w)
    return img


def offset_crop(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    offset_y: float,
    offset_x: float,
) -> np.ndarray:
    """Crop *img* to (target_h, target_w) using normalised [0, 1] offsets."""
    h, w = img.shape[:2]
    max_top = max(h - target_h, 0)
    max_left = max(w - target_w, 0)
    top = int(offset_y * max_top)
    left = int(offset_x * max_left)
    return img[top : top + target_h, left : left + target_w]


def enhance_contrast_pil_like(
    img: npt.NDArray[np.float32], factor: float
) -> npt.NDArray[np.float32]:
    """
    Enhances the contrast of a numpy image similar to PIL's ImageEnhance.Contrast.
    Args:
        img: np.ndarray (H, W) or (H, W, C), uint8
        factor: contrast factor (1.0 = original image)
    """
    img = img.astype(np.float32)

    mean = np.mean(img, axis=(0, 1), keepdims=True)

    out = mean + factor * (img - mean)
    out = np.clip(out, 0, 1)

    return out.astype(np.float32)


def texture_generator(
    texture: Texture,
    target_h: int = 256,
    target_w: int = 256,
    offset_y: float | None = None,
    offset_x: float | None = None,
) -> Texture:
    """Crop a texture to (target_h, target_w) and augment it.

    The spatial crop position is controlled by *offset_y* / *offset_x*
    (normalised 0-1 values).  When they are `None` a fresh random
    position is sampled, which is the legacy behaviour.

    Args:
        texture: Source texture (not mutated).
        target_h: Desired crop / output height in pixels.
        target_w: Desired crop / output width in pixels.
        offset_y: Normalised vertical crop offset in [0, 1].
            ``None`` → random.
        offset_x: Normalised horizontal crop offset in [0, 1].
            ``None`` → random.

    Returns:
        :class:`Texture` whose image is the augmented crop.
    """
    if offset_y is None:
        offset_y = random.random()
    if offset_x is None:
        offset_x = random.random()

    tex = texture.copy()
    should_erode = texture.should_erode()

    # 1. Ensure texture is at least as large as the requested crop
    tex.image = ensure_min_size(tex.image, target_h, target_w)

    # 2. Early Exit: 15 % chance or skip-resize texture group → crop and return
    if random.uniform(0.0, 1.0) < 0.15 or texture.should_skip_resize():
        tex.image = offset_crop(tex.image, target_h, target_w, offset_y, offset_x)
        return tex

    # 3. Rotate the full texture *before* cropping to preserve content
    rotation_angle = random.randint(0, 360)
    rotated = rotate(tex.image, rotation_angle)

    # 4. Center-crop to the largest axis-aligned rect inside the
    #    rotated original rectangle (removes black borders)
    th, tw = tex.image.shape[:2]
    max_w, max_h = rotated_rect_with_max_area(tw, th, math.radians(rotation_angle))
    cropped_rotated = center_crop(rotated, int(max_w), int(max_h))

    # 5. Ensure rotated result covers the target (fallback for extreme angles)
    cropped_rotated = ensure_min_size(cropped_rotated, target_h, target_w)

    # 6. Crop to target size using normalised offsets
    tex.image = offset_crop(cropped_rotated, target_h, target_w, offset_y, offset_x)

    # 7. Optional zoom-in branch: enlarge texture then crop back to target size.
    # This keeps output dimensions fixed while increasing apparent texture scale.
    if random.uniform(0.0, 1.0) < 0.33:
        zoom_scale = random.uniform(1.3, 1.6)
        zoom_h = max(int(math.ceil(target_h * zoom_scale)), target_h + 1)
        zoom_w = max(int(math.ceil(target_w * zoom_scale)), target_w + 1)
        zoomed = resize(tex.image, zoom_h, zoom_w)
        tex.image = offset_crop(zoomed, target_h, target_w, offset_y, offset_x)

    # 8. Morphological erosion for applicable texture groups (25% chance).
    if should_erode and random.uniform(0.0, 1.0) < 0.25:
        erosion_kernel_size = random.randint(0, 1) * 2 + 1
        tex.image = np.asarray(
            cv2.erode(
                tex.image,
                np.ones((erosion_kernel_size, erosion_kernel_size), np.float32),
                iterations=1,
            ),
            dtype=np.float32,
        )

    # 9. Contrast enhancement (70 % probability, factor 2-4)
    if random.uniform(0.0, 1.0) < 0.7:
        tex.image = enhance_contrast_pil_like(tex.image, random.uniform(2.0, 4.0))

    return tex


## Used for the generate the texture for moving line
def moving_line_texture_generator(
    texture_input: Texture,
    last_texture: Optional[Texture],
    target_h=256,
    target_w=256,
) -> Texture:
    """
    Generates a scratch texture.
    If last_texture is provided, it simulates temporal coherence
    (Brownian motion) by shifting the previous texture.
    If last_texture is None, it initializes a new scratch by cropping
    a random region from the input texture.
    """
    # Case A: Initialize a new scratch
    if last_texture is None:
        augmented_texture = texture_input.copy()
        H, W = augmented_texture.resolution

        img = augmented_texture.image
        if img.ndim == 3 and img.shape[2] >= 4:
            # RGBA: scratch content lives where alpha > 0
            texture_mask = (img[:, :, 3] > 0.01).astype(np.float32)
        else:
            # Grayscale / RGB: dark pixels relative to the mean
            if img.ndim == 3:
                gray = np.mean(img[:, :, :3], axis=2)
            else:
                gray = img
            texture_mean = float(np.mean(gray))
            texture_mask = (gray < (texture_mean - 25.0 / 255.0)).astype(np.float32)

        texture_mask[:, :40] = 0.0
        texture_mask[:, -40:] = 0.0

        valid_points = np.where(texture_mask)
        if len(valid_points[0]) == 0:
            logger.warning("Texture mask has no valid points, resizing directly.")

            augmented_texture.resize(target_h, target_w)
            return augmented_texture

        anchor_idx = random.randint(0, len(valid_points[0]) - 1)
        anchor_point_y, anchor_point_x = (
            valid_points[0][anchor_idx],
            valid_points[1][anchor_idx],
        )

        bounding_box_size = random.randint(315, 560)  ## Original: [130,360]

        shift_x = random.randint(
            max(bounding_box_size - W + anchor_point_x, 0),
            min(anchor_point_x, bounding_box_size),
        )
        shift_y = random.randint(
            max(bounding_box_size - H + anchor_point_y, 0),
            min(anchor_point_y, bounding_box_size),
        )

        left_up_x = anchor_point_x - shift_x
        left_up_y = anchor_point_y - shift_y

        augmented_texture.crop(
            left_up_y, left_up_x, bounding_box_size, bounding_box_size
        )

        augmented_texture.resize(target_h, target_w)
        return augmented_texture

    # Case B: Evolve existing scratch (Temporal Coherence)
    else:
        augmented_last_texture = last_texture.copy()
        texture_np = np.array(last_texture.image)
        shift_amount = random.randint(5, 15)

        if random.uniform(0, 1) > 0.5:
            shift_amount = -shift_amount

        texture_np = np.roll(texture_np, shift_amount, axis=1)

        if shift_amount > 0:
            texture_np[:, :shift_amount] = 0
        else:
            texture_np[:, shift_amount:] = 0

        augmented_last_texture.image = texture_np
        return augmented_last_texture
