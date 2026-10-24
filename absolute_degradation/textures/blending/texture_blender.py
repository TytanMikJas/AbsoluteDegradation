import random

import cv2
import numpy as np
import numpy.typing as npt

from absolute_degradation.textures.blending.blend_mode import BlendMode
from absolute_degradation.textures.blending.blend_models import (
    addition,
    multiply,
    subtract,
)
from absolute_degradation.textures.texture import Texture
from absolute_degradation.textures.texture_augmentation import (
    moving_line_texture_generator,
    texture_generator,
)
from absolute_degradation.textures.texture_type import TextureType


def texture_blending(
    image: npt.NDArray[np.float32],
    texture: Texture,
    blend_mode: BlendMode | None = None,
    augment: bool = True,
    verbose: bool = False,
    return_mask: bool = False,
) -> npt.NDArray[np.float32] | tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Blends a static texture onto the image.
    Args:
        image: np.ndarray (H, W, 3) RGB or (H, W) Grayscale, float32 [0-1] image.
        texture: Texture object containing the texture to blend.
        verbose: bool for logging details.
        blend_mode: Optional BlendMode enum specifying the blending mode. If None, a random mode is chosen.
            if texture is of type VERTICAL_SCRATCHES, BlendMode.ADDITION is used automatically.
        augment: bool indicating whether to apply augmentation to the texture.
    Returns:
        scratched: np.ndarray (H, W, 3) RGB or (H, W) Grayscale float32 [0-1] image with blended texture.
        If ``return_mask`` is True, also returns a soft coverage mask (H, W) float32 in [0, 1]
        derived from per-pixel alpha composition and opacity.
    """
    w, h = image.shape[1], image.shape[0]
    if augment:
        texture_to_use = texture_generator(texture, h, w)
    else:
        texture_to_use = texture

    if texture.texture_type == TextureType.VERTICAL_SCRATCHES:
        blend_mode = BlendMode.ADDITION
    elif blend_mode is None:
        blend_mode = BlendMode.random_mode()

    opacity = random.uniform(0.8, 1.0)

    return apply_blending(
        image,
        texture_to_use,
        blend_mode,
        opacity=opacity,
        verbose=verbose,
        return_mask=return_mask,
    )


def moving_line_texture_blending(
    image: npt.NDArray[np.float32],
    texture: Texture,
    last_texture: Texture | None,
    mode: BlendMode,
    verbose: bool = False,
    return_mask: bool = False,
) -> (
    tuple[npt.NDArray[np.float32], Texture]
    | tuple[npt.NDArray[np.float32], Texture, npt.NDArray[np.float32]]
):
    """
    Blends a dynamic (moving) texture onto the image.
    Args:
        image: npt.NDArray[np.float32] (H, W, 3) RGB or (H, W) Grayscale, float32 [0-1] image.
        texture: Texture object containing the texture to blend.
        last_texture: Texture object from the previous frame for temporal consistency.
        mode: BlendMode enum specifying the blending mode.
        verbose: bool for logging details.
    Returns:
        scratched_rgb: npt.NDArray[np.float32] (H, W, 3) RGB float32 [0-1] image with blended texture.
        texture_processed: Texture object after processing for blending.
        If ``return_mask`` is True, also returns a soft coverage mask (H, W) float32 in [0, 1].
    """
    w, h = image.shape[1], image.shape[0]
    texture_processed = moving_line_texture_generator(texture, last_texture, h, w)
    opacity = random.uniform(0.8, 1.0)
    blend_result = apply_blending(
        image,
        texture_processed,
        mode,
        opacity=opacity,
        verbose=verbose,
        return_mask=return_mask,
    )
    if return_mask:
        assert isinstance(blend_result, tuple)
        scratched_rgb, mask = blend_result
        return scratched_rgb, texture_processed, mask

    assert not isinstance(blend_result, tuple)
    scratched_rgb = blend_result
    return scratched_rgb, texture_processed


def apply_blending(
    image: npt.NDArray[np.float32],
    processed_texture: Texture,
    mode: BlendMode,
    opacity: float,
    verbose: bool = False,
    return_mask: bool = False,
) -> npt.NDArray[np.float32] | tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Applies blending of a texture onto an image with specified mode and opacity.
    Args:
        image: np.ndarray (H, W, 3) RGB or (H, W) Grayscale, float32 [0-1] image.
        processed_texture: Texture object containing the texture to blend.
        mode: BlendMode enum specifying the blending mode.
        opacity: float specifying the opacity of the texture.
        verbose: bool for logging details.
    Returns:
        Blended RGB image.
        If ``return_mask`` is True, also returns an intensity-aware mask (H, W)
        float32 in [0, 1], computed as coverage(alpha, opacity) weighted by
        texture RGB effect strength (independent of background image).
    """
    if image.ndim == 2:
        img_rgba = 255 * cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    else:
        img_rgba = 255 * cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    tex_rgba = 255 * processed_texture.image_to_rgba()

    if verbose:
        print(f"Blending mode: {mode}")

    match mode:
        case BlendMode.ADDITION:
            blended_rgba = addition(img_rgba, tex_rgba, opacity)
        case BlendMode.SUBTRACT:
            blended_rgba = subtract(img_rgba, tex_rgba, opacity)
        case BlendMode.MULTIPLY:
            blended_rgba = multiply(img_rgba, tex_rgba, opacity)
        case _:
            raise ValueError(f"Unsupported blend mode: {mode}")

    blended_rgb = blended_rgba[:, :, :3] / 255.0
    if not return_mask:
        return blended_rgb

    img_in_alpha = img_rgba[:, :, 3] / 255.0
    layer_alpha = tex_rgba[:, :, 3] / 255.0
    comp_alpha = np.minimum(img_in_alpha, layer_alpha) * opacity

    if np.all(img_in_alpha >= (1.0 - 1e-6)):
        coverage_mask = np.clip(comp_alpha, 0.0, 1.0).astype(np.float32)
    else:
        new_alpha = img_in_alpha + (1.0 - img_in_alpha) * comp_alpha
        ratio = np.divide(
            comp_alpha,
            new_alpha,
            out=np.zeros_like(comp_alpha, dtype=np.float32),
            where=new_alpha > 0,
        )
        coverage_mask = np.clip(ratio, 0.0, 1.0).astype(np.float32)

    # # Strength of visual change introduced by blending (brightness/content effect).
    # input_rgb = img_rgba[:, :, :3] / 255.0
    # photometric_delta = np.mean(np.abs(blended_rgb - input_rgb), axis=2).astype(
    #     np.float32
    # )
    # texture_mask = np.clip(coverage_mask * photometric_delta, 0.0, 1.0).astype(
    #     np.float32
    # )
    # return blended_rgb, texture_mask

    texture_rgb = tex_rgba[:, :, :3] / 255.0
    texture_rgb_strength = np.mean(1.0 - texture_rgb, axis=2).astype(np.float32)

    texture_mask = np.clip(coverage_mask * texture_rgb_strength, 0.0, 1.0).astype(
        np.float32
    )
    return blended_rgb, texture_mask
