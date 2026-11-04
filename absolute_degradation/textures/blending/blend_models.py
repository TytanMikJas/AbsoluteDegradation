"""
Module provides various blending functions to apply textures onto images.
Blending modes include addition, subtract, and multiply.
"""

import numpy as np


def addition(img_in: np.ndarray, img_layer: np.ndarray, opacity: float) -> np.ndarray:
    """Apply addition blending mode of a layer on an image."""
    assert img_in.dtype.kind == "f"
    assert img_layer.dtype.kind == "f"
    assert img_in.shape[2] == 4
    assert img_layer.shape[2] == 4
    assert 0.0 <= opacity <= 1.0

    img_in = img_in / 255.0
    img_layer[:, :, :3] = 1.0 - img_layer[:, :, :3] / 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = img_in[:, :, :3] + img_layer[:, :, :3]

    ratio_rs = np.reshape(
        np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]]
    )
    img_out = np.clip(comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs), 0.0, 1.0)

    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))
    return img_out * 255.0


def multiply(img_in: np.ndarray, img_layer: np.ndarray, opacity: float) -> np.ndarray:
    """Apply multiply blending mode of a layer on an image."""
    assert img_in.dtype.kind == "f"
    assert img_layer.dtype.kind == "f"
    assert img_in.shape[2] == 4
    assert img_layer.shape[2] == 4
    assert 0.0 <= opacity <= 1.0

    img_in = img_in / 255.0
    img_layer = img_layer / 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.clip(img_layer[:, :, :3] * img_in[:, :, :3], 0.0, 1.0)

    ratio_rs = np.reshape(
        np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]]
    )
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)

    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))
    return img_out * 255.0


def subtract(img_in: np.ndarray, img_layer: np.ndarray, opacity: float) -> np.ndarray:
    """Apply subtract blending mode."""
    assert img_in.dtype.kind == "f"
    assert img_layer.dtype.kind == "f"
    assert img_in.shape[2] == 4
    assert img_layer.shape[2] == 4
    assert 0.0 <= opacity <= 1.0

    img_in = img_in / 255.0
    img_layer[:, :, :3] = 1.0 - img_layer[:, :, :3] / 255.0
    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = img_in[:, :, :3] - img_layer[:, :, :3]

    ratio_rs = np.reshape(
        np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]]
    )
    img_out = np.clip(comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs), 0.0, 1.0)

    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))
    return img_out * 255.0


def _compose_alpha(
    img_in: np.ndarray, img_layer: np.ndarray, opacity: float
) -> np.ndarray:
    """Calculate alpha composition ratio between two images."""
    comp_alpha = np.minimum(img_in[:, :, 3], img_layer[:, :, 3]) * opacity
    new_alpha = img_in[:, :, 3] + (1.0 - img_in[:, :, 3]) * comp_alpha

    np.seterr(divide="ignore", invalid="ignore")
    ratio = comp_alpha / new_alpha
    ratio[np.isnan(ratio)] = 0.0
    return ratio
