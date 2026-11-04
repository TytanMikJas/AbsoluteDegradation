import random

import cv2
import numpy as np
import numpy.typing as npt


def apply_random_scaling(
    image: npt.NDArray[np.float32], target_width: int, target_height: int
) -> npt.NDArray[np.float32]:
    """
    Resizes a NumPy image array ([0, 1] range) using a random interpolation method.

    Args:
        image: Input image as a numpy array.
        target_width: The desired output width.
        target_height: The desired output height.
    """
    interpolation_methods = [
        cv2.INTER_CUBIC,
        cv2.INTER_LINEAR,
        cv2.INTER_LANCZOS4,
    ]

    selected_method = random.choice(interpolation_methods)
    # TODO: maybe rewrite it to DistortionStep with ability to shift probs of methods
    resized_image = cv2.resize(
        image, (target_width, target_height), interpolation=selected_method
    )

    return resized_image  # type: ignore
