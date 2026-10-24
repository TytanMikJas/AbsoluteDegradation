import random

import cv2
import numpy as np
import numpy.typing as npt

from absolute_degradation.distortions.steps.abstract_distort_step import (
    AbstractDistortStep,
    MaskStepMixin,
)


class SizeMatchingStep(AbstractDistortStep, MaskStepMixin):
    """
    Class for resizing the image to match the ground truth size.
    """

    def __init__(self, probability: float = 1) -> None:
        super().__init__(probability)
        self._last_target_size: tuple[int, int] | None = None
        self._last_interpolation_method: int | None = None

    def rebuild(self) -> None:
        pass

    def distort(
        self, image: npt.NDArray[np.float32], ground_truth: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        orig_x, orig_y = ground_truth.shape[1], ground_truth.shape[0]

        interpolation_methods = [
            cv2.INTER_CUBIC,
            cv2.INTER_LINEAR,
            cv2.INTER_LANCZOS4,
        ]
        selected_method = random.choice(interpolation_methods)
        self._last_target_size = (orig_x, orig_y)
        self._last_interpolation_method = selected_method

        image = cv2.resize(
            image, (orig_x, orig_y), interpolation=selected_method
        ).astype(np.float32)

        return image, ground_truth

    def apply_to_mask(self, mask: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Applies the same resize (shape + interpolation) used for image distortion."""
        if self._last_target_size is None or self._last_interpolation_method is None:
            return mask

        resized_mask = cv2.resize(
            mask,
            self._last_target_size,
            interpolation=self._last_interpolation_method,
        )
        return np.clip(resized_mask, 0.0, 1.0).astype(np.float32)
