import random

import cv2
import numpy as np
import numpy.typing as npt

from absolute_degradation.distortions.steps.abstract_distort_step import (
    AbstractDistortStep,
    MaskStepMixin,
)


class DownsampleStep(AbstractDistortStep, MaskStepMixin):
    """Class for downsampling step of the pipeline."""

    def __init__(
        self,
        down_scale_factor_bounds: tuple[float, float],
        up_scale_factor_bounds: tuple[float, float],
        random_value: float | None = None,
        down_scale_threshold: float = 0.7,
        up_scale_threshold: float = 0.8,
        default_scale_factor: float = 1.0,
        probability: float = 1.0,
    ) -> None:
        """
        Initializes DownsampleStep
        Args:
            down_scale_factor (float): Downscaling factor.
            up_scale (float): Upscaling factor.
            random_value (float): Random value to decide scaling.
            down_scale_threshold (float): Threshold below which downscaling is applied.
            up_scale_threshold (float): Threshold above which upscaling is applied.
            default_scale_factor (float): Default scaling factor when no scaling is applied.
            probability (float): Probability of applying this distortion.
        """
        super().__init__(probability=probability)
        self._down_scale_threshold = down_scale_threshold
        self._up_scale_threshold = up_scale_threshold
        self._default_scale_factor = default_scale_factor

        self._down_scale_factor_bounds = down_scale_factor_bounds
        self._up_scale_factor_bounds = up_scale_factor_bounds
        self._random_value_original = random_value
        self._last_target_size: tuple[int, int] | None = None
        self._last_interpolation_method: int | None = None

    def rebuild(self) -> None:
        """sample distortion parameters for new video sequence"""
        self._down_scale_factor = random.uniform(
            self._down_scale_factor_bounds[0], self._down_scale_factor_bounds[1]
        )
        self._up_scale_factor = random.uniform(
            self._up_scale_factor_bounds[0], self._up_scale_factor_bounds[1]
        )
        self._random_value = (
            self._random_value_original
            if self._random_value_original is not None
            else random.random()
        )

    @property
    def scale_factor(self) -> float:
        """
        Returns the scaling factor based on the random value and thresholds.
        """
        if self._random_value < self._down_scale_threshold:
            return self._down_scale_factor
        elif self._random_value > self._up_scale_threshold:
            return self._up_scale_factor
        else:
            return self._default_scale_factor

    def distort(
        self,
        image: npt.NDArray[np.float32],
        ground_truth: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Apply downsampling to the image using nearest neighbor interpolation.
        Args:
            image (npt.NDArray[np.float32]): Input image H x W x C [0-1].
        Returns:
            npt.NDArray[np.float32]: Downsampled image H x W x C [0-1].
        """
        # downsample the mask aswell
        w, h = image.shape[1], image.shape[0]
        scale_factor = self.scale_factor
        new_w = int(scale_factor * w)
        new_h = int(scale_factor * h)

        interpolation_methods = [
            cv2.INTER_CUBIC,
            cv2.INTER_LINEAR,
            cv2.INTER_LANCZOS4,
        ]
        selected_method = random.choice(interpolation_methods)
        self._last_target_size = (new_w, new_h)
        self._last_interpolation_method = selected_method

        img = cv2.resize(image, (new_w, new_h), interpolation=selected_method)
        return img, ground_truth  # type: ignore

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
