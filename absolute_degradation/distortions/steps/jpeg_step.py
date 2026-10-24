import random

import cv2
import numpy as np
import numpy.typing as npt

from absolute_degradation.distortions.steps.abstract_distort_step import (
    AbstractDistortStep,
    MaskStepMixin,
)


class JpegStep(AbstractDistortStep, MaskStepMixin):
    """
    Class for JPEG distortion step of the pipeline.
    """

    def __init__(
        self,
        quality_center_bounds: tuple[int, int],
        quality_variance: int | tuple[int, int] = 15,
        probability: float = 1.0,
    ) -> None:
        """
        Initializes JpegStep
        Args:
            quality_center_bounds (tuple[int, int]): Min/Max of the clip-level JPEG quality center.
            quality_variance (int | tuple[int, int]): Per-frame deviation from the clip center.
                If int v, quality is sampled from [center - v, center + v].
                If tuple (a, b), quality is sampled from [center - a, center + b].
            probability (float): Probability of applying this distortion.
        """
        super().__init__(probability=probability)
        self._quality_center_bounds = quality_center_bounds
        self._quality_variance = quality_variance
        self._last_quality_value: int | None = None

    def rebuild(self) -> None:
        """Sample clip-level JPEG center quality."""
        self._quality_center = random.randint(
            self._quality_center_bounds[0], self._quality_center_bounds[1]
        )
        self._last_quality_value = None

    def _sample_quality_for_frame(self) -> int:
        """Per-frame +/- variance around clip center, clipped to [40, 100]."""
        if isinstance(self._quality_variance, tuple):
            variance = random.randint(
                -self._quality_variance[0], self._quality_variance[1]
            )
        else:
            variance = random.randint(-self._quality_variance, self._quality_variance)
        return int(np.clip(self._quality_center + variance, 40, 100))

    def distort(
        self,
        image: npt.NDArray[np.float32],
        ground_truth: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Apply JPEG distortion to the image using OpenCV.
        Args:
            image (np.ndarray): Input image H x W x C [0-1].
            ground_truth (np.ndarray): Ground truth image H x W x C [0-1].
        Returns:
            tuple[np.ndarray, np.ndarray]: Distorted image H x W [0-1] (Grayscale) and ground truth.
        """
        image_uint_8 = (image * 255).clip(0, 255).astype(np.uint8)

        quality_value = self._sample_quality_for_frame()
        self._last_quality_value = quality_value
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_value]
        result, encimg = cv2.imencode(".jpg", image_uint_8, encode_param)

        if not result:
            raise RuntimeError("Could not encode image to JPEG")

        image_jpeg: npt.NDArray[np.uint8] = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)  # type: ignore

        distorted_image = image_jpeg.astype(np.float32) / 255.0

        return distorted_image, ground_truth

    def apply_to_mask(self, mask: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Returns the mask unchanged.
        """
        return mask
