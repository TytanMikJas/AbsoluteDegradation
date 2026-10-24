import cv2
import numpy as np
import numpy.typing as npt

from absolute_degradation.distortions.steps.abstract_distort_step import (
    AbstractDistortStep,
)


class GroundTruthSharpenStep(AbstractDistortStep):
    """
    Ground-truth-only USM sharpening.
    """

    def __init__(
        self,
        weight: float = 0.5,
        radius: int = 50,
        threshold: int = 10,
        probability: float = 1.0,
    ) -> None:
        super().__init__(probability)
        self.weight = weight
        self.radius = radius
        self.threshold = threshold

    def rebuild(self) -> None:
        pass

    def distort(
        self,
        image: npt.NDArray[np.float32],
        ground_truth: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        if ground_truth.dtype == np.uint8:
            gt_f32 = ground_truth.astype(np.float32) / 255.0
        else:
            gt_f32 = ground_truth.astype(np.float32)

        sharpened_gt = self._add_sharpening(
            gt_f32,
            weight=self.weight,
            radius=self.radius,
            threshold=self.threshold,
        )
        return image, np.clip(sharpened_gt, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _add_sharpening(
        img: npt.NDArray[np.float32],
        weight: float = 0.5,
        radius: int = 50,
        threshold: int = 10,
    ) -> npt.NDArray[np.float32]:
        if radius % 2 == 0:
            radius += 1
        blur = cv2.GaussianBlur(img, (radius, radius), 0)
        residual = img - blur
        mask = np.abs(residual) * 255 > threshold
        mask = mask.astype(np.float32)
        soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

        sharpened = img + weight * residual
        sharpened = np.clip(sharpened, 0, 1)
        return soft_mask * sharpened + (1 - soft_mask) * img
