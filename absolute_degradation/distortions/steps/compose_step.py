import random

import numpy as np
import numpy.typing as npt

from absolute_degradation.distortions.steps.abstract_distort_step import (
    AbstractDistortStep,
    DistortOutput,
    MaskStepMixin,
)


class ComposeDistortStep(AbstractDistortStep, MaskStepMixin):
    def __init__(
        self,
        distortions: list[AbstractDistortStep],
        permute: bool = False,
        probability: float = 1,
    ) -> None:
        super().__init__(probability)
        self._distortions = distortions
        self._permute = permute

    def rebuild(self) -> None:
        """sample distortion parameters for new video sequence"""
        if self._permute:
            random.shuffle(self._distortions)

        for distortion in self._distortions:
            distortion.rebuild()

    def distort(
        self, image: npt.NDArray[np.float32], ground_truth: npt.NDArray[np.float32]
    ) -> DistortOutput:
        aggregated_mask: npt.NDArray[np.float32] | None = None

        for distortion in self._distortions:
            distortion_output = distortion(image, ground_truth)
            if len(distortion_output) == 3:
                image, ground_truth, step_mask = distortion_output
                if step_mask.shape != image.shape[:2]:
                    raise RuntimeError(
                        f"{distortion.__class__.__name__} returned mask with shape "
                        f"{step_mask.shape}, expected {image.shape[:2]}."
                    )
                if aggregated_mask is None:
                    aggregated_mask = step_mask.astype(np.float32)
                else:
                    aggregated_mask = 1.0 - (1.0 - aggregated_mask) * (1.0 - step_mask)
            else:
                image, ground_truth = distortion_output
                if (
                    aggregated_mask is not None
                    and distortion.was_applied_last_call
                    and isinstance(distortion, MaskStepMixin)
                ):
                    transformed_mask = distortion.apply_to_mask(aggregated_mask)
                    if transformed_mask is not None:
                        aggregated_mask = transformed_mask

            if aggregated_mask is not None and aggregated_mask.shape != image.shape[:2]:
                raise RuntimeError(
                    f"Mask/image shape mismatch after {distortion.__class__.__name__}: "
                    f"mask={aggregated_mask.shape}, image={image.shape[:2]}. "
                    "Implement apply_to_mask() for this distortion."
                )

        if aggregated_mask is not None:
            return (
                image,
                ground_truth,
                np.clip(aggregated_mask, 0.0, 1.0).astype(np.float32),
            )
        return image, ground_truth

    def apply_to_mask(self, mask: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply nested mask-aware transforms in the same order as image distortions."""
        transformed_mask = mask
        for distortion in self._distortions:
            if distortion.was_applied_last_call and isinstance(
                distortion, MaskStepMixin
            ):
                candidate_mask = distortion.apply_to_mask(transformed_mask)
                if candidate_mask is not None:
                    transformed_mask = candidate_mask
        return np.clip(transformed_mask, 0.0, 1.0).astype(np.float32)
