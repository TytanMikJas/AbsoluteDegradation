import albumentations as A
import numpy as np
import numpy.typing as npt

from absolute_degradation.distortions.steps.abstract_distort_step import (
    AbstractDistortStep,
)


class ColorJitterStep(AbstractDistortStep):
    def __init__(
        self,
        brightness_bounds: tuple[float, float] = (0.8, 1.2),
        contrast_bounds: tuple[float, float] = (0.9, 1.0),
        saturation_bounds: tuple[float, float] = (1.0, 1.0),
        hue_bounds: tuple[float, float] = (0.0, 0.0),
        probability: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            brightness_bounds: Range of the brightness factor.
            contrast_bounds: Range of the contrast factor.
            saturation_bounds: Range of the saturation factor.
            hue_bounds: Range of the hue shift.
            probability: Chance (0-1) of applying the jitter to a frame.
            seed: Seed for the internal Albumentations RNG. Albumentations keeps
                its own random generator, which ``random.seed`` and
                ``np.random.seed`` do not affect, so without a seed the jitter
                parameters are not reproducible. ``None`` keeps the
                Albumentations default (non-deterministic) seeding.
        """
        super().__init__(probability)

        self._transform = A.ColorJitter(
            brightness=brightness_bounds,
            contrast=contrast_bounds,
            saturation=saturation_bounds,
            hue=hue_bounds,
            p=1.0,  # Handled by the step probability
        )
        if seed is not None:
            self._transform.set_random_seed(seed)

    def rebuild(self) -> None:
        pass

    def distort(
        self, image: npt.NDArray[np.float32], ground_truth: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return self._transform(image=image)["image"], ground_truth
