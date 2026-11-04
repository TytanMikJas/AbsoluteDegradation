from typing import Literal

import numpy as np
import numpy.typing as npt

from absolute_degradation.distortions.steps.abstract_distort_step import (
    AbstractDistortStep,
)


class QuantizationStep(AbstractDistortStep):
    def __init__(
        self, output_type: Literal["float", "int"] = "float", probability: float = 1.0
    ) -> None:
        super().__init__(probability)
        self._output_type = output_type

    def rebuild(self) -> None:
        pass

    def distort(
        self, image: npt.NDArray[np.float32], ground_truth: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        image = (255 * image).clip(0, 255).astype(np.uint8)

        if self._output_type == "int":
            ground_truth = (255 * ground_truth).clip(0, 255).astype(np.uint8)
            return image, ground_truth

        image = image.astype(np.float32) / 255.0

        return image, ground_truth
