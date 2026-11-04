import random

import numpy as np
import numpy.typing as npt

from absolute_degradation.distortions.steps.abstract_distort_step import (
    AbstractDistortStep,
)


class CropStep(AbstractDistortStep):
    """Crops both image and ground truth to a fixed H x W patch."""

    def __init__(
        self,
        crop_h: int = 256,
        crop_w: int = 256,
        probability: float = 1.0,
    ) -> None:
        """
        Args:
            crop_h: Height of the crop (pixels).
            crop_w: Width of the crop (pixels).
            probability: Chance (0-1) that the crop is applied.
        """
        super().__init__(probability)
        self.crop_h = crop_h
        self.crop_w = crop_w
        self._top: int = 0
        self._left: int = 0

    def rebuild(self) -> None:
        """Sample normalised crop offsets for a new clip.

        Actual pixel coordinates are resolved on the first `distort` call
        because image dimensions are unknown until then.
        """
        self._offset_y: float = random.random()
        self._offset_x: float = random.random()
        self._coords_resolved = False

    def _resolve_coords(self, h: int, w: int) -> None:
        """Convert normalised offsets to pixel coordinates for given dims."""
        max_top = max(h - self.crop_h, 0)
        max_left = max(w - self.crop_w, 0)
        self._top = int(self._offset_y * max_top)
        self._left = int(self._offset_x * max_left)
        self._coords_resolved = True

    def distort(
        self,
        image: npt.NDArray[np.float32],
        ground_truth: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        h, w = image.shape[:2]

        if not self._coords_resolved:
            self._resolve_coords(h, w)

        t, left = self._top, self._left
        ch, cw = self.crop_h, self.crop_w

        image = image[t : t + ch, left : left + cw].copy()
        ground_truth = ground_truth[t : t + ch, left : left + cw].copy()

        return image, ground_truth
