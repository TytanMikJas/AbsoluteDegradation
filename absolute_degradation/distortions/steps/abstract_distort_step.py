from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

DistortOutput = (
    tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
    | tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]
)


class AbstractDistortStep(ABC):
    """
    Class for distortion step of the pipeline.
    """

    def __init__(self, probability: float = 1.0) -> None:
        """
        Initializes AbstractDistortStep
        Args:
            probability (float): Probability of applying this distortion step.
        """
        super().__init__()
        self.probability = probability
        self._was_applied_last_call = False

    def __call__(
        self, image: npt.NDArray[np.float32], ground_truth: npt.NDArray[np.float32]
    ) -> DistortOutput:
        """
        Applies distortion step with a certain probability.
        Args:
            image (npt.NDArray[np.float32]): Input image H x W x C [0-1].
            ground_truth (npt.NDArray[np.float32]): Ground truth image H x W x C [0-1].
        Returns:
            tuple[npt.NDArray[np.float32]]: Pair of distorted image (H x W x C [0-1]) and the ground truth
        """
        if np.random.rand() < self.probability:
            self._was_applied_last_call = True
            return self.distort(image, ground_truth)
        self._was_applied_last_call = False
        return image, ground_truth

    @property
    def was_applied_last_call(self) -> bool:
        """Whether `distort()` was executed in the most recent `__call__`."""
        return self._was_applied_last_call

    @abstractmethod
    def distort(
        self,
        image: npt.NDArray[np.float32],
        ground_truth: npt.NDArray[np.float32],
    ) -> DistortOutput:
        """
        Applies defined distort of the input image.
        Args:
            image (npt.NDArray[np.float32]): Input image H x W x C [0-1].
            ground_truth (npt.NDArray[np.float32]): Ground truth image H x W x C [0-1].
        Returns:
            tuple[npt.NDArray[np.float32]]: Pair of distorted image (H x W x C [0-1]) and the ground truth
        """
        pass

    @abstractmethod
    def rebuild(
        self,
    ) -> None:
        """Reinitializes any internal parameters before processing a new video."""
        pass


class MaskStepMixin(ABC):
    """Interface for distortion steps that can transform an existing mask."""

    @abstractmethod
    def apply_to_mask(
        self,
        mask: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Apply the same spatial/intensity transform to a mask as to the image."""
        pass
