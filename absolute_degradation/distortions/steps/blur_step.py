import random
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt

from absolute_degradation.distortions.noise_blur import (
    anisotropic_Gaussian,
    fspecial_gaussian,
)
from absolute_degradation.distortions.steps.abstract_distort_step import (
    AbstractDistortStep,
    MaskStepMixin,
)


class FixedBlurDistortStep(AbstractDistortStep, MaskStepMixin):
    """
    For applying fixed blur distortion step. Applies isotropic or anisotropic gaussian blur to the input image.
    used in Bring Old Films Back to Life
    """

    def __init__(
        self,
        kernel_size_bounds: tuple[int, int],
        l1_center: float | None = None,
        l2_center: float | None = None,
        theta_center: float | None = None,
        mode: Literal["anisotropic", "isotropic"] | None = None,
        weight_anisotropic: float = 2.0,
        weight_isotropic: float = 2.0,
        probability: float = 1.0,
    ) -> None:
        """
        Initializes FixedBlurDistortStep
        Args:
            l1_center (float): Center value for l1 (axis 1 of gaussian kernel).
                Used in anisotropic and isotropic gaussian blur.
            l2_center (float): Center value for l2 (axis 2 of gaussian kernel).
                Used in anisotropic gaussian blur.
            theta_center (float): Center value for theta (rotation angle of gaussian kernel).
                Used in anisotropic gaussian blur.
            kernel_size (int): Size of the gaussian kernel.
            mode (Literal["anisotropic", "isotropic"]): Type of gaussian blur to apply.
            weight_anisotropic (float): Weight for anisotropic gaussian blur.
            weight_isotropic (float): Weight for isotropic gaussian blur.
            probability (float): Probability of applying this distortion.
        All center values are augmented with uniform random value
        """
        super().__init__(probability=probability)
        self._kernel_size_bounds = kernel_size_bounds
        self._l1_center_original = l1_center
        self._l2_center_original = l2_center
        self._theta_center_original = theta_center
        self._mode_original = mode
        self._weight_anisotropic = weight_anisotropic
        self._weight_isotropic = weight_isotropic

    def rebuild(self) -> None:
        """sample distortion parameters for new video sequence"""
        l1_center = (
            self._l1_center_original
            if self._l1_center_original is not None
            else random.uniform(0.4, 1.0)
        )
        l2_center = (
            self._l2_center_original
            if self._l2_center_original is not None
            else random.uniform(0.4, 1.0)
        )
        theta_center = (
            self._theta_center_original
            if self._theta_center_original is not None
            else random.uniform(0.0, 1.0)
        )

        self._l1_final = np.clip(l1_center + (random.random() - 0.5) * 0.2, 0.4, 1.0)
        self._l2_final = np.clip(l2_center + (random.random() - 0.5) * 0.2, 0.4, 1.0)
        self._theta_final = (
            np.clip(theta_center + (random.random() - 0.5) * 0.2, 0.0, 1.0) * np.pi
        )

        self._mode = (
            self._mode_original
            if self._mode_original
            else random.choice(["anisotropic", "isotropic"])
        )
        self._kernel_size = (
            2 * random.randint(self._kernel_size_bounds[0], self._kernel_size_bounds[1])
            + 3
        )

    def _get_anisotropic_gaussian_kernel(self) -> npt.NDArray[np.float32]:
        """Generates an anisotropic (elliptical) Gaussian kernel."""
        l1 = self._weight_anisotropic * self._l1_final
        l2 = self._weight_anisotropic * self._l2_final
        return anisotropic_Gaussian(
            ksize=self._kernel_size, theta=self._theta_final, l1=l1, l2=l2
        )

    def _get_isotropic_gaussian_kernel(self) -> npt.NDArray[np.float32]:
        """Generates an isotropic (circular) Gaussian kernel."""
        sigma = self._weight_isotropic * self._l1_final
        k = fspecial_gaussian(self._kernel_size, sigma)
        return k

    def distort(
        self,
        image: npt.NDArray[np.float32],
        ground_truth: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Applies anisotropic or isotropic Gaussian blur based on params."""

        if self._mode == "anisotropic":
            k = self._get_anisotropic_gaussian_kernel()
        else:
            k = self._get_isotropic_gaussian_kernel()

        return cv2.filter2D(image, -1, k, borderType=cv2.BORDER_REFLECT), ground_truth  # type: ignore

    def apply_to_mask(self, mask: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply blur distortion to a mask."""
        if self._mode == "anisotropic":
            k = self._get_anisotropic_gaussian_kernel()
        else:
            k = self._get_isotropic_gaussian_kernel()

        blurred_mask = cv2.filter2D(mask, -1, k, borderType=cv2.BORDER_REFLECT)
        return np.clip(blurred_mask, 0.0, 1.0).astype(np.float32)
