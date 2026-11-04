from typing import Literal

import numpy as np
import numpy.typing as npt

from absolute_degradation.distortions.steps.abstract_distort_step import (
    AbstractDistortStep,
    MaskStepMixin,
)


def _scale_crop(
    shift: int, padding: int, image_len: int, gt_len: int
) -> tuple[int, int]:
    """
    Converts padding and shift given in GT pixels to image pixels.

    Args:
        shift: Crop start in GT pixels (padding + offset, in [0, 2 * padding]).
        padding: Padding in GT pixels.
        image_len: Image size along the cropped axis.
        gt_len: Ground truth size along the cropped axis.
    Returns:
        (padding, shift) in image pixels, shift clipped to [0, 2 * padding].
    """
    if image_len == gt_len:
        return padding, shift
    scale = image_len / gt_len
    pad = int(round(padding * scale))
    img_shift = pad + int(round((shift - padding) * scale))
    return pad, int(np.clip(img_shift, 0, 2 * pad))


class YAxisMotionStep(AbstractDistortStep, MaskStepMixin):
    """
    Applies vertical jitter/motion distortion to video frames using a
    stochastic process for smooth, continuous movement. Uses 2nd order
    Ornstein-Uhlenbeck process for movement with
    varying shakeness throughout the video.
    Note that this function reduces the height of the frame by 2*padding.
    """

    def __init__(
        self,
        padding: int = 25,
        theta_avg: float = 0.25,
        theta_speed: float = 0.02,
        theta_std: float = 1.0,
        sigma_bounds: tuple[float, float] = (10, 20),
        ground_truth_motion: Literal["fixed", "moving"] = "fixed",
        probability: float = 1,
    ) -> None:
        """
        Initializes the vertical motion step with stochastic drift parameters.

        Args:
            padding: Max pixel offset for cropping; defines the vertical "shake" range.
            theta_avg: Baseline mean-reversion speed (how strongly it pulls to 0).
            theta_speed: How fast the reversion speed (theta) changes over time.
            theta_std: How much shakeness can vary throughout the video
            sigma_bounds: Min/max range for the diffusion (shake intensity).
            ground_truth_motion: If "moving", GT is cropped with the same shift as the frame. Otherwise it's center cropped.
            probability: Chance (0-1) of applying this step to a sequence.
        """
        super().__init__(probability)
        self._padding = padding
        self._theta_avg = theta_avg
        self._theta_speed = theta_speed
        self._theta_std = theta_std
        self._sigma_bounds = sigma_bounds
        self._ground_truth_motion = ground_truth_motion
        self._theta = self._theta_avg
        self.prev_shift: int = 0
        self.prev_pad: int = padding

    def rebuild(self) -> None:
        """sample distortion parameters for new video sequence"""
        self._sigma = np.random.uniform(self._sigma_bounds[0], self._sigma_bounds[1])
        self._mu_x = np.random.randn() * np.sqrt(
            np.abs(self._sigma / (2 * self._theta))
        )

    def get_theta(self) -> float:
        """
        Updates the current theta via a random walk and returns it.

        Returns:
            The updated mean-reversion speed, clipped to maintain stability.
        """
        dtheta = (
            self._theta_speed * (self._theta_avg - self._theta)
            + np.sqrt(2 * self._theta_std * self._theta_speed) * np.random.randn()
        )
        self._theta += dtheta

        return np.clip(self._theta, 0.25 * self._theta_avg, 2.5 * self._theta_avg)

    def get_shift(self, theta) -> int:
        """
        Calculates the vertical pixel offset based on the current state of self._mu_x process.

        Args:
            theta: The current mean-reversion speed.
        Returns:
            The total vertical shift in pixels including padding.
        """
        dx = (
            theta * (0 - self._mu_x)
            + np.sqrt(2 * theta * self._sigma) * np.random.randn()
        )
        self._mu_x += dx

        self._mu_x = np.clip(self._mu_x, -self._padding, self._padding)

        return self._padding + int(self._mu_x)

    def distort(
        self, image: npt.NDArray[np.float32], ground_truth: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Crops the frame vertically based on calculated shift;
        optionally shifts ground truth to match the frame shift.

        Args:
            image: Input image array [H, W, C].
            ground_truth: The target/clean image array.
        Returns:
            A tuple of (distorted_frame, processed_ground_truth)
            after applying vertical camera shakeness.
        """
        theta = self.get_theta()

        shift = self.get_shift(theta)

        pad, img_shift = _scale_crop(
            shift, self._padding, image.shape[0], ground_truth.shape[0]
        )
        self.prev_shift = img_shift
        self.prev_pad = pad

        distorted_frame = image[img_shift : img_shift + image.shape[0] - 2 * pad]

        if self._ground_truth_motion == "moving":
            ground_truth = ground_truth[
                shift : shift + ground_truth.shape[0] - 2 * self._padding
            ]
        else:
            ground_truth = ground_truth[
                self._padding : ground_truth.shape[0] - self._padding
            ]

        return distorted_frame, ground_truth

    def apply_to_mask(self, mask: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply motion distortion to a mask."""
        mask = mask[
            self.prev_shift : self.prev_shift + mask.shape[0] - 2 * self.prev_pad
        ]
        return mask


class XAxisMotionStep(AbstractDistortStep, MaskStepMixin):
    """
    Applies horizontal jitter/motion distortion to video frames,
    simulating camera shake or lateral instability.
    Note that this function reduces the width of the frame by 2*padding.
    """

    def __init__(
        self,
        padding: int = 8,
        theta: float = 1,
        sigma_bounds: tuple[float, float] = (0.5, 1),
        ground_truth_motion: Literal["fixed", "moving"] = "fixed",
        probability: float = 1,
    ) -> None:
        """
        Initializes the horizontal motion step

        Args:
            padding: Max horizontal pixel offset for cropping.
            theta: Constant mean-reversion speed for lateral movement.
            sigma_bounds: Range for horizontal noise intensity.
            ground_truth_motion: If "moving", GT follows the horizontal jitter.
            probability: Chance (0-1) of applying this step.
        """
        super().__init__(probability)
        self._padding = padding
        self._theta = theta
        self._sigma_bounds = sigma_bounds
        self._ground_truth_motion = ground_truth_motion
        self.prev_shift: int = 0
        self.prev_pad: int = padding

    def rebuild(self) -> None:
        """sample distortion parameters for new video sequence"""
        self._sigma = np.random.uniform(self._sigma_bounds[0], self._sigma_bounds[1])
        self._mu_x = np.random.randn() * np.sqrt(
            np.abs(self._sigma / (2 * self._theta))
        )

    def get_shift(self) -> int:
        """
        Calculates the vertical pixel offset based on the current state of self._mu_x process.

        Returns:
            The total horizontal shift in pixels including padding.
        """
        dx = (
            self._theta * (0 - self._mu_x)
            + np.sqrt(2 * self._theta * self._sigma) * np.random.randn()
        )
        self._mu_x += dx

        self._mu_x = np.clip(self._mu_x, -self._padding, self._padding)

        return self._padding + int(self._mu_x)

    def distort(
        self, image: npt.NDArray[np.float32], ground_truth: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Crops the image horizontally based on calculated shift;
        optionally shifts ground truth to match the image shift.

        Args:
            image: Input image array [H, W, C].
            ground_truth: The target/clean image array.
        Returns:
            A tuple of (distorted_frame, processed_ground_truth)
            after applying horizontal camera shakeness.
        """
        shift = self.get_shift()

        pad, img_shift = _scale_crop(
            shift, self._padding, image.shape[1], ground_truth.shape[1]
        )
        self.prev_shift = img_shift
        self.prev_pad = pad

        distorted_frame = image[:, img_shift : img_shift + image.shape[1] - 2 * pad]

        if self._ground_truth_motion == "moving":
            ground_truth = ground_truth[
                :, shift : shift + ground_truth.shape[1] - 2 * self._padding
            ]
        else:
            ground_truth = ground_truth[
                :, self._padding : ground_truth.shape[1] - self._padding
            ]

        return distorted_frame, ground_truth

    def apply_to_mask(self, mask: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Apply motion distortion to a mask."""
        mask = mask[
            :, self.prev_shift : self.prev_shift + mask.shape[1] - 2 * self.prev_pad
        ]
        return mask
