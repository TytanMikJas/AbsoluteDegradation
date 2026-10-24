import numpy as np
import numpy.typing as npt

from absolute_degradation.distortions.steps.abstract_distort_step import (
    AbstractDistortStep,
)


class GammaCorrectionStep(AbstractDistortStep):
    """
    Applies gamma correction to an image.

    Gamma is sampled from a log-normal distribution:
        X ~ N(log_mean, log_std)
        gamma = exp(X)
    which keeps sampling centered around gamma=1 in log-space when log_mean=0.
    """

    def __init__(
        self,
        log_mean: float = 0.0,
        log_std: float = 0.125,
        gamma_clip: tuple[float, float] = (0.2, 4),
        probability: float = 1.0,
    ) -> None:
        """
        Initializes the GammaCorrectionStep.
        Args:
            log_mean (float): Mean of the normal distribution in log-space.
            log_std (float): Standard deviation in log-space.
            gamma_clip (tuple): Min and max values to clip sampled gamma.
            probability (float): Probability of applying this distortion.
        """
        super().__init__(probability=probability)
        self._log_mean = log_mean
        self._log_std = log_std
        self._gamma_clip = gamma_clip

    def rebuild(self) -> None:
        pass

    @property
    def gamma(self) -> float:
        x = np.random.normal(loc=self._log_mean, scale=self._log_std)
        sampled_gamma = float(np.exp(x))
        return float(np.clip(sampled_gamma, self._gamma_clip[0], self._gamma_clip[1]))

    def distort(
        self,
        image: npt.NDArray[np.float32],
        ground_truth: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Shifts the luminance of the input image by a specified value.
        Args:
            image (npt.NDArray[np.float32]): Input image H x W x C [0-1].
            ground_truth (npt.NDArray[np.float32]): Input image H x W x C [0-1].
        Returns:
            npt.NDArray[np.float32]: Gamma corrected image H x W x C [0-1].
        """
        gamma = self.gamma
        distorted_image = np.power(image.clip(0.0, 1.0), gamma)
        distorted_image = distorted_image.clip(0.0, 1.0)

        return distorted_image, ground_truth
