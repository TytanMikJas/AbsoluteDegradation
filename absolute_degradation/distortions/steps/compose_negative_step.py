import numpy as np
import numpy.typing as npt

from .compose_step import ComposeDistortStep


class ComposeNegativeStep(ComposeDistortStep):
    """
    Composes multiple distortions, but applies them on negative images.
    Mimics the negative film recording process, where the colors are inverted on the film,
    and then inverted back during copying.
    """

    def distort(
        self, image: npt.NDArray[np.float32], ground_truth: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Converts input frame to complementary (negative), applies distortions, and converts back to original color space (positive).
        """
        negative_image = np.clip(1.0 - image, 0.0, 1.0)
        distorted_negative_image, distorted_ground_truth = super().distort(
            negative_image, ground_truth
        )

        distorted_image = np.clip(1.0 - distorted_negative_image, 0.0, 1.0)
        return distorted_image, distorted_ground_truth
