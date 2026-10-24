import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from hydra.utils import to_absolute_path

from absolute_degradation.distortions.noise_type import NoiseType
from absolute_degradation.distortions.steps.abstract_distort_step import (
    AbstractDistortStep,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class FilmGrainStep(AbstractDistortStep):
    """
    Physics-inspired Film Grain Generator.
    Implements:
    1. Toroidal Shifting (Boiling): O(1) temporal noise generation via random rolling.
    2. Noise Basis: Blue Noise (if provided) for realistic Silver Halide crystal
        distribution, or White Gaussian Noise as fallback.
    3. Signal-Dependent Variance (Heteroscedasticity): H&D Curve approximation
        where grain intensity scales with pixel intensity.
    4. Scanner Noise Floor: Additive electronic noise from the digitization process.
    """

    def __init__(
        self,
        probability: float = 1.0,
        iso_range: tuple[float, float] = (0.05, 0.2),
        grain_size_range: tuple[float, float] = (1.0, 1.5),
        gamma_range: tuple[float, float] = (0.3, 0.7),
        scanner_noise_sigma_range: tuple[float, float] = (0.002, 0.01),
        noise_std_bounds: tuple[float, float] = (-0.05, 0.05),
        noise_variance_range: float | tuple[float, float] = 0.05,
        blue_noise_path: str | None = "data/blue_noise/blue_noise_2048_5ch.npy",
        use_film_grain_only: bool = False,
        heteroscedastic_film_grain_prob: float = 0.25,
        homoscedastic_film_grain_prob: float = 0.25,
        gaussian_noise_prob: float = 0.25,
        speckle_noise_prob: float = 0.25,
    ) -> None:
        """
        Args:
            probability: Chance of applying the step.
            iso_range: Range (min, max) for film speed / grain intensity scaling factor.
                    Sampled uniformly each application. Typical: (0.05, 0.15).
            grain_size_range: Range (min, max) for relative grain size.
                    1.0 = fine, >1.0 = coarse. Typical: (1.0, 1.4).
            gamma_range: Range (min, max) for signal dependence exponent.
                    0.5 = Poissonian/Square-Root law.
            scanner_noise_sigma_range: Range (min, max) for std dev of additive
                    electronic noise (signal-independent).
            noise_std_bounds (float): Min/Max values for noise standard deviation.
            noise_variance_range (float | tuple[float, float]): Range for noise standard deviation variance.
                    If float, interpreted as (-value, value). If tuple, interpreted as (min, max).
            blue_noise_path: Optional path to pre-computed Blue Noise texture (.npy).
                            Blue noise better simulates the hard-core repulsion of
                            silver halide crystals. If None, uses White Gaussian Noise.
            heteroscedastic_film_grain_prob: Probability of using heteroscedastic (signal-dependent) film grain.
            homoscedastic_film_grain_prob: Probability of using homoscedastic (uniform) film grain.
            gaussian_noise_prob: Probability of using additive Gaussian noise.
            speckle_noise_prob: Probability of using multiplicative speckle noise.
        """
        super().__init__(probability=probability)

        self.iso_range = iso_range
        self.grain_size_range = grain_size_range
        self.gamma_range = gamma_range
        self.scanner_noise_sigma_range = scanner_noise_sigma_range
        self._noise_std_bounds = noise_std_bounds
        self._noise_variance_range = noise_variance_range
        self.noise_basis = self._load_or_generate_noise_basis(blue_noise_path)
        self.use_film_grain_only = use_film_grain_only

        self.heteroscedastic_film_grain_prob = heteroscedastic_film_grain_prob
        self.homoscedastic_film_grain_prob = homoscedastic_film_grain_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.speckle_noise_prob = speckle_noise_prob

        self._current_noise_type = NoiseType.HETEROSCEDASTIC_FILM_GRAIN

    def rebuild(self) -> None:
        """Sample distortion parameters for new video sequence."""

        self._noise_std_center = random.uniform(
            self._noise_std_bounds[0], self._noise_std_bounds[1]
        )

        if self.use_film_grain_only:
            self._current_noise_type = NoiseType.HETEROSCEDASTIC_FILM_GRAIN
        else:
            probs = [
                self.heteroscedastic_film_grain_prob,
                self.homoscedastic_film_grain_prob,
                self.gaussian_noise_prob,
                self.speckle_noise_prob,
            ]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                probs = [0.25, 0.25, 0.25, 0.25]

            self._current_noise_type = random.choices(
                NoiseType.all_types(), weights=probs
            )[0]

    @property
    def noise_std(self) -> float:
        """
        Returns augmented noise standard deviation.
        Clipped between 5/255 and 10/255.
        """
        std_variance = random.uniform(-0.5, 0.5) / 255.0
        return float(
            np.clip(self._noise_std_center + std_variance, 5.0 / 255.0, 10.0 / 255.0)
        )

    def _load_or_generate_noise_basis(
        self, blue_noise_path: str | None
    ) -> npt.NDArray[np.float32]:
        """
        Load blue noise texture or generate white Gaussian noise as fallback.

        Blue noise (Void-and-Cluster distribution) is preferred because it simulates
        the hard-core repulsion of silver halide crystals in the emulsion - crystals
        cannot occupy the same physical space, creating a uniform but random distribution
        lacking the low-frequency clumps found in white noise.

        Args:
            blue_noise_path: Path to .npy file containing blue noise texture
                            (H x W or H x W x C). Generate with tools/generate_blue_noise.py.

        Returns:
            Noise basis array (H x W or H x W x C), each channel normalized to
            zero-mean, unit variance.

        Raises:
            ValueError: If the file exists but contains no usable noise
                        (constant channel, NaN/inf or wrong shape).
        """
        if blue_noise_path:
            # First try resolving relative to the package root
            # (so the default "data/blue_noise/..." always works),
            # then fall back to Hydra's CWD-based resolution.
            abs_path = _PACKAGE_ROOT / blue_noise_path
            if not abs_path.exists():
                abs_path = Path(to_absolute_path(blue_noise_path))
            if abs_path.exists():
                noise: npt.NDArray[np.float32] = np.load(abs_path).astype(
                    np.float32, copy=False
                )
                return self._validate_noise_basis(noise, abs_path)
            else:
                warnings.warn(
                    f"Blue noise file not found at {abs_path}. Using White Gaussian Noise fallback."
                )

        warnings.warn(
            "Blue noise basis not found. Using White Gaussian Noise fallback. "
            "For more realistic grain, provide blue_noise_path."
        )
        noise = np.random.randn(2048, 2048).astype(np.float32, copy=False)
        return (noise - noise.mean()) / (noise.std() + 1e-8)

    @staticmethod
    def _validate_noise_basis(
        noise: npt.NDArray[np.float32], path: Path
    ) -> npt.NDArray[np.float32]:
        """
        Checks a loaded noise texture and normalizes every channel to zero-mean,
        unit variance. A broken file (e.g. all zeros) would otherwise silently
        disable film grain, so it fails loudly instead.
        """
        if noise.ndim not in (2, 3):
            raise ValueError(
                f"Blue noise at {path} must be HxW or HxWxC, got shape {noise.shape}."
            )
        if not np.isfinite(noise).all():
            raise ValueError(f"Blue noise at {path} contains NaN/inf values.")

        mean = noise.mean(axis=(0, 1), keepdims=True)
        std = noise.std(axis=(0, 1), keepdims=True)
        if (std < 1e-6).any():
            raise ValueError(
                f"Blue noise at {path} has constant channel(s) (std={std.ravel()}). "
                "Regenerate it with tools/generate_blue_noise.py."
            )
        return ((noise - mean) / std).astype(np.float32, copy=False)

    def _apply_speckle_noise(
        self, image: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Applies multiplicative Speckle noise. I + Noise * I, where Noise ~ N(0, std)."""
        gauss = np.random.normal(loc=0, scale=self.noise_std, size=image.shape)
        noisy = image + gauss * image
        noisy = np.clip(noisy, 0, 1).astype(np.float32)
        return noisy

    def _apply_gaussian_noise(
        self, image: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Applies additive Gaussian noise. I + Noise, where Noise ~ N(0, std)."""
        gauss = np.random.normal(loc=0, scale=self.noise_std, size=image.shape)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 1).astype(np.float32)
        return noisy

    def _apply_film_grain(
        self, image: npt.NDArray[np.float32], use_heteroscedastic: bool = True
    ) -> npt.NDArray[np.float32]:
        """
        Applies physics-inspired film grain noise.

        The noise model follows the General Signal-Dependent Noise formula:
            g(x,y) = f(x,y) + k * f(x,y)^(gamma) * u(x,y) + w(x,y)

        Where:
            - f(x,y) is the clean signal
            - k is the ISO scaling factor
            - gamma is the signal dependence exponent (0.5 for Poissonian statistics)
            - u(x,y) is the normalized grain noise (zero-mean, unit variance)
            - w(x,y) is the scanner/electronic noise floor

        Args:
            image: H x W grayscale image (Float32) in range [0, 1].
            use_heteroscedastic: If True, use signal-dependent (heteroscedastic) noise.
                                If False, use uniform (homoscedastic) noise.

        Returns:
            Degraded image with film grain, clipped to [0, 1].
        """
        iso = np.random.uniform(*self.iso_range)
        grain_size = np.random.uniform(*self.grain_size_range)
        scanner_noise_sigma = np.random.uniform(*self.scanner_noise_sigma_range)
        gamma = np.random.uniform(*self.gamma_range)

        h, w = image.shape

        if image.min() < 0 or image.max() > 1:
            image = np.clip(image, 0.0, 1.0)

        # --- Step 1: Toroidal Shift (The "Boiling" Effect) ---
        # Film grain is temporally independent - every frame has a unique crystal
        # arrangement. Instead of generating new noise (expensive), we roll the
        # existing texture with random offsets. This guarantees temporal independence
        # (zero correlation between frames) with O(1) computational cost.

        if self.noise_basis.ndim == 3:
            channel_idx = np.random.randint(0, self.noise_basis.shape[2])
            noise_basis_2d = self.noise_basis[:, :, channel_idx]
        else:
            noise_basis_2d = self.noise_basis

        basis_h, basis_w = noise_basis_2d.shape
        ox = np.random.randint(0, basis_w)
        oy = np.random.randint(0, basis_h)

        # Circular roll maintains noise properties at edges (toroidal topology)
        rolled = np.roll(noise_basis_2d, shift=(oy, ox), axis=(0, 1))

        # --- Step 2: Grain Morphology (Resizing) ---
        # Calculate crop size based on grain_size simulation.
        # If grain_size is 2.0, we crop half the resolution and zoom up,
        # effectively making the grain particles appear larger.
        crop_h = int(min(h / grain_size, basis_h))
        crop_w = int(min(w / grain_size, basis_w))

        noise_crop = rolled[:crop_h, :crop_w]

        grain_layer: npt.NDArray[np.float32]
        if grain_size != 1.0:
            # Zoom/interpolation acts as a low-pass filter, making grain "softer"
            # or "coarser". This simulates the difference between fine-grain (low ISO)
            # and coarse-grain (high ISO) film stocks.
            grain_layer = cv2.resize(
                noise_crop,
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32, copy=False)  # type: ignore
        else:
            grain_layer = noise_crop

        if grain_layer.shape[0] < h or grain_layer.shape[1] < w:
            # Pad if undersized (wrap mode maintains toroidal properties)
            pad_h = max(0, h - grain_layer.shape[0])
            pad_w = max(0, w - grain_layer.shape[1])
            grain_layer = np.pad(grain_layer, ((0, pad_h), (0, pad_w)), mode="wrap")

        # Crop to exact size
        grain_layer = grain_layer[:h, :w]

        # Linear interpolation is a low-pass filter and blue noise lives in the high
        # frequencies, so resizing drops its std to ~0.6. Renormalize so that iso
        # (variance_map) keeps its meaning regardless of grain_size.
        grain_layer = (grain_layer - grain_layer.mean()) / (grain_layer.std() + 1e-8)

        # --- Step 3: Signal Dependence (Heteroscedasticity vs Homoscedasticity) ---
        # - Heteroscedastic (Poissonian statistics from grain counting).
        #   In negative film, high-density areas (scene highlights) have the most grain.
        # - Homoscedastic: Uniform noise intensity across the image (simpler model).

        if use_heteroscedastic:
            variance_map = iso * np.power(np.maximum(image, 1e-3), gamma)
        else:
            variance_map = iso

        # --- Step 4: Scanner Noise (Electronic Noise Floor) ---
        # Signal-independent additive noise from the digitization process.
        # This is white Gaussian noise representing sensor read noise.
        scanner_floor = np.random.randn(h, w).astype(np.float32, copy=False)
        scanner_floor *= scanner_noise_sigma

        # --- Step 5: Composition (SDN Model) ---
        # g(x,y) = f(x,y) + k*f(x,y)^(gamma)*u(x,y) + w(x,y)
        # grain_layer has zero-mean, unit variance (renormalized above),
        # so variance_map correctly scales it.
        noisy_img = image + (grain_layer * variance_map) + scanner_floor

        return np.clip(noisy_img, 0.0, 1.0).astype(np.float32)

    def distort(
        self, image: npt.NDArray[np.float32], ground_truth: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Apply noise to a grayscale image.

        The noise type is sampled once per video sequence in rebuild() and remains
        temporally consistent across all frames in the clip:
        - Heteroscedastic Film grain: Signal-dependent noise where variance scales with intensity
        - Homoscedastic Film grain: Film grain with uniform noise intensity
        - Gaussian noise: Simple additive Gaussian noise
        - Speckle noise: Multiplicative speckle noise

        Args:
            image: H x W grayscale image (Float32) in range [0, 1].
            ground_truth: Corresponding ground truth image (passed through unchanged).

        Returns:
            Tuple of (degraded image with noise, ground_truth), both clipped to [0, 1].
        """
        match self._current_noise_type:
            case NoiseType.HETEROSCEDASTIC_FILM_GRAIN:
                return self._apply_film_grain(
                    image, use_heteroscedastic=True
                ), ground_truth
            case NoiseType.HOMOSCEDASTIC_FILM_GRAIN:
                return self._apply_film_grain(
                    image, use_heteroscedastic=False
                ), ground_truth
            case NoiseType.GAUSSIAN:
                return self._apply_gaussian_noise(image), ground_truth
            case NoiseType.SPECKLE:
                return self._apply_speckle_noise(image), ground_truth
            case _:
                raise ValueError(f"Unknown noise type: {self._current_noise_type}")
