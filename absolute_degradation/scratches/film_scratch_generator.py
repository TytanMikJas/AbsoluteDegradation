from dataclasses import dataclass

import numba
import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class ScratchConfig:
    probability: float = 0.3
    count_range: tuple[int, int] = (4, 6)
    width_range_microns: tuple[float, float] = (5.0, 60.0)
    scan_resolution_dpi: int = 2000
    intensity_range: tuple[float, float] = (0.2, 0.8)
    weave_freq_hz: float = 2.0
    weave_amplitude_pixels: float = 5.0
    flicker_freq_hz: float = 12.0
    prob_curved: float = 0.1
    lifespan_range: tuple[int, int] = (15, 60)
    spawn_probability: float = 0.3
    prob_white: float = 0.8


@numba.jit(nopython=True, cache=True, fastmath=True)
def _generate_kernel_atlas(max_width: float, step: float) -> tuple[np.ndarray, int]:
    """
    Pre-calculates kernel lookup table.
    """
    num_kernels = int(max_width / step) + 2
    max_radius = int(max_width * 2.0)
    max_len = 2 * max_radius + 1

    atlas = np.zeros((num_kernels, max_len), dtype=np.float32)

    for i in range(num_kernels):
        w_px = i * step
        if w_px < 0.1:
            continue

        sigma = w_px * 1.5
        radius = int(w_px * 2.0)
        center_idx = max_radius

        for r in range(-radius, radius + 1):
            x = r
            scaled_x = x / (w_px / 2.0)
            pi_scaled = np.pi * scaled_x
            if abs(pi_scaled) < 1e-8:
                sinc_val = 1.0
            else:
                sinc_val = np.sin(pi_scaled) / pi_scaled

            val = (sinc_val * sinc_val) * np.exp(-0.5 * (x / sigma) ** 2)
            atlas[i, center_idx + r] = val

    return atlas, max_radius


@numba.jit(nopython=True, cache=True, fastmath=True)
def _render_scratch_atlas_signed(
    scratch_layer: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    width_indices: np.ndarray,
    intensities: np.ndarray,
    sign: int,
    kernel_atlas: np.ndarray,
    atlas_center: int,
    w: int,
) -> None:
    """
    Renders scratch into int16 accumulator.
    Negative values darken the image, Positive values brighten it.
    """
    n_pixels = len(y_coords)

    for i in range(n_pixels):
        intensity = intensities[i]
        if intensity < 0.01:
            continue

        val_base = intensity * 255.0 * sign

        y = y_coords[i]
        x_center = int(round(x_coords[i]))
        w_idx = width_indices[i]

        kernel_row = kernel_atlas[w_idx]

        safe_rad = w_idx + 2
        k_start_idx = atlas_center - safe_rad
        x_start = x_center - safe_rad
        x_end = x_center + safe_rad + 1

        if x_start < 0:
            k_start_idx += 0 - x_start
            x_start = 0
        if x_end > w:
            x_end = w

        if x_start >= x_end:
            continue

        for j in range(x_end - x_start):
            x_pos = x_start + j
            k_val = kernel_row[k_start_idx + j]

            delta = k_val * val_base

            scratch_layer[y, x_pos] += np.int16(delta)


@numba.jit(nopython=True, cache=True, fastmath=True)
def _blend_frames_signed_roi(
    frame: np.ndarray,
    scratch_layer: np.ndarray,  # int16
    x_min: int,
    x_max: int,
) -> np.ndarray:
    """
    Blends int16 scratch layer onto uint8 frame.
    Handles clamping for both darkening and brightening.
    """
    h, w = frame.shape[:2]
    is_color = frame.ndim == 3
    output = frame.copy()

    if x_min < 0:
        x_min = 0
    if x_max > w:
        x_max = w
    if x_min >= x_max:
        return output

    for y in range(h):
        for x in range(x_min, x_max):
            delta = scratch_layer[y, x]

            if delta != 0:
                if is_color:
                    for c in range(3):
                        val = int(frame[y, x, c]) + delta
                        if val < 0:
                            val = 0
                        elif val > 255:
                            val = 255
                        output[y, x, c] = np.uint8(val)
                else:
                    val = int(frame[y, x]) + delta
                    if val < 0:
                        val = 0
                    elif val > 255:
                        val = 255
                    output[y, x] = np.uint8(val)

    return output


class FilmScratchGenerator:
    def __init__(
        self,
        fps: int = 24,
        config: ScratchConfig = ScratchConfig(),
        seed: int | None = None,
    ):
        self.fps = fps
        self.rng = np.random.default_rng(seed)
        self.config = config

        # Atlas Generation
        self.atlas_step = 0.5
        max_supported_width = 100.0
        self.kernel_atlas, self.atlas_center = _generate_kernel_atlas(
            max_supported_width, self.atlas_step
        )

        self._reset_noise_buffer()

    def _reset_noise_buffer(self) -> None:
        """Regenerate the noise buffer and reset the read index."""
        self._noise_buffer = self.rng.standard_normal(100000).astype(np.float32)
        self._noise_idx = 0

    def _get_noise_slice(self, size: int) -> np.ndarray:
        if self._noise_idx + size >= len(self._noise_buffer):
            self._noise_idx = 0
        sl = self._noise_buffer[self._noise_idx : self._noise_idx + size]
        self._noise_idx += size
        return sl

    def _generate_temporal_drift(
        self, num_frames: int, freq_hz: float, amp: float
    ) -> np.ndarray:
        period_frames = max(1, int(self.fps / freq_hz))
        num_control_points = max(2, num_frames // period_frames + 2)
        control_points = self.rng.uniform(-1, 1, size=num_control_points) * amp
        x_control = np.linspace(0, num_frames, num_control_points)
        cs = CubicSpline(x_control, control_points)
        return cs(np.arange(num_frames))

    def _generate_spatial_path(
        self, height: int, center_x: float, drift_offset: float, is_curved: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        y_coords = np.arange(height)
        if not is_curved:
            jitter = self._get_noise_slice(height) * 0.3
            jitter = np.convolve(jitter, np.ones(5) / 5, mode="same")
            path_x = center_x + drift_offset + jitter
        else:
            num_knots = 5
            knot_y = np.linspace(0, height, num_knots)
            knot_x_offsets = self.rng.uniform(-10, 10, size=num_knots)
            cs = CubicSpline(knot_y, knot_x_offsets)
            path_x = center_x + drift_offset + cs(y_coords)
        return y_coords, path_x.astype(np.float32)

    def _create_scratch_instance(
        self, h: int, w: int, n_frames: int, start_frame: int = 0
    ) -> dict | None:
        width_microns = self.rng.uniform(*self.config.width_range_microns)
        width_px = max(1.0, (width_microns / 25400.0) * self.config.scan_resolution_dpi)

        lifespan = self.rng.integers(*self.config.lifespan_range)
        birth_frame = start_frame
        death_frame = min(start_frame + lifespan, n_frames)
        lifespan_actual = death_frame - birth_frame
        if lifespan_actual <= 0:
            return None

        drift = self._generate_temporal_drift(
            lifespan_actual,
            self.config.weave_freq_hz,
            self.config.weave_amplitude_pixels,
        )
        flicker = 1.0 + self._generate_temporal_drift(
            lifespan_actual, self.config.flicker_freq_hz, 0.2
        )

        min_seg = int(0.5 * h)
        max_seg = int(1 * h)
        seg_height = self.rng.integers(min_seg, max_seg + 1)
        y_start = self.rng.integers(0, h - seg_height + 1)
        y_end = y_start + seg_height

        return {
            "center_x": self.rng.uniform(0, w),
            "drift": drift,
            "flicker": flicker,
            "base_width_px": width_px,
            "base_intensity": self.rng.uniform(*self.config.intensity_range),
            "is_curved": self.rng.random() < self.config.prob_curved,
            "y_start": y_start,
            "y_end": y_end,
            "birth_frame": birth_frame,
            "death_frame": death_frame,
            "sign": 1 if self.rng.random() < self.config.prob_white else -1,
        }

    def apply_vertical_scratches(
        self, frames: list[np.ndarray], return_mask: bool = False
    ) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]:
        if not frames:
            if return_mask:
                return frames, []
            return frames

        if self.rng.random() >= self.config.probability:
            if return_mask:
                zero_masks = [
                    np.zeros(frame.shape[:2], dtype=np.float32) for frame in frames
                ]
                return frames, zero_masks
            return frames

        self._reset_noise_buffer()
        n_frames = len(frames)
        h, w = frames[0].shape[:2]

        active_scratches = []
        max_scratches = self.config.count_range[1]

        initial_count = self.rng.integers(self.config.count_range[0], max_scratches + 1)
        for _ in range(initial_count):
            s = self._create_scratch_instance(h, w, n_frames, 0)
            if s:
                active_scratches.append(s)

        output_frames = []
        output_masks: list[np.ndarray] = []
        scratch_layer = np.zeros((h, w), dtype=np.int16)

        for t in range(n_frames):
            active_scratches = [s for s in active_scratches if s["death_frame"] > t]
            if (
                len(active_scratches) < max_scratches
                and self.rng.random() < self.config.spawn_probability
            ):
                s = self._create_scratch_instance(h, w, n_frames, t)
                if s:
                    active_scratches.append(s)

            frame_min_x, frame_max_x = w, 0
            has_work = False
            scratch_layer.fill(0)

            for s in active_scratches:
                rel_t = t - s["birth_frame"]
                if rel_t < 0 or rel_t >= len(s["drift"]):
                    continue

                seg_h = s["y_end"] - s["y_start"]
                if seg_h <= 0:
                    continue

                y_coords, path_x = self._generate_spatial_path(
                    seg_h, s["center_x"], s["drift"][rel_t], s["is_curved"]
                )
                y_coords += s["y_start"]

                noise_w = self._get_noise_slice(seg_h) * 0.5
                width_px_vec = np.clip(
                    s["base_width_px"] + np.cumsum(noise_w), 1.0, 90.0
                )
                width_indices = (width_px_vec / self.atlas_step).astype(np.int32)

                noise_i = self._get_noise_slice(seg_h)
                noise_i = (noise_i - noise_i.min()) / (
                    noise_i.max() - noise_i.min() + 1e-6
                )

                intensity_vec = (
                    s["base_intensity"]
                    * np.clip(s["flicker"][rel_t], 0.5, 1.5)
                    * (0.7 + 0.6 * noise_i)
                ).astype(np.float32)

                dash_vals = self._get_noise_slice(seg_h)
                intensity_vec[dash_vals < -1.4] = 0.0

                _render_scratch_atlas_signed(
                    scratch_layer,
                    y_coords,
                    path_x,
                    width_indices,
                    intensity_vec,
                    s["sign"],
                    self.kernel_atlas,
                    self.atlas_center,
                    w,
                )

                curr_min = int(np.min(path_x) - 10)
                curr_max = int(np.max(path_x) + 10)
                if curr_min < frame_min_x:
                    frame_min_x = curr_min
                if curr_max > frame_max_x:
                    frame_max_x = curr_max
                has_work = True

            if has_work:
                out_frame = _blend_frames_signed_roi(
                    frames[t], scratch_layer, frame_min_x, frame_max_x
                )
                output_frames.append(out_frame)
                if return_mask:
                    scratch_mask = np.clip(
                        np.abs(scratch_layer).astype(np.float32) / 255.0,
                        0.0,
                        1.0,
                    )
                    output_masks.append(scratch_mask)
            else:
                output_frames.append(frames[t].copy())
                if return_mask:
                    output_masks.append(np.zeros((h, w), dtype=np.float32))

        if return_mask:
            return output_frames, output_masks
        return output_frames
