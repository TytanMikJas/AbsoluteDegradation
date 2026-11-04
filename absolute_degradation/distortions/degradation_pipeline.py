import logging
import random
from collections.abc import Iterable

import numpy as np

from absolute_degradation.degradation_degree import DegradationDegree
from absolute_degradation.distortions.steps.abstract_distort_step import (
    AbstractDistortStep,
    DistortOutput,
)
from absolute_degradation.scratches.film_scratch_generator import FilmScratchGenerator

logger = logging.getLogger(__name__)


class DegradationPipeline:
    """
    Orchestrates the application of frame-by-frame video degradations.

    This pipeline selects a distortion intensity (Light, Medium, or Heavy) and
    processes a sequence of frames through a consistent degradation strategy,
    ensuring temporal coherence by rebuilding distortion parameters only at the
    start of a new stream.
    """

    def __init__(
        self,
        light_steps: AbstractDistortStep,
        medium_steps: AbstractDistortStep,
        heavy_steps: AbstractDistortStep,
        probabilities: tuple[float, ...] = (0.33, 0.34, 0.33),
        scratch_generator: FilmScratchGenerator | None = None,
        include_mask: bool = False,
    ) -> None:
        """
        Args:
            light_steps: Distortion sequence for subtle quality reduction.
            medium_steps: Distortion sequence for moderate quality reduction.
            heavy_steps: Distortion sequence for severe quality reduction.
            probabilities: A list of three floats representing the selection weight
                for [LIGHT, MEDIUM, HEAVY] degrees respectively.
            scratch_generator: Optional FilmScratchGenerator instance. When
                provided, the generator's `config.probability` controls the
                chance (0-1) that film scratches will be applied to a given video.
            include_mask: If True, ``__call__`` returns degradation masks.
        """
        self.pipelines = {
            DegradationDegree.LIGHT: light_steps,
            DegradationDegree.MEDIUM: medium_steps,
            DegradationDegree.HEAVY: heavy_steps,
        }

        self.probabilities = probabilities
        self.scratch_generator = scratch_generator
        self.include_mask = include_mask

        for step in self.pipelines.values():
            self._propagate_include_mask(step, include_mask)

    @staticmethod
    def _propagate_include_mask(step: AbstractDistortStep, include_mask: bool) -> None:
        """Recursively set include_mask on steps that support it."""
        if hasattr(step, "include_mask"):
            setattr(step, "include_mask", include_mask)

        nested_steps = getattr(step, "_distortions", None)
        if isinstance(nested_steps, Iterable) and not isinstance(
            nested_steps, (str, bytes)
        ):
            for substep in nested_steps:
                DegradationPipeline._propagate_include_mask(substep, include_mask)

    def sample_degradation_degree(self) -> DegradationDegree:
        """
        Randomly selects a degradation level based on the defined probabilities.

        Returns:
            The selected DegradationDegree enum (LIGHT, MEDIUM, or HEAVY).
        """
        return random.choices(
            [
                DegradationDegree.LIGHT,
                DegradationDegree.MEDIUM,
                DegradationDegree.HEAVY,
            ],
            weights=self.probabilities,
            k=1,
        )[0]

    def __call__(
        self, stream, degree: DegradationDegree | None = None
    ) -> tuple[list, list] | tuple[list, list, list]:
        """
        Applies a consistent degradation strategy to a stream of video frames.

        Args:
            stream: An iterable sequence of frames (e.g., a list or generator of image arrays).
            degree: Optional; Force a specific degradation level. If None, one is sampled
                randomly using internal probabilities.

        Returns:
            A tuple containing:
                - degraded_frames: A list of processed/distorted frames.
                - ground_truths: A list of clean frames, potentially cropped or
                    aligned to match the distorted output.
                - degradation_masks (optional): A list of float32 masks (H, W)
                    in [0, 1], returned only when ``include_mask=True``.
        """
        degraded_frames = []
        ground_truths = []
        degradation_masks: list[np.ndarray] = []

        method = self.pipelines[degree or self.sample_degradation_degree()]
        method.rebuild()

        for frame in stream:
            ground_truth = frame.copy()
            step_mask: np.ndarray | None = None

            method_output: DistortOutput = method(frame, ground_truth)
            if len(method_output) == 3:
                degraded_frame, ground_truth, step_mask = method_output
            else:
                degraded_frame, ground_truth = method_output

            degraded_frames.append(degraded_frame)
            ground_truths.append(ground_truth)

            if self.include_mask:
                if step_mask is None:
                    # possibly will change after Motion Step Moved
                    step_mask = np.zeros(degraded_frame.shape[:2], dtype=np.float32)
                elif step_mask.shape != degraded_frame.shape[:2]:
                    raise RuntimeError(
                        "Mask/image shape mismatch at pipeline output: "
                        f"mask={step_mask.shape}, frame={degraded_frame.shape[:2]}. "
                        "Ensure geometry-changing distortions implement apply_to_mask()."
                    )
                degradation_masks.append(
                    np.clip(step_mask, 0.0, 1.0).astype(np.float32)
                )

        if self.scratch_generator:
            if self.include_mask:
                scratch_result = self.scratch_generator.apply_vertical_scratches(
                    degraded_frames,
                    return_mask=True,
                )
                assert isinstance(scratch_result, tuple)
                degraded_frames, scratch_masks = scratch_result
                degradation_masks = [
                    np.clip(1.0 - (1.0 - m) * (1.0 - s), 0.0, 1.0).astype(np.float32)
                    for m, s in zip(degradation_masks, scratch_masks)
                ]
            else:
                scratch_only = self.scratch_generator.apply_vertical_scratches(
                    degraded_frames
                )
                assert isinstance(scratch_only, list)
                degraded_frames = scratch_only

        if self.include_mask:
            return degraded_frames, ground_truths, degradation_masks
        return degraded_frames, ground_truths
