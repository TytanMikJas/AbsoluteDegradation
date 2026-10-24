import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from absolute_degradation.distortions.steps.abstract_distort_step import (
    AbstractDistortStep,
    DistortOutput,
)
from absolute_degradation.textures.blending.blend_mode import BlendMode
from absolute_degradation.textures.blending.texture_blender import (
    moving_line_texture_blending,
    texture_blending,
)
from absolute_degradation.textures.texture import ColorSpace, Texture
from absolute_degradation.textures.texture_augmentation import texture_generator
from absolute_degradation.textures.texture_type import TextureType
from absolute_degradation.utils.image_utils import rgb2gray
from absolute_degradation.utils.io_utils import getfilelist


@dataclass
class _TextureSlot:
    """A single independently-cached texture with its own lifetime."""

    texture: Texture
    blend_mode: BlendMode
    frames_remaining: int


class ArtifactsDistortStep(AbstractDistortStep):
    """
    Adds visual artifacts (like dust, scratches, or moving lines) to a video.

    This step overlays random texture images onto the frames to simulate film
    damage or digital noise. It can handle "moving lines" that persist across
    frames for realism and applies various blending modes to mix the textures
    into the original video.
    """

    def __init__(
        self,
        texture_root: str,
        moving_lines_root: str,
        include_mask: bool = False,
        min_textures_per_frame: int = 1,
        max_textures_per_frame: int = 3,
        moving_line_probability: float = 0.2,
        lock_blending_mode_per_frame: bool = True,
        texture_durations: tuple[int, ...] = (1, 3, 5),
        texture_probs: tuple[float, ...] = (0.87, 0.1, 0.03),
        probability: float = 1,
    ) -> None:
        """
        Args:
            texture_root: Folder path containing standard artifact textures.
            moving_lines_root: Folder path containing textures for moving line effects.
            include_mask: If True, compute and return texture coverage masks.
            min_textures_per_frame: The minimum number of random textures to apply to each frame.
            max_textures_per_frame: The maximum number of random textures to apply to each frame.
            moving_line_probability: The chance (0-1) that a "moving line" effect will be active for the video.
            lock_blending_mode_per_frame: If True, all textures on a single frame use the same blending style.
            texture_durations: How many consecutive frames each texture set can persist for.
            texture_probs: Probability weights for each duration in texture_durations (must sum to 1).
            probability: The chance (0-1) that this entire artifact step will be used for a sequence.
        """
        super().__init__(probability)
        self.include_mask = include_mask
        self._min_textures_per_frame = min_textures_per_frame
        self._max_textures_per_frame = max_textures_per_frame
        self._moving_line_probability = moving_line_probability
        self._lock_blending_mode_per_frame = lock_blending_mode_per_frame
        self._texture_durations = texture_durations
        self._texture_probs = texture_probs
        self._validate_texture_probs()

        self._moving_line_texture_paths = TextureType.MOVING_LINE.get_files(
            Path(moving_lines_root)
        )
        self._texture_templates = getfilelist(Path(texture_root))

    def _validate_texture_probs(self) -> None:
        """Validate that texture_probs and texture_durations are consistent.

        Mirrors the old `TextureConfig.__post_init__` validation so YAML-provided
        lists are checked at step construction time.
        """
        if len(self._texture_probs) != len(self._texture_durations):
            raise ValueError(
                f"texture_probs length ({len(self._texture_probs)}) must equal "
                f"texture_durations length ({len(self._texture_durations)})"
            )

        if any(p < 0 for p in self._texture_probs):
            raise ValueError("texture_probs must contain non-negative values")

        total = sum(self._texture_probs)
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(
                f"texture_probs must sum to 1.0 (got {total:.12f}); "
                "adjust values or normalize before passing to ArtifactsDistortStep"
            )

    def rebuild(self) -> None:
        """Sample distortion parameters for new video sequence."""
        self._moving_line_flag = random.uniform(0, 1) < self._moving_line_probability

        self._line_texture: Texture | None = None
        self._line_mode: BlendMode = BlendMode.ADDITION
        self._last_line_texture: Texture | None = None
        self._texture_slots: list[_TextureSlot] = []

        self._tex_crop_offset_y: float = random.random()
        self._tex_crop_offset_x: float = random.random()

        if self._moving_line_flag and self._moving_line_texture_paths:
            selected_texture_path = random.choice(self._moving_line_texture_paths)
            self._line_texture = Texture(
                selected_texture_path, colorspace=ColorSpace.RGBA
            )
            self._line_mode = BlendMode.random_mode()

    def _make_texture_slot(
        self,
        image: npt.NDArray[np.float32],
        locked_blend_mode: BlendMode | None = None,
    ) -> _TextureSlot:
        """Create a single texture slot with its own independently-sampled lifetime."""
        h, w = image.shape[0], image.shape[1]
        selected_texture_path = random.choice(self._texture_templates)
        raw_tex = Texture(selected_texture_path, colorspace=ColorSpace.RGBA)
        texture = texture_generator(
            raw_tex,
            h,
            w,
            offset_y=self._tex_crop_offset_y,
            offset_x=self._tex_crop_offset_x,
        )

        blend_mode = locked_blend_mode or BlendMode.random_mode()
        duration = int(np.random.choice(self._texture_durations, p=self._texture_probs))
        return _TextureSlot(
            texture=texture, blend_mode=blend_mode, frames_remaining=duration
        )

    def _refresh_expired_slots(self, image: npt.NDArray[np.float32]) -> None:
        """Prune expired slots and spawn new ones up to a per-frame target.

        Each frame independently samples how many textures it wants.
        Expired slots are removed first, then new slots are created to fill the
        gap between surviving slots and the desired count. If more slots are
        still alive than the target, they are kept - they will expire naturally.
        """
        self._texture_slots = [
            slot for slot in self._texture_slots if slot.frames_remaining > 0
        ]

        num_desired_textures = random.randint(
            self._min_textures_per_frame, self._max_textures_per_frame
        )
        if self._moving_line_flag and self._line_texture:
            num_desired_textures = max(num_desired_textures - 1, 0)

        new_slots_needed = max(num_desired_textures - len(self._texture_slots), 0)
        if new_slots_needed == 0:
            return

        locked_blend_mode = (
            BlendMode.random_mode() if self._lock_blending_mode_per_frame else None
        )
        for _ in range(new_slots_needed):
            self._texture_slots.append(
                self._make_texture_slot(image, locked_blend_mode)
            )

    def distort(
        self,
        image: npt.NDArray[np.float32],
        ground_truth: npt.NDArray[np.float32],
    ) -> DistortOutput:
        """
        Randomly picks and blends textures onto the current image.

        Textures persist for a random number of consecutive frames controlled by
        ``texture_durations`` and ``texture_probs`` (sampled once per group).

        Args:
            image: The current video frame to be distorted.
            ground_truth: The clean version of the image (remains unchanged).

        Returns:
            If include_mask=False:
                - distorted image
                - original ground truth
            If include_mask=True:
                - distorted image
                - original ground truth
                - texture mask (H, W) float32 in [0, 1]
        """
        texture_mask: npt.NDArray[np.float32] | None = None
        if self.include_mask:
            texture_mask = np.zeros(image.shape[:2], dtype=np.float32)

        if self._moving_line_flag and self._line_texture:
            if self.include_mask:
                moving_result = moving_line_texture_blending(
                    image,
                    self._line_texture,
                    self._last_line_texture,
                    self._line_mode,
                    return_mask=True,
                )
                assert len(moving_result) == 3
                image, self._last_line_texture, moving_mask = moving_result
                assert texture_mask is not None
                texture_mask = 1.0 - (1.0 - texture_mask) * (1.0 - moving_mask)
            else:
                moving_result_no_mask = moving_line_texture_blending(
                    image,
                    self._line_texture,
                    self._last_line_texture,
                    self._line_mode,
                )
                assert len(moving_result_no_mask) == 2
                image, self._last_line_texture = moving_result_no_mask

        self._refresh_expired_slots(image)

        for slot in self._texture_slots:
            if self.include_mask:
                image, slot_mask = texture_blending(
                    image,
                    slot.texture,
                    blend_mode=slot.blend_mode,
                    augment=False,
                    return_mask=True,
                )
                assert texture_mask is not None
                texture_mask = 1.0 - (1.0 - texture_mask) * (1.0 - slot_mask)
            else:
                slot_result_no_mask = texture_blending(
                    image,
                    slot.texture,
                    blend_mode=slot.blend_mode,
                    augment=False,
                )
                assert not isinstance(slot_result_no_mask, tuple)
                image = slot_result_no_mask
            slot.frames_remaining -= 1

        image = rgb2gray(image)

        if not self.include_mask:
            return image, ground_truth

        assert texture_mask is not None
        texture_mask = np.clip(texture_mask, 0.0, 1.0).astype(np.float32)
        return image, ground_truth, texture_mask
