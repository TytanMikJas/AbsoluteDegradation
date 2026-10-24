import logging
from pathlib import Path

from absolute_degradation.degradation_degree import DegradationDegree
from absolute_degradation.distortions.degradation_pipeline import DegradationPipeline
from absolute_degradation.utils.io_utils import (
    get_image_generator,
    save_frame,
)

logger = logging.getLogger(__name__)


def process_video(
    video_dir: Path,
    degradation_pipeline: DegradationPipeline,
    output_root: Path,
    degree: DegradationDegree,
    valid_extensions: list[str],
) -> None:
    """
    Process a single video directory: load frames, apply degradation, and save results.
    """
    frame_files = sorted(
        [
            f
            for f in video_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
    )

    if not frame_files:
        logger.warning(f"No valid images found in {video_dir}. Skipping.")
        return

    image_gen = get_image_generator(frame_files)

    degraded_frames, gt_frames = degradation_pipeline(
        image_gen,
        degree=degree,
    )

    degraded_dir = output_root / "degraded" / video_dir.name
    gt_dir = output_root / "gt" / video_dir.name

    for name, deg_img, gt_img in zip(frame_files, degraded_frames, gt_frames):
        save_frame(degraded_dir / name.name, deg_img)
        save_frame(gt_dir / name.name, gt_img)
