import logging
import random
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from absolute_degradation.degradation_degree import DegradationDegree
from absolute_degradation.pipeline import process_video

logger = logging.getLogger(__name__)


def set_seed(seed: int | None) -> None:
    """Set seed for reproducibility."""
    if seed is None:
        return
    logging.info(f"Using seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)


@hydra.main(config_path="../config", config_name="conf", version_base="1.1")
def main(cfg: DictConfig) -> None:
    cfg.input_path = to_absolute_path(cfg.input_path)
    cfg.output_path = to_absolute_path(cfg.output_path)
    cfg.texture_path = to_absolute_path(cfg.texture_path)
    set_seed(cfg.seed)

    OmegaConf.register_new_resolver("div", lambda a, b: float(a) / float(b))
    OmegaConf.register_new_resolver("sum", lambda a, b: int(a) + int(b))
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b))

    if not Path(cfg.input_path).exists() or not Path(cfg.input_path).is_dir():
        logger.error(
            f"Input path does not exist or is not a directory: {cfg.input_path}"
        )
        raise SystemExit(1)

    if not Path(cfg.texture_path).exists() or not Path(cfg.texture_path).is_dir():
        logger.error(
            f"Texture path does not exist or is not a directory: {cfg.texture_path}"
        )
        raise SystemExit(1)

    video_dirs = sorted([d for d in Path(cfg.input_path).iterdir() if d.is_dir()])

    logger.info(f"Found {len(video_dirs)} videos to process.")

    degradation_pipeline = hydra.utils.instantiate(cfg.degradation_pipeline)

    for i, video_dir in enumerate(tqdm(video_dirs, desc="Processing Videos")):
        degree = DegradationDegree.from_REDS_index(i)
        process_video(
            video_dir,
            degradation_pipeline,
            Path(cfg.output_path),
            degree,
            cfg.valid_extensions,
        )


if __name__ == "__main__":
    main()
