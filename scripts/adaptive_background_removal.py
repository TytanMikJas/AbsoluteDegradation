import logging
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from absolute_degradation.textures.preprocessing.background_remover import (
    remove_background_with_adaptive_threshold,
)
from absolute_degradation.utils.io_utils import getfilelist

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../config", config_name="background_removal_config", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    input_root = Path(to_absolute_path(cfg.raw_texture_path)).resolve()
    output_root = Path(to_absolute_path(cfg.texture_out_path)).resolve()

    threshold_args = cfg.get(
        "threshold_args",
        {},
    )
    default_block_size = threshold_args.get("block_size", 35)
    default_C = threshold_args.get("C", 10)
    if not input_root.exists() or not input_root.is_dir():
        logger.error(f"Input path does not exist or is not a directory: {input_root}")
        raise SystemExit(1)

    if output_root.exists() and not output_root.is_dir():
        logger.error(f"Output path is not a directory: {output_root}")
        raise SystemExit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    textures_paths = getfilelist(input_root, extensions=cfg.valid_extensions)
    logger.info(f"Found {len(textures_paths)} textures to process.")

    texture_groups: dict[str, list[Path]] = {}
    for texture_path in textures_paths:
        group = texture_path.relative_to(input_root).parts[0]
        texture_groups.setdefault(group, []).append(texture_path)

    skip_groups = set(cfg.get("texture_skip_groups", []))
    for texture_group, texture_paths in texture_groups.items():
        if texture_group in skip_groups:
            logger.info(f"Skipping texture group: {texture_group}")
            continue

        for texture_path in tqdm(
            texture_paths, desc=f"Removing Backgrounds {texture_group}"
        ):
            relative_path = texture_path.relative_to(input_root)
            relative_path = relative_path.with_suffix(".png")
            output_path = output_root / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            remove_background_with_adaptive_threshold(
                texture_path,
                output_path,
                block_size=default_block_size,
                C=default_C,
            )


if __name__ == "__main__":
    main()
