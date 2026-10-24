from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

logger = logging.getLogger(__name__)
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _register_resolvers() -> None:
    """Register the custom OmegaConf resolvers used by the YAML configs."""
    for name, fn in [
        ("div", lambda a, b: float(a) / float(b)),
        ("sum", lambda a, b: int(a) + int(b)),
        ("mul", lambda a, b: int(a) * int(b)),
    ]:
        OmegaConf.register_new_resolver(name, fn, replace=True)


def _load_and_compose(
    overrides: dict[str, Any] | None = None,
    ablation: str | None = None,
) -> DictConfig:
    """
    Load the AbsoluteDegradation YAML configs and compose them into a
    single resolved DictConfig — mirroring what Hydra does for ``scripts/degrade.py``.

    Args:
        overrides: Key-value pairs merged on top of the root config before
            resolution (e.g. ``{"texture_path": "/abs/path", "seed": 123}``).
        ablation: Optional ablation folder name under ``config/ablations``.

    Returns:
        A fully composed and override-applied (but still lazy-resolved)
        DictConfig ready for ``hydra.utils.instantiate(cfg.degradation_pipeline)``.
    """
    _register_resolvers()

    config_root = _CONFIG_DIR / "ablations" / ablation if ablation else _CONFIG_DIR
    if ablation:
        logger.warning(
            "AbsoluteDegradation ablation active: '%s' (config root: %s)",
            ablation,
            config_root,
        )
    else:
        logger.info("AbsoluteDegradation baseline config root: %s", config_root)

    conf: DictConfig = OmegaConf.load(config_root / "conf.yaml")  # type: ignore[assignment]
    logger.warning("AbsoluteDegradation loaded conf: %s", config_root / "conf.yaml")

    pipeline_cfg: DictConfig = OmegaConf.load(  # type: ignore[assignment]
        config_root / "degradation_pipeline" / "default.yaml"
    )
    logger.warning(
        "AbsoluteDegradation loaded pipeline default: %s",
        config_root / "degradation_pipeline" / "default.yaml",
    )
    for level in ("light_steps", "medium_steps", "heavy_steps"):
        level_path = config_root / "degradation_pipeline" / level / "default.yaml"
        level_cfg = OmegaConf.load(level_path)
        logger.warning("AbsoluteDegradation loaded %s: %s", level, level_path)
        pipeline_cfg[level] = level_cfg

    if "defaults" in pipeline_cfg:
        with open_dict(pipeline_cfg):
            del pipeline_cfg["defaults"]

    conf["degradation_pipeline"] = pipeline_cfg

    if overrides:
        logger.info(f"Applying overrides to config: {overrides}")
        conf.merge_with(OmegaConf.create(overrides))

    return conf


def create_pipeline_from_config(
    texture_path: str | Path,
    seed: int | None = None,
    crop_size: int | None = None,
    include_mask: bool = False,
    overrides: dict[str, Any] | None = None,
    ablation: str | None = None,
):
    """
    Build a DegradationPipeline by loading the canonical YAML configs.

    This is the entry point that AbsoluteCinema's ``degradation.yaml`` calls
    via ``_target_``.  All distortion parameters come from the YAML files in
    ``AbsoluteDegradation/config/`` — the same files that ``scripts/degrade.py`` uses.

    Args:
        texture_path: Absolute path to ``AbsoluteDegradation/data/textures``.
        seed: Random seed forwarded to the scratch generator. ``None`` = random.
        crop_size: Override for ``crop_size`` (default in YAML is 256).
        include_mask: If True, instantiated pipeline returns degradation masks.
        overrides: Any additional key-value overrides applied to the root config.
        ablation: Optional ablation folder name under ``config/ablations``.

    Returns:
        A fully instantiated ``DegradationPipeline``.
    """
    ovr: dict[str, Any] = {"texture_path": str(texture_path)}
    if seed is not None:
        ovr["seed"] = seed
    if crop_size is not None:
        ovr["crop_size"] = crop_size
    if overrides:
        ovr.update(overrides)

    cfg = _load_and_compose(ovr, ablation=ablation)
    with open_dict(cfg.degradation_pipeline):
        cfg.degradation_pipeline.include_mask = include_mask

    OmegaConf.resolve(cfg)
    return instantiate(cfg.degradation_pipeline)
