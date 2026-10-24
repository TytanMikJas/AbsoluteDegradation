from enum import Enum
from pathlib import Path

from absolute_degradation.utils.io_utils import getfilelist


class TextureType(Enum):
    """
    Enum for different types of textures.
    """

    # TODO: add more texture types when determined
    MOVING_LINE = "moving_line"
    VERTICAL_SCRATCHES = "vertical_scratches"
    PURE_NOISE = "pure_noise"
    UNKNOWN_TYPE = "unknown_type"

    def get_path(self, texture_root: Path) -> Path | None:
        """
        Returns the file path associated with the texture type.
        Args:
            texture_root (Path): The root directory where textures are stored.
        Returns:
            Path | None: The path to the texture type directory, or None if unknown.
        """
        texture_paths = {
            TextureType.MOVING_LINE: texture_root / "001",
            TextureType.VERTICAL_SCRATCHES: texture_root / "011",
            TextureType.PURE_NOISE: texture_root / "008",
        }
        return texture_paths.get(self)

    def get_files(
        self, texture_root: Path, extensions=("png", "jpg", "jpeg")
    ) -> list[Path]:
        texture_dir = self.get_path(texture_root)
        if texture_dir is None or not texture_dir.exists():
            raise FileNotFoundError(f"Texture directory not found: {texture_dir}")
        return getfilelist(texture_dir, extensions)

    @classmethod
    def from_path(cls, texture_path: Path) -> "TextureType":
        """
        Determines the texture type based on the given file path.
        Args:
            texture_path (Path): The path to the texture file.
        Returns:
            TextureType: The corresponding texture type.
        Raises:
            ValueError: If the texture type cannot be determined from the path.
        """
        if texture_path.parent.name == "001":
            return cls.MOVING_LINE
        if texture_path.parent.name == "011":
            return cls.VERTICAL_SCRATCHES
        if texture_path.parent.name == "008":
            return cls.PURE_NOISE
        return cls.UNKNOWN_TYPE
