from copy import deepcopy
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from absolute_degradation.textures.texture_type import TextureType
from absolute_degradation.utils.io_utils import (
    read_image,
    read_image_grayscale,
    read_texture,
)

EROSION_GROUP = ("002", "003", "004", "005", "006", "007", "009", "012")
SKIP_RESIZE_FOLDER = "008"


class ColorSpace(Enum):
    GRAYSCALE = "grayscale"
    RGB = "rgb"
    RGBA = "rgba"


class Texture:
    """
    Base container class for texture
    """

    def __init__(
        self, texture_path: Path, colorspace: ColorSpace = ColorSpace.GRAYSCALE
    ) -> None:
        """
        Args:
            texture_path (Path): Path to texture file or directory.
            colorspace (ColorSpace): The color space of the texture.
        """
        self.texture_path = texture_path
        self.texture_type = TextureType.from_path(texture_path)
        self.colorspace = colorspace
        self.image: npt.NDArray[np.float32] = self._load_texture()

    @property
    def resolution(self) -> tuple[int, int]:
        """Returns the resolution of the texture image as (height, width)."""
        return self.image.shape[:2]  # type: ignore

    def _load_texture(self) -> npt.NDArray[np.float32]:
        if not self.texture_path.exists():
            raise FileNotFoundError(f"Texture file not found: {self.texture_path}")
        match self.colorspace:
            case ColorSpace.RGB:
                return read_image(self.texture_path)
            case ColorSpace.RGBA:
                return read_texture(self.texture_path)
            case ColorSpace.GRAYSCALE:
                return read_image_grayscale(self.texture_path)
            case _:
                raise ValueError(f"Unsupported colorspace: {self.colorspace}")

    def should_erode(self) -> bool:
        """
        Checks if the texture can be eroded based on its folder name.
        Returns:
            bool: True if the texture can be eroded, False otherwise.
        """
        return self.texture_path.parent.name in EROSION_GROUP

    def should_skip_resize(self) -> bool:
        """
        Checks if the texture should skip resizing based on its folder name.
        Returns:
            bool: True if resizing should be skipped, False otherwise.
        """
        return self.texture_path.parent.name == SKIP_RESIZE_FOLDER

    def copy(self) -> "Texture":
        return deepcopy(self)

    def resize(self, target_h: int, target_w: int) -> None:
        """
        Resizes the texture image to the target height and width.
        Uses Lanczos interpolation.
        Args:
            target_h (int): Target height.
            target_w (int): Target width.
        """
        from absolute_degradation.textures.texture_augmentation import resize

        self.image = resize(self.image, target_h, target_w)

    def crop(self, top: int, left: int, height: int, width: int) -> None:
        """
        Crops the texture image to the specified region.
        Args:
            top (int): Top coordinate of the crop.
            left (int): Left coordinate of the crop.
            height (int): Height of the crop.
            width (int): Width of the crop.
        """

        self.image = self.image[top : top + height, left : left + width]

    def image_to_rgba(self) -> npt.NDArray[np.float32]:
        """
        Gets image of the texture image in RGBA format.
        Handles both grayscale and RGB images.
        Returns:
            npt.NDArray[np.float32]: The texture image in RGBA format.
        """
        match self.colorspace:
            case ColorSpace.RGBA:
                assert self.image.ndim == 3 and self.image.shape[2] == 4, (
                    "RGBA image should have 4 channels"
                )
                return self.image.copy()  # type: ignore
            case ColorSpace.RGB:
                assert self.image.ndim == 3 and self.image.shape[2] == 3, (
                    "RGB image should have 3 channels"
                )
                return cv2.cvtColor(self.image, cv2.COLOR_RGB2RGBA)  # type: ignore
            case ColorSpace.GRAYSCALE:
                assert self.image.ndim == 2, "Grayscale image should have 2 dimensions"
                return cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGBA)  # type: ignore
            case _:
                raise ValueError(f"Unsupported colorspace: {self.colorspace}")
