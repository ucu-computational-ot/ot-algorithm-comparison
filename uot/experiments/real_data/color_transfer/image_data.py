import os
import numpy as np
from PIL import Image
from jax import numpy as jnp
from uot.data.dataset_loader import (
    load_image_as_color_grid,
    convert_rgb_to_color_space,
    convert_color_space_to_rgb,
)
from uot.utils.types import ArrayLike

from uot.utils.logging import logger

class ImageData:

    image_dir = None

    def __init__(
        self,
        name: str,
        bin_num: int = 32,
        *,
        color_space: str = "rgb",
        active_channels: list[int] | None = None,
    ):
        self.name = name
        self._bin_num = bin_num
        self._image_path = os.path.join(ImageData.images_dir, name)
        self.color_space = color_space
        self.active_channels = active_channels
        self._np_grid = None
        self._np_image = None
        self._np_color_image = None
        self._np_rgb_image = None
        self._jax_grid = None
        self._jax_image = None

    def _load_grid(self):
        if self._np_grid is None:
            logger.info(
                "Loading color histogram for %s (bins=%d)...",
                self._image_path,
                self._bin_num,
            )
            self._np_grid = load_image_as_color_grid(
                self._image_path,
                bins_per_channel=self._bin_num,
                color_space=self.color_space,
                active_channels=self.active_channels,
            )
            logger.info("Finished loading color histogram for %s", self._image_path)
        return self._np_grid

    def _load_image(self):
        if self._np_image is None:
            color_image = self._load_color_image()
            if self.active_channels is None:
                self._np_image = color_image
            else:
                self._np_image = color_image[..., self.active_channels]
        return self._np_image

    def _load_rgb_image(self):
        if self._np_rgb_image is None:
            logger.info("Loading raw image data for %s...", self._image_path)
            im = Image.open(self._image_path)
            self._np_rgb_image = np.asarray(im, dtype=np.float64) / 255.0
            logger.info("Finished loading raw image data for %s", self._image_path)
        return self._np_rgb_image

    def _load_color_image(self):
        if self._np_color_image is None:
            rgb = self._load_rgb_image()
            self._np_color_image = convert_rgb_to_color_space(rgb, self.color_space)
        return self._np_color_image
    
    def get_grid(self, use_jax: bool = False) -> ArrayLike:
        """Get the color grid of the image, either as numpy or jax array"""
        if use_jax:
            if self._jax_grid is None:
                self._jax_grid = self._load_grid().get_jax()
            return self._jax_grid
        return self._load_grid()

    def get_image(self, use_jax: bool = False) -> ArrayLike:
        """Get the image as numpy or jax array"""
        if use_jax:
            if self._jax_image is None:
                self._jax_image = jnp.array(self._load_image())
            return self._jax_image
        return self._load_image()

    def get_color_image(self) -> ArrayLike:
        """Get the full image in the configured color space."""
        return self._load_color_image()

    def get_rgb_image(self) -> ArrayLike:
        """Get the image in RGB space."""
        return self._load_rgb_image()

    def color_to_rgb(self, color_image: ArrayLike) -> ArrayLike:
        """Convert a color-space image to RGB."""
        return convert_color_space_to_rgb(np.asarray(color_image), self.color_space)

    def get_image_shape(self) -> tuple[int, int, int]:
        """Get the shape of the image"""
        return self._load_image().shape

    def free_memory(
        self,
        keep_numpy: bool = True,
        *,
        keep_grid: bool | None = None,
        keep_image: bool | None = None,
    ) -> None:
        """Release cached arrays to reduce memory usage."""
        self._jax_grid = None
        self._jax_image = None
        if keep_grid is None:
            keep_grid = keep_numpy
        if keep_image is None:
            keep_image = keep_numpy
        if not keep_grid:
            self._np_grid = None
        if not keep_image:
            self._np_image = None
            self._np_color_image = None
            self._np_rgb_image = None

    @classmethod
    def set_image_dir(cls, image_dir: str):
        """Set the directory where images are stored"""
        cls.images_dir = image_dir
