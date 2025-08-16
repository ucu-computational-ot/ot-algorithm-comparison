import os
from PIL import Image
from jax import numpy as jnp
from uot.data.dataset_loader import load_image_as_color_grid
from uot.utils.types import ArrayLike

from uot.utils.logging import logger

class ImageData:

    image_dir = None

    def __init__(self, name: str, bin_num: int = 32):
        self.name = name
        logger.info('before load_image_as_color_grid')
        self._np_grid = load_image_as_color_grid(os.path.join(ImageData.images_dir, name), bins_per_channel=bin_num)
        logger.info('after load_image_as_color_grid')
        # self._np_image = read_image(os.path.join(ImageData.images_dir, name))
        im = Image.open(os.path.join(ImageData.images_dir, name))
        self._np_image = jnp.asarray(im, dtype=jnp.float32) / 255.0
        self._jax_grid = None
        self._jax_image = None
    
    def get_grid(self, use_jax: bool = False) -> ArrayLike:
        """Get the color grid of the image, either as numpy or jax array"""
        if use_jax:
            if self._jax_grid is None:
                self._jax_grid = self._np_grid.get_jax()
            return self._jax_grid
        return self._np_grid

    def get_image(self, use_jax: bool = False) -> ArrayLike:
        """Get the image as numpy or jax array"""
        if use_jax:
            if self._jax_image is None:
                self._jax_image = jnp.array(self._np_image)
            return self._jax_image
        return self._np_image

    def get_image_shape(self) -> tuple[int, int, int]:
        """Get the shape of the image"""
        return self._np_image.shape

    @classmethod
    def set_image_dir(cls, image_dir: str):
        """Set the directory where images are stored"""
        cls.images_dir = image_dir

