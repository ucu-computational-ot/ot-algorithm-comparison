import numpy as np
from jax import numpy as jnp
from uot.problems.two_marginal import TwoMarginalProblem


class ColorTransferProblem(TwoMarginalProblem):
    """
    A class representing a color transfer problem, which is a specific type of two-marginal problem.
    It inherits from the TwoMarginalProblem class.
    """

    def __init__(
            self,
            name,
            mu,
            nu,
            cost_fn,
            source_image,
            target_image,
            source_image_name: str | None = None,
            target_image_name: str | None = None,
            color_space: str = "rgb",
            active_channels: list[int] | None = None,
            bins_per_channel: int | None = None,
            *args, **kwargs):
        super().__init__(name, mu, nu, cost_fn, *args, **kwargs)
        self._source_image = None
        self._target_image = None
        self._source_data = source_image
        self._target_data = target_image
        if hasattr(source_image, "color_space"):
            color_space = source_image.color_space
        if hasattr(source_image, "active_channels"):
            active_channels = source_image.active_channels
        self.color_space = color_space
        self.active_channels = active_channels
        if not hasattr(source_image, "get_image") and not callable(source_image):
            self._source_image = source_image
            self._source_data = None
        if not hasattr(target_image, "get_image") and not callable(target_image):
            self._target_image = target_image
            self._target_data = None
        self.source_image_name = source_image_name
        self.target_image_name = target_image_name
        self.bins_per_channel = bins_per_channel
        self._source_full = None
        self._target_full = None
        self._source_rgb = None
        self._target_rgb = None

    @property
    def source_image(self) -> jnp.ndarray:
        if self._source_image is None:
            if self._source_data is None:
                return None
            if hasattr(self._source_data, "get_image"):
                self._source_image = self._source_data.get_image()
            elif callable(self._source_data):
                self._source_image = self._source_data()
            else:
                self._source_image = self._source_data
        return self._source_image

    @property
    def target_image(self) -> jnp.ndarray:
        if self._target_image is None:
            if self._target_data is None:
                return None
            if hasattr(self._target_data, "get_image"):
                self._target_image = self._target_data.get_image()
            elif callable(self._target_data):
                self._target_image = self._target_data()
            else:
                self._target_image = self._target_data
        return self._target_image

    @property
    def source_full(self):
        if self._source_full is None and hasattr(self._source_data, "get_color_image"):
            self._source_full = self._source_data.get_color_image()
        return self._source_full

    @property
    def target_full(self):
        if self._target_full is None and hasattr(self._target_data, "get_color_image"):
            self._target_full = self._target_data.get_color_image()
        return self._target_full

    @property
    def source_rgb(self):
        if self._source_rgb is None and hasattr(self._source_data, "get_rgb_image"):
            self._source_rgb = self._source_data.get_rgb_image()
        return self._source_rgb

    @property
    def target_rgb(self):
        if self._target_rgb is None and hasattr(self._target_data, "get_rgb_image"):
            self._target_rgb = self._target_data.get_rgb_image()
        return self._target_rgb

    def to_rgb_image(self, active_image):
        if hasattr(self._source_data, "color_to_rgb"):
            if self.active_channels is None:
                full = active_image
            else:
                base = np.array(self.source_full, copy=True)
                base[..., self.active_channels] = np.asarray(active_image)
                full = base
            return self._source_data.color_to_rgb(full)
        return active_image

    def to_dict(self):
        d = super().to_dict()
        d.update({
            'source_image': self.source_image,
            'target_image': self.target_image,
            'source_image_name': self.source_image_name,
            'target_image_name': self.target_image_name,
            'color_space': self.color_space,
            'active_channels': self.active_channels,
            'bins_per_channel': self.bins_per_channel,
        })
        return d

    def free_memory(self):
        super().free_memory()
        self._source_image = None
        self._target_image = None
        self._source_full = None
        self._target_full = None
        self._source_rgb = None
        self._target_rgb = None
        if hasattr(self._source_data, "free_memory"):
            self._source_data.free_memory(keep_grid=True, keep_image=False)
        if hasattr(self._target_data, "free_memory"):
            self._target_data.free_memory(keep_grid=True, keep_image=False)
