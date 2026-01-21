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
            bins_per_channel: int | None = None,
            *args, **kwargs):
        super().__init__(name, mu, nu, cost_fn, *args, **kwargs)
        self._source_image = None
        self._target_image = None
        self._source_data = source_image
        self._target_data = target_image
        if not hasattr(source_image, "get_image") and not callable(source_image):
            self._source_image = source_image
            self._source_data = None
        if not hasattr(target_image, "get_image") and not callable(target_image):
            self._target_image = target_image
            self._target_data = None
        self.source_image_name = source_image_name
        self.target_image_name = target_image_name
        self.bins_per_channel = bins_per_channel

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

    def to_dict(self):
        d = super().to_dict()
        d.update({
            'source_image': self.source_image,
            'target_image': self.target_image,
            'source_image_name': self.source_image_name,
            'target_image_name': self.target_image_name,
            'bins_per_channel': self.bins_per_channel,
        })
        return d

    def free_memory(self):
        super().free_memory()
        self._source_image = None
        self._target_image = None
        if hasattr(self._source_data, "free_memory"):
            self._source_data.free_memory(keep_grid=True, keep_image=False)
        if hasattr(self._target_data, "free_memory"):
            self._target_data.free_memory(keep_grid=True, keep_image=False)
