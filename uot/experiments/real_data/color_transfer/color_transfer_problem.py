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
            source_image: jnp.ndarray,
            target_image: jnp.ndarray,
            *args, **kwargs):
        super().__init__(name, mu, nu, cost_fn, *args, **kwargs)
        self.source_image = source_image
        self.target_image = target_image

    def to_dict(self):
        d = super().to_dict()
        d.update({
            'source_image': self.source_image,
            'target_image': self.target_image
        })
        return d

    