from abc import ABC, abstractmethod
from typing import List, Callable
import numpy as np
from uot.data.measure import DiscreteMeasure
from uot.problems.two_marginal import TwoMarginalProblem
from uot.utils.generator_helpers import get_gmm_pdf
from uot.utils.generate_nd_grid import generate_nd_grid
import jax.numpy as jnp
from uot.utils.types import ArrayLike


class ProblemGenerator(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs) -> List:
        "Return a list of MarginalProblem objects."
        pass


class GaussianMixtureGenerator(ProblemGenerator):
    def __init__(
        self,
        name: str,
        dim: int,
        num_components: int,
        n_points: int,
        num_datasets: int,
        borders: tuple[float, float],
        cost_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
        use_jax: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        # TODO: arbitrary dim?
        if dim not in [1, 2, 3]:
            raise ValueError("dim must be 1, 2 or 3")
        self._name = name
        self._dim = dim
        self._num_components = num_components
        self._n_points = n_points
        self._num_datasets = num_datasets
        self._borders = borders
        self._cost_fn = cost_fn
        self._use_jax = use_jax
        self._seed = seed

    def generate(self) -> List[TwoMarginalProblem]:
        pdfs_num = 2 * self._num_datasets
        axes_support = self._get_axes(self._n_points)
        points = generate_nd_grid(axes_support)
        mixtures_pdfs = [
            get_gmm_pdf(
                self._dim,
                num_components=self._num_components,
                use_jax=self._use_jax,
                seed=self._seed,
            )
            for _ in range(pdfs_num)
        ]

        problems: List[TwoMarginalProblem] = []
        for i in range(self._num_datasets):
            mu = DiscreteMeasure(
                points=points,
                weights=mixtures_pdfs[2 * i](points),
            )
            nu = DiscreteMeasure(
                points=points,
                weights=mixtures_pdfs[2 * i + 1](points),
            )
            problem = TwoMarginalProblem(
                name=self._name,
                mu=mu,
                nu=nu,
                cost_fn=self._cost_fn,
            )
            problems.append(problem)
        return problems

    def _get_axes(self, n_points: int) -> List[ArrayLike]:
        lib = jnp if self._use_jax else np
        ax = lib.linspace(self._borders[0], self._borders[1], n_points)
        axs = [ax for _ in range(self._dim)]
        return axs
