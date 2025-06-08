import os
import uuid
import pickle
from abc import ABC, abstractmethod
from typing import List, Callable

import numpy as np
from uot.data.measure import DiscreteMeasure
from uot.problems.base_problem import MarginalProblem
from uot.problems.two_marginal import TwoMarginalProblem
from uot.utils.generator_helpers import get_gmm_pdf
from uot.utils.generate_nd_grid import generate_nd_grid
import jax.numpy as jnp
from uot.utils.types import ArrayLike


class ProblemGenerator(ABC):

    DUMP_PATH = "synthetic"

    def __init__(
        self,
        name: str,
        dim: int,
        n_points: int,
        num_datasets: int,
        borders: tuple[float, float],
        cost_fn: Callable[[ArrayLike, ArrayLike], ArrayLike],
        use_jax: bool = False,
        seed: int = 42
    ):
        super().__init__()
        # TODO: arbitrary dim?
        if dim not in [1, 2, 3]:
            raise ValueError("dim must be 1, 2 or 3")

        self._name = name
        self._dim = dim
        self._n_points = n_points
        self._num_datasets = num_datasets
        self._borders = borders
        self._cost_fn = cost_fn
        self._use_jax = use_jax
        self._seed = seed

    def __eq__(self, other: "ProblemGenerator"):
        return self.__dict__ == other.__dict__

    def generate(self, *args, **kwargs) -> List:
        loaded_problems = self._load_problems()
        if loaded_problems is None:
            generated_problems = self._generate()
            self._dump_problems(generated_problems)
            return generated_problems
        return loaded_problems

    @abstractmethod
    def _generate(self):
        pass

    def _get_axes(self, n_points: int) -> List[ArrayLike]:
        lib = jnp if self._use_jax else np
        ax = lib.linspace(self._borders[0], self._borders[1], n_points)
        axs = [ax for _ in range(self._dim)]
        return axs

    def _load_problems(self) -> list[MarginalProblem] | None:
        for filename in os.listdir(ProblemGenerator.DUMP_PATH):
            filepath = os.path.join(ProblemGenerator.DUMP_PATH, filename)

            with open(filepath, 'rb') as f:
                generator, problems = pickle.load(f)
            
            if generator == self:
                return problems

        return None

    def _dump_problems(self, problems: list[MarginalProblem]):
        dump_filepath = os.path.join(ProblemGenerator.DUMP_PATH, 
                                     f"{str(uuid.uuid4())}.pkl")

        with open(dump_filepath, 'wb') as f:
            pickle.dump((self, problems), f)


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
        super().__init__(
            name=name,
            dim=dim,
            n_points=n_points,
            num_datasets=num_datasets,
            borders=borders,
            cost_fn=cost_fn,
            use_jax=use_jax,
            seed=seed,
        )

        self._num_components = num_components

    def _generate(self) -> List[TwoMarginalProblem]:
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
