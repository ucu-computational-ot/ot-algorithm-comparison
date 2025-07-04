from typing import Any
from collections.abc import Iterator
from uot.problems.two_marginal import TwoMarginalProblem
from uot.problems.problem_generator import ProblemGenerator


class PairedGenerator(ProblemGenerator):
    """
    Compose two separate marginal generators into a single two-marginal problem generator.

    Configuration example:
      paired:
        generator_a:
          class: uot.problems.generators.CauchyGenerator
          params:
            dim: 1
            n_points: 32
            borders: [0, 1]
            cost_fn: uot.utils.costs.cost_euclid_squared
            use_jax: true
            seed: 42
        generator_b:
          class: uot.problems.generators.GaussianMixtureGenerator
          params:
            dim: 1
            num_components: 3
            n_points: 32
            borders: [0, 1]
            cost_fn: uot.utils.costs.cost_euclid_squared
            use_jax: true
            seed: 24
        num_datasets: 30

    Yields TwoMarginalProblem instances with mu from A and nu from B.
    """

    def __init__(
        self,
        name: str,
        gen_a_cfg: dict[str, Any],
        gen_b_cfg: dict[str, Any],
        num_datasets: int
    ):
        super().__init__()
        self._name = name
        if num_datasets <= 0:
            raise ValueError("num_datasets must be a positive integer")
        self._num_datasets = num_datasets

        gen_a_params = gen_a_cfg.get('params', {})
        gen_a_params.update({'num_datasets': num_datasets})
        gen_b_params = gen_b_cfg.get('params', {})
        gen_b_params.update({'num_datasets': num_datasets})

        self._gen_a: ProblemGenerator = gen_a_cfg['class'](
            name=name + '-A',
            **gen_a_params,
        )
        self._gen_b: ProblemGenerator = gen_b_cfg['class'](
            name=name + '-B',
            **gen_b_params,
        )

    def generate(self) -> Iterator[TwoMarginalProblem]:
        iter_a: TwoMarginalProblem = self._gen_a.generate()
        iter_b: TwoMarginalProblem = self._gen_b.generate()

        for _ in range(self._num_datasets):
            mu_problem = next(iter_a)
            nu_problem = next(iter_b)
            mu, _ = mu_problem.get_marginals()
            nu, _ = nu_problem.get_marginals()

            yield TwoMarginalProblem(
                name=self._name,
                mu=mu,
                nu=nu,
                cost_fn=mu_problem._cost_fn,
            )
