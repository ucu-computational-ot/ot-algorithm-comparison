from uot.problems.problem_generator import ProblemGenerator
from uot.problems.two_marginal import TwoMarginalProblem
import numpy as np

def generator_to_weights_list(
        generator: ProblemGenerator,
        include_zeros: bool = True,
        ):
    """Return a shared support array and a list of weight tuples per problem."""
    support = None
    weights_list = []
    for problem in generator.generate():
        if not isinstance(problem, TwoMarginalProblem):
            raise TypeError(
                f"Expected TwoMarginalProblem from generator, got {type(problem).__name__}"
            )
        marginals = problem.get_marginals()
        pts_weights = [
            marginal.to_discrete(include_zeros=include_zeros)
            for marginal in marginals
        ]
        supports, weights = zip(*pts_weights)
        if support is None:
            support = supports[0]
        for idx, marginal_support in enumerate(supports):
            if np.asarray(marginal_support).shape != np.asarray(support).shape or \
               not np.allclose(np.asarray(marginal_support), np.asarray(support)):
                raise ValueError(
                    f"All marginals must share the same support; mismatch at index {idx}"
                )
        weights_list.append(weights)
    return support, weights_list
