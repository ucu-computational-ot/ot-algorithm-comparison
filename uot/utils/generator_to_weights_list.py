from uot.problems.problem_generator import ProblemGenerator
from uot.problems.two_marginal import TwoMarginalProblem
import numpy as np
import jax
import jax.numpy as jnp

def _is_jax_array(x) -> bool:
    return isinstance(x, jax.Array)


def _stack_grid_support(axes):
    use_jax = any(_is_jax_array(ax) for ax in axes)
    xp = jnp if use_jax else np
    grid_axes = [xp.asarray(ax).ravel() for ax in axes]
    if len(grid_axes) == 0:
        return xp.zeros((1, 0), dtype=xp.float32)
    mesh = xp.meshgrid(*grid_axes, indexing="ij")
    return xp.stack(mesh, axis=-1)


def _axes_match(ref_axes, new_axes) -> bool:
    if len(ref_axes) != len(new_axes):
        return False
    for ref_axis, new_axis in zip(ref_axes, new_axes):
        ref_arr = np.asarray(ref_axis)
        new_arr = np.asarray(new_axis)
        if ref_arr.shape != new_arr.shape or not np.allclose(ref_arr, new_arr):
            return False
    return True


def _collect_weights(
        generator: ProblemGenerator,
        include_zeros: bool,
        ):
    grid_mode = getattr(generator, "_measure_mode", None) == "grid"
    support = None
    support_axes = None
    num_marginals = None
    weights_list = []
    for problem in generator.generate():
        if not isinstance(problem, TwoMarginalProblem):
            raise TypeError(
                f"Expected TwoMarginalProblem from generator, got {type(problem).__name__}"
            )
        marginals = problem.get_marginals()
        if num_marginals is None:
            num_marginals = len(marginals)
        elif len(marginals) != num_marginals:
            raise ValueError("All problems must have the same number of marginals")
        if grid_mode:
            grid_weights = []
            for idx, marginal in enumerate(marginals):
                if not hasattr(marginal, "for_grid_solver"):
                    raise TypeError(
                        "Grid mode requires marginals to implement for_grid_solver"
                    )
                axes, weights_nd = marginal.for_grid_solver()
                if support_axes is None:
                    support_axes = axes
                    support = _stack_grid_support(axes)
                elif not _axes_match(support_axes, axes):
                    raise ValueError(
                        f"All marginals must share the same grid axes; mismatch at index {idx}"
                    )
                grid_weights.append(weights_nd)
            weights_list.extend(grid_weights)
            # weights_list.append(tuple(grid_weights))
        else:
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
            weights_list.extend(weights)
            # weights_list.append(weights)
    if num_marginals is None:
        num_marginals = 0
    return support, weights_list, grid_mode, num_marginals


def generator_to_weights_list(
        generator: ProblemGenerator,
        include_zeros: bool = True,
        ):
    """Return a shared support array and a flat list of weights.

    In grid mode, the support is a stacked meshgrid with shape
    (n0, n1, ..., nd-1, d), and include_zeros is ignored because the grid
    structure is preserved.
    """
    support, weights_list, _, _ = _collect_weights(generator, include_zeros=include_zeros)
    return support, weights_list


def generator_to_weights_array(
        generator: ProblemGenerator,
        include_zeros: bool = True,
        ):
    """Return a shared support array and stacked weights (mu then nu for each problem)."""
    if include_zeros is False:
        raise ValueError("generator_to_weights_array requires include_zeros=True")
    support, weights_list, _, num_marginals = _collect_weights(generator, include_zeros=True)
    if not weights_list:
        return support, np.asarray([])
    use_jax = _is_jax_array(weights_list[0])
    xp = jnp if use_jax else np
    if num_marginals != 2:
        raise ValueError("generator_to_weights_array expects exactly two marginals per problem")
    return support, xp.stack(weights_list, axis=0)
