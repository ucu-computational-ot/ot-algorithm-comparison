from typing import NamedTuple, Tuple, Optional

import jax
import jax.numpy as jnp

from .solver_log import display_iteration_stats
from .utils import (
    CachedQuadraticProgramInfo,
    ConvergenceInformation,
    InfeasibilityInformation,
    IterationStats,
    PdhgSolverState,
    PointType,
    QuadraticProgrammingProblem,
    ScaledQpProblem, OTProblem, BarycenterProblem, PrimalShape,
)


def compute_dual_objective(
    marginals: jnp.ndarray,
    dual_solution: jnp.ndarray,
    primal_solution_P: jnp.ndarray,
    primal_obj_product: jnp.ndarray,
):
    """Compute the dual objective.

    Parameters
    ----------
    variable_lower_bound : jnp.ndarray
        Lower bound of variables.
    variable_upper_bound : jnp.ndarray
        Upper bound of variables.
    reduced_costs : jnp.ndarray
        Reduced costs.
    right_hand_side : jnp.ndarray
        Right hand side of the constraints.
    primal_solution : jnp.ndarray
        Primal solution.
    dual_solution : jnp.ndarray
        Dual solution.
    primal_obj_product : jnp.ndarray
        Product of the primal solution and the objective.
    objective_constant : float
        Constant in the objective.

    Returns
    -------
    float
        the dual objective
    """
    M = marginals.shape[0]
    v = dual_solution[M:]

    dual_linear = jnp.sum(marginals * v)

    quad_term = jnp.sum(primal_obj_product * primal_solution_P)

    base_dual_objective = (
        dual_linear
        - 0.5 * quad_term
    )
    return base_dual_objective


def compute_reduced_costs_from_primal_gradient(
    grad_P: jnp.ndarray,
    primal_obj_product: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Kernel to compute the reduced costs from the primal gradient.

    Parameters
    ----------
    primal_gradient : jnp.ndarray
        Primal gradient vector.
    isfinite_variable_lower_bound : jnp.ndarray
        Boolean array indicating finite lower bounds.
    isfinite_variable_upper_bound : jnp.ndarray
        Boolean array indicating finite upper bounds.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Reduced costs and reduced costs violation vectors.
    """
    primal_gradient = grad_P + primal_obj_product
    rc_pos = jnp.maximum(primal_gradient, 0.0)
    rc_neg = primal_gradient - rc_pos
    return rc_pos, rc_neg


# Note: the order of the calculations can be improved.
def compute_convergence_information(
    problem: BarycenterProblem,
    primal_iterate: PrimalShape,
    dual_iterate: jnp.ndarray,
    eps_ratio: float,
    primal_product: jnp.ndarray,
    dual_product: PrimalShape,
    primal_obj_product: jnp.ndarray,
    reference_solution: Optional[jnp.ndarray],
    norm_ord: int,
) -> ConvergenceInformation:
    """
    Compute convergence information of the given primal and dual solutions.

    Relative versions of the residuals are defined as
      relative_residual = residual / (eps_ratio + norm),
    where
      eps_ratio = eps_abs / eps_rel
      residual = one of the residuals (l{2,_inf}_{primal,dual}_residual)
      norm = the relative norm (l{2,_inf} norm of
             {constraint_bounds,primal_linear_objective} respectively).

    1. If eps_rel = 0.0, these will all be 0.0.
    2. If eps_rel > 0.0, the absolute and relative termination
    criteria translate to relative_residual <= eps_rel.

    NOTE: The usefulness of these relative residuals is based on their
    relationship to TerminationCriteria. If the TerminationCriteria change
    consider adding additional iteration measures here.


    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        Quadratic programming problem instance.
    qp_cache : CachedQuadraticProgramInfo
        Cached quadratic program information.
    primal_iterate : jnp.ndarray
        Primal iterate vector.
    dual_iterate : jnp.ndarray
        Dual iterate vector.
    eps_ratio : float
        Epsilon ratio for relative measures.
    primal_product : jnp.ndarray
        Primal product vector.
    dual_product : jnp.ndarray
        Dual product vector.
    primal_obj_product : jnp.ndarray
        Primal objective product vector.

    Returns
    -------
    ConvergenceInformation
        Computed convergence information.
    """

    M = problem.marginals.shape[0]

    primal_linear_objective = jnp.sum(
        (problem.weights[:, None, None] * problem.cost) * primal_iterate.P
    )

    primal_objective = (
        primal_linear_objective
        + 0.5 * jnp.sum(primal_iterate.P * primal_obj_product)
    )


    row_errors = primal_product[:M]
    col_errors = primal_product[M:] - problem.marginals
    primal_residual_norm = jnp.linalg.norm(
       jnp.concatenate([row_errors, col_errors])
    )

    primal_solution_norm = jnp.linalg.norm(primal_iterate.P)

    grad_P = problem.weights[:, None, None] * problem.cost - dual_product.P

    reduced_costs, reduced_costs_violation = compute_reduced_costs_from_primal_gradient(
       grad_P, primal_obj_product
    )

    dual_objective = compute_dual_objective(
        problem.marginals,
        dual_iterate,
        primal_iterate.P,
        primal_obj_product
    )
    # dual_residual_vec = jnp.zeros_like(dual_iterate)
    dual_residual_norm = jnp.linalg.norm(
        reduced_costs_violation
    )
    dual_solution_norm = jnp.linalg.norm(dual_iterate)

    relative_primal_residual_norm = primal_residual_norm / (
        eps_ratio
        + jnp.maximum(
            jnp.linalg.norm(problem.marginals),
            jnp.linalg.norm(primal_product),
        )
    )
    relative_dual_residual_norm = dual_residual_norm / (
        eps_ratio
        + jnp.maximum(
            jnp.maximum(
                problem.cost_matrix_norm,
                jnp.linalg.norm(primal_obj_product),
            ),
            jnp.linalg.norm(dual_product.P),
        )
    )
    corrected_dual_obj_value = jax.lax.cond(
        dual_residual_norm == 0.0, lambda: dual_objective, lambda: -jnp.inf
    )
    absolute_optimality_gap = jnp.abs(primal_objective - dual_objective)
    relative_optimality_gap = absolute_optimality_gap / (
        eps_ratio + jnp.maximum(jnp.abs(primal_objective), jnp.abs(dual_objective))
    )

    l2_difference = jnp.inf
    objective_difference = jnp.inf

    if reference_solution is not None:
        l2_difference = jnp.linalg.norm(primal_iterate - reference_solution)
        objective_difference = jnp.abs(
            jnp.sum(problem.weights[:, None, None] * problem.cost * reference_solution) - primal_linear_objective
        )

    return ConvergenceInformation(
        PointType.POINT_TYPE_AVERAGE_ITERATE,
        primal_objective,
        dual_objective,
        primal_linear_objective,
        corrected_dual_obj_value,
        primal_residual_norm,
        dual_residual_norm,
        relative_primal_residual_norm,
        relative_dual_residual_norm,
        absolute_optimality_gap,
        relative_optimality_gap,
        primal_solution_norm,
        dual_solution_norm,
        l2_difference,
        objective_difference,
    )


def compute_infeasibility_information(
    problem: QuadraticProgrammingProblem,
    primal_ray_estimate: jnp.ndarray,
    dual_ray_estimate: jnp.ndarray,
    primal_ray_estimate_product: jnp.ndarray,
    dual_ray_estimate_product: jnp.ndarray,
    primal_ray_estimate_obj_product: jnp.ndarray,
):
    """
    Compute infeasibility information of the given primal and dual solutions.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        Quadratic programming problem instance.
    primal_ray_estimate : jnp.ndarray
        Primal ray estimate vector.
    dual_ray_estimate : jnp.ndarray
        Dual ray estimate vector.
    primal_ray_estimate_product : jnp.ndarray
        Primal ray estimate product vector.
    dual_ray_estimate_product : jnp.ndarray
        Dual ray estimate product vector.
    primal_ray_estimate_obj_product : jnp.ndarray
        Primal ray estimate objective product vector.

    Returns
    -------
    InfeasibilityInformation
        Computed infeasibility information.
    """
    # Assume InfeasibilityInformation is a namedtuple
    primal_ray_inf_norm = jnp.linalg.norm(primal_ray_estimate, ord=jnp.inf)
    scaled_primal_ray_estimate, scaled_primal_ray_estimate_product = jax.lax.cond(
        primal_ray_inf_norm == 0.0,
        lambda _: (primal_ray_estimate, primal_ray_estimate_product),
        lambda _: (
            primal_ray_estimate / primal_ray_inf_norm,
            primal_ray_estimate_product / primal_ray_inf_norm,
        ),
        operand=None,
    )

    lower_variable_violation = jnp.maximum(
        (-1 / problem.isfinite_variable_lower_bound + 1) - scaled_primal_ray_estimate,
        0.0,
    )
    upper_variable_violation = jnp.maximum(
        scaled_primal_ray_estimate - (1 / problem.isfinite_variable_upper_bound - 1),
        0.0,
    )

    constraint_violation = jax.lax.select(
        problem.equalities_mask,
        jnp.zeros_like(problem.right_hand_side) - scaled_primal_ray_estimate_product,
        jnp.maximum(
            jnp.zeros_like(problem.right_hand_side)
            - scaled_primal_ray_estimate_product,
            0.0,
        ),
    )

    max_primal_ray_infeasibility = jnp.linalg.norm(
        jnp.concatenate(
            [constraint_violation, lower_variable_violation, upper_variable_violation]
        ),
        ord=jnp.inf,
    )
    primal_ray_linear_objective = jnp.dot(
        problem.objective_vector, scaled_primal_ray_estimate
    )
    reduced_costs, reduced_costs_violation = compute_reduced_costs_from_primal_gradient(
        -dual_ray_estimate_product,
        problem.isfinite_variable_lower_bound,
        problem.isfinite_variable_upper_bound,
    )
    dual_objective = compute_dual_objective(
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        reduced_costs,
        problem.right_hand_side,
        primal_ray_estimate,
        dual_ray_estimate,
        primal_ray_estimate_obj_product,
        problem.objective_constant,
    )
    dual_residual = jnp.where(
        problem.inequalities_mask, jnp.maximum(-dual_ray_estimate, 0.0), 0.0
    )
    l_inf_dual_residual = jnp.linalg.norm(
        jnp.concatenate([dual_residual, reduced_costs_violation]), ord=jnp.inf
    )

    scaling_factor = jnp.maximum(
        jnp.linalg.norm(scaled_primal_ray_estimate, ord=jnp.inf),
        jnp.linalg.norm(reduced_costs, ord=jnp.inf),
    )
    max_dual_ray_infeasibility, dual_ray_objective = jax.lax.cond(
        scaling_factor == 0.0,
        lambda: (0.0, 0.0),
        lambda: (l_inf_dual_residual / scaling_factor, dual_objective / scaling_factor),
    )

    return InfeasibilityInformation(
        PointType.POINT_TYPE_AVERAGE_ITERATE,
        max_primal_ray_infeasibility,
        primal_ray_linear_objective,
        max_dual_ray_infeasibility,
        dual_ray_objective,
    )


def evaluate_unscaled_iteration_stats(
    problem: BarycenterProblem,
    solver_state: PdhgSolverState,
    cumulative_time: float,
    eps_ratio: float,
    norm_ord: float,
    average: bool = True,
    infeasibility_detection: bool = True,
    reference_solution: Optional[jnp.ndarray] = None,
):
    """
    Compute the iteration stats of the unscaled primal and dual solutions.

    Parameters
    ----------
    scaled_problem : ScaledQpProblem
        Scaled quadratic programming problem instance.
    qp_cache : CachedQuadraticProgramInfo
        Cached quadratic program information.
    solver_state : PdhgSolverState
        The current solver state.
    cumulative_time : float
        Cumulative time in seconds.
    eps_ratio : float
        eps_abs / eps_rel
    norm_ord : float
        Order of the norm.
    average : bool
        Whether to use the average solution.
    infeasibility_detection : bool
        Whether to detect infeasibility.

    Returns
    -------
    IterationStats
        Computed iteration statistics for the unscaled problem.
    """
    (
        unscaled_primal_solution,
        unscaled_dual_solution,
        unscaled_primal_product,
        unscaled_dual_product,
        unscaled_primal_obj_product,
    ) = jax.lax.cond(
        average == True,
        lambda: (
            solver_state.avg_primal_solution,
            solver_state.avg_dual_solution,
            solver_state.avg_primal_product,
            solver_state.avg_dual_product,
            solver_state.avg_primal_obj_product,
        ),
        lambda: (
            solver_state.current_primal_solution,
            solver_state.current_dual_solution,
            solver_state.current_primal_product,
            solver_state.current_dual_product,
            solver_state.current_primal_obj_product,
        ),
    )
    convergence_information = compute_convergence_information(
        problem,
        unscaled_primal_solution,
        unscaled_dual_solution,
        eps_ratio,
        unscaled_primal_product,
        unscaled_dual_product,
        unscaled_primal_obj_product,
        reference_solution,
        norm_ord,
    )
    # infeasibility_information = jax.lax.cond(
    #     infeasibility_detection,
    #     lambda: compute_infeasibility_information(
    #         scaled_problem.original_qp,
    #         unscaled_primal_solution,
    #         unscaled_dual_solution,
    #         unscaled_primal_product,
    #         unscaled_dual_product,
    #         unscaled_primal_obj_product,
    #     ),
    #     lambda: InfeasibilityInformation(
    #         PointType.POINT_TYPE_AVERAGE_ITERATE, 1.0, 1.0, 1.0, 1.0
    #     ),
    # )
    current_iteration_stats = IterationStats(
        iteration_number=solver_state.num_iterations,
        convergence_information=convergence_information,
        cumulative_rejected_steps=0,  # cumulative_rejected_steps
        cumulative_time_sec=cumulative_time,
        step_size=solver_state.step_size,
        primal_weight=solver_state.primal_weight,
        method_specific_stats={},
    )
    display_iteration_stats(current_iteration_stats, solver_state)
    return current_iteration_stats


def should_log_iteration_status(iteration: int, params: NamedTuple) -> bool:
    """
    Determine if the iteration statistics should be printed based on the
    termination status, current iteration number, and display frequency.

    Parameters
    ----------
    iteration : int
        Current iteration number.
    params : NamedTuple
        Parameters for the solver.

    Returns
    -------
    bool
        Whether to print the iteration stats.
    """
    num_of_evaluations = (iteration - 1) // params.termination_evaluation_frequency
    # Print stats every display_frequency * termination_evaluation_frequency iterations
    return num_of_evaluations % params.display_frequency == 0