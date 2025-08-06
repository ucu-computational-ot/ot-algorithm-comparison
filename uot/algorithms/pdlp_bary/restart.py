import logging

import jax
from jax import numpy as jnp

from .iteration_stats_utils import (
    compute_dual_objective,
    compute_reduced_costs_from_primal_gradient,
    compute_convergence_information
)
from .solver_log import jax_debug_log
from .utils import (
    PdhgSolverState,
    QuadraticProgrammingProblem,
    RestartInfo,
    RestartParameters,
    RestartScheme,
    RestartToCurrentMetric,
    SaddlePointOutput,
    ScaledQpProblem,
    TerminationStatus, BarycenterProblem, PrimalShape,
)

logger = logging.getLogger(__name__)

def unscaled_saddle_point_output(
    scaled_problem: BarycenterProblem,
    primal_solution: jnp.ndarray,
    dual_solution: jnp.ndarray,
    termination_status: TerminationStatus,
    iterations_completed: int,
    convergence_information,
    timing_info,
) -> SaddlePointOutput:
    """
    Return the unscaled primal and dual solutions.

    Parameters
    ----------
    scaled_problem : ScaledQpProblem
        The scaled quadratic programming problem.
    primal_solution : jnp.ndarray
        The primal solution vector.
    dual_solution : jnp.ndarray
        The dual solution vector.
    termination_status : TerminationStatus
        Reason for termination.
    iterations_completed : int
        Number of iterations completed.
    convergence_information : ConvergenceInformation
        Convergence information.
    timing_info : dict
        Timing information.

    Returns
    -------
    SaddlePointOutput
        The unscaled primal and dual solutions along with other details.
    """
    original_primal_solution = primal_solution
    original_dual_solution = dual_solution

    return SaddlePointOutput(
        primal_solution=original_primal_solution,
        dual_solution=original_dual_solution,
        termination_status=termination_status,
        iteration_count=iterations_completed,
        primal_objective=convergence_information.primal_objective,
        dual_objective=convergence_information.dual_objective,
        corrected_dual_objective=convergence_information.corrected_dual_objective,
        primal_residual_norm=convergence_information.primal_residual_norm,
        dual_residual_norm=convergence_information.dual_residual_norm,
        relative_primal_residual_norm=convergence_information.relative_primal_residual_norm,
        relative_dual_residual_norm=convergence_information.relative_dual_residual_norm,
        absolute_optimality_gap=convergence_information.absolute_optimality_gap,
        relative_optimality_gap=convergence_information.relative_optimality_gap,
        timing_info=timing_info,
    )


def weighted_norm(vec: jnp.ndarray, weights: float) -> float:
    """
    Compute the weighted norm of a vector.

    Parameters
    ----------
    vec : jnp.ndarray
        The input vector.
    weights : float
        The weight to apply.

    Returns
    -------
    float
        The weighted norm of the vector.
    """
    # tmp = jax.lax.cond(jnp.all(vec == 0.0), lambda: 0.0, lambda: jnp.linalg.norm(vec))
    tmp = jax.lax.cond(jnp.all(vec == 0.0), lambda: 0.0, lambda: jnp.linalg.norm(vec))
    return jnp.sqrt(weights) * tmp


def compute_weight_kkt_residual(
    problem: BarycenterProblem,
    primal_iterate: PrimalShape,
    dual_iterate: jnp.ndarray,
    primal_product: jnp.ndarray,
    dual_product: PrimalShape,
    primal_obj_product: jnp.ndarray,
    primal_weight: float,
    norm_ord: float,
) -> float:
    """
    Compute the weighted KKT residual for restarting based on the current iterate values.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The quadratic programming problem.
    primal_iterate : jnp.ndarray
        Current primal iterate.
    dual_iterate : jnp.ndarray
        Current dual iterate.
    primal_product : jnp.ndarray
        Primal product vector.
    dual_product: jnp.ndarray
        Dual product vector.
    primal_obj_product : jnp.ndarray
        Primal objective product.
    primal_weight : float
        Weight factor for primal.
    norm_ord : float
        Order of the norm.

    Returns
    -------
    float
        The weighted KKT residual.
    """

    conv_info = compute_convergence_information(
        problem,
        primal_iterate,
        dual_iterate,
        1,
        primal_product,
        dual_product,
        primal_obj_product,
        reference_solution=None,
        norm_ord=norm_ord,
    )

    weighted_kkt_residual = jnp.linalg.norm(
        jnp.array(
            [
                primal_weight * conv_info.primal_residual_norm,
                1 / primal_weight * conv_info.dual_residual_norm,
                conv_info.absolute_optimality_gap,
            ]
        ),
        ord=norm_ord,
    )
    # relative_weighted_kkt_residual = jnp.linalg.norm(
    #     jnp.array(
    #         [
    #             primal_weight * conv_info.relative_primal_residual_norm,
    #             1 / primal_weight * conv_info.relative_dual_residual_norm,
    #             conv_info.relative_optimality_gap,
    #         ]
    #     ),
    #     ord=norm_ord,
    # )
    return weighted_kkt_residual
    # return jax.lax.cond(
    #     problem.is_lp,
    #     lambda: weighted_kkt_residual,
    #     lambda: relative_weighted_kkt_residual,
    # )


def construct_restart_parameters(
    restart_scheme: str,
    restart_to_current_metric: str,
    restart_frequency_if_fixed: int,
    artificial_restart_threshold: float,
    sufficient_reduction_for_restart: float,
    necessary_reduction_for_restart: float,
    primal_weight_update_smoothing: float,
) -> RestartParameters:
    """
    Constructs the restart parameters for an optimization algorithm.

    Parameters
    ----------
    restart_scheme : str
        The restart scheme to use.
    restart_to_current_metric : str
        The metric for restarting.
    restart_frequency_if_fixed : int
        Fixed frequency for restart.
    artificial_restart_threshold : float
        Threshold for artificial restart.
    sufficient_reduction_for_restart : float
        Sufficient reduction for restart.
    necessary_reduction_for_restart : float
        Necessary reduction for restart.
    primal_weight_update_smoothing : float
        Smoothing factor for updating the primal weight.

    Returns
    -------
    RestartParameters
        The constructed restart parameters.
    """
    assert restart_frequency_if_fixed > 1, "Restart frequency must be greater than 1."
    assert (
        0.0 < artificial_restart_threshold <= 1.0
    ), "Threshold must be between 0 and 1."
    assert (
        0.0 < sufficient_reduction_for_restart <= necessary_reduction_for_restart <= 1.0
    ), "Reduction parameters must be in the range (0, 1]."
    assert (
        0.0 <= primal_weight_update_smoothing <= 1.0
    ), "Smoothing must be between 0 and 1."

    return RestartParameters(
        restart_scheme,
        restart_to_current_metric,
        restart_frequency_if_fixed,
        artificial_restart_threshold,
        sufficient_reduction_for_restart,
        necessary_reduction_for_restart,
        primal_weight_update_smoothing,
    )


def should_do_adaptive_restart_kkt(
    problem: BarycenterProblem,
    kkt_candidate_residual: float,
    restart_params: RestartParameters,
    last_restart_info: RestartInfo,
    primal_weight: float,
    norm_ord: float,
) -> bool:
    """
    Checks if an adaptive restart should be triggered based on KKT residual reduction.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The quadratic programming problem instance.
    kkt_candidate_residual : float
        The current KKT residual of the candidate solution.
    restart_params : RestartParameters
        Parameters for restart logic.
    last_restart_info : RestartInfo
        Information from the last restart.
    primal_weight : float
        The weight for the primal variable norm.
    norm_ord : float
        The order of the norm.

    Returns
    -------
    bool
        True if a restart should occur, False otherwise.
    """
    kkt_last_residual = compute_weight_kkt_residual(
        problem,
        last_restart_info.primal_solution,
        last_restart_info.dual_solution,
        last_restart_info.primal_product,
        last_restart_info.dual_product,
        last_restart_info.primal_obj_product,
        primal_weight,
        norm_ord,
    )

    # Stop gradient since kkt_last_residual might be zero.
    kkt_reduction_ratio = jax.lax.stop_gradient(
        jax.lax.cond(
            kkt_last_residual > jnp.finfo(float).eps,
            lambda: kkt_candidate_residual / kkt_last_residual,
            lambda: 1.0,
        )
    )
    do_restart = jax.lax.cond(
        (kkt_reduction_ratio < restart_params.necessary_reduction_for_restart)
        & (
            (kkt_reduction_ratio < restart_params.sufficient_reduction_for_restart)
            | (kkt_reduction_ratio > last_restart_info.reduction_ratio_last_trial)
        ),
        lambda: True,
        lambda: False,
    )
    return do_restart, kkt_reduction_ratio


def restart_criteria_met_kkt(
    restart_params, problem, solver_state, last_restart_info, norm_ord
):
    # Computational expensive!!!
    current_kkt_res = compute_weight_kkt_residual(
        problem,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        solver_state.current_primal_product,
        solver_state.current_dual_product,
        solver_state.current_primal_obj_product,
        solver_state.primal_weight,
        norm_ord,
    )
    avg_kkt_res = compute_weight_kkt_residual(
        problem,
        solver_state.avg_primal_solution,
        solver_state.avg_dual_solution,
        solver_state.avg_primal_product,
        solver_state.avg_dual_product,
        solver_state.avg_primal_obj_product,
        solver_state.primal_weight,
        norm_ord,
    )
    reset_to_average = jax.lax.cond(
        restart_params.restart_to_current_metric == RestartToCurrentMetric.KKT_GREEDY,
        lambda: current_kkt_res >= avg_kkt_res,
        lambda: True,
    )
    candidate_kkt_residual = jax.lax.cond(
        reset_to_average, lambda: avg_kkt_res, lambda: current_kkt_res
    )

    restart_length = solver_state.solutions_count
    kkt_do_restart, kkt_reduction_ratio = should_do_adaptive_restart_kkt(
        problem,
        candidate_kkt_residual,
        restart_params,
        last_restart_info,
        solver_state.primal_weight,
        norm_ord,
    )
    do_restart = jax.lax.cond(
        (
            restart_length
            >= (
                restart_params.artificial_restart_threshold
                * solver_state.num_iterations
            )
        )
        | (
            (restart_params.restart_scheme == RestartScheme.FIXED_FREQUENCY)
            & (restart_length >= restart_params.restart_frequency_if_fixed)
        )
        | (
            (restart_params.restart_scheme == RestartScheme.ADAPTIVE_KKT)
            & (kkt_do_restart)
        ),
        lambda: True,
        lambda: False,
    )
    return do_restart, reset_to_average, kkt_reduction_ratio


def perform_restart(
    solver_state,
    reset_to_average,
    last_restart_info,
    kkt_reduction_ratio,
    restart_params,
):
    restart_length = solver_state.solutions_count
    (
        restarted_primal_solution,
        restarted_dual_solution,
        restarted_primal_product,
        restarted_dual_product,
        restarted_primal_obj_product,
    ) = jax.lax.cond(
        reset_to_average,
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
    if logging.root.level <= logging.DEBUG:
        jax_debug_log(
            "Restarted after {} iterations",
            restart_length,
            logger=logger,
            level=logging.DEBUG,
        )

    primal_norm_params = 1 / solver_state.step_size * solver_state.primal_weight
    dual_norm_params = 1 / solver_state.step_size / solver_state.primal_weight
    primal_distance_moved_last_restart_period = weighted_norm(
        solver_state.avg_primal_solution.P - last_restart_info.primal_solution.P,
        primal_norm_params,
    ) / jnp.sqrt(solver_state.primal_weight)
    dual_distance_moved_last_restart_period = weighted_norm(
        solver_state.avg_dual_solution - last_restart_info.dual_solution,
        dual_norm_params,
    ) * jnp.sqrt(solver_state.primal_weight)
    new_last_restart_info = RestartInfo(
        primal_solution=restarted_primal_solution,
        dual_solution=restarted_dual_solution,
        primal_product=restarted_primal_product,
        dual_product=restarted_dual_product,
        primal_diff=solver_state.delta_primal,
        dual_diff=solver_state.delta_dual,
        primal_diff_product=solver_state.delta_primal_product,
        primal_obj_product=restarted_primal_obj_product,
        last_restart_length=restart_length,
        primal_distance_moved_last_restart_period=primal_distance_moved_last_restart_period,
        dual_distance_moved_last_restart_period=dual_distance_moved_last_restart_period,
        reduction_ratio_last_trial=kkt_reduction_ratio,
    )

    new_primal_weight = compute_new_primal_weight(
        new_last_restart_info,
        solver_state.primal_weight,
        restart_params.primal_weight_update_smoothing,
    )

    # The initial point of the restart will not counted into the average.
    # The weight (step size) of the initial point is zero.
    restarted_solver_state = PdhgSolverState(
        current_primal_solution=restarted_primal_solution,
        current_dual_solution=restarted_dual_solution,
        current_primal_product=restarted_primal_product,
        current_dual_product=restarted_dual_product,
        current_primal_obj_product=restarted_primal_obj_product,
        avg_primal_solution=PrimalShape(
            P=jnp.zeros_like(restarted_primal_solution.P),
            a=jnp.zeros_like(restarted_primal_solution.a)
        ),
        avg_dual_solution=jnp.zeros_like(restarted_dual_solution),
        avg_primal_product=jnp.zeros_like(restarted_dual_solution),
        avg_dual_product=PrimalShape(
            P=jnp.zeros_like(restarted_primal_solution.P),
            a=jnp.zeros_like(restarted_primal_solution.a)
        ),
        avg_primal_obj_product=jnp.zeros_like(restarted_primal_solution.P),
        delta_primal=PrimalShape(
            P=jnp.zeros_like(restarted_primal_solution.P),
            a=jnp.zeros_like(restarted_primal_solution.a)
        ),
        delta_dual=jnp.zeros_like(restarted_dual_solution),
        delta_primal_product=jnp.zeros_like(restarted_dual_solution),
        reg=solver_state.reg,
        solutions_count=0,
        weights_sum=0.0,
        step_size=solver_state.step_size,
        primal_weight=new_primal_weight,
        numerical_error=solver_state.numerical_error,
        num_steps_tried=solver_state.num_steps_tried,
        num_iterations=solver_state.num_iterations,
        termination_status=solver_state.termination_status,
    )

    return restarted_solver_state, new_last_restart_info


def run_restart_scheme(
    problem: BarycenterProblem,
    solver_state: PdhgSolverState,
    last_restart_info: RestartInfo,
    restart_params: RestartParameters,
    norm_ord: float,
):
    """
    Check restart criteria based on current and average KKT residuals.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The quadratic programming problem instance.
    solver_state : PdhgSolverState
        The current solver state.
    last_restart_info : RestartInfo
        Information from the last restart.
    restart_params : RestartParameters
        Parameters for controlling restart behavior.
    norm_ord : float
        Order of the norm.

    Returns
    -------
    tuple
        The new solver state, and the new last restart info.
    """

    do_restart, reset_to_average, kkt_reduction_ratio = jax.lax.cond(
        jnp.logical_or(
            solver_state.solutions_count == 0,
            restart_params.restart_scheme == RestartScheme.NO_RESTARTS
        ),
        lambda: (False, False, last_restart_info.reduction_ratio_last_trial),
        lambda: restart_criteria_met_kkt(
            restart_params, problem, solver_state, last_restart_info, norm_ord
        ),
    )
    return jax.lax.cond(
        do_restart,
        lambda: perform_restart(
            solver_state,
            reset_to_average,
            last_restart_info,
            kkt_reduction_ratio,
            restart_params,
        ),
        lambda: (solver_state, last_restart_info),
    )

def run_restart_scheme_feasibility_polishing(
    problem: QuadraticProgrammingProblem,
    current_solver_state: PdhgSolverState,
    restart_solver_state: PdhgSolverState,
    last_restart_info: RestartInfo,
    restart_params: RestartParameters,
):
    """
    Check restart criteria based on current and average KKT residuals.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The quadratic programming problem instance.
    current_solver_state : PdhgSolverState
        The current solver state, i.e. (x_k, y_k).
    restart_solver_state : PdhgSolverState
        The solver state to check restart criteria, i.e. (x_k, 0) or (0, y_k).
    last_restart_info : RestartInfo
        Information from the last restart.
    restart_params : RestartParameters
        Parameters for controlling restart behavior.

    Returns
    -------
    tuple
        The new solver state, and the new last restart info.
    """

    do_restart, reset_to_average, kkt_reduction_ratio = jax.lax.cond(
        restart_solver_state.solutions_count == 0,
        lambda: (False, False, last_restart_info.reduction_ratio_last_trial),
        lambda: restart_criteria_met_kkt(
            restart_params, problem, restart_solver_state, last_restart_info
        ),
    )
    return jax.lax.cond(
        do_restart,
        lambda: perform_restart(
            restart_solver_state,
            reset_to_average,
            last_restart_info,
            kkt_reduction_ratio,
            restart_params,
        ),
        lambda: (current_solver_state, last_restart_info),
    )


def compute_new_primal_weight(
    last_restart_info: RestartInfo,
    primal_weight: float,
    primal_weight_update_smoothing: float,
) -> float:
    """
    Compute primal weight at restart.

    Parameters
    ----------
    last_restart_info : RestartInfo
        Information about the last restart.
    primal_weight : float
        The current primal weight.
    primal_weight_update_smoothing : float
        Smoothing factor for weight update.

    Returns
    -------
    float
        The updated primal weight.
    """
    primal_distance = last_restart_info.primal_distance_moved_last_restart_period
    dual_distance = last_restart_info.dual_distance_moved_last_restart_period
    new_primal_weight = jax.lax.cond(
        (primal_distance > jnp.finfo(float).eps)
        & (dual_distance > jnp.finfo(float).eps),
        lambda: jnp.exp(
            primal_weight_update_smoothing * jnp.log(dual_distance / primal_distance)
            + (1 - primal_weight_update_smoothing) * jnp.log(primal_weight)
        ),
        lambda: primal_weight,
    )
    return new_primal_weight


def select_initial_primal_weight(
    problem: BarycenterProblem,
    primal_norm_params: float,
    dual_norm_params: float,
    primal_importance: float,
) -> float:
    """
    Initialize primal weight.

    Parameters
    ----------
    problem : QuadraticProgrammingProblem
        The quadratic programming problem instance.
    primal_norm_params : float
        Primal norm parameters.
    dual_norm_params : float
        Dual norm parameters.
    primal_importance : float
        Importance factor for primal weight.

    Returns
    -------
    float
        The initial primal weight.
    """
    rhs_vec_norm = weighted_norm(problem.marginals, dual_norm_params)
    obj_vec_norm = weighted_norm(problem.weights[:, None, None] * problem.cost, primal_norm_params)
    primal_weight = jax.lax.cond(
        (obj_vec_norm > 0.0) & (rhs_vec_norm > 0.0),
        lambda x: x * (obj_vec_norm / rhs_vec_norm),
        lambda x: x,
        operand=primal_importance,
    )
    if logging.root.level == logging.DEBUG:
        jax_debug_log(
            "Initial primal weight = {primal_weight}",
            primal_weight=primal_weight,
            logger=logger,
            level=logging.DEBUG,
        )
    return primal_weight