import timeit
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, List, NamedTuple, Optional, Union
import functools

import numpy as np
from jax.tree_util import register_dataclass

import chex
import jax
from jax import jit
from jax import numpy as jnp
from jax.experimental.sparse import BCOO, BCSR
from matplotlib import pyplot as plt


@chex.dataclass
class PrimalShape:
    P: jnp.ndarray # Shape (M, n, n)
    a: jnp.ndarray # Shape (n, )
    def __add__(self, other):
        return jax.tree_util.tree_map(lambda x, y: x + y, self, other)
    def __sub__(self, other):
        return jax.tree_util.tree_map(lambda x, y: x - y, self, other)
    def __mul__(self, alpha):
        return jax.tree_util.tree_map(lambda x, y: x * alpha, self)
    __rmul__ = __mul__         # ←–––––––– important line
    def __neg__(self):
        return jax.tree_util.tree_map(lambda x: -x, self)

class TerminationStatus(IntEnum):
    """
    TerminationStatus explains why the solver stopped.

    Attributes
    ----------
    UNSPECIFIED : Int
        Default value.
    OPTIMAL : Int
    PRIMAL_INFEASIBLE : Int
        Note: In this situation, the dual could be either unbounded or infeasible.
    DUAL_INFEASIBLE : Int
        Note: In this situation, the primal could be either unbounded or infeasible.
    TIME_LIMIT : Int
    ITERATION_LIMIT : Int
    NUMERICAL_ERROR : Int
    INVALID_PROBLEM : Int
        Indicates that the solver detected invalid problem data.
    OTHER : Int
    """

    UNSPECIFIED = auto()
    OPTIMAL = auto()
    PRIMAL_INFEASIBLE = auto()
    DUAL_INFEASIBLE = auto()
    TIME_LIMIT = auto()
    ITERATION_LIMIT = auto()
    NUMERICAL_ERROR = auto()
    INVALID_PROBLEM = auto()
    OTHER = auto()


class TerminationCriteria(NamedTuple):
    """
    A description of solver termination criteria.

    Let p correspond to the norm we are using as specified by optimality_norm.
    If the algorithm terminates with termination_status = OPTIMAL then the following hold:
    (1) | primal_objective - dual_objective | <= eps_abs + eps_rel * ( | primal_objective | + | dual_objective | )
    (2) norm(primal_residual, p) <= eps_abs + eps_rel * norm(right_hand_side, p)
    (3) norm(dual_residual, p) <= eps_abs + eps_rel * norm(objective_vector, p)

    It is possible to prove that a solution satisfying the above conditions also satisfies SCS's optimality conditions (see link above) with ϵ_pri = ϵ_dual = ϵ_gap = eps_abs = eps_rel. (ϵ_pri, ϵ_dual, and ϵ_gap are SCS's parameters).

    If the following two conditions hold we say that we have obtained an
    approximate dual ray, which is an approximate certificate of primal
    infeasibility.
    (1) dual_ray_objective > 0.0,
    (2) max_dual_ray_infeasibility / dual_ray_objective <= eps_primal_infeasible.

    If the following three conditions hold we say we have obtained an
    approximate primal ray, which is an approximate certificate of dual
    infeasibility.
    (1) primal_ray_linear_objective < 0.0,
    (2) max_primal_ray_infeasibility / (-primal_ray_linear_objective) <= eps_dual_infeasible,
    (3) primal_ray_quadratic_norm / (-primal_ray_linear_objective) <= eps_dual_infeasible.

    Attributes
    ----------
    eps_abs : float
        Absolute tolerance on the duality gap, primal feasibility, and dual feasibility.
    eps_rel : float
        Relative tolerance on the duality gap, primal feasibility, and dual feasibility.
    eps_primal_infeasible : float
        Used to identify approximate dual ray, which is an approximate certificate of primal infeasibility.
    eps_dual_infeasible : float
        Used to identify approximate primal ray, which is an approximate certificate of dual infeasibility.
    time_sec_limit : float
        Time limit for the solver. Corresponding termination_status = TIME_LIMIT.
    iteration_limit : int
        Iteration limit for the solver. Corresponding termination_status = ITERATION_LIMIT.
    """

    eps_abs: float = 1.0e-6
    eps_rel: float = 1.0e-6
    eps_primal_infeasible: float = 1.0e-8
    eps_dual_infeasible: float = 1.0e-8
    # time_sec_limit: float = float("inf")
    iteration_limit: int = jnp.iinfo(jnp.int32).max


class CachedQuadraticProgramInfo(NamedTuple):
    primal_linear_objective_norm: float
    primal_right_hand_side_norm: float


@dataclass
class TwoSidedQpProblem:
    """
    A data class representing a quadratic programming problem with two-sided constraints.

    Attributes
    ----------
    variable_lower_bound : List[float]
        Lower bounds on the variables.
    variable_upper_bound : List[float]
        Upper bounds on the variables.
    constraint_lower_bound : List[float]
        Lower bounds on the constraints.
    constraint_upper_bound : List[float]
        Upper bounds on the constraints.
    constraint_matrix : sparse.BCSR
        The constraint matrix in BCSR format.
    objective_constant : float
        The constant term in the objective.
    objective_vector : List[float]
        The objective vector.
    objective_matrix : sparse.BCSR
        The objective matrix in BCSR format.
    """

    variable_lower_bound: List[float]
    variable_upper_bound: List[float]
    constraint_lower_bound: List[float]
    constraint_upper_bound: List[float]
    constraint_matrix: Union[BCSR, BCOO, jnp.ndarray]
    objective_constant: float
    objective_vector: List[float]
    objective_matrix: Union[BCSR, BCOO, jnp.ndarray]


@functools.partial(
    register_dataclass,
    data_fields=[
        "num_variables",
        "num_constraints",
        "variable_lower_bound",
        "variable_upper_bound",
        "isfinite_variable_lower_bound",
        "isfinite_variable_upper_bound",
        "objective_matrix",
        "objective_vector",
        "objective_constant",
        "constraint_matrix",
        "constraint_matrix_t",
        "right_hand_side",
        "num_equalities",
        "equalities_mask",
        "inequalities_mask",
    ],
    meta_fields=["is_lp"],
)
@dataclass
class QuadraticProgrammingProblem:
    """
    A QuadraticProgrammingProblem specifies a quadratic programming problem
    with the following format:

    minimize 1/2 x' * objective_matrix * x + objective_vector' * x + objective_constant

    s.t. constraint_matrix[1:num_equalities, :] * x = right_hand_side[1:num_equalities]
         constraint_matrix[(num_equalities + 1):end, :] * x >= right_hand_side[(num_equalities + 1):end]
         variable_lower_bound <= x <= variable_upper_bound

    Attributes:
    -----------
    variable_lower_bound : jnp.ndarray
        The vector of variable lower bounds.
    variable_upper_bound : jnp.ndarray
        The vector of variable upper bounds.
    objective_matrix : BCSR
        The symmetric and positive semidefinite matrix that defines the quadratic term in the objective.
    objective_vector : jnp.ndarray
        The linear coefficients of the objective function.
    objective_constant : float
        The constant term of the objective function.
    constraint_matrix : BCSR
        The matrix of coefficients in the linear constraints.
    right_hand_side : jnp.ndarray
        The vector of right-hand side values in the linear constraints.
    num_equalities : int
        The number of equalities in the problem.
    equalities_mask : jnp.ndarray
        A boolean mask indicating which constraints are equalities.
    inequalities_mask : jnp.ndarray
        A boolean mask indicating which constraints are inequalities.
    is_lp : bool
        Indicates whether the problem is a linear program (True) or a quadratic program (False).
    """

    num_variables: int
    num_constraints: int
    variable_lower_bound: jnp.ndarray
    variable_upper_bound: jnp.ndarray
    isfinite_variable_lower_bound: jnp.ndarray
    isfinite_variable_upper_bound: jnp.ndarray
    objective_matrix: Union[BCSR, BCOO, jnp.ndarray]
    objective_vector: jnp.ndarray
    objective_constant: float
    constraint_matrix: Union[BCSR, BCOO, jnp.ndarray]
    constraint_matrix_t: Union[BCSR, BCOO, jnp.ndarray]
    right_hand_side: jnp.ndarray
    num_equalities: int
    equalities_mask: jnp.ndarray
    inequalities_mask: jnp.ndarray
    is_lp: bool


class PresolveInfo(NamedTuple):
    original_primal_size: int
    original_dual_size: int
    empty_rows: List[int]
    empty_columns: List[int]
    variable_lower_bound: jnp.ndarray
    variable_upper_bound: jnp.ndarray


class RestartScheme(IntEnum):
    """
    RestartScheme IntEnum
    -  `NO_RESTARTS`: No restarts are performed.
    -  `FIXED_FREQUENCY`: does a restart every [restart_frequency] iterations where [restart_frequency] is a user-specified number.
    -  `ADAPTIVE_KKT`: a heuristic based on the KKT residual to decide when to restart.
    """

    NO_RESTARTS = auto()
    FIXED_FREQUENCY = auto()
    ADAPTIVE_KKT = auto()


class RestartToCurrentMetric(IntEnum):
    """
    RestartToCurrentMetric IntEnum
    - `NO_RESTART_TO_CURRENT`: Always reset to the average.
    - `KKT_GREEDY`: Decide between the average current based on which has a smaller KKT.
    """

    NO_RESTART_TO_CURRENT = auto()
    KKT_GREEDY = auto()


class SaddlePointOutput(NamedTuple):
    """
    A class to store the output of a saddle point computation.

    Attributes
    ----------
    primal_solution : jnp.ndarray
        The output primal solution vector.
    dual_solution : jnp.ndarray
        The output dual solution vector.
    termination_status : TerminationStatus
        One of the possible values from the TerminationStatus IntEnum.
    iteration_count : int
        The total number of algorithmic iterations for the solve.
    primal_objective: float
        The primal objective value.
    dual_objective: float
        The dual objective value.
    corrected_dual_objective: float
        The corrected dual objective value.
    primal_residual_norm: float
        The norm of the primal residual.
    dual_residual_norm: float
        The norm of the dual residual.
    relative_primal_residual_norm: float
        The relative norm of the primal residual.
    relative_dual_residual_norm: float
        The relative norm of the dual residual.
    absolute_optimality_gap: float
        The absolute optimality gap.
    relative_optimality_gap: float
        The relative optimality gap.
    """

    primal_solution: jnp.ndarray
    dual_solution: jnp.ndarray
    termination_status: TerminationStatus
    iteration_count: int
    primal_objective: float
    dual_objective: float
    corrected_dual_objective: float
    primal_residual_norm: float
    dual_residual_norm: float
    relative_primal_residual_norm: float
    relative_dual_residual_norm: float
    absolute_optimality_gap: float
    relative_optimality_gap: float
    timing_info: dict


@chex.dataclass
class RestartInfo:
    """
    A class to keep track of restart information.

    Attributes
    ----------
    primal_solution : jnp.ndarray
        The primal solution recorded at the last restart point.
    dual_solution : jnp.ndarray
        The dual solution recorded at the last restart point.
    primal_product : jnp.ndarray
        Primal product vector.
    primal_product : jnp.ndarray
        Dual product vector.
    last_restart_length : int
        The length of the last restart interval.
    primal_distance_moved_last_restart_period : float
        The primal distance moved from the restart point two restarts ago.
    dual_distance_moved_last_restart_period : float
        The dual distance moved from the restart point two restarts ago.
    reduction_ratio_last_trial : float
        Reduction in the potential function that was achieved last time a restart was attempted.
    """

    primal_solution: PrimalShape
    dual_solution: jnp.ndarray
    primal_diff: Optional[PrimalShape] = None  # used in r2HPDHG
    dual_diff: Optional[jnp.ndarray] = None  # used in r2HPDHG
    primal_diff_product: Optional[jnp.ndarray] = None  # used in r2HPDHG
    primal_product: Optional[jnp.ndarray] = None  # used in raPDHG
    dual_product: Optional[PrimalShape] = None  # used in raPDHG
    primal_obj_product: Optional[jnp.ndarray] = None
    last_restart_length: int = 1
    primal_distance_moved_last_restart_period: float = 0.0
    dual_distance_moved_last_restart_period: float = 0.0
    reduction_ratio_last_trial: float = 1.0


class RestartParameters(NamedTuple):
    """
    A class to store the parameters related to restarting the algorithm.

    Attributes
    ----------
    restart_scheme : RestartScheme
        Specifies what type of restart scheme is used.
    restart_to_current_metric : RestartToCurrentMetric
        Specifies how to decide between restarting to the average or current.
    restart_frequency_if_fixed : int
        The frequency of restarts if using a fixed frequency scheme.
    artificial_restart_threshold : float
        Fraction of iterations without a restart to trigger an artificial restart.
    sufficient_reduction_for_restart : float
        Threshold improvement to trigger a restart.
    necessary_reduction_for_restart : float
        Necessary threshold improvement for a restart.
    primal_weight_update_smoothing : float
        Controls the exponential smoothing of log(primal_weight).
    """

    restart_scheme: RestartScheme
    restart_to_current_metric: RestartToCurrentMetric
    restart_frequency_if_fixed: int
    artificial_restart_threshold: float
    sufficient_reduction_for_restart: float
    necessary_reduction_for_restart: float
    primal_weight_update_smoothing: float


class ScaledQpProblem(NamedTuple):
    """
    A ScaledQpProblem specifies an original quadratic programming problem, a scaled
    quadratic programming problem, and the scaling vector, which requires to satisfy the
    condition that original_qp = unscale_problem(scaled_qp, constraint_rescaling, variable_rescaling).

    Attributes
    ----------
    original_qp : QuadraticProgrammingProblem
        The original quadratic programming problem.
    scaled_qp : QuadraticProgrammingProblem
        The scaled quadratic programming problem.
    constraint_rescaling : jnp.ndarray
        The constraint rescaling vector.
    variable_rescaling : jnp.ndarray
        The variable rescaling vector.
    """

    original_qp: QuadraticProgrammingProblem
    scaled_qp: QuadraticProgrammingProblem
    constraint_rescaling: jnp.ndarray
    variable_rescaling: jnp.ndarray


class RestartChoice(IntEnum):
    """
    RestartChoice specifies whether a restart was performed on a given iteration.

    Attributes
    ----------
    RESTART_CHOICE_UNSPECIFIED : Int
        Default value.
    RESTART_CHOICE_NO_RESTART : Int
        No restart on this iteration.
    RESTART_CHOICE_WEIGHTED_AVERAGE_RESET : Int
        The weighted average of iterates is cleared and reset to the current point.
        From a mathematical perspective, this is equivalent to restarting the algorithm
        but picking the restart point to be the current iterate.
    RESTART_CHOICE_RESTART_TO_AVERAGE : Int
        The algorithm is restarted at the average of iterates since the last restart.
    """

    RESTART_CHOICE_UNSPECIFIED = auto()
    RESTART_CHOICE_NO_RESTART = auto()
    RESTART_CHOICE_WEIGHTED_AVERAGE_RESET = auto()
    RESTART_CHOICE_RESTART_TO_AVERAGE = auto()
    RESTART_CHOICE_RESTART_TO_CURRENT = auto()


class PointType(IntEnum):
    """
    PointType identifies the type of point used to compute the fields in a given
    structure. See ConvergenceInformation and InfeasibilityInformation.

    Values
    ------
    POINT_TYPE_UNSPECIFIED : auto
        Default value.
    POINT_TYPE_CURRENT_ITERATE : auto
        Current iterate (x_k, y_k).
    POINT_TYPE_ITERATE_DIFFERENCE : auto
        Difference of iterates (x_{k+1} - x_k, y_{k+1} - y_k).
    POINT_TYPE_AVERAGE_ITERATE : auto
        Average of iterates since the last restart.
    POINT_TYPE_NONE : auto
        There is no corresponding point.
    """

    POINT_TYPE_UNSPECIFIED = auto()
    POINT_TYPE_CURRENT_ITERATE = auto()
    POINT_TYPE_ITERATE_DIFFERENCE = auto()
    POINT_TYPE_AVERAGE_ITERATE = auto()
    POINT_TYPE_NONE = auto()


class ConvergenceInformation(NamedTuple):
    """
    Information measuring how close a candidate is to establishing feasibility and
    optimality; see also TerminationCriteria.

    Attributes
    ----------
    candidate_type : PointType
        Type of the candidate point described by this ConvergenceInformation.
    primal_objective : float
        The primal objective. The primal need not be feasible.
    dual_objective : float
        The dual objective. The dual need not be feasible. The dual objective should
        include the contributions from the reduced costs.
    corrected_dual_objective : float
        If possible (e.g., when all primal variables have lower and upper bounds), a
        correct dual bound. Set to negative infinity if no corrected dual bound is available.
    l_inf_primal_residual : float
        The maximum violation of any primal constraint, i.e., the l_∞ norm of the violations.
    l2_primal_residual : float
        The l_2 norm of the violations of primal constraints.
    l_inf_dual_residual : float
        The maximum violation of any dual constraint, i.e., the l_∞ norm of the violations.
    l2_dual_residual : float
        The l_2 norm of the violations of dual constraints.
    relative_l_inf_primal_residual : float
        Relative l_∞ norm of the primal constraint violations.
    relative_l2_primal_residual : float
        Relative l_2 norm of the primal constraint violations.
    relative_l_inf_dual_residual : float
        Relative l_∞ norm of the dual constraint violations.
    relative_l2_dual_residual : float
        Relative l_2 norm of the dual constraint violations.
    relative_optimality_gap : float
        Relative optimality gap:
        |primal_objective - dual_objective| / (eps_ratio + |primal_objective| + |dual_objective|).
    l_inf_primal_variable : float
        The maximum absolute value of the primal variables, i.e., the l_∞ norm.
        Useful to detect when the primal iterates are diverging.
    l2_primal_variable : float
        The l_2 norm of the primal variables.
    l_inf_dual_variable : float
        The maximum absolute value of the dual variables, i.e., the l_∞ norm.
        Useful to detect when the dual iterates are diverging.
    l2_dual_variable : float
        The l_2 norm of the dual variables.
    """

    candidate_type: PointType = PointType.POINT_TYPE_UNSPECIFIED
    primal_objective: float = 0.0
    dual_objective: float = 0.0
    primal_linear_objective: float = 0.0
    corrected_dual_objective: float = float("-inf")
    primal_residual_norm: float = 0.0
    dual_residual_norm: float = 0.0
    relative_primal_residual_norm: float = 0.0
    relative_dual_residual_norm: float = 0.0
    absolute_optimality_gap: float = 0.0
    relative_optimality_gap: float = 0.0
    primal_solution_norm: float = 0.0
    dual_solution_norm: float = 0.0
    l2_difference: jnp.ndarray = jnp.inf
    objective_difference: jnp.ndarray = jnp.inf


class InfeasibilityInformation(NamedTuple):
    """
    Information measuring how close a point is to establishing primal or dual
    infeasibility (i.e., has no solution); see also TerminationCriteria.

    Attributes
    ----------
    candidate_type : PointType
        Type of the point used to compute the InfeasibilityInformation.

    max_primal_ray_infeasibility : float
        Let `x_ray` be the algorithm's estimate of the primal extreme ray where
        `x_ray` is a vector scaled such that its infinity norm is one.
        A simple and typical choice of `x_ray` is `x_ray = x / ||x||_∞` where `x`
        is the current primal iterate. For this value, compute the maximum absolute
        error in the primal linear program with the right-hand side set to zero.

    primal_ray_linear_objective : float
        The value of the linear part of the primal objective (ignoring additive
        constants) evaluated at `x_ray`, i.e., `c' * x_ray` where `c` is the
        objective coefficient vector.

    max_dual_ray_infeasibility : float
        Let `(y_ray, r_ray)` be the algorithm's estimate of the dual and reduced
        cost extreme ray where `(y_ray, r_ray)` is a vector scaled such that its
        infinity norm is one. A simple and typical choice of `y_ray` is
        `(y_ray, r_ray) = (y, r) / max(||y||_∞, ||r||_∞)` where `y` is the current
        dual iterate and `r` is the current dual reduced costs. Consider the
        quadratic program we are solving but with the objective (both quadratic and
        linear terms) set to zero. This forms a linear program (label this linear
        program (1)) with no objective. Take the dual of (1) and compute the maximum
        absolute value of the constraint error for `(y_ray, r_ray)` to obtain the
        value of `max_dual_ray_infeasibility`.

    dual_ray_objective : float
        The objective of the linear program labeled (1) in the previous paragraph.
    """

    candidate_type: PointType = PointType.POINT_TYPE_UNSPECIFIED
    max_primal_ray_infeasibility: float = 0.0
    primal_ray_linear_objective: float = 0.0
    max_dual_ray_infeasibility: float = 0.0
    dual_ray_objective: float = 0.0


@chex.dataclass
class IterationStats:
    """
    A data class to store statistics for each iteration of a quadratic program
    solver. All values assume that the primal quadratic program is a minimization
    problem and the dual is a maximization problem. Problems should be transformed
    to this form if they are not already.

    Attributes
    ----------
    iteration_number : int
        The iteration number at which these stats were recorded. By convention,
        iteration counts start at 1, and the stats correspond to the solution
        *after* the iteration. Therefore, stats from iteration 0 are the stats
        at the starting point.

    convergence_information : ConvergenceInformation
        A set of statistics measuring how close a point is to establishing primal
        and dual feasibility and optimality. This field is repeated since there might
        be several different points that are considered.

    infeasibility_information : InfeasibilityInformation
        A set of statistics measuring how close a point is to establishing primal or
        dual infeasibility (i.e., has no solution). This field is repeated since there
        might be several different points that could establish infeasibility.

    cumulative_rejected_steps : int
        The total number of rejected steps (e.g., within a line search procedure) since
        the start of the solve.

    cumulative_time_sec : float
        The amount of time passed since we started solving the problem (total time).

    step_size : float
        Step size used at this iteration. Note that the step size used for the primal
        update is step_size / primal_weight, while the one used for the dual update is
        step_size * primal_weight.

    primal_weight : float
        Primal weight controlling the relation between primal and dual step sizes.
        See the 'step_size' field for a detailed description.

    method_specific_stats : Dict[str, float]
        A dictionary containing method-specific statistics with string keys and float values.
    """

    iteration_number: int = 0
    convergence_information: ConvergenceInformation = None
    infeasibility_information: InfeasibilityInformation = None
    cumulative_rejected_steps: int = 0
    cumulative_time_sec: float = 0.0
    step_size: float = 0.0
    primal_weight: float = 0.0
    method_specific_stats: Dict[str, float] = field(default_factory=dict)


class AdaptiveStepsizeParams(NamedTuple):
    reduction_exponent: float
    growth_exponent: float
    step_size_limit_coef: float = 1.0


class ConstantStepsizeParams(NamedTuple):
    pass


class PdhgParameters(NamedTuple):
    l_inf_ruiz_iterations: int
    l2_norm_rescaling: bool
    pock_chambolle_alpha: Union[float, None]
    primal_importance: float
    scale_invariant_initial_primal_weight: bool
    verbose: bool
    debug: bool
    # record_iteration_stats: bool
    termination_evaluation_frequency: int
    termination_criteria: TerminationCriteria
    restart_params: RestartParameters
    step_size_policy_params: Union[AdaptiveStepsizeParams, ConstantStepsizeParams]
    display_frequency: int


@chex.dataclass
class PrimalShape:
    P: jnp.ndarray # Shape (M, n, n)
    a: jnp.ndarray # Shape (n, )
    def __add__(self, other):
        return jax.tree_util.tree_map(lambda x, y: x + y, self, other)
    def __sub__(self, other):
        return jax.tree_util.tree_map(lambda x, y: x - y, self, other)
    def __mul__(self, alpha):
        return jax.tree_util.tree_map(lambda x: x * alpha, self)
    def __neg__(self):
        return jax.tree_util.tree_map(lambda x: -x, self)

@chex.dataclass
class PdhgSolverState:
    current_primal_solution: PrimalShape
    current_dual_solution: jnp.ndarray
    current_primal_product: jnp.ndarray
    current_dual_product: PrimalShape
    current_primal_obj_product: jnp.ndarray
    reg: jnp.ndarray
    solutions_count: int
    step_size: float
    primal_weight: float
    numerical_error: bool
    # total_number_iterations: int
    num_steps_tried: int
    num_iterations: int
    termination_status: TerminationStatus
    weights_sum: float = 0.0
    # Average solutions are used in raPDHG
    avg_primal_solution: Optional[PrimalShape] = None
    avg_dual_solution: Optional[jnp.ndarray] = None
    avg_primal_product: Optional[jnp.ndarray] = None
    avg_dual_product: Optional[PrimalShape] = None
    avg_primal_obj_product: Optional[jnp.ndarray] = None
    # Initial solutions and delta information are used in r2HPDHG
    delta_primal: Optional[PrimalShape] = None
    delta_dual: Optional[jnp.ndarray] = None
    delta_primal_product: Optional[jnp.ndarray] = None
    initial_step_size: float = 0.0


class OTProblem(NamedTuple):
    cost_matrix: jnp.ndarray
    marginals: jnp.ndarray
    cost_matrix_norm: float
    mu: jnp.ndarray
    nu: jnp.ndarray
    norm_A: float
    n: int

# Primal shape (n, n) --> Dual shape (2, n)
def ot_apply_A(P: jnp.ndarray) -> jnp.ndarray:
    row_sums = jnp.sum(P, axis=1)
    col_sums = jnp.sum(P, axis=0)
    return jnp.array([row_sums, col_sums])

# Dual shape (2, n) --> Primal shape (n, n)
def ot_apply_AT(dual: jnp.ndarray) -> jnp.ndarray:
    return dual[0, :, None] + dual[1, None, :]


def create_ot_problem(C: jnp.ndarray, mu: jnp.ndarray, nu: jnp.ndarray) -> OTProblem:
    n = mu.shape[0]
    return OTProblem(
        cost_matrix=C,
        cost_matrix_norm=jnp.linalg.norm(C),
        marginals=jnp.array([mu, nu]),
        mu=mu,
        nu=nu,
        n=int(n),
        norm_A=jnp.sqrt(2 * n)
    )
class BarycenterProblem(NamedTuple):
    cost: jnp.ndarray       # Shape (M, n, n)
    marginals: jnp.ndarray  # Shape (M, n)
    weights: jnp.ndarray    # Shape (M, )
    n: int
    norm_A: float
    cost_matrix_norm: float

class BarycenterProblemBatch(NamedTuple):
    cost: jnp.ndarray       # Shape (M, n, n)
    marginals: jnp.ndarray  # Shape (M, n)
    weights: jnp.ndarray    # Shape (M, )
    n: jnp.ndarray
    norm_A: jnp.ndarray
    cost_matrix_norm: jnp.ndarray

# Primal State --> Dual shape (2*M, n)
def barry_apply_A(x: PrimalShape):
    """Return stacked (row‑part, col‑part) with shape (2*M, n)."""
    row_part = jnp.sum(x.P, axis=2) - x.a           # (M, n)
    col_part = jnp.sum(x.P, axis=1)               # (M, n)
    return jnp.concatenate([row_part, col_part], axis=0)  # (2*M, n)

# Dual shape (2*M, n) --> Primal State
def barry_apply_AT(y: jnp.ndarray) -> PrimalShape:
    """
    y has shape (2*M, n).  First M rows are Δu_k, next M rows are Δv_k.
    """
    M = y.shape[0] // 2
    du = y[:M]           # (M, n)
    dv = y[M:]           # (M, n)

    # build ΔP
    delta_P = du[:, :, None] + dv[:, None, :]   # broadcast to (M, n, n)

    # build Δa
    delta_a = jnp.sum(du, axis=0)              # (n,)

    return PrimalShape(P=delta_P, a=delta_a)

def create_barycenter_problem(
    cost_matrix: jnp.ndarray,
    marginals: jnp.ndarray,
    weights: jnp.ndarray,
):
    M = marginals.shape[0]
    n = np.prod(marginals.shape[1:])
    return BarycenterProblem(
        cost=jnp.broadcast_to(cost_matrix[None, ...], (M, *cost_matrix.shape)),
        cost_matrix_norm=M * jnp.linalg.norm(cost_matrix),
        marginals=marginals,
        weights=weights,
        n=n,
        norm_A=jnp.sqrt(0.5 * (2*n + M + jnp.sqrt(4*n*n + M*M)))
    )

def create_batched_barycenter_problem(
    cost_matrix: jnp.ndarray, # shape (n, n)
    marginals_batch: jnp.ndarray, # shape (K, M, n)
    weights_batch: jnp.ndarray, # shape (K, M)
):
   K, M, n = marginals_batch.shape
   assert cost_matrix.shape == (n, n), "cost‑matrix shape mismatch"
   assert weights_batch.shape == (K, M), "weights shape mismatch"

   norm_A_val = jnp.sqrt(0.5 * (2 * n + M + jnp.sqrt(4 * n * n + M * M)))
   cost_matrix_norm = M * jnp.linalg.norm(cost_matrix)

   return BarycenterProblemBatch(
        cost=jnp.broadcast_to(cost_matrix[None, None, ...], (K, M, n, n)),
        cost_matrix_norm=jnp.full((K,), cost_matrix_norm),
        marginals=marginals_batch,
        weights=weights_batch,
        n=jnp.full((K,), n, dtype=jnp.int32),
        norm_A=jnp.full((K,), norm_A_val)
   )


class TimingData(dict):
    def record_time(self, code_block_name, elapsed_time):
        # Adds elapsed time to the specified code block's total
        self[code_block_name] = self.get(code_block_name, 0) + elapsed_time

    def get_block_time(self, code_block_name):
        # Returns the total elapsed time for the specified code block
        return self.get(code_block_name, 0)

    def get_main_elapsed_time(self):
        # Calculates the elapsed time since the main timer started
        if self.main_timer_start_time is None:
            raise ValueError("Main timer has not been started.")
        return timeit.default_timer() - self.main_timer_start_time


@contextmanager
def time_code(timing_data_obj, code_block_name, is_main_timer=False):
    """Starts timer at entry, stores elapsed time at exit.

    If `is_main_timer=True`, the start time is stored in the timing_data_obj,
    allowing calculation of total elapsed time using `get_main_elapsed_time()`.
    """
    start_time = timeit.default_timer()
    if is_main_timer:
        timing_data_obj.main_timer_start_time = start_time
    yield
    elapsed_time = timeit.default_timer() - start_time
    timing_data_obj.record_time(code_block_name, elapsed_time)

def save_conv_info_to_dict(conv_info: ConvergenceInformation):
    return conv_info._asdict()

def blank_conv_info():
    conv_info = ConvergenceInformation()
    return conv_info._asdict()

def plot_convergence(array_iters=[], labels=[], filename=None, plot_every=1):
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(20, 14))

    has_ref_sol = (array_iters[0]["l2_difference"] != jnp.inf).any()

    fig.suptitle("Convergence plots for Quadratic PDHG")

    array_primal_residuals = [iters["primal_residual_norm"][::plot_every] for iters in array_iters]
    array_rel_primal_residuals = [iters["relative_primal_residual_norm"][::plot_every] for iters in array_iters]
    array_dual_residuals = [iters["dual_residual_norm"][::plot_every] for iters in array_iters]
    array_rel_dual_residuals = [iters["relative_dual_residual_norm"][::plot_every] for iters in array_iters]
    array_gaps = [iters["absolute_optimality_gap"][::plot_every] for iters in array_iters]
    array_rel_gaps = [iters["relative_optimality_gap"][::plot_every] for iters in array_iters]

    if has_ref_sol:
        array_obj_differences = [iters["objective_difference"][::plot_every] for iters in array_iters]
        array_l2_differences = [iters["l2_difference"][::plot_every] for iters in array_iters]

    min_len = min(map(len, array_primal_residuals))
    x = jnp.arange(min_len)

    if labels is None:
        labels = [f"Solution {i}" for (i, _) in enumerate(array_iters)]

    # Axis 1 Absolute tolerance Primal Feasibility
    for (primal_residuals, label) in zip(array_primal_residuals, labels):
        ax1.plot(x, primal_residuals[:min_len], label=label)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_title("Primal feasibility")

    for (rel_primal_residuals, label) in zip(array_rel_primal_residuals, labels):
        ax2.plot(x, rel_primal_residuals[:min_len], label=label)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.set_title("Relative Primal feasibility")

    ax3.set_yscale('log')
    for (dual_residuals, label) in zip(array_dual_residuals, labels):
        if float(0) in dual_residuals:
            ax3.set_yscale('linear')
        ax3.plot(x, dual_residuals[:min_len], label=label)
    ax3.legend()
    ax3.set_title("Dual feasibility")

    ax4.set_yscale('log')
    for (rel_dual_residuals, label) in zip(array_rel_dual_residuals, labels):
        if float(0) in rel_dual_residuals:
            ax4.set_yscale('linear')
        ax4.plot(x, rel_dual_residuals[:min_len], label=label)
    ax4.legend()
    ax4.set_title("Relative Dual feasibility")

    for (gaps, label) in zip(array_gaps, labels):
        ax5.plot(x, gaps[:min_len], label=label)
    ax5.set_yscale('log')
    ax5.legend()
    ax5.set_title("Optimality gaps")

    for (rel_gaps, label) in zip(array_rel_gaps, labels):
        ax6.plot(x, rel_gaps[:min_len], label=label)
    ax6.set_yscale('log')
    ax6.legend()
    ax6.set_title("Relative Optimality gaps")

    if has_ref_sol:
        for (obj_differences, label) in zip(array_obj_differences, labels):
            ax7.plot(x, obj_differences[:min_len], label=label)
        ax7.set_yscale('log')
        ax7.legend()
        ax7.set_title("Objective differences with LP solution")

        for (l2_differences, label) in zip(array_l2_differences, labels):
            ax8.plot(x, l2_differences[:min_len], label=label)
        ax8.set_yscale('log')
        ax8.legend()
        ax8.set_title("L2 differences with LP solution")
    else:
        fig.delaxes(ax7)
        fig.delaxes(ax8)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(f"plots/pdlp-gpu-{filename}.png")
    else:
        plt.plot()

def cut_dict_at_first_zero(arr_dict):
    """
    Assume every non‑None 1‑D array in `arr_dict` is identical up to
    some index K, and every entry from K onward is zero.
    Returns a *new* dict with each array cut at K.
    """
    if arr_dict is None:
        return None
    # 1) pick any representative array
    first_key, ref = next((k, v) for k, v in arr_dict.items() if v is not None)

    # 2) find the first zero index K  (implicitly assumes array has at least one zero)
    #    jnp.argmax(ref == 0) → position of the first `True`
    K = int(jax.device_get(jnp.argmax(ref == 0)))

    # 3) build a new dict with arrays sliced at K
    trimmed = {
        k: (v[:K] if isinstance(v, jax.Array) else v)
        for k, v in arr_dict.items()
    }
    return trimmed
