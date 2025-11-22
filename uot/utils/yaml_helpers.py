import os

from uot.problems.iterator import ProblemIterator
from uot.problems.store import ProblemStore
from uot.utils.import_helpers import import_object
from uot.utils.exceptions import InvalidConfigurationException
from uot.solvers.solver_config import SolverConfig
from uot.experiments.experiment import Experiment


def load_solvers(config: dict) -> list[SolverConfig]:
    solvers_configs = config['solvers']
    params = config['param-grids']

    loaded_solvers_configs = []

    for solver_name, solver_config in solvers_configs.items():
        solver_class = solver_config['solver']
        params_grid_name = solver_config.get('param-grid', {})
        is_jit = solver_config['jit']
        use_cost_matrix = solver_config.get('use-cost-matrix', True)

        solver = import_object(solver_class)

        solver_config = SolverConfig(
            name=solver_name,
            solver=solver,
            param_grid=params[params_grid_name] if params_grid_name else [{}],
            is_jit=is_jit,
            use_cost_matrix=use_cost_matrix,
        )

        loaded_solvers_configs.append(solver_config)

    return loaded_solvers_configs


def load_problems(config: dict) -> list[ProblemIterator]:
    iterators = []
    problemsets_names = config.get('problems', {'names': []})['names']
    problemsets_dir = config.get('problems', {'dir': None})['dir']

    for problemset_name in problemsets_names:
        store_path = os.path.join(problemsets_dir, problemset_name)

        if not os.path.exists(store_path):
            raise InvalidConfigurationException(
                f"There is no problem store on path {store_path}")

        problems_store = ProblemStore(store_path)
        problems_iterator = ProblemIterator(problems_store)

        if not len(problems_iterator):
            raise InvalidConfigurationException(
                f"There is no problems on path {store_path}")

        iterators.append(problems_iterator)

    return iterators


def load_experiment(config: dict) -> Experiment:
    experiment_name = config['experiment']['name']
    experiment_function = import_object(config['experiment']['function'])

    return Experiment(name=experiment_name,
                      solve_fn=experiment_function)
