import ot
import jax
import time
import multiprocessing
import numpy as np
import pandas as pd
import itertools as it
import jax.numpy as jnp
from uot.dataset import Measure, generate_coefficients, generate_measures, get_grids, load_from_file, save_to_file
from uot.analysis import get_agg_table
import os.path


def get_q_const(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    m = y.shape[0]

    return ot.dist(x.reshape((n, -1)), y.reshape((m, -1)))

def get_exact_solution(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, float]:
    T = ot.emd(a, b, C)
    return T, np.sum(T * C)

def generate_two_fold_problems(grid, measures: list[Measure], name: str, one_cost=False):
    ot_problems = []
    if one_cost:
        source_points, _ = measures[0].to_histogram()
        target_points, _ = measures[1].to_histogram()
        C = get_q_const(source_points, target_points)
    else:
        C = get_q_const
    for source_measure, target_measure in it.combinations(measures, 2):
        ot_problem = OTProblem(name="Simple transport", 
                                     source_measure=source_measure,
                                     target_measure=target_measure,
                                     C=C)
        ot_problem.kwargs.update({'name': name})
        ot_problems.append(ot_problem)
    return ot_problems

def create_problemset(dim: int, distributions: dict[str, int], grid_size: int, coefficients = None):
    """
    Generates a set of OT problems based on the specified dimensions, distributions, and grid size.

    Args:
        dim (int): Dimensionality of the dataset (1, 2, or 3).
        distributions (dict): Dictionary containing distribution types and their counts.
        grid_size (int): Size of the grid for the measures.

    Returns:
        list[OTProblem]: A list of OTProblem objects.
    """
    grids = get_grids(dim, [grid_size])
    if coefficients is None:
        coefficients = generate_coefficients(dim, distributions)
    measures = generate_measures(dim, coefficients, grids)
    name = f"{'x'.join([str(grid_size)] * dim)} {dim}D {'_'.join(sorted(distributions))}"

    try:
        problems = generate_two_fold_problems(None, measures[name.replace('_', '|')], name=name)
    except KeyError as e:
        print(f"KeyError: {name.replace('_', '|')} not found in measures. Available keys: {list(measures.keys())}")
        return

    return problems

def get_problemset(name: str, coeffs = None, create: bool = False):
    size, dim, distributions = name.split(' ')
    distributions = sorted(distributions.split('|'))
    dim = int(dim[0])

    if dim not in [1, 2, 3]:
        raise ValueError(f"Invalid dimension: {dim}. Expected 1, 2, or 3.")

    if len(size.split('x')) != dim:
        raise ValueError(f"Invalid size format: {size}. Expected format: {'x'.join(['<size>'] * dim)}.")

    filename = f"./datasets/{dim}D/{size}_{'_'.join(distributions)}.pkl"
    
    if os.path.exists(filename) and not create:
        return load_from_file(filename)

    elif create:
        if os.path.exists(filename):
            filename = filename.replace('.pkl', '_1.pkl')
            i = 2
            while os.path.exists(filename):
                filename = filename.replace(f"_{i-1}.pkl", f"_{i}.pkl")
                i += 1

    distribution_counts = {distribution: 10 // len(distributions) for distribution in distributions}
    problems = create_problemset(dim, distribution_counts, int(size.split('x')[0]), coeffs)

    if not problems:
        raise ValueError(f"Failed to create problems for {name}. Check the parameters.")

    save_to_file(problems, filename)
    return problems

def generate_two_fold_problems_lazy(grid, measures_generator, name: str, one_cost=False):
    """
    Lazily generates two-fold OT problems from a generator of measures.

    Args:
        grid (list[np.ndarray]): The grid used for the measures.
        measures_generator (generator): A generator yielding Measure objects.
        name (str): Name of the OT problems.
        one_cost (bool): Whether to compute a single cost matrix for all problems.

    Yields:
        OTProblem: An OTProblem object for each pair of measures.
    """

    if one_cost:
        first_measure = next(measures_generator)
        second_measure = next(measures_generator)
        source_points, _ = first_measure.to_histogram()
        target_points, _ = second_measure.to_histogram()
        C = get_q_const(source_points, target_points)
        measures_generator = it.chain([first_measure, second_measure], measures_generator)  # Reinsert the first two measures
    else:
        C = get_q_const

    for source_measure, target_measure in it.combinations(measures_generator, 2):
        ot_problem = OTProblem(
            name="Simple transport",
            source_measure=source_measure,
            target_measure=target_measure,
            C=C
        )
        ot_problem.kwargs.update({'name': name})
        yield ot_problem


class OTProblem:

    def __init__(self, name: str, source_measure: Measure, target_measure: Measure, C: np.ndarray, kwargs: dict = None):
        self.name = name
        self.source_measure = source_measure
        self.target_measure = target_measure
        self._C = C
        
        self._exact_cost = None
        self._exact_map = None 

        self.kwargs = kwargs if kwargs is not None else {}
    
    @property
    def C(self):
        if callable(self._C):
            self._C = self._C(self.source_measure.get_flat_support(),
                              self.target_measure.get_flat_support())
            return self._C
        return self._C
    
    @property
    def a(self):
        return self.source_measure.to_histogram()[1]
    
    @property
    def b(self):
        return self.target_measure.to_histogram()[1]

    def to_jax_arrays(self, regularization=1e-30):
        C = self.C
        C /= self.C.max()

        a = jnp.array(self.a + regularization)
        b = jnp.array(self.b + regularization)
        C = jnp.array(C + regularization)
        return a, b, C

    @property
    def exact_cost(self):
        if self._exact_cost is None:
            self._exact_map, self._exact_cost = get_exact_solution(self.a, self.b, self.C)
        return self._exact_cost

    @property
    def exact_map(self):
        if self._exact_map is None:
            self._exact_map, self._exact_cost = get_exact_solution(self.a, self.b, self.C)
        return self._exact_map

    def __hash__(self):
        return hash(self.name) + hash(str(self.source_measure.kwargs)) + hash(str(self.target_measure.kwargs))
        
    def __eq__(self, other):
        if not isinstance(other, OTProblem):
            return False
        
        return self.name == other.name and \
               self.source_measure == other.source_measure and \
               self.target_measure == other.target_measure and \
               np.array_equal(self.C, other.C) and \
               self.kwargs == other.kwargs

    def __str__(self):
        return f"<OTProblem: {self.name} source={self.source_measure}, target={self.target_measure}>"

    def to_dict(self) -> dict:
        problem_dict = {'name': self.name, 'source_measure_name': self.source_measure.name, 'target_measure_name': self.target_measure.name}
        source_kwargs = { f"source_{key}": value for key, value in self.source_measure.kwargs.items() }
        target_kwargs = { f"target_{key}": value for key, value in self.target_measure.kwargs.items() }
        problem_dict.update(source_kwargs)
        problem_dict.update(target_kwargs)
        problem_dict.update(self.kwargs)
        return problem_dict


class RunResult:
    
    def __init__(self, name: str, result_df: pd.DataFrame, run_kwargs: dict):
        self.name = name
        self.result_df = result_df
        self.run_kwargs = run_kwargs

    def display_result(self):
        self.display_header() 
        print(self.result_df)

    def display_agg(self):
        self.display_header()
        print(get_agg_table(self.result_df))

    def get_agg(self):
        return get_agg_table(self.result_df)

    def display_header(self):
        print("Name", self.name)
        for key, value in self.run_kwargs.items():
            print(f"{key}: {value}")
        print('='*100)

    def export(self, filepath: str) -> None:
        self.result_df.to_csv(filepath)


class Experiment:

    def __init__(self, name: str, run_function: callable):
        self.name = name
        self.run_function = run_function

    def run_experiment(self, ot_problems: list[OTProblem], progress_callback: callable = None, **kwargs) -> dict:
        results = {}
        for ot_problem in ot_problems:
            results[ot_problem] = self.run_function(ot_problem, **kwargs)
            if progress_callback is not None:
                progress_callback()
        
        return results
    
    def run_single(self, ot_problem: OTProblem) -> dict:
        return self.run_function(ot_problem)

    
class ExperimentSuite:
    MAX_RESULTS_IN_WORKER = 50

    def __init__(self, experiments: list[Experiment]):
        self.experiments = experiments

    def run_suite(self, name: str, ot_problems: list[OTProblem], njobs: int = 1,
                  progress_callback: callable = None, **kwargs) -> pd.DataFrame:
        if njobs == 1:
            return self._run_suite(name, ot_problems, progress_callback=progress_callback, **kwargs)
        else:
            return self._run_suite_multiprocess(ot_problems, njobs)
    
    def _run_suite_multiprocess(self, ot_problems: list[Experiment], njobs: int):

        def _worker(queue, tasks):
            results = []
            for task in tasks:
                experiment, ot_problem, ot_identifier = task
                results.append((ot_identifier, experiment.run_single(ot_problem)))

                if len(results) == self.MAX_RESULTS_IN_WORKER:
                    queue.put(results)
                    results = []

            queue.put(results)

        # added problems identifiers to reduce time for copying data from processes
        ot_problems_ids = { ot_problem: identifier for identifier, ot_problem in enumerate(ot_problems) }
        ids_to_ot_problem = dict(zip(ot_problems_ids.values(), ot_problems_ids.keys()))

        tasks = [ (experiment, ot_problem, ot_problems_ids[ot_problem]) for ot_problem in ot_problems for experiment in self.experiments ]

        q = multiprocessing.Queue()
        tasks_per_worker = len(tasks) // njobs
        processes = [ multiprocessing.Process(target=_worker, args=(q, tasks[i * tasks_per_worker: (i+1) * tasks_per_worker]))
                      for i in range(njobs) ]

        for p in processes: 
            p.start()

        ot_problems_results = {ot_problem: {} for ot_problem in ot_problems}
        while any(p.is_alive() for p in processes) or not q.empty():
            time.sleep(0.5)
            try:
                results = q.get_nowait()
                for (ot_identifier, result) in results:
                    ot_problem = ids_to_ot_problem[ot_identifier]
                    ot_problems_results[ot_problem].update(result)
            except multiprocessing.queues.Empty:
                pass
        
        df_rows = []
        for ot_problem, results in ot_problems_results.items():
            row_dict = ot_problem.to_dict() | results
            df_rows.append(row_dict)

        return pd.DataFrame(df_rows)

    def _run_suite(self, name, ot_problems: list[OTProblem], progress_callback: callable = None, **kwargs) -> RunResult:
        results = []
        for experiment in self.experiments:
            results.append(experiment.run_experiment(ot_problems, progress_callback=progress_callback, **kwargs))

        df_rows = []
        for ot_problem in ot_problems:
            row_dict = ot_problem.to_dict() 
            for result in results:
                row_dict.update(result[ot_problem])
            df_rows.append(row_dict)

        result = RunResult(name=name, result_df=pd.DataFrame(df_rows), run_kwargs=kwargs)

        return result
