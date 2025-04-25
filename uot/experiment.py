import ot
import jax
import time
import multiprocessing
import numpy as np
import pandas as pd
import itertools as it
import jax.numpy as jnp
import open3d as o3d
from uot.dataset import Measure, generate_coefficients, generate_measures, get_grids, load_from_file, save_to_file, Measure
from uot.analysis import get_agg_table
from tqdm import tqdm
import os.path
import os

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

def generate_data_problems(data_type: str, num_points: int, num_samples: int = 10, create_grids: bool = True):
    """
    Generates OT problems from data files in the Data folder.
    
    Args:
        data_type (str): The type of data to use (e.g., 'CauchyDensity', 'ClassicImages')
        num_points (int): The number of points/resolution (e.g., 32, 64, 128, 256, 512)
        num_samples (int): Maximum number of samples to use (default: 10)
        create_grids (bool): Whether to create grid points for the measures (default: True)
    
    Returns:
        list[OTProblem]: A list of OTProblem objects created from the data

    Full list of data types:
        - WhiteNoise
        - Cauchy density
        - GRFmoderate
        - GRFrough
        - GRFsmooth
        - LogGRF
        - LogitGRF
        - MicroscopyImages
        - Shapes
        - ClassicImages
    """
    data_folder = os.path.join("Data", data_type)
    
    if not os.path.exists(data_folder):
        raise ValueError(f"Data folder '{data_folder}' does not exist")
    
    file_pattern = f"data{num_points}_"
    data_files = [f for f in os.listdir(data_folder) if file_pattern in f]
    
    if len(data_files) > num_samples:
        data_files = data_files[:num_samples]
    
    measures = []
    for file_name in data_files:
        file_path = os.path.join(data_folder, file_name)
        data = pd.read_csv(file_path, header=None).values
        
        data = data / data.sum()
        
        if create_grids:
            
            x = np.linspace(0, 1, data.shape[1])
            y = np.linspace(0, 1, data.shape[0])
            grid = np.meshgrid(x, y)
            
            measure = Measure(
                name=f"{data_type}_{file_name.replace('.csv', '')}",
                support=grid,
                distribution=data,
                kwargs={"data_type": data_type, "file_name": file_name}
            )
        else:
            flat_data = data.flatten()
            
            indices = np.arange(len(flat_data))
            
            measure = Measure(
                name=f"{data_type}_{file_name.replace('.csv', '')}",
                support=[indices],
                distribution=flat_data,
                kwargs={"data_type": data_type, "file_name": file_name}
            )
        
        measures.append(measure)
    
    problems = []
    for source_measure, target_measure in it.combinations(measures, 2):
        ot_problem = OTProblem(
            name=f"{data_type} {num_points}x{num_points}",
            source_measure=source_measure,
            target_measure=target_measure,
            C=get_q_const
        )
        ot_problem.kwargs.update({"data_type": data_type, "num_points": num_points})
        problems.append(ot_problem)
    
    return problems

def generate_3d_mesh_problems(num_points: int = None, num_samples: int = 10, color_mode: str = "rgb_avg"):
    """
    Generates OT problems from 3D colored mesh files in the Reference_3D_colored_meshes folder.
    
    Args:
        num_points (int, optional): The number of points to sample from each 3D mesh. 
                                   If None, uses the actual number of vertices in each mesh.
        num_samples (int): Maximum number of mesh files to use (default: 10)
        color_mode (str): How to handle color channels:
                          - "r", "g", "b": Use a single color channel
                          - "rgb_avg": Use all three color channels combined
                          - "separate": Create separate problems for each color channel
    
    Returns:
        list[OTProblem]: A list of OTProblem objects created from the 3D meshes
    """
    mesh_folder = "Reference_3D_colored_meshes"
    
    if not os.path.exists(mesh_folder):
        raise ValueError(f"Mesh folder '{mesh_folder}' does not exist")
    
    mesh_files = [f for f in os.listdir(mesh_folder) if f.endswith('.ply')]
    
    if len(mesh_files) > num_samples:
        mesh_files = mesh_files[:num_samples]
    
    if color_mode.lower() == "r":
        channels = [0]
        channel_names = ["red"]
    elif color_mode.lower() == "g":
        channels = [1]
        channel_names = ["green"]
    elif color_mode.lower() == "b":
        channels = [2]
        channel_names = ["blue"]
    elif color_mode.lower() == "rgb_avg":
        channels = [None]
        channel_names = ["rgb_avg"]
    elif color_mode.lower() == "separate":
        channels = [0, 1, 2]
        channel_names = ["red", "green", "blue"]
    else:
        raise ValueError(f"Invalid color_mode: {color_mode}. Must be 'r', 'g', 'b', 'rgb_avg', or 'separate'")
    
    measures_by_channel = {channel_name: [] for channel_name in channel_names}
    
    for file_name in mesh_files:
        file_path = os.path.join(mesh_folder, file_name)
        
        try:
            mesh = o3d.io.read_triangle_mesh(file_path)
            
            mesh_num_points = len(np.asarray(mesh.vertices))
            sampling_points = num_points if num_points is not None else mesh_num_points
            
            if sampling_points > mesh_num_points:
                print(f"Warning: Requested {sampling_points} points but {file_name} only has {mesh_num_points} vertices. Using all vertices.")
                sampled_points = mesh
            else:
                sampled_points = mesh.sample_points_uniformly(sampling_points)
            
            points = np.asarray(sampled_points.points)
            colors = np.asarray(sampled_points.colors)
            
            for channel, channel_name in zip(channels, channel_names):
                if channel is None:
                    distribution = colors.mean(axis=1)
                    distribution = distribution / distribution.sum()
                    
                    measure = Measure(
                        name=f"3DMesh_{file_name.replace('.ply', '')}_rgb_avg",
                        support=[points[:, 0], points[:, 1], points[:, 2]],
                        distribution=distribution,
                        kwargs={
                            "mesh_name": file_name, 
                            "color_mode": "rgb_avg", 
                            "num_points": len(points)
                        }
                    )
                    measures_by_channel["rgb_avg"].append(measure)
                else:
                    distribution = colors[:, channel]
                    distribution = distribution / distribution.sum()
                
                    measure = Measure(
                        name=f"3DMesh_{file_name.replace('.ply', '')}_{channel_name}",
                        support=[points[:, 0], points[:, 1], points[:, 2]],
                        distribution=distribution,
                        kwargs={
                            "mesh_name": file_name, 
                            "color_channel": channel,
                            "color_name": channel_name,
                            "num_points": len(points)
                        }
                    )
                    measures_by_channel[channel_name].append(measure)
                
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    

    all_problems = []
    
    for channel_name, measures in measures_by_channel.items():
        for source_measure, target_measure in it.combinations(measures, 2):
            source_points = source_measure.kwargs.get("num_points")
            target_points = target_measure.kwargs.get("num_points")
            
            ot_problem = OTProblem(
                name=f"3D_Colored_Mesh_{channel_name}_{source_points}x{target_points}pts",
                source_measure=source_measure,
                target_measure=target_measure,
                C=get_q_const
            )
            ot_problem.kwargs.update({
                "data_type": "3D_Colored_Mesh", 
                "source_points": source_points,
                "target_points": target_points,
                "color_mode": channel_name
            })
            all_problems.append(ot_problem)
    
    return all_problems

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


def run_experiment(suite: 'ExperimentSuite', problemsets_names: list[str], solvers: dict[str, callable]) -> dict[str, 'RunResult']:
    problem_sets = [
        get_problemset(problemset_name) for problemset_name in problemsets_names
    ]

    problems = [problem for problem_set in problem_sets for problem in problem_set]
    results = {}

    with tqdm(total=len(solvers) * len(suite.experiments) * len(problems), desc="Running experiments") as pbar:
        progress_callback = lambda: pbar.update(1)

        for solver_name, solver in solvers.items():
            solver_result = suite.run_suite(name=solver_name, ot_problems=problems,
                                                    progress_callback=progress_callback, solver=solver)
            results[solver_name] = solver_result
    
    return results


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
        self.df = result_df
        self.run_kwargs = run_kwargs

    def display_result(self):
        self.display_header() 
        print(self.df)

    def display_agg(self):
        self.display_header()
        print(get_agg_table(self.df))

    def get_agg(self):
        return get_agg_table(self.df)

    def display_header(self):
        print("Name", self.name)
        for key, value in self.run_kwargs.items():
            print(f"{key}: {value}")
        print('='*100)

    def export(self, filepath: str) -> None:
        self.df.to_csv(filepath)


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
