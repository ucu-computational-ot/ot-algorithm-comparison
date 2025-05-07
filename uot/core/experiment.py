import ot
import gc
import os
import os.path
import numpy as np
import pandas as pd
import itertools as it
import jax.numpy as jnp
import open3d as o3d
from functools import partial
from uot.core.dataset import Measure, generate_coefficients, generate_measures, get_grids, Measure
from tqdm import tqdm


def get_q_const(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    m = y.shape[0]

    return ot.dist(x.reshape((n, -1)), y.reshape((m, -1)))

def get_exact_solution(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, float]:
    T = ot.emd(a, b, C, numItermax=10000000, numThreads=os.cpu_count())
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
        ot_problem.kwargs.update({'dataset': name})
        ot_problems.append(ot_problem)
    return ot_problems

def generate_data_problems(data_type: str, num_points: int, num_samples: int = 10):
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
    data_folder = os.path.join("datasets", "DOTmark_1.0", "Data", data_type)
    
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
        
        x = np.linspace(0, 1, data.shape[1])
        y = np.linspace(0, 1, data.shape[0])
        grid = np.meshgrid(x, y)
        
        measure = Measure(
            name=f"{data_type}_{file_name.replace('.csv', '')}",
            support=grid,
            distribution=data,
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
        ot_problem.kwargs.update({'dataset': f"{data_type} {num_points}x{num_points}"})
        ot_problem.kwargs.update({"data_type": data_type, "num_points": num_points})
        problems.append(ot_problem)
    
    return problems

def generate_3d_mesh_problems(num_points: int = None, num_meshes: int = 10, color_mode: str = "r"):
    """
    Generates OT problems from 3D colored mesh files in the Reference_3D_colored_meshes folder.
    
    Args:
        num_points (int, optional): The number of points to sample from each 3D mesh. 
                                   If None, uses the actual number of vertices in each mesh.
        num_meshes (int): Maximum number of mesh files to use (default: 10)
        color_mode (str): How to handle color channels:
                          - "r", "g", "b": Use a single color channel
                          - "separate": Create separate problems for each color channel
    
    Returns:
        list[OTProblem]: A list of OTProblem objects created from the 3D meshes
    """
    mesh_folder = os.path.join("datasets", "color_meshes")
    
    if not os.path.exists(mesh_folder):
        raise ValueError(f"Mesh folder '{mesh_folder}' does not exist")
    
    mesh_files = [f for f in os.listdir(mesh_folder) if f.endswith('.ply')]
    
    if len(mesh_files) > num_meshes:
        mesh_files = mesh_files[:num_meshes]
    
    if color_mode.lower() == "r":
        channels = [0]
        channel_names = ["red"]
    elif color_mode.lower() == "g":
        channels = [1]
        channel_names = ["green"]
    elif color_mode.lower() == "b":
        channels = [2]
        channel_names = ["blue"]
    elif color_mode.lower() == "separate":
        channels = [0, 1, 2]
        channel_names = ["red", "green", "blue"]
    else:
        raise ValueError(f"Invalid color_mode: {color_mode}. Must be 'r', 'g', 'b', or 'separate'")
    
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
                'dataset': f"3D_Colored_Mesh_{channel_name}_{source_points}x{target_points}pts",
                "data_type": "3D_Colored_Mesh", 
                "source_points": source_points,
                "target_points": target_points,
                "color_mode": channel_name
            })
            all_problems.append(ot_problem)
    
    return all_problems

def get_distribution_problemset(name: str, coeffs=None):
    size, dim_str, distributions_str = name.split(' ')
    distributions = sorted(distributions_str.split('|'))
    dim = int(dim_str[0])

    if dim not in [1, 2, 3]:
        raise ValueError(f"Invalid dimension: {dim}. Expected 1, 2, or 3.")

    size_parts = size.split('x')
    if len(size_parts) != dim:
        raise ValueError(f"Invalid size format: {size}. Expected format: {'x'.join(['<size>'] * dim)}.")
    
    grid_size = int(size_parts[0])
    
    distribution_counts = {distribution: 10 // len(distributions) for distribution in distributions}
    
    if coeffs is None:
        coeffs = generate_coefficients(dim, distribution_counts)
    
    grids = get_grids(dim, [grid_size], start=-10, end=10)
    measures = generate_measures(dim, coeffs, grids)
    
    measure_key = f"{'x'.join([str(grid_size)] * dim)} {dim}D {'|'.join(distributions)}"
    
    try:
        problems = generate_two_fold_problems(None, measures[measure_key], name=measure_key)
        return problems
    except KeyError as e:
        available_keys = list(measures.keys())
        raise KeyError(f"Key '{measure_key}' not found in measures. Available keys: {available_keys}")

def get_problemset(problem_spec, coeffs=None, **kwargs):
    if isinstance(problem_spec, str):
        return get_distribution_problemset(problem_spec, coeffs)
    
    if not isinstance(problem_spec, tuple) or len(problem_spec) < 3:
        raise ValueError("Problem spec must be either a string or a tuple (type, name, num_points[, dims])")
    
    dimensionality, name, num_points = problem_spec[:3]
    dims = problem_spec[3] if len(problem_spec) > 3 else 1
    
    if dimensionality == 1:
        if dims == 1:
            size_str = f"{num_points}"
        elif dims == 2:
            size_str = f"{num_points}x{num_points}"
        elif dims == 3:
            size_str = f"{num_points}x{num_points}x{num_points}"
        else:
            raise ValueError(f"Invalid dimension: {dims}. Expected 1, 2, or 3.")
        
        problem_str = f"{size_str} {dims}D {name}"
        return get_distribution_problemset(problem_str, coeffs)

    elif dimensionality == 2:
        data_type = name
        num_samples = kwargs.get("num_samples", 10)
        return generate_data_problems(data_type=data_type, num_points=num_points, num_samples=num_samples)
    
    elif dimensionality == 3:
        color_mode = name
        num_meshes = kwargs.get("num_meshes", 10)
        return generate_3d_mesh_problems(num_points=num_points, color_mode='r', num_meshes=num_meshes)
   
    else:
        raise ValueError(f"Unknown problem type: {dimensionality}. Expected 'distribution', '3d_mesh', or 'data'")

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


def run_experiment(experiment: 'Experiment',
                   jit_algorithms=None,
                   solvers: dict[str, callable] = None,
                   problemsets_names: list[tuple] = None,
                   folds: int = 1) -> pd.DataFrame:
    """
    Executes a series of experiments using specified solvers on a set of problems.
    Args:
        experiment (Experiment): The experiment object that defines how to run the experiments.
        solvers (dict[str, callable]): A dictionary where keys are solver names and values are tuples 
            containing the solver function and optional sets of keyword arguments for the solver.
        problemsets_names (list[tuple]): A list of problem set names to retrieve and use in the experiments.
        jit_algorithms (list[str], optional): A list of algorithm names to exclude from the results. Defaults to None.
        folds (int, optional): The number of folds for cross-validation. Defaults to 1.
    Returns:
        pd.DataFrame: A DataFrame containing the results of the experiments, including solver names 
        and any additional parameters used.
    """

    problem_sets = [get_problemset(name) for name in problemsets_names]
    problems = [problem for problemset in problem_sets for problem in problemset]

    problems *= folds

    for problem in problems:
        source_distribution = problem.source_measure.distribution
        target_distribution = problem.target_measure.distribution
        
        if np.any(np.logical_or(np.isnan(source_distribution), np.isinf(source_distribution))):
            print(f"Detected problem in: {problem}")
            print("Source distribution contains NaN or Inf")
            print(source_distribution)
            raise ValueError("Nan or Inf in source distributions")
        
        if np.any(np.logical_or(np.isnan(target_distribution), np.isinf(target_distribution))):
            print(f"Detected problem in: {problem}")
            print("Source distribution contains NaN or Inf")
            print(target_distribution)
            raise ValueError("Nan or Inf in target distributions")
    
    dfs = []

    solvers_number = sum(len(kwargs) if kwargs else 1 for _, kwargs in solvers.values())

    with tqdm(total=solvers_number * len(problems), desc="Running experiments") as pbar:
        progress_callback = lambda: pbar.update(1)

        for solver_name, solver in solvers.items():
            
            solver_function, kwargs_sets = solver
            kwargs_sets = kwargs_sets if kwargs_sets else [{}]
            solvers = [(partial(solver_function, **kwargs), kwargs) for kwargs in kwargs_sets]

            for solver, kwargs in solvers:
                pbar.set_description(f"Solver: {solver_name}({kwargs})")
                solver_result = experiment.run_experiment(ot_problems=problems, progress_callback=progress_callback, solver=solver)

                solver_result['name'] = solver_name
                
                for kwarg_name, value in kwargs.items():
                    solver_result[kwarg_name] = value
            
            dfs.append(solver_result)

    df = pd.concat(dfs)

    for dataset in df.dataset.unique():
        for algorithm_name in jit_algorithms:
            algorithm_results = df[(df.dataset == dataset) & (df.name == algorithm_name)]
            if len(algorithm_results):
                df.drop(algorithm_results.index[0], inplace=True)

    return df


class OTProblem:

    def __init__(self, name: str, source_measure: Measure, target_measure: Measure, C: np.ndarray, kwargs: dict = None):
        self.name = name
        self.source_measure = source_measure
        self.target_measure = target_measure

        self._C_cache = None
        self._C = C

        self._exact_cost = None
        self._exact_map = None 

        self.kwargs = kwargs if kwargs is not None else {}
    
    @property
    def C(self):
        if callable(self._C) and self._C_cache is None:
            self._C_cache = self._C(self.source_measure.get_flat_support(),
                               self.target_measure.get_flat_support())
            self._C_cache /= self._C_cache.max()
            return self._C_cache
        return self._C_cache
    
    @property
    def a(self):
        return self.source_measure.to_histogram()[1]
    
    @property
    def b(self):
        return self.target_measure.to_histogram()[1]

    def free_memory(self):
        del self._C_cache
        self._C_cache = None
        self._exact_cost = None
        self._exact_map = None

    def to_jax_arrays(self):
        a = jnp.array(self.a)
        b = jnp.array(self.b)
        C = jnp.array(self.C)

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
        problem_dict = {'source_measure_name': self.source_measure.name, 'target_measure_name': self.target_measure.name}
        source_kwargs = { f"source_{key}": value for key, value in self.source_measure.kwargs.items() }
        target_kwargs = { f"target_{key}": value for key, value in self.target_measure.kwargs.items() }
        problem_dict.update(source_kwargs)
        problem_dict.update(target_kwargs)
        problem_dict.update(self.kwargs)
        return problem_dict


class Experiment:

    def __init__(self, name: str, run_function: callable):
        self.name = name
        self.run_function = run_function

    def run_experiment(self, ot_problems: list[OTProblem], progress_callback: callable = None, solver = None) -> dict:
        results = {}
        for i, ot_problem in enumerate(ot_problems):

            run_function = partial(self.run_function, solver=solver)

            results[ot_problem] = run_function(ot_problem)
            if progress_callback is not None:
                progress_callback()
            ot_problem.free_memory()
            if i % 100 == 0:
                gc.collect()

        df_rows = []
        for ot_problem in ot_problems:
            row_dict = ot_problem.to_dict() 
            row_dict.update(results[ot_problem])
            df_rows.append(row_dict)

        return pd.DataFrame(df_rows)
    
    def run_single(self, ot_problem: OTProblem) -> dict:
        return self.run_function(ot_problem)
