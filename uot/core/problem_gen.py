import os
import ot
import random
import open3d as o3d
import itertools as it
import numpy as np
import pandas as pd
from uot.core.dataset import Measure, generate_coefficients, generate_measures, get_grids
import jax.numpy as jnp



def get_exact_solution(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, float]:
    T = ot.emd(a, b, C, numItermax=10000000, numThreads=os.cpu_count())
    return T, np.sum(T * C)


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


def get_q_const(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    m = y.shape[0]

    return ot.dist(x.reshape((n, -1)), y.reshape((m, -1)))


def generate_data_problems(data_type: str, num_points: int, num_samples: int = 10):
    """
    Generates OT problems from data files in the Data folder.
    
    Args:
        data_type (str): The type of data to use (e.g., 'CauchyDensity', 'ClassicImages')
        num_points (int): The number of points/resolution (e.g., 32, 64, 128, 256, 512)
        num_samples (int): Maximum number of samples to use (default: 10)
    
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

    color_map = {
        "r": ([0], ["red"]),
        "g": ([1], ["green"]),
        "b": ([2], ["blue"]),
        "separate": ([0, 1, 2], ["red", "green", "blue"]),
    }
        
    try:
        channels, channel_names = color_map[color_mode]
    except KeyError:
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


def get_distribution_problemset(name: str, number = 10):

    seed_value = hash(name) % 2 ** 32
    
    random.seed(seed_value)
    np.random.seed(seed_value)

    size, dim_str, distributions_str = name.split(' ')
    distributions = sorted(distributions_str.split('|'))
    distributions_str = '|'.join(distributions)
    dim = int(dim_str[0])
    name = f"{size} {dim_str} {distributions_str}"

    if dim not in [1, 2, 3]:
        raise ValueError(f"Invalid dimension: {dim}. Expected 1, 2, or 3.")

    size_parts = size.split('x')
    if len(size_parts) != dim:
        raise ValueError(f"Invalid size format: {size}. Expected format: {'x'.join(['<size>'] * dim)}.")
    
    grid_size = int(size_parts[0])
    
    ot_problems = []
    C = get_q_const


    if len(distributions) == 1:
        distributions.append(distributions[0])
    
    distributions = it.cycle(it.combinations(distributions, 2))

    for _ in range(number):
        source_measure, target_measure = next(distributions)
        if source_measure != target_measure:
            distr_counts = {source_measure: 1, target_measure: 1}
        else:
            distr_counts = {source_measure: 2}
        coeffs = generate_coefficients(dim, distr_counts)
        grids = get_grids(dim, [grid_size])

        try: 
            measures = list(generate_measures(dim, coeffs, grids).values())[0]
        except KeyError:
            raise ValueError(f"Invalid measure name: {name}. Available measures: {list(generate_measures(dim, coeffs, grids).keys())}")
        
        ot_problem = OTProblem(name="Simple transport", 
                                     source_measure=measures[0],
                                     target_measure=measures[1],
                                     C=C)
        ot_problem.kwargs.update({'dataset': name})
        ot_problems.append(ot_problem)
    return ot_problems


def get_problemset(problem_spec, **kwargs):
    number = kwargs.get("number", 10)

    if isinstance(problem_spec, str):
        return get_distribution_problemset(problem_spec, number)
    
    if not isinstance(problem_spec, tuple) or len(problem_spec) < 3:
        raise ValueError("Problem spec must be either a string or a tuple (type, name, num_points[, dims])")
    
    problem_type, name, num_points = problem_spec[:3]
    dims = problem_spec[3] if len(problem_spec) > 3 else 1
    
    if problem_type == 'distribution':
        if dims in [1, 2, 3]:
            size_str = "x".join([str(num_points)] * dims)
        else:
            raise ValueError(f"Invalid dimension: {dims}. Expected 1, 2, or 3.")
        
        problem_str = f"{size_str} {dims}D {name}"
        return get_distribution_problemset(problem_str, number=number)

    elif problem_type == 'data':
        data_type = name
        num_samples = kwargs.get("num_samples", 10)
        return generate_data_problems(data_type=data_type, num_points=num_points, num_samples=num_samples)
    
    elif problem_type == '3d_mesh':
        color_mode = name if name in ["r", "g", "b", "separate"] else "r"
        num_meshes = kwargs.get("num_meshes", 10)
        return generate_3d_mesh_problems(num_points=num_points, color_mode=color_mode, num_meshes=num_meshes)
   
    else:
        raise ValueError(f"Unknown problem type: {problem_type}. Expected 'distribution', '3d_mesh', or 'data'")
