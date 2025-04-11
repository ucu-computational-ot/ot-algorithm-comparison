import os
import open3d as o3d
import numpy as np
import pandas as pd
import scipy.stats as stats
import inspect

import matplotlib.pyplot as plt
from matplotlib import cm

def compare_measure_kwargs(source: dict, target: dict):
    if source.keys() != target.keys():
        return False  

    for key in source:
        if isinstance(source[key], np.ndarray) and isinstance(target[key], np.ndarray):
            if not np.array_equal(source[key], target[key]):
                return False  
        elif source[key] != target[key]:
            return False

    return True  

class Measure:

    def __init__(self, name, support: list[np.ndarray], distribution: np.ndarray, kwargs=None):
        self.name = name
        self.support = support
        self.distribution = distribution
        self.kwargs = kwargs
    
    def to_histogram(self):
        """
        Converts the support and distribution data into histogram format.
        """
        support = self.get_flat_support()
        distribution = self.distribution.ravel()
        return support, distribution
    
    def get_flat_support(self):
        return np.vstack([coordinate.ravel() for coordinate in self.support]).T
        
    def __str__(self):
        """
        Returns a string representation of the Measure object, displaying its name and kwargs.
        """
        kwargs_str = ", ".join(f"{key}={value}" for key, value in (self.kwargs or {}).items())
        return f"Measure(name='{self.name}', kwargs={{ {kwargs_str} }})".replace('\n', '')

    def __repr__(self):
        return str(self) + '\n'
    
    def __hash__(self):
        points, distribution = self.to_histogram()
        return hash(self.name) & hash(tuple(points.flatten())) & \
               hash(tuple(distribution.flatten())) & hash(tuple(self.kwargs.keys())) & \
               hash(str(self.kwargs.values())) 
    
    def __eq__(self, other):
        if not isinstance(other, Measure):
            return False
        return self.name == other.name and \
               np.array_equal(self.support, other.support) and \
               np.array_equal(self.distribution, other.distribution) and \
               compare_measure_kwargs(self.kwargs, other.kwargs)

    def plot(self):
        if len(self.support) == 1:
            plt.figure(figsize=(8, 4))
            plt.plot(self.support, self.distribution, label=self.name)
            plt.xlabel("Support")
            plt.ylabel("Distribution")
            plt.title(f"1D Distribution: {self.name}")
            plt.legend()
            plt.grid(True)
            plt.show()
        elif len(self.support) == 2:
            print(self.distribution)
            x, y = self.support
            plt.figure(figsize=(8, 6))
            plt.imshow(self.distribution, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis')
            plt.colorbar(label="Density")
            plt.title("2D Gaussian Distribution")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.grid(False)
            plt.show()
        elif len(self.support) == 3:
            points, distribution = self.to_histogram()
            normalized_distribution =  (distribution - np.min(distribution)) / (np.max(distribution) - np.min(distribution))
            colormap = cm.get_cmap('viridis')
            colors = colormap(normalized_distribution)[:, :3]  
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([point_cloud])
        

def generate_gaussian_pdf(x, mean=0, std=1):
    pdf = stats.norm.pdf(x, loc=mean, scale=std)
    return pdf / pdf.sum()


def generate_2d_gaussian_pdf(x, y, mean=(0, 0), cov=((1, 0), (0, 1))):
    """
    Generates a 2D Gaussian PDF on a mesh grid.

    Args:
        x (np.array): X-coordinates of the mesh grid.
        y (np.array): Y-coordinates of the mesh grid.
        mean (tuple): Mean of the Gaussian distribution (default is (0, 0)).
        cov (tuple): Covariance matrix of the Gaussian distribution 
                     (default is identity matrix).

    Returns:
        np.array: 2D Gaussian PDF values on the mesh grid.

    Usage:
        x = np.linspace(-2, 2, 2)
        y = np.linspace(-2, 2, 2)

        x, y = np.meshgrid(x, y)

        gaussian = generate_2d_gaussian_pdf(x, y)
    """
    pos = np.stack((x, y), axis=-1)
    rv = stats.multivariate_normal(mean=mean, cov=cov)
    pdf = rv.pdf(pos)
    return pdf / pdf.sum()


def generate_3d_gaussian_pdf(x, y, z, mean=(0, 0, 0), cov=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
    """
    Generates a 3D Gaussian PDF on a mesh grid.

    Args:
        x (np.array): X-coordinates of the mesh grid.
        y (np.array): Y-coordinates of the mesh grid.
        z (np.array): Z-coordinates of the mesh grid.
        mean (tuple): Mean of the Gaussian distribution (default is (0, 0, 0)).
        cov (tuple): Covariance matrix of the Gaussian distribution 
                     (default is identity matrix).

    Returns:
        np.array: 3D Gaussian PDF values on the mesh grid.

    Usage:
        x = np.linspace(-2, 2, 2)
        y = np.linspace(-2, 2, 2)
        z = np.linspace(-2, 2, 2)

        x, y, z = np.meshgrid(x, y, z)

        gaussian = generate_3d_gaussian_pdf(x, y, z)
    """
    pos = np.stack((x, y, z), axis=-1)
    rv = stats.multivariate_normal(mean=mean, cov=cov)
    pdf = rv.pdf(pos)
    return pdf / pdf.sum()


def generate_1d_gaussians_ds(x: np.ndarray):
    means = np.arange(-6, 6)
    std = np.linspace(0.1, 2, 10)
    return [Measure(name="1D Gaussian", support=[x], distribution=generate_gaussian_pdf(x, mean=mean, std=std),
                    kwargs={'mean': mean, 'std': std})  for mean, std in zip(means, std)]


def generate_2d_gaussians_ds(x, y):
    means = [
        [-4, -4],
        [-3, -3],
        [-2, -2],
        [-1, -1],
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5]
    ]

    covariances = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0.5, 0], [0, 0.5]]),
        np.array([[2, 0], [0, 2]]),
        np.array([[1, 0.5], [0.5, 1]]),
        np.array([[1, -0.5], [-0.5, 1]]),
        np.array([[1.5, 0.3], [0.3, 1.5]]),
        np.array([[0.8, -0.2], [-0.2, 0.8]]),
        np.array([[1.2, 0.6], [0.6, 1.2]]),
        np.array([[0.6, -0.4], [-0.4, 0.6]]),
        np.array([[1.8, 0.2], [0.2, 1.8]])
    ]
    return [Measure(name="2D Gaussian", support=[x, y], distribution=generate_2d_gaussian_pdf(x, y, mean=mean, cov=covariances),
                    kwargs={'mean': mean, 'cov': covariances})  for mean, covariances in zip(means, covariances)]


def generate_3d_gaussians_ds(x, y, z):
    means = [
        [-4, -4, -4],
        [-3, -3, -3],
        [-2, -2, -2],
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5]
    ]

    covariances = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]),
        np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]]),
        np.array([[1, -0.5, 0], [-0.5, 1, 0], [0, 0, 1]]),
        np.array([[1.5, 0.3, 0], [0.3, 1.5, 0], [0, 0, 1.5]]),
        np.array([[0.8, -0.2, 0], [-0.2, 0.8, 0], [0, 0, 0.8]]),
        np.array([[1.2, 0.6, 0], [0.6, 1.2, 0], [0, 0, 1.2]]),
        np.array([[0.6, -0.4, 0], [-0.4, 0.6, 0], [0, 0, 0.6]]),
        np.array([[1.8, 0.2, 0], [0.2, 1.8, 0], [0, 0, 1.8]])
    ]

    return [Measure(name="3D Gaussian", support=[x, y, z], distribution=generate_3d_gaussian_pdf(x, y, z, mean=mean, cov=covariances),
                    kwargs={'mean': mean, 'cov': covariances})  for mean, covariances in zip(means, covariances)]


def generate_gamma_pdf(x, shape=1, scale=1):
    return stats.gamma.pdf(x, a=shape, scale=scale)

def generate_beta_pdf(x, alpha=1, beta=1):
    return stats.beta.pdf(x, a=alpha, b=beta)

def generate_uniform_pdf(x, lower=0, upper=1):
    return stats.uniform.pdf(x, loc=lower, scale=upper - lower)

def generate_cauchy_pdf(x, loc=0, scale=1):
    return stats.cauchy.pdf(x, loc=loc, scale=scale)

def generate_normalized_white_noise(size, mean=0, std=1):
    noise = np.random.normal(loc=mean, scale=std, size=size)
    return (noise - np.mean(noise)) / np.std(noise)



def get_ts_dataset() -> list[np.array]:
    """
    Reads a time series dataset from a CSV file, processes it by region, and returns 
    a list of normalized time series data.

    Returns:
        list: A list of numpy arrays, where each array represents the normalized 
              time series data for a specific region.
    """
    df = pd.read_csv('datasets/ts_dataset.csv')
    regions = df.region.unique()
    time_series = []
    for region in regions:
        ts = df[df.region == region].sort_values(by="year")['tincidence']
        ts /= ts.sum()
        time_series.append(ts.to_numpy())
    return time_series


def load_2d_data(label: str, resolution: int):
    """
    Loads 2D data from the specified dataset based on the label and resolution.

    Args:
        label (str): The label identifying the dataset to load. 
                     Must be one of the predefined keys in `labels_to_folders`.
        resolution (int): The resolution of the data to filter files by.

    Returns:
        list: A list of 2D numpy arrays, each representing a dataset loaded from the files.
    """
    data_folder = os.path.join("datasets/DOTmark_1.0/Data", label)
    filepaths = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if str(resolution) in file]
    datasets = [pd.read_csv(filepath, header=None, index_col=None).to_numpy() for filepath in filepaths] 
    datasets = [dataset / dataset.sum() for dataset in datasets]
    return datasets
 

def generate_3d_gaussian_pdf(x, y, z, mean=(0, 0, 0), cov=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
    """
    Generates a 3D Gaussian PDF on a mesh grid.

    Args:
        x (np.array): X-coordinates of the mesh grid.
        y (np.array): Y-coordinates of the mesh grid.
        z (np.array): Z-coordinates of the mesh grid.
        mean (tuple): Mean of the Gaussian distribution (default is (0, 0, 0)).
        cov (tuple): Covariance matrix of the Gaussian distribution 
                     (default is identity matrix).

    Returns:
        np.array: 3D Gaussian PDF values on the mesh grid.

    Usage:
        x = np.linspace(-2, 2, 2)
        y = np.linspace(-2, 2, 2)
        z = np.linspace(-2, 2, 2)

        x, y, z = np.meshgrid(x, y, z)

        gaussian = generate_3d_gaussian_pdf(x, y, z)
    """
    pos = np.stack((x, y, z), axis=-1)
    rv = stats.multivariate_normal(mean=mean, cov=cov)
    return rv.pdf(pos)


def generate_normalized_white_noise_3d(x, y, z, mean=0, std=1):
    """
    Generates normalized white noise on a 3D mesh grid.

    Args:
        x (np.array): X-coordinates of the mesh grid.
        y (np.array): Y-coordinates of the mesh grid.
        z (np.array): Z-coordinates of the mesh grid.
        mean (float): Mean of the white noise distribution (default is 0).
        std (float): Standard deviation of the white noise distribution (default is 1).

    Returns:
        np.array: Normalized white noise values on the 3D mesh grid.
    """
    shape = x.shape
    noise = np.random.normal(loc=mean, scale=std, size=shape)
    return (noise - np.mean(noise)) / np.std(noise)


def load_3d_data(label: str, color: int, n: int):
    """
    Loads 3D data from a specified file, samples points from a 3D mesh, and computes a color-based distribution.

    Args:
        label (str): A label for the data (not used in the function).
        color (int): The index of the color channel to use (e.g., 0 for red, 1 for green, 2 for blue).
        n (int): The number of points to sample from the 3D mesh.

    Returns:
        tuple: A tuple containing:
            - points (numpy.ndarray): An array of sampled 3D points.
            - distribution (numpy.ndarray): A normalized distribution based on the specified color channel.
    """
    file_path = f"datasets/color_meshes/{label}.ply"
    mesh = o3d.io.read_triangle_mesh(file_path)
    sampled_points = mesh.sample_points_uniformly(n)

    points = np.asarray(sampled_points.points)
    colors = np.asarray(sampled_points.colors)

    distribution = colors[:, color] / np.sum(colors[:, color])

    return points, distribution

def generate_ds(dim: int, distributions: list[str], grid: list[np.ndarray], number: int, **kwargs):
    """
    Generates a dataset for 1D, 2D, or 3D distributions with multiple distribution types.

    Args:
        dim (int): Dimensionality of the dataset (1, 2, or 3).
        distributions (list[str]): List of distribution types (e.g., ['gaussian', 'gamma', 'beta']).
        grid (list[np.ndarray]): List of mesh grid arrays for the dimensions.
        number (int): Total number of measures to generate.
        **kwargs: Additional parameters for the distributions.

    Returns:
        list[Measure]: A list of Measure objects with randomly generated characteristics.
    """
    if dim not in [1, 2, 3]:
        raise ValueError("dim must be 1, 2, or 3.")

    if len(grid) != dim:
        raise ValueError(f"Expected {dim} grid arrays, but got {len(grid)}.")
    
    basic_ranges = {
        'mean_range': (-5, 5),
        'std_range': (0.5, 2),
        'shape_range': (1, 5),
        'scale_range': (0.1, 2),
        'loc_range': (-5, 5),
        'scale_range': (0.1, 2),
        'alpha_range': (0.1, 5),
        'beta_range': (0.1, 5),
    }

    distribution_map = {
        'gaussian': {
            1: generate_gaussian_pdf,
            2: generate_2d_gaussian_pdf,
            3: generate_3d_gaussian_pdf
        },
        'gamma': {
            1: generate_gamma_pdf,
        },
        'beta': {
            1: generate_beta_pdf,
        },
        'uniform': {
            1: generate_uniform_pdf,
        },
        'cauchy': {
            1: generate_cauchy_pdf,
        }
    }

    for distribution in distributions:
        if distribution not in distribution_map:
            raise ValueError(f"Unsupported distribution: {distribution}.")
        if distribution_map[distribution].get(dim) is None:
            raise ValueError(f"Unsupported distribution for {dim}D: {distribution}.")

    measures = []
    num_distributions = len(distributions)

    for i in range(number):
        distribution = distributions[i % num_distributions]

        vars_data = inspect.signature(distribution_map[distribution][dim]).parameters
        vars = list(vars_data.keys() - {'x', 'y', 'z'})

        inputs = {}

        if dim == 1:

            for var in vars:
                if f'{var}_range' in kwargs:
                    inputs[var] = np.random.uniform(*kwargs[f'{var}_range'])

                elif f'{var}_range' in basic_ranges:
                    inputs[var] = np.random.uniform(*basic_ranges[f'{var}_range'])
                
                elif '=' not in str(vars_data[var]):
                    raise ValueError(f"Missing range for {var}.")

            pdf = distribution_map[distribution][dim](grid[0], **inputs)
            measures.append(Measure(name=f"{dim}D {distribution}", support=[grid[0]], distribution=pdf, kwargs=inputs))
        
        else:
            for var in vars:

                if var == 'cov':
                    if 'cov_range' in kwargs:
                        inputs[var] = generate_random_covariance(dim, kwargs['cov_range'])
                    else:
                        inputs[var] = generate_random_covariance(dim)
                    
                    continue

                if f'{var}_range' in kwargs:
                    inputs[var] = tuple(np.random.uniform(*kwargs[f'{var}_range']) for _ in range(dim))

                elif f'{var}_range' in basic_ranges:
                    inputs[var] = tuple(np.random.uniform(*basic_ranges[f'{var}_range']) for _ in range(dim))
                
                elif '=' not in str(vars_data[var]):
                    raise ValueError(f"Missing range for {var}.")

            pdf = distribution_map[distribution][dim](*grid, **inputs)
            measures.append(Measure(name=f"{dim}D {distribution}", support=grid, distribution=pdf, kwargs=inputs))
    
    return measures


def generate_random_covariance(dim: int, cov_range: tuple = (0.5, 2), off_diag_range: tuple = (-0.5, 0.5)):
    """
    Generates a random symmetric positive definite covariance matrix.

    Args:
        dim (int): Dimensionality of the covariance matrix (2 or 3).
        cov_range (tuple): Range for the diagonal elements (variances).

    Returns:
        np.ndarray: A random symmetric positive definite covariance matrix.
    """
    if dim not in [2, 3]:
        raise ValueError("Covariance matrix generation is only supported for 2D or 3D.")

    diag = np.random.uniform(*cov_range, size=dim)

    off_diag = np.random.uniform(*off_diag_range, size=(dim, dim))
    off_diag = (off_diag + off_diag.T) / 2
    np.fill_diagonal(off_diag, 0)

    cov = np.diag(diag) + off_diag

    min_eigenvalue = np.min(np.linalg.eigvalsh(cov))
    if min_eigenvalue <= 0:
        cov += np.eye(dim) * (abs(min_eigenvalue) + 1e-6)

    return cov
