import open3d as o3d
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import product

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
            plt.plot(self.support[0], self.distribution, label=self.name)
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
    pdf_sum = pdf.sum()
    if pdf_sum > 1e-10:
        return pdf / pdf_sum
    else:
        return np.ones_like(pdf) / len(pdf)


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


def generate_gamma_pdf(x, shape=1, scale=1):
    pdf = stats.gamma.pdf(x, a=shape, scale=scale)
    return pdf / pdf.sum()
    

def generate_beta_pdf(x, alpha=1, beta=1):
    pdf = stats.beta.pdf(x, a=alpha, b=beta)
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0, neginf=0)
    return pdf / pdf.sum()

def generate_uniform_pdf(x, lower=0, width=1):
    pdf = stats.uniform.pdf(x, loc=lower, scale=width)
    return pdf / pdf.sum()

def generate_cauchy_pdf(x, loc=0, scale=1):
    pdf = stats.cauchy.pdf(x, loc=loc, scale=scale)
    return pdf / pdf.sum()

def generate_normalized_white_noise(x, mean=0, std=1):
    size = x.shape
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

def generate_normalized_white_noise_2d(x, y, mean=0, std=1):
    """
    Generates normalized white noise on a 2D mesh grid.

    Args:
        x (np.array): X-coordinates of the mesh grid.
        y (np.array): Y-coordinates of the mesh grid.
        mean (float): Mean of the white noise distribution (default is 0).
        std (float): Standard deviation of the white noise distribution (default is 1).

    Returns:
        np.array: Normalized white noise values on the 2D mesh grid.
    """
    shape = x.shape
    mean = mean[0]
    std = std[0]
    noise = np.random.normal(loc=mean, scale=std, size=shape)
    return (noise - np.mean(noise)) / np.std(noise)


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
    mean = mean[0]
    std = std[0]
    noise = np.random.normal(loc=mean, scale=std, size=shape)
    return (noise - np.mean(noise)) / np.std(noise)

def generate_coefficients(dim: int, distributions: dict[str, int]):
    if dim not in [1, 2, 3]:
        raise ValueError("dim must be 1, 2, or 3.")

    basic_ranges = {
        'mean_range': (-5, 5),
        'std_range': (0.5, 4.0),
        'shape_range': (1, 5),
        'scale_range': (0.2, 2.0),
        'loc_range': (-5, 5),
        'alpha_range': (0.5, 8),
        'beta_range': (0.5, 8),
        'width_range': (2, 8),
        'lower_range': (-8, 5),
    }

    distribution_parameters = {
        'gaussian': ('mean', 'std'),
        'gamma': ('shape', 'scale'),
        'beta': ('alpha', 'beta'),
        'uniform': ('lower', 'width'),
        'cauchy': ('loc', 'scale'),
        'white-noise': ('mean', 'std'),
    }

    results = {}

    for distribution in distributions:
        if distribution not in distribution_parameters:
            raise ValueError(f"Unsupported distribution: {distribution}")

        num_to_generate = distributions[distribution]
        param_names = distribution_parameters[distribution]
        param_ranges = []

        for param in param_names:
            if f"{param}_range" not in basic_ranges:
                raise ValueError(f"Missing range for {param}.")
            
            num_points = max(10, num_to_generate * 2)
            values = np.linspace(*basic_ranges[f"{param}_range"], num_points)
            
            if dim == 1:
                param_ranges.append(values)
            else:
                param_ranges.append(list(product(values, repeat=dim)))

        if distribution == 'gaussian' and dim > 1:
            mean_values = np.linspace(*basic_ranges['mean_range'], max(10, num_to_generate))
            mean_choices = list(product(mean_values, repeat=dim))
            
            step = max(1, len(mean_choices) // num_to_generate)
            selected_means = mean_choices[::step][:num_to_generate]
            
            result = []
            
            diag_values = np.linspace(0.5, 3.0, 10)
            offdiag_values = np.linspace(-0.3, 0.3, 7)
            
            for i, mean in enumerate(selected_means):
                diag_indices = [(i + j) % len(diag_values) for j in range(dim)]
                offdiag_idx = i % len(offdiag_values)
                
                diag = np.array([diag_values[idx] for idx in diag_indices])
                cov = np.diag(diag)
                
                indices = np.triu_indices(dim, k=1)
                for idx in range(len(indices[0])):
                    row, col = indices[0][idx], indices[1][idx]
                    val = offdiag_values[offdiag_idx]
                    cov[row, col] = cov[col, row] = val
                
                min_eig = np.min(np.linalg.eigvalsh(cov))
                if min_eig <= 0:
                    cov += np.eye(dim) * (abs(min_eig) + 1e-6)
                
                result.append((tuple(np.round(mean, 2)), np.round(cov, 2)))
            
            results[distribution] = result
        else:
            all_combinations = list(product(*param_ranges))
            
            step = max(1, len(all_combinations) // num_to_generate)
            selected_combinations = all_combinations[::step][:num_to_generate]

            rounded_combinations = [
                tuple(np.round(combination, 2)) for combination in selected_combinations
            ]
            results[distribution] = rounded_combinations

    return results


def generate_grid(dim: int, grid_size: int, start: int = -10, end: int = 10):
    """
    Generates a mesh grid for the specified dimensionality and size.

    Args:
        dim (int): Dimensionality of the grid (1, 2, or 3).
        grid_size (int): Number of points along each dimension.
        start (int): Starting point of the grid (default is -5).
        end (int): Ending point of the grid (default is 5).

    Returns:
        list[np.ndarray]: A list of mesh grid arrays for the specified dimensions.
    """
    if dim not in [1, 2, 3]:
        raise ValueError("dim must be 1, 2, or 3.")

    if dim == 1:
        return [np.linspace(start, end, grid_size)]

    elif dim == 2:
        x = np.linspace(start, end, grid_size)
        y = np.linspace(start, end, grid_size)
        return np.meshgrid(x, y)

    elif dim == 3:
        x = np.linspace(start, end, grid_size)
        y = np.linspace(start, end, grid_size)
        z = np.linspace(start, end, grid_size)
        return np.meshgrid(x, y, z)


def get_grids(dim: list[int], grid_sizes: list[int], start: int = -10, end: int = 10):
    '''
    Generates a list of mesh grids for the specified dimensions and sizes.
    '''
    grids = {}
    for grid_size in grid_sizes:
        grids[f"{'x'.join([str(grid_size)] * dim)} {dim}D"] = generate_grid(dim, grid_size, start, end)
    
    return grids

def generate_measures(dim: int, coefficients: dict[str, list[tuple]], grids: list[np.ndarray]):
    """
    Generates measures based on the provided coefficients and grid.

    Args:
        dim (int): Dimensionality of the measures (1, 2, or 3).
        coefficients (dict): Coefficients for the distributions.
        grid (list[np.ndarray]): List of mesh grid arrays for the dimensions.

    Returns:
        list[Measure]: A list of Measure objects with randomly generated characteristics.
    """
    if dim not in [1, 2, 3]:
        raise ValueError("dim must be 1, 2, or 3.")
    
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
        },
        'white-noise': {
            1: generate_normalized_white_noise,
            2: generate_normalized_white_noise_2d,
            3: generate_normalized_white_noise_3d
        }
    }

    output = {}

    for name, grid in grids.items():

        entry = f"{name} {'|'.join(sorted(param for param in coefficients))}"
        output[entry] = []

        for distribution in coefficients:

            func = distribution_map[distribution].get(dim)
            if func is None:
                raise ValueError(f"Unsupported distribution for {dim}D: {distribution}.")

            for params in coefficients[distribution]:
                pdf = func(*grid, *params)
                mes = Measure(name=f"{name} {distribution}", support=grid, distribution=pdf, kwargs={distribution: params})
                output[entry].append(mes)
        
    return output

if __name__ == "__main__":
    coeffs = generate_coefficients(1, {'gaussian': 1, 'gamma': 1, 'beta': 1, 'uniform': 1, 'cauchy': 1, 'white-noise': 1})
    grids = get_grids(1, [64])
    measures = generate_measures(1, coeffs, grids)
    for key, value in measures.items():
        for measure in value:
            print(measure)
            measure.plot()