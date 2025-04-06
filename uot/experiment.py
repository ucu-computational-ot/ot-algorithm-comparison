import ot
import itertools as it
import numpy as np


def get_q_const(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    m = y.shape[0]

    return ot.dist(x.reshape((n, -1)), y.reshape((m, -1)))

def get_exact_solution(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, float]:
    T = ot.emd(a, b, C)
    return T, np.sum(T * C)

def get_histograms(grid, source, target):
    points = np.vstack([coordinate.ravel() for coordinate in grid]).T
    source_hisgogram = source.ravel()
    target_histogram = target.ravel()
    return points, source_hisgogram, target_histogram

def generate_two_fold_problems(grid, distributions):
    ot_problems = []
    points = np.vstack([coordinate.ravel() for coordinate in grid]).T
    cost = get_q_const(points, points)

    for a, b in it.combinations(distributions, 2):
        source_histogram, target_hisgogram = a.ravel(), b.ravel()
        ot_problems.append(OTProblem(name="Simple transport", a=source_histogram, b=target_hisgogram, C=cost))

    return ot_problems

class OTProblem:

    def __init__(self, name: str, a: np.ndarray, b: np.ndarray, C: np.ndarray, kwargs: dict = None):
        self.name = name
        self.a = a
        self.b = b
        self.C = C
        self.kwargs = kwargs
 
    def __hash__(self):
        return hash(tuple(self.a) + tuple(self.b) + tuple(self.C.flatten()))
    
    def __str__(self):
        return f"<OTProblem: {self.name}>"

class Experiment:

    def __init__(self, name: str, ot_problems: list[OTProblem], run_function: callable):
        self.name = name
        self.problems = ot_problems
        self.run_function = run_function

    def run_experiment(self) -> dict:
        results = {}
        for ot_problem in self.problems:
            results[ot_problem] = self.run_function(ot_problem)
        
        return results
