import ot
import itertools as it
import numpy as np
from uot.dataset import Measure

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

def generate_two_fold_problems(grid, measures: list[Measure]):
    ot_problems = []
    source_points, _ = measures[0].to_histogram()
    target_points, _ = measures[1].to_histogram()
    C = get_q_const(source_points, target_points)
    for source_measure, target_measure in it.combinations(measures, 2):
        ot_problems.append(OTProblem(name="Simple transport", 
                                     source_measure=source_measure,
                                     target_measure=target_measure,
                                     C=C))
    return ot_problems

class OTProblem:

    def __init__(self, name: str, source_measure: Measure, target_measure: Measure, C: np.ndarray, kwargs: dict = None):
        self.name = name
        self.source_measure = source_measure
        self.target_measure = target_measure
        self.C = C
        self.kwargs = kwargs
 
    def __hash__(self):
        return hash(tuple(self.a) + tuple(self.b) + tuple(self.C.flatten()))
    
    def __str__(self):
        return f"<OTProblem: {self.name} source={self.source_measure}, target={self.target_measure}>"

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
