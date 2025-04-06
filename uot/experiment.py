import ot
import numpy as np


def get_q_const(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    m = y.shape[0]

    return ot.dist(x.reshape((n, -1)), y.reshape((m, -1)))

def get_exact_solution(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, float]:
    T = ot.emd(a, b, C)
    return T, np.sum(T * C)

class OTProblem:

    def __init__(self, name: str, a: np.ndarray, b: np.ndarray, C: np.ndarray, kwargs: dict = None):
        self.name = name
        self.a = a
        self.b = b
        self.C = C
        self.kwargs = kwargs
 
    def __hash__(self):
        return hash(tuple(self.a) + tuple(self.b))

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
