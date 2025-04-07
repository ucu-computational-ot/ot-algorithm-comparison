import ot
import numpy as np
import pandas as pd
import itertools as it
from uot.dataset import Measure

def get_q_const(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    m = y.shape[0]

    return ot.dist(x.reshape((n, -1)), y.reshape((m, -1)))

def get_exact_solution(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, float]:
    T = ot.emd(a, b, C)
    return T, np.sum(T * C)

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
        self.kwargs = kwargs if kwargs is not None else {}
 
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
    

class Experiment:

    def __init__(self, name: str, run_function: callable):
        self.name = name
        self.run_function = run_function

    def run_experiment(self, ot_problems: list[OTProblem]) -> dict:
        results = {}
        for ot_problem in ot_problems:
            results[ot_problem] = self.run_function(ot_problem)
        
        return results
    
class ExperimentSuite:

    def __init__(self, experiments: list[Experiment]):
        self.experiments = experiments

    def run_suite(self, ot_problems: list[OTProblem]) -> pd.DataFrame:
        results = []
        for experiment in self.experiments:
            results.append(experiment.run_experiment(ot_problems))

        df_rows = []
        for ot_problem in ot_problems:
            row_dict = ot_problem.to_dict() 
            for result in results:
                row_dict.update(result[ot_problem])
            df_rows.append(row_dict)

        return pd.DataFrame(df_rows)
             
