from uot.problems.base_problem import MarginalProblem


class MultiMarginalProblem(MarginalProblem):
    def __init__(self, name, measures, cost_fns):
        super().__init__(name, measures, cost_fns)
