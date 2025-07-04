from abc import ABC, abstractmethod
from uot.problems.base_problem import MarginalProblem
from collections.abc import Iterator


class ProblemGenerator(ABC):

    @abstractmethod
    def generate(self, *args, **kwargs) -> Iterator[MarginalProblem]:
        "Return a list of MarginalProblem objects."
        pass
