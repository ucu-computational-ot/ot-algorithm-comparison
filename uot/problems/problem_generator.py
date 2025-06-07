from abc import ABC, abstractmethod
from typing import List


class ProblemGenerator(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs) -> List:
        "Return a list of MarginalProblem objects."
        pass
