from abc import ABC, abstractmethod
from collections.abc import Sequence

from uot.data.measure import BaseMeasure
from uot.utils.types import ArrayLike


class BaseSolver(ABC):
    @abstractmethod
    def solve(
        self,
        marginals: Sequence[BaseMeasure],
        costs: Sequence[ArrayLike],
        *args,
        **kwargs
    ) -> dict:
        """
        Solves (multi-)marginal OT problem.
        Returns a dict of results and metrics.
        """
        pass
