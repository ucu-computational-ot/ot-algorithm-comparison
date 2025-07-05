from collections.abc import Iterator
from uot.problems.store import ProblemStore
from uot.problems.base_problem import MarginalProblem


class ProblemIterator(Iterator[MarginalProblem]):
    """
    Lazily iterates over pickled problems in a ProblemStore.
    Only one problem is loaded into memory at a time.
    """

    def __init__(self, store: ProblemStore):
        self._problems = store.all_problems()
        self._idx = 0
        self._store = store

    def __iter__(self) -> "ProblemIterator":
        return self

    def __len__(self) -> int:
        return len(self._problems)

    def __next__(self) -> MarginalProblem:
        if self._idx >= len(self._problems):
            raise StopIteration
        fn = self._problems[self._idx]
        self._idx += 1
        return self._store.load(fn)
