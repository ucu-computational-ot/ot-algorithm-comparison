
from pathlib import Path
from typing import Iterator
from uot.problems.store import ProblemStore
from uot.problems.base_problem import MarginalProblem


class ProblemIterator(Iterator[MarginalProblem]):
    """
    Lazily iterates over pickled problems in a ProblemStore.
    Only one problem is loaded into memory at a time.
    """

    def __init__(self, store: ProblemStore):
        self._files = store.all_files()
        self._idx = 0
        self._store = store

    def __iter__(self) -> "ProblemIterator":
        return self

    def __len__(self) -> int:
        return len(self._files)

    def __next__(self) -> MarginalProblem:
        if self._idx >= len(self._files):
            raise StopIteration
        fn = self._files[self._idx]
        self._idx += 1
        return self._store.load(fn)
