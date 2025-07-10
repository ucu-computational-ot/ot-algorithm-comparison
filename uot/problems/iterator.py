from collections.abc import Iterator
from uot.problems.store import ProblemStore
from uot.problems.base_problem import MarginalProblem
from uot.problems.problem_generator import ProblemGenerator
from uot.utils.logging import setup_logger

logger = setup_logger(__name__)


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


class OnlineProblemIterator(Iterator[MarginalProblem]):
    def __init__(
            self,
            generator: ProblemGenerator,
            num: int,
            cache_gt: bool = False
    ):
        self._num = num
        self._generator = generator
        self._idx = 0
        self._cache_gt = cache_gt
        self._gen_iterator = self._generator.generate()

    def __iter__(self) -> "OnlineProblemIterator":
        return self

    def __len__(self) -> int:
        return self._num

    def __next__(self) -> MarginalProblem:
        if self._idx >= self._num:
            raise StopIteration
        self._idx += 1
        prob = next(self._gen_iterator)
        if self._cache_gt:
            prob.get_costs()
            prob.get_exact_cost()
        logger.info(f"Generated problem {prob}")
        return prob

    def __getstate__(self):
        # Called by pickle.dumps(self)
        state = self.__dict__.copy()
        # remove the actual generator; it's not pickleable
        state.pop('_gen_iterator', None)
        return state

    def __setstate__(self, state):
        # Called by pickle.loads(...)
        self.__dict__.update(state)
        # recreate the iterator fresh
        self._gen_iterator = self._generator.generate()
        # advance it to where we were
        for _ in range(self._idx):
            next(self._iterator)
