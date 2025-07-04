from pathlib import Path
from uot.problems.iterator import ProblemIterator
from uot.problems.store import ProblemStore


def load_problems_from_dir(path: str) -> list[ProblemIterator]:
    path = Path(path)
    iterators = []
    for problemset_path in path.iterdir():
        problems_store = ProblemStore(problemset_path)
        problems_iterator = ProblemIterator(problems_store)
        iterators.append(problems_iterator)
    return iterators
