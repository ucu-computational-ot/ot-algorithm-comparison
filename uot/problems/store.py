import pickle
from pathlib import Path
from uot.problems.base_problem import MarginalProblem
import hashlib


class ProblemStore:
    """
    Storage of the MarginalProblem via pickle.
    Problems are hashed for comparison and filename generation.
    Identical problems map to the same filename.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"<ProblemStore path={str(self.path)}>"

    def _key(self, problem: MarginalProblem) -> str:
        blob = pickle.dumps(problem, protocol=4)
        hash = hashlib.sha1(blob).hexdigest()
        return f"{hash}.pkl"

    def exists(self, problem: MarginalProblem) -> bool:
        return (self.path / self._key(problem)).exists()

    def load(self, pth: Path) -> MarginalProblem:
        with open(pth, "rb") as f:
            return pickle.load(f)

    def save(self, problem: MarginalProblem) -> None:
        pth = self._key(problem)
        with open(self.path / pth, "wb") as f:
            pickle.dump(problem, f, protocol=4)

    def all_problems(self) -> list[Path]:
        return sorted(self.path.glob("*.pkl"))
