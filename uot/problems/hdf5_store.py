"""HDF5-backed storage for :class:`MarginalProblem` objects."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from jax import numpy as jnp

from uot.data.measure import DiscreteMeasure
from uot.problems.base_problem import MarginalProblem
from uot.problems.two_marginal import TwoMarginalProblem
from uot.utils.import_helpers import import_object


class HDF5ProblemStore:
    """Store :class:`MarginalProblem` objects in a structured HDF5 file."""

    ROOT = "problems"

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if self.path.suffix.lower() != ".h5":
            self.path = self.path.with_suffix(".h5")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._file = h5py.File(self.path, mode="a")
        self._file.require_group(self.ROOT)

    def __repr__(self) -> str:
        return f"<HDF5ProblemStore path={self.path}>"

    # ------------------------------------------------------------------
    def _meta_for(self, problem: MarginalProblem) -> dict:
        meta = problem.to_dict()
        meta["problem_class"] = (
            problem.__class__.__module__ + "." + problem.__class__.__name__
        )
        # assume single cost function for now
        # TODO: handle pairwise cost function (multiple costs ideally)
        if problem.cost_fns:
            fn = problem.cost_fns[0]
            meta["cost_fn"] = fn.__module__ + "." + fn.__name__
        return meta

    def _key(self, problem: MarginalProblem) -> str:
        return problem.key()

    # ------------------------------------------------------------------
    def exists(self, problem: MarginalProblem) -> bool:
        key = self._key(problem)
        return key in self._file[self.ROOT]

    def all_problems(self) -> list[str]:
        return sorted(list(self._file[self.ROOT].keys()))

    # ------------------------------------------------------------------
    def save(self, problem: MarginalProblem) -> None:
        """Serialize ``problem`` into the HDF5 file."""

        key = self._key(problem)
        base = self._file[self.ROOT].require_group(key)

        meta = self._meta_for(problem)
        base.attrs.clear()
        for k, v in meta.items():
            base.attrs[k] = v
        base.attrs["marginals_count"] = len(problem.measures)

        # TODO: later as we introduce multi-marginal problem or
        #       barycenter problems it would be better to use factory

        # ------------------ marginals ------------------
        marg_group = base.require_group("marginals")
        for child in list(marg_group.keys()):
            del marg_group[child]

        for i, m in enumerate(problem.measures):
            pts, wts = m.to_discrete()
            chunks = (min(pts.shape[0], 1024), pts.shape[1])
            grp = marg_group.create_group(str(i))
            grp.create_dataset(
                "points",
                data=np.asarray(pts),
                compression="gzip",
                compression_opts=4,
                chunks=chunks,
            )
            grp.create_dataset(
                "weights",
                data=np.asarray(wts),
                compression="gzip",
                compression_opts=4,
                chunks=(min(wts.shape[0], 1024),),
            )

        # ------------------ cost ----------------------
        cost_group = base.require_group("cost")
        for child in list(cost_group.keys()):
            del cost_group[child]

        cost_list = [np.asarray(c) for c in problem.get_costs()]
        if len(cost_list) == 1:
            arr = cost_list[0]
            name = "matrix" if arr.ndim == 2 else "tensor"
            chunks = arr.shape if arr.ndim < 2 else ((min(arr.shape[0], 256),) + arr.shape[1:])
            cost_group.create_dataset(
                name,
                data=arr,
                compression="gzip",
                compression_opts=4,
                chunks=chunks,
            )
        else:
            pw = cost_group.create_group("pairwise")
            for idx, arr in enumerate(cost_list):
                pw.create_dataset(
                    str(idx),
                    data=arr,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(min(arr.shape[0], 256),) + arr.shape[1:]
                )

        # ------------------ optional ground truth -----
        try:
            cost_val = problem.get_exact_cost()
            coup = problem.get_exact_coupling()
        except Exception:
            cost_val = None
            coup = None

        if coup is not None:
            ex = base.require_group("exact")
            if "coupling" in ex:
                del ex["coupling"]
            ex.create_dataset("coupling", data=np.asarray(coup))
            if cost_val is not None:
                ex.attrs["cost"] = float(cost_val)

        self._file.flush()

    # ------------------------------------------------------------------
    def load(self, key: str) -> MarginalProblem:
        """Load a problem instance by ``key``."""

        base = self._file[self.ROOT][key]
        meta = {k: base.attrs[k] for k in base.attrs}

        # TODO: refactor this into factory as soon as we introduce
        #       multi-marginal or barycenter

        # reconstruct marginals as DiscreteMeasure
        marginals = []
        mg = base["marginals"]
        for i in sorted(mg.keys(), key=int):
            grp = mg[i]
            pts = jnp.array(grp["points"][...])
            wts = jnp.array(grp["weights"][...])
            marginals.append(DiscreteMeasure(pts, wts, name=f"m{i}"))

        # cost arrays
        cost_group = base["cost"]
        cost_arrays = []
        if "matrix" in cost_group:
            cost_arrays.append(jnp.array(cost_group["matrix"][...]))
        elif "tensor" in cost_group:
            cost_arrays.append(jnp.array(cost_group["tensor"][...]))
        elif "pairwise" in cost_group:
            for nm in sorted(cost_group["pairwise"].keys(), key=int):
                cost_arrays.append(jnp.array(cost_group["pairwise"][nm][...]))

        cls_path = meta.get("problem_class")
        prob_cls = import_object(cls_path) if cls_path else TwoMarginalProblem
        cost_fn_path = meta.get("cost_fn")
        cost_fn = import_object(cost_fn_path) if cost_fn_path else None
        name = meta.get("dataset") or meta.get("name", key)

        if prob_cls is TwoMarginalProblem:
            if len(marginals) != 2 or cost_fn is None:
                raise ValueError("Invalid HDF5 data for TwoMarginalProblem")
            prob = TwoMarginalProblem(
                name, marginals[0], marginals[1], cost_fn)
            if cost_arrays:
                prob._C = [cost_arrays[0]]
        else:
            # generic fallback for other problem types
            msg = "Only TwoMarginalProblem loading supported"
            raise NotImplementedError(msg)

        if "exact" in base:
            ex = base["exact"]
            prob._exact_coupling = jnp.array(ex["coupling"][...])
            prob._exact_cost = float(ex.attrs.get("cost", jnp.inf))

        return prob

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._file:
            self._file.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        """Return the number of stored problems."""
        return len(self._file[self.ROOT].keys())

    def __getstate__(self):
        """
        When pickling (e.g. deepcopy), remove the non-serializable _file handle.
        """
        state = self.__dict__.copy()
        state.pop("_file", None)
        return state

    def __setstate__(self, state):
        """
        When unpickling, restore attributes and re-open the HDF5 file.
        """
        self.__dict__.update(state)
        import h5py
        # reopen in append mode
        self._file = h5py.File(self.path, mode="a")
