import dash
from dash import Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.filters import FILTER_INPUTS
from utils.data import get_filtered
from utils.data import SOLVERS_COLOR_MAP
from dataset import LEGEND_ORDER


INSTANCE_KEYS = ["distribution", "size", "dim"]
# TAU_MAX_DEFAULT = 1000.0
TAU_MAX_DEFAULT = 100.0
N_TAU = 200
MAX_COLS = 3
# R_M = 1e6  # large ratio for missing/failed runs
# R_M = 1e6  # failure cap ratio (Dolan–Moré)
R_M = 1e4  # failure cap ratio (Dolan–Moré)


def _runtime_column(df: pd.DataFrame) -> str:
    if "runtime" in df.columns: return "runtime"
    if "time" in df.columns:    return "time"
    raise KeyError("Neither 'runtime' nor 'time' present in dataframe.")


def _compute_ratios_per_run(
    df: pd.DataFrame,
    runtime_col: str,
    r_M: float = R_M,
    sort_key: str | None = None,
) -> pd.DataFrame:
    """
    Per-run Dolan–Moré ratios WITHOUT averaging.
    Repeats are aligned by an integer run index per (instance, solver),
    then normalization is done across solvers for the same (instance, run_ix).

    If `sort_key` is provided (e.g. 'time_counter'), we sort by it inside each
    (instance, solver) before assigning run indices to get stable alignment.

    Returns columns: ['solver', 'ratio'] + INSTANCE_KEYS + ['run_ix']
    """
    cols = ["solver", runtime_col] + INSTANCE_KEYS + ([sort_key] if sort_key and sort_key in df.columns else [])
    d = df[cols].copy()

    # numeric clean
    d[runtime_col] = pd.to_numeric(d[runtime_col], errors="coerce")
    d = d.dropna(subset=["solver"])

    # assign run index PER (instance, solver)
    grp_keys = INSTANCE_KEYS + ["solver"]
    if sort_key and sort_key in d.columns:
        d = d.sort_values(grp_keys + [sort_key])
    # run_ix = 0,1,2,... within each (instance, solver)
    d["run_ix"] = d.groupby(grp_keys, dropna=False).cumcount()

    # valid runtimes
    valid = d[np.isfinite(d[runtime_col]) & (d[runtime_col] > 0)].copy()

    # compress to one time per (instance, solver, run_ix) in case of duplicates
    per_solver_run = (
        valid.groupby(INSTANCE_KEYS + ["solver", "run_ix"], dropna=False, sort=False)[runtime_col]
             .min()              # or .first() if you prefer
             .rename("runtime")
             .reset_index()
    )

    if per_solver_run.empty:
        return pd.DataFrame(columns=["solver", "ratio"] + INSTANCE_KEYS + ["run_ix"])
    #── NEW: equalize repeats per instance (intersection) ───────────────
    # count repeats per (instance, solver)
    rep_counts = (per_solver_run
                  .groupby(INSTANCE_KEYS + ["solver"], dropna=False, sort=False)["run_ix"]
                  .max().add(1).rename("n_rep").reset_index())
    # min repeats across solvers for each instance
    nmin = (rep_counts.groupby(INSTANCE_KEYS, dropna=False)["n_rep"]
            .min().rename("n_min").reset_index())
    # keep only run_ix < n_min for that instance
    per_solver_run = (per_solver_run
                      .merge(nmin, on=INSTANCE_KEYS, how="left", validate="many_to_one"))
    per_solver_run = per_solver_run[per_solver_run["run_ix"] < per_solver_run["n_min"]]
    per_solver_run = per_solver_run.drop(columns="n_min")
    if per_solver_run.empty:
        return pd.DataFrame(columns=["solver", "ratio"] + INSTANCE_KEYS + ["run_ix"])
    # ────────────────────────────────────────────────────────────────────

    # best runtime across solvers for SAME (instance, run_ix)
    best_per_run = (
        per_solver_run.groupby(INSTANCE_KEYS + ["run_ix"], dropna=False, sort=False)["runtime"]
                      .min()
                      .rename("best_runtime")
                      .reset_index()
    )
    best_per_run = best_per_run[np.isfinite(best_per_run["best_runtime"]) & (best_per_run["best_runtime"] > 0)]
    if best_per_run.empty:
        return pd.DataFrame(columns=["solver", "ratio"] + INSTANCE_KEYS + ["run_ix"])

    # ── NEW: build grid only over solvers PRESENT for that instance ─────
    solvers_per_inst = per_solver_run[INSTANCE_KEYS + ["solver"]].drop_duplicates()
    all_runs = best_per_run[INSTANCE_KEYS + ["run_ix"]].drop_duplicates()
    grid = all_runs.merge(solvers_per_inst, on=INSTANCE_KEYS, how="inner")
    # ────────────────────────────────────────────────────────────────────

    # build full solver × (instance, run_ix) grid so missing solvers get r_M
    # all_solvers = pd.Index(d["solver"].dropna().astype(str).unique())
    # all_runs = best_per_run[INSTANCE_KEYS + ["run_ix"]].drop_duplicates()
    #
    # grid = (
    #     all_runs.assign(_k=1)
    #     .merge(pd.DataFrame({"solver": all_solvers, "_k": 1}), on="_k")
    #     .drop(columns="_k")
    # )

    # attach runtimes and bests
    grid = grid.merge(
        per_solver_run,
        on=INSTANCE_KEYS + ["solver", "run_ix"],
        how="left",
        validate="many_to_one",
    ).merge(
        best_per_run,
        on=INSTANCE_KEYS + ["run_ix"],
        how="left",
        validate="many_to_one",
    )

    # ratios; default to failure cap
    grid["ratio"] = r_M
    ok = (
        np.isfinite(grid["runtime"]) & (grid["runtime"] > 0) &
        np.isfinite(grid["best_runtime"]) & (grid["best_runtime"] > 0)
    )
    grid.loc[ok, "ratio"] = grid.loc[ok, "runtime"] / grid.loc[ok, "best_runtime"]

    out = grid[["solver", "ratio"] + INSTANCE_KEYS + ["run_ix"]].copy()
    bad = ~np.isfinite(out["ratio"]) | (out["ratio"] <= 0)
    if bad.any():
        out.loc[bad, "ratio"] = r_M

    return out.reset_index(drop=True)


def _choose_tau_grid(ratios: pd.DataFrame, tau_max: float | None) -> np.ndarray:
    if ratios.empty:
        return np.array([1.0, 1.1])
    if tau_max is None:
        # robust upper bound: cap by default, but don’t squash genuine spread
        finite_max = np.nanmax(ratios["ratio"].values)
        # if extreme, use a quantile to avoid a single outlier blowing out the axis
        q95 = np.nanquantile(ratios["ratio"].values, 0.95)
        upper = float(np.clip(max(min(finite_max, q95), 2.0), 2.0, TAU_MAX_DEFAULT))
        tau_max = upper
    tau_max = max(1.01, float(tau_max))
    return np.geomspace(1.0, tau_max, N_TAU)


def _profile_curves_runs(
    ratios: pd.DataFrame,
    run_col: str = "run_ix",              # default to aligned index from _compute_ratios_per_run_aligned
    group_keys=None,
    tau_max=None,
):
    """
    Performance profiles where *each run* counts as its own instance.
    `ratios` must include columns: ['solver', 'ratio'] + INSTANCE_KEYS + [run_col].
    Returns:
        results: dict {group_key_tuple: (taus, {solver -> Phi(τ) array})}
        solver_order: stable global solver order
    """
    # sanity
    needed = {"solver", "ratio", *INSTANCE_KEYS, run_col}
    if ratios.empty or not needed.issubset(ratios.columns):
        return {}, []

    # τ grid
    taus = _choose_tau_grid(ratios, tau_max)

    # stable global solver order (first appearance)
    solvers = list(dict.fromkeys(ratios["solver"].tolist()))
    solver_order = [solver for solver in LEGEND_ORDER if solver in solvers]

    # select groups (overall if none)
    groups = [((), ratios)] if not group_keys else list(ratios.groupby(group_keys, dropna=False))

    # each (instance, run) is a unique “problem”
    instance_keys = INSTANCE_KEYS + [run_col]
    results = {}

    for gkey, gdf in groups:
        # number of (instance, run) pairs in this panel
        inst = gdf[instance_keys].drop_duplicates()
        N = len(inst)
        if N == 0:
            results[gkey] = (taus, {})
            continue

        # only solvers present in this panel
        solvers_here = list(dict.fromkeys(gdf["solver"].tolist()))
        profiles = {}

        # Φ_s(τ) = (1/N) * |{ (i,run): r_{s,i,run} ≤ τ }|
        # ratios already have one row per (solver, instance, run_col)
        for s in solvers_here:
            r = gdf.loc[gdf["solver"] == s, "ratio"].to_numpy()
            if r.size == 0:
                continue
            counts = (r[None, :] <= taus[:, None]).sum(axis=1)
            profiles[s] = counts / float(N)

        results[gkey] = (taus, profiles)

    return results, solver_order
