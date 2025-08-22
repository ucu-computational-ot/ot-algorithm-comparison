import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from convergence_thresholds import (
    MAX_ITER_BY_SOLVER,
    TOL_BY_SOLVER,
)
import logging

load_dotenv()

if os.getenv("DATASET_LOCATION") is None:
    raise ValueError("DATASET_LOCATION environment variable not set")
DATASET_LOCATION = Path(os.getenv("DATASET_LOCATION"))
DATASET_FOLDERS = os.getenv("DATASET_FOLDERS", ".").split(",")
_raw = os.getenv("LIMIT_TO_FILES", "none")
# build a list of real filenames, dropping “none” or empties
LIMIT_TO_FILES = [
    fn.strip() for fn in _raw.split(",")
    if fn.strip() and fn.strip().lower() != "none"
]

SOLVERS_MAP = {
    "gradient": "SGD",
    "gradient-log": "Log-Domain Gradient Ascent (ADAM optimizer)",
    "gradient-plain": "Vanilla Gradient Ascent",
    "lbfgs": "LBFGS",
    "lp": "Simplex",
    "sinkhorn": "Vanilla Sinkhorn",
    "sinkhorn-log": "Log-Domain Sinkhorn",
    "sinkhorn-normed": "Sinkhorn (normalized cost matrix)",
    "sinkhorn-normed-log": "Log-Domain Sinkhorn (normalized cost matrix)",
}

DISTRIBUTIONS_MAP = {
    "cauchy": "Cauchy",
    "exponential": "Exponential",
    "gaussian-1c": "Gaussian",

    "gaussian-2c": "Gaussian (2 comp.)",
    "gaussian-4c": "Gaussian (4 comp.)",
    "gaussian-6c": "Gaussian (6 comp.)",

    "gen-hyperb-mixture-2c": "Gen.Hyperb.Mixt. (2 comp.)",
    "gen-hyperb-mixture-4c": "Gen.Hyperb.Mixt. (4 comp.)",
    "gen-hyperb-mixture-6c": "Gen.Hyperb.Mixt. (6 comp.)",

    "students-2df": "Students (2 deg.f.)",
    "students-4df": "Students (4 deg.f.)",
    "students-6df": "Students (6 deg.f.)",

    "cauchy-vs-gaussian": "Cauchy to Gaussian",
    "exp-vs-cauchy": "Exponential to Cauchy",
    "exp-vs-gaussian": "Exponential to Gaussian",
}


def load_all_csvs(base_dir: Path, subdirs: list[str]) -> pd.DataFrame:
    """
    Reads every .csv file in each of the specified subdirectories (non-recursively)
    under base_dir, concatenates them, and returns one DataFrame.
    """
    df_list = []
    for sub in subdirs:
        folder = base_dir / sub
        if not folder.exists():
            logging.warn(f"Warning: {folder!r} does not exist, skipping.")
            continue

        # Gather all CSV files directly under this folder
        for csv_path in folder.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path)
                df["__source_file__"] = str(csv_path.relative_to(base_dir))
                df_list.append(df)
            except Exception as e:
                logging.error(f"Failed to read {csv_path}: {e}")

    if not df_list:
        raise ValueError("No CSV files were loaded.")
    # Concatenate and reset index
    return pd.concat(df_list, ignore_index=True)


def load_all_df():
    df = load_all_csvs(DATASET_LOCATION, DATASET_FOLDERS)
    if LIMIT_TO_FILES:
        df = df[df["__source_file__"].isin(LIMIT_TO_FILES)]

    for file in df["__source_file__"]:
        logging.info(f"Loaded {file}")

    # first, verify the raw columns exist
    raw_cols = {
        "dataset", "iterations", "error", "time", "time_counter",
        "peak_gpu_mem", "combined_peak_gpu_ram",
        "peak_gpu_util_pct", "mean_gpu_util_pct",
        "peak_ram_MiB", "combined_peak_ram_MiB",
        "max_cpu_util_pct", "mean_cpu_util_pct",
        "status", "name", "reg"
    }
    missing = raw_cols - set(df.columns)
    if missing:
        logging.warning(f"Missing expected raw columns: {missing}")

    # now rename to your standardized names
    df = df.rename(columns={
        "combined_peak_gpu_ram": "combined_peak_gpu_mem",
        "peak_gpu_util_pct":    "peak_util_pct",
        "mean_gpu_util_pct":    "mean_util_pct",
        "peak_ram_MiB":         "peak_ram_mem",
        "combined_peak_ram_MiB": "combined_peak_ram_mem",
        "max_cpu_util_pct":     "peak_cpu_util_pct",
        "mean_cpu_util_pct":    "mean_cpu_util_pct",  # if you use this
    })

    # finally, ensure the standardized set that the dashboard expects
    cols_to_ensure = {
        "dataset", "iterations", "error", "time", "time_counter",
        "peak_gpu_mem", "combined_peak_gpu_mem",
        "peak_util_pct", "mean_util_pct",
        "peak_ram_mem", "combined_peak_ram_mem",
        "peak_cpu_util_pct", "mean_cpu_util_pct",
        "status", "name", "reg",
    }
    missing_std = cols_to_ensure - set(df.columns)
    if missing_std:
        logging.warning(f"Missing standardized columns: {missing_std}")

    return df


def preprocess(dataframe: pd.DataFrame):
    replacements = {
        "iterations": "none",
        "reg": 0,
        "maxiter": 'none',
        "tol": 'na',
    }
    dataframe.fillna(value=replacements, inplace=True)
    dataframe['reg'] = dataframe['reg'].astype(float)
    dataframe['size'] = dataframe['dataset'].str.extract(r'(\d+)p').astype(int)
    dataframe['dim'] = dataframe['dataset'].str.extract(r'(\d)D-').astype(int)
    dataframe.rename(columns={
        'name': 'solver',
    }, inplace=True)
    dataframe['solver'] = dataframe['solver'].map(lambda x: SOLVERS_MAP.get(x, x))
    dataframe['runtime'] = dataframe['time_counter'].copy()
    dataframe['distribution'] = dataframe['dataset'].str.extract(r'\dD-(.+)-\d+p')
    dataframe['distribution'] = dataframe['distribution'].map(lambda x: DISTRIBUTIONS_MAP.get(x, x))


def select_dimension(df: pd.DataFrame, dim: int) -> pd.DataFrame:
    return df[df['dataset'].str.startswith(dim + 'D') == True]


def filter_converged(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where the solver converged.

    A row is dropped if:
      - error is NaN, OR
      - (iterations >= maxiter) AND (error > tol)

    Notes
    -----
    - Uses per-row `maxiter` (and optional per-row `tol`) from the DataFrame.
    - If `maxiter` is missing/NaN → treated as +inf (i.e., cannot 'hit' max).
    - If `tol` is missing/NaN or absent → treated as -inf (i.e., any error is acceptable
      unless maxiter was hit, which still requires error > tol to count as failure).
    """
    if df.empty:
        return df.copy()

    good = df.copy()

    # Coerce numerics
    good["iterations"] = pd.to_numeric(good.get("iterations", np.nan), errors="coerce")
    good["error"]      = pd.to_numeric(good.get("error", np.nan), errors="coerce")
    maxiter_series     = pd.to_numeric(good.get("maxiter", np.nan), errors="coerce")
    # Optional tolerance column; fall back to -inf
    if "tol" in good.columns:
        tol_series = pd.to_numeric(good["tol"], errors="coerce")
    else:
        tol_series = pd.Series(np.nan, index=good.index)

    # 1) Drop NaN errors (non-converged by definition)
    good = good[~good["error"].isna()].copy()
    if good.empty:
        return good.reset_index(drop=True)

    # 2) Fill missing thresholds
    maxiter_filled = maxiter_series.reindex(good.index).fillna(np.inf)
    tol_filled     = tol_series.reindex(good.index).fillna(-np.inf)

    # 3) Failure: hit (or exceeded) its own maxiter AND still above tol
    #    (treat NaN iterations as 0 to avoid spurious failures)
    iters = good["iterations"].fillna(0)
    failures = (iters >= maxiter_filled) & (good["error"] > tol_filled)

    # 4) Keep only non-failures
    return good[~failures].reset_index(drop=True)
