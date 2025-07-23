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
    dataframe['runtime'] = dataframe['time_counter'].copy()
    dataframe['distribution'] = dataframe['dataset'].str.extract(r'\dD-(.+)-\d+p')


def select_dimension(df: pd.DataFrame, dim: int) -> pd.DataFrame:
    return df[df['dataset'].str.startswith(dim + 'D') == True]


def filter_converged(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only those rows where the solver actually converged.
    A row is dropped if:
      - error is NaN, or
      - iterations >= solver’s max_iter AND error > solver’s tol.
    Solvers not in the dicts are assumed to always converge (unless error is NaN).
    """
    # copy and coerce types
    good = df.copy()
    # ensure numeric
    good["iterations"] = pd.to_numeric(good["iterations"], errors="coerce").fillna(0).astype(int)
    good["error"] = pd.to_numeric(good["error"], errors="coerce")

    # 1) drop NaN errors
    good = good[~good["error"].isna()]

    # 2) map per‑solver thresholds, defaulting to “never fail”
    max_iters = good["solver"].map(MAX_ITER_BY_SOLVER).fillna(np.inf)
    tols = good["solver"].map(TOL_BY_SOLVER).fillna(-np.inf)

    # 3) failure iff it actually hit its max and is still above tol
    failures = (good["iterations"] >= max_iters) & (good["error"] > tols)

    # 4) keep only non‑failures
    return good[~failures].reset_index(drop=True)
