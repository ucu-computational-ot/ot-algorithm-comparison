import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
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

    cols_to_ensure = {
        'dataset', 'iterations', 'error', 'time', 'time_counter',
        'peak_gpu_mem', 'peak_util_pct', 'mean_util_pct', 'name',
        'status', 'reg',
    }
    missing_columns = cols_to_ensure.difference(set(df.columns))
    if len(missing_columns) != 0:
        print(f"Some columns are missing: {missing_columns}")
        # TODO: handle missing columns
        # well that never gonna happen but let's just be sure
        pass
    return df


def preprocess(dataframe: pd.DataFrame):
    replacements = {
        "iterations": "none",
        "reg": 0,
        "maxiter": 'none',
        "tol": 'na',
    }
    dataframe.fillna(value=replacements, inplace=True)
    dataframe['size'] = dataframe['dataset'].str.extract(r'(\d+)p')
    dataframe['dim'] = dataframe['dataset'].str.extract(r'(\d)D-')
    dataframe.rename(columns={
        'name': 'solver',
    }, inplace=True)
    dataframe['runtime'] = dataframe['time_counter'].copy()
    dataframe['distribution'] = dataframe['dataset'].str.extract(r'\dD-(.+)-\d+p')


def select_dimension(df: pd.DataFrame, dim: int) -> pd.DataFrame:
    return df[df['dataset'].str.startswith(dim + 'D') == True]
