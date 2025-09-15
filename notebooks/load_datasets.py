import pandas as pd
from pathlib import Path

def load_all_csvs(base_dir: Path, subdirs: list[str]) -> pd.DataFrame:
    """
    Reads every .csv file in each of the specified subdirectories (non-recursively)
    under base_dir, concatenates them, and returns one DataFrame.
    """
    df_list = []
    for sub in subdirs:
        folder = base_dir / sub
        if not folder.exists():
            print(f"Warning: {folder!r} does not exist, skipping.")
            continue

        # Gather all CSV files directly under this folder
        for csv_path in folder.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path)
                df["__source_file__"] = str(csv_path.relative_to(base_dir))
                df_list.append(df)
            except Exception as e:
                print(f"Failed to read {csv_path}: {e}")

    if not df_list:
        raise ValueError("No CSV files were loaded.")
    # Concatenate and reset index
    return pd.concat(df_list, ignore_index=True)

if __name__ == "__main__":
    # Base directory containing your `results/` folder
    base = Path("./results/remote")

    # List exactly the subfolders you want to include
    subs = [
        "1d",
        "2d",
        "3d",
        "problem_set_1",
    ]

    all_data = load_all_csvs(base, subs)
    print(f"Loaded {len(all_data)} rows from {all_data['__source_file__'].nunique()} files.")