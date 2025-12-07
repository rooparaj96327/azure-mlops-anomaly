# src/train.py

import argparse
import os
import pandas as pd

def load_data(input_path: str) -> pd.DataFrame:
    """
    Load preprocessed data. In Azure ML, input may be a folder or a file.
    """
    if os.path.isdir(input_path):
        csv_path = os.path.join(input_path, "processed_data.csv")
    else:
        csv_path = input_path

    print(f"Loading data from: {csv_path}")
    return pd.read_csv(csv_path)


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select final features to use in model training.
    Removes identifiers and overly strong/leaking internal metrics.
    Keeps operational performance metrics.
    """

    # Columns identified earlier as NOT useful for modeling
    drop_cols = [
        "run_id",
        "test_name",
        # Add problematic Go memory metrics if present
        "go_memstats_alloc_bytes",
        "go_memstats_alloc_bytes_total",
        "go_memstats_buck_hash_sys_bytes",
        "go_memstats_frees_total",
        "go_memstats_gc_sys_bytes",
        "go_memstats_heap_alloc_bytes",
        "go_memstats_heap_idle_bytes",
        "go_memstats_heap_inuse_bytes",
        "go_memstats_heap_objects",
        "go_memstats_heap_released_bytes",
    ]

    # Drop only columns that exist in the dataframe
    drop_cols = [col for col in drop_cols if col in df.columns]

    print(f"Dropping columns: {drop_cols}")
    df = df.drop(columns=drop_cols)

    print("Final feature columns:")
    print(df.columns)

    return df


def save_output(df: pd.DataFrame, output_path: str):
    """
    Save the selected feature dataset to Azure ML's output folder.
    """
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "selected_features.csv")
    df.to_csv(output_file, index=False)
    print(f"Selected feature dataset saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True,
                        help="Path to processed_data.csv or its folder.")
    parser.add_argument("--output_data", type=str, required=True,
                        help="Output folder to save selected_features.csv")
    
    args = parser.parse_args()

    df = load_data(args.input_data)
    df_selected = select_features(df)
    save_output(df_selected, args.output_data)