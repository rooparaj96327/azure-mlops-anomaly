# src/prep.py

import argparse
import pandas as pd
from sklearn.utils import resample
import os

def preprocess(input_path, output_path):

    print("Loading dataset from:", input_path)
    df = pd.read_csv(input_path)

    # Check class distribution
    print("Class distribution BEFORE balancing:")
    print(df['is_anomaly'].value_counts())

    # Separate classes
    df_majority = df[df['is_anomaly'] == 1]   # anomalies (majority class)
    df_minority = df[df['is_anomaly'] == 0]   # normal (minority class)

    # Downsample anomaly class to match minority class size
    print("Downsampling majority class...")
    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),   # match minority class count
        random_state=42
    )

    # Combine balanced dataset
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    # Shuffle rows
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Class distribution AFTER balancing:")
    print(df_balanced['is_anomaly'].value_counts())

    # Save output
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "processed_data.csv")

    df_balanced.to_csv(output_file, index=False)
    print(f"Processed dataset saved to: {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True,
                        help="Path to the raw dataset CSV")
    parser.add_argument("--output_data", type=str, required=True,
                        help="Path for saving processed dataset")

    args = parser.parse_args()

    preprocess(args.input_data, args.output_data)