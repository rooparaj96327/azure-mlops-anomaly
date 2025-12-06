# src/train.py

import argparse
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import joblib


def load_data(training_data_path: str) -> pd.DataFrame:
    """
    Load the processed dataset.

    If training_data_path is a folder (Azure ML style), it looks for
    'processed_data.csv' inside it.
    If it's a file path, it reads that file directly.
    """
    if os.path.isdir(training_data_path):
        csv_path = os.path.join(training_data_path, "processed_data.csv")
    else:
        csv_path = training_data_path

    print(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def split_features_labels(df: pd.DataFrame):
    """
    Split dataframe into X (features) and y (target).
    Assumes the target column is 'is_anomaly'.
    """
    X = df.drop(columns=["is_anomaly"])
    y = df["is_anomaly"]
    return X, y


def train_model(X, y):
    """
    Train a RandomForest classifier and return the model and evaluation metrics.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Evaluation metrics on test set:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return model, metrics


def save_model(model, output_dir: str):
    """
    Save the trained model to the specified output directory.
    Azure ML will pick up anything saved in this folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_data",
        type=str,
        required=True,
        help="Path to processed training data (file or folder).",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        required=True,
        help="Output directory to save the trained model.",
    )

    args = parser.parse_args()

    # 1. Load data
    df = load_data(args.training_data)

    # 2. Split features/labels
    X, y = split_features_labels(df)

    # 3. Train model and evaluate
    model, metrics = train_model(X, y)

    # 4. Save model
    save_model(model, args.model_output)