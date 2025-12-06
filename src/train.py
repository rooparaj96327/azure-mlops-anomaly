# src/train.py

import argparse
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import joblib


def load_data(training_data_path: str) -> pd.DataFrame:
    if os.path.isdir(training_data_path):
        csv_path = os.path.join(training_data_path, "selected_features.csv")
    else:
        csv_path = training_data_path

    print(f"Loading training data from: {csv_path}")
    return pd.read_csv(csv_path)


def split_features_labels(df: pd.DataFrame):
    X = df.drop(columns=["is_anomaly"])
    y = df["is_anomaly"]
    return X, y


def get_model(model_type: str):
    model_type = model_type.lower()

    if model_type == "logistic":
        print("Using Logistic Regression...")
        return LogisticRegression(max_iter=500)

    elif model_type == "randomforest":
        print("Using RandomForestClassifier...")
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    elif model_type == "xgboost":
        print("Using XGBClassifier...")
        return XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_and_evaluate(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nModel Evaluation:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}\n")
    print(classification_report(y_test, y_pred))

    return model, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def save_model(model, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True,
                        help="Choose: logistic | randomforest | xgboost")

    args = parser.parse_args()

    df = load_data(args.training_data)
    X, y = split_features_labels(df)

    model = get_model(args.model_type)

    trained_model, metrics = train_and_evaluate(model, X, y)

    save_model(trained_model, args.model_output)