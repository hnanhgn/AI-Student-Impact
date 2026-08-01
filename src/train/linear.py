import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.append(str(root))

from configs import data_config as cfg
from src.process_data.engineering import add_engineered_features

# ---------- feature sets ----------
base_features = cfg.features  # 5 features, same as RF raw

engineered_only = [
    "Weekly_GenAI_Hours_Sq",
    "PreGPA_Sq",
    "PreGPA_x_Tool",
    "AI_Engagement_Index",
]


# ---------- helpers ----------
def load_split_data(data_dir: str = cfg.processed_dir):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(data_dir, "X_val.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))

    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    y_val = pd.read_csv(os.path.join(data_dir, "y_val.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()

    return X_train, X_val, X_test, y_train, y_val, y_test


def _compute_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# ---------- train / evaluate ----------
def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test,
                       feature_cols: list):
    """Train LinearRegression on *feature_cols* and return metrics on
    train / val / test sets."""
    X_tr = X_train[feature_cols]
    X_v = X_val[feature_cols]
    X_te = X_test[feature_cols]

    model = LinearRegression()
    model.fit(X_tr, y_train)

    metrics = {
        "train": _compute_metrics(y_train, model.predict(X_tr)),
        "val":   _compute_metrics(y_val,   model.predict(X_v)),
        "test":  _compute_metrics(y_test,  model.predict(X_te)),
    }

    return {
        "model": model,
        "metrics": metrics,
        "features": feature_cols,
    }


def train_engineered(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train LinearRegression with base + 4 stepwise-engineered features."""
    X_train_fe, means = add_engineered_features(X_train)
    X_val_fe, _       = add_engineered_features(X_val, means=means)
    X_test_fe, _      = add_engineered_features(X_test, means=means)

    feature_cols = base_features + engineered_only
    return train_and_evaluate(X_train_fe, X_val_fe, X_test_fe,
                              y_train, y_val, y_test,
                              feature_cols=feature_cols)


# ---------- printing ----------
def print_results(name: str, result: dict):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    for split in ("train", "val", "test"):
        m = result["metrics"][split]
        print(f"  {split.capitalize():6s}  |  MSE: {m['mse']:.6f}  "
              f"MAE: {m['mae']:.6f}  R2: {m['r2']:.6f}")

    print(f"\n  Coefficients:")
    for feat, coef in zip(result["features"], result["model"].coef_):
        print(f"    {feat}: {coef:.6f}")
    print(f"  Intercept: {result['model'].intercept_:.6f}")


# ---------- main ----------
def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_split_data()

    # 1) Base 5-feature model (parity with RF raw)
    result_base = train_and_evaluate(
        X_train, X_val, X_test, y_train, y_val, y_test,
        feature_cols=base_features,
    )
    print_results("Linear Regression - 5 Base Features", result_base)

    # 2) Base + 4 engineered features (stepwise selection)
    result_eng = train_engineered(
        X_train, X_val, X_test, y_train, y_val, y_test,
    )
    print_results("Linear Regression - Base + Engineered (9 Features)", result_eng)

    # 3) Compare test MSE
    mse_base = result_base["metrics"]["test"]["mse"]
    mse_eng  = result_eng["metrics"]["test"]["mse"]
    delta    = mse_base - mse_eng
    pct      = delta / mse_base * 100

    print(f"\n{'-' * 60}")
    print(f"  Test MSE comparison")
    print(f"    Base (5 feat):        {mse_base:.6f}")
    print(f"    Engineered (9 feat):  {mse_eng:.6f}")
    print(f"    Improvement:          {delta:.6f}  ({pct:+.2f}%)")
    print(f"{'-' * 60}")


if __name__ == "__main__":
    main()