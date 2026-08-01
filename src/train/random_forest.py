import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from configs import data_config as cfg
from src.process_data.engineering import (
    add_engineered_features,
    build_preprocessor,
    select_base_features,
)


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


def _prepare_features(X_train, X_val, X_test, use_engineered: bool = False):
    """Select features and scale. Returns scaled arrays + preprocessor."""
    if use_engineered:
        X_train_fe, means = add_engineered_features(X_train)
        X_val_fe, _       = add_engineered_features(X_val, means=means)
        X_test_fe, _      = add_engineered_features(X_test, means=means)
    else:
        X_train_fe = select_base_features(X_train)
        X_val_fe   = select_base_features(X_val)
        X_test_fe  = select_base_features(X_test)

    preprocessor = build_preprocessor(use_engineered=use_engineered)
    X_train_sc = preprocessor.fit_transform(X_train_fe)
    X_val_sc   = preprocessor.transform(X_val_fe)
    X_test_sc  = preprocessor.transform(X_test_fe)

    return X_train_sc, X_val_sc, X_test_sc


# ---------- evaluate (default RF) ----------
def evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test,
                   use_engineered: bool = False):
    X_train_sc, X_val_sc, X_test_sc = _prepare_features(
        X_train, X_val, X_test, use_engineered=use_engineered,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X_train_sc, y_train)

    metrics = {
        "train": _compute_metrics(y_train, model.predict(X_train_sc)),
        "val":   _compute_metrics(y_val,   model.predict(X_val_sc)),
        "test":  _compute_metrics(y_test,  model.predict(X_test_sc)),
    }

    return metrics, model


# ---------- tuned RF (RandomizedSearchCV) ----------
PARAM_DISTRIBUTIONS = {
    "n_estimators":      [100, 200, 300, 400, 500],
    "max_depth":         [5, 10, 15, 20, 30, None],
    "min_samples_leaf":  [1, 2, 4, 8, 16],
    "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
}


def evaluate_tuned_model(X_train, X_val, X_test, y_train, y_val, y_test,
                         use_engineered: bool = False,
                         n_iter: int = 30,
                         cv: int = 3):
    X_train_sc, X_val_sc, X_test_sc = _prepare_features(
        X_train, X_val, X_test, use_engineered=use_engineered,
    )

    base_rf = RandomForestRegressor(
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        random_state=cfg.random_state,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_train_sc, y_train)

    best_model = search.best_estimator_

    metrics = {
        "train": _compute_metrics(y_train, best_model.predict(X_train_sc)),
        "val":   _compute_metrics(y_val,   best_model.predict(X_val_sc)),
        "test":  _compute_metrics(y_test,  best_model.predict(X_test_sc)),
    }

    return metrics, best_model, search.best_params_


# ---------- printing ----------
def print_results(name: str, metrics: dict, best_params: dict = None):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    for split in ("train", "val", "test"):
        m = metrics[split]
        print(f"  {split.capitalize():6s}  |  MSE: {m['mse']:.6f}  "
              f"MAE: {m['mae']:.6f}  R2: {m['r2']:.6f}")

    if best_params:
        print(f"\n  Best hyper-parameters:")
        for k, v in best_params.items():
            print(f"    {k}: {v}")


# ---------- main ----------
def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_split_data()

    # 1) Default RF — raw features
    metrics_raw, _ = evaluate_model(
        X_train, X_val, X_test, y_train, y_val, y_test,
        use_engineered=False,
    )
    print_results("RF Default - Raw Features", metrics_raw)

    # 2) Tuned RF — raw features (RandomizedSearchCV)
    print("\n[...] Running RandomizedSearchCV (this may take a minute)...")
    metrics_tuned, _, best_params = evaluate_tuned_model(
        X_train, X_val, X_test, y_train, y_val, y_test,
        use_engineered=False,
    )
    print_results("RF Tuned - Raw Features", metrics_tuned, best_params)

    # 3) Default RF — engineered features
    metrics_eng, _ = evaluate_model(
        X_train, X_val, X_test, y_train, y_val, y_test,
        use_engineered=True,
    )
    print_results("RF Default - Engineered Features", metrics_eng)

    # 4) Compare
    print(f"\n{'-' * 60}")
    print(f"  Test MSE comparison")
    print(f"    RF Default (raw):        {metrics_raw['test']['mse']:.6f}")
    print(f"    RF Tuned   (raw):        {metrics_tuned['test']['mse']:.6f}")
    print(f"    RF Default (engineered): {metrics_eng['test']['mse']:.6f}")
    print(f"{'-' * 60}")


if __name__ == "__main__":
    main()