import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.append(str(root))

from configs import data_config as cfg

features = ["Pre_Semester_GPA", "Weekly_GenAI_Hours", "Tool_Diversity"]
target = cfg.target


def load_data(path: str = cfg.raw_path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_features(df: pd.DataFrame):
    X = df[features].copy()
    y = df[target]
    return X, y


def train_baseline_linear_regression(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = cfg.random_state,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def print_results(result: dict):
    print("\n=== Baseline Linear Regression ===")
    print(f"MSE: {result['metrics']['mse']:.6f}")
    print(f"MAE: {result['metrics']['mae']:.6f}")
    print(f"R2 : {result['metrics']['r2']:.6f}")

    print("\nCoefficients:")
    for feat, coef in zip(result["X_train"].columns, result["model"].coef_):
        print(f"  {feat}: {coef:.6f}")

    print(f"\nIntercept: {result['model'].intercept_:.6f}")


def main():
    df = load_data()
    X, y = build_features(df)
    result = train_baseline_linear_regression(X, y)
    print_results(result)


if __name__ == "__main__":
    main()