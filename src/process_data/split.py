"""
make_dataset.py
Entry-point script: đọc dữ liệu đã clean -> chia Train/Validation/Test -> lưu ra CSV.

Cách chạy:
    python -m src.data.make_dataset
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from configs import data_config as config
from src.process_data.clean import clean_pipeline


def split_train_val_test(
    df: pd.DataFrame,
    target: str = config.target,
    stratify_col: str = config.stratify,
    train_ratio: float = config.train_ratio,
    val_ratio: float = config.val_ratio,
    test_ratio: float = config.test_ratio,
    random_state: int = config.random_state,
):
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9
    X = df.drop(columns=[target])
    y = df[target]

    temp_ratio = val_ratio + test_ratio
    test_ratio_within_temp = test_ratio / temp_ratio

    stratify_full = X[stratify_col] if stratify_col else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=stratify_full,
    )

    stratify_temp = X_temp[stratify_col] if stratify_col else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_ratio_within_temp,
        random_state=random_state,
        stratify=stratify_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def report_split_summary(X_train, X_val, X_test, stratify_col=config.stratify):
    total = len(X_train) + len(X_val) + len(X_test)
    print(f"Train : {len(X_train):>6} dòng ({len(X_train)/total:.1%})")
    print(f"Val   : {len(X_val):>6} dòng ({len(X_val)/total:.1%})")
    print(f"Test  : {len(X_test):>6} dòng ({len(X_test)/total:.1%})")
    print(f"Tổng  : {total:>6} dòng")

    if stratify_col:
        print(f"Phân bố '{stratify_col}' (%)")
        summary = pd.DataFrame({
            "Train": X_train[stratify_col].value_counts(normalize=True),
            "Val": X_val[stratify_col].value_counts(normalize=True),
            "Test": X_test[stratify_col].value_counts(normalize=True),
        }).round(3)
        print(summary)


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, out_dir: str = config.processed_dir):
    os.makedirs(out_dir, exist_ok=True)

    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(out_dir, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)

    y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(out_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(out_dir, "y_test.csv"), index=False)

    print(f"Saved in {out_dir}")


def main():
    df_clean = clean_pipeline()
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(df_clean)
    report_split_summary(X_train, X_val, X_test)
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    main()