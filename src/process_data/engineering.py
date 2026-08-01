import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from configs import data_config as cfg

base_numeric_feature = [f for f in cfg.features if f not in cfg.binary_feature]

engineered_feature = [
    "Weekly_GenAI_Hours_Sq",   
    "PreGPA_Sq",               
    "PreGPA_x_Tool",          
    "AI_Engagement_Index",    
]


def select_base_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[cfg.features].copy()


def add_engineered_features(df: pd.DataFrame, means: dict = None):
    df = df.copy()

    cols_to_center = ["Weekly_GenAI_Hours", "Tool_Diversity", "Pre_Semester_GPA"]

    if means is None:
        means = {col: df[col].mean() for col in cols_to_center}

    centered = {col: df[col] - means[col] for col in cols_to_center}
    df["Weekly_GenAI_Hours_Sq"] = centered["Weekly_GenAI_Hours"] ** 2
    df["PreGPA_Sq"] = centered["Pre_Semester_GPA"] ** 2
    df["PreGPA_x_Tool"] = centered["Pre_Semester_GPA"] * centered["Tool_Diversity"]
    df["AI_Engagement_Index"] = centered["Weekly_GenAI_Hours"] * centered["Tool_Diversity"]

    return df, means


def build_features(df: pd.DataFrame, means: dict = None):
    df_base = select_base_features(df)
    df_final, means_used = add_engineered_features(df_base, means=means)
    return df_final, means_used


def get_all_feature_names() -> list:
    return base_numeric_feature + engineered_feature + cfg.binary_feature


def build_preprocessor(use_engineered: bool = False) -> ColumnTransformer:
    numeric_cols = base_numeric_feature + (engineered_feature if use_engineered else [])
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("bin", "passthrough", cfg.binary_feature),
        ]
    )
    return preprocessor


if __name__ == "__main__":
    import os

    X_train = pd.read_csv(os.path.join(cfg.processed_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(cfg.processed_dir, "X_val.csv"))

    X_train_fe, means = build_features(X_train)
    X_val_fe, _ = build_features(X_val, means=means)

    print(X_train_fe.columns.tolist())
    print(means)
    print(X_train_fe.head())