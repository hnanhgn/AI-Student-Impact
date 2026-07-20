import pandas as pd
import configs.data_config as config


def load_raw_data(path: str = config.raw_path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Paid_Subscription"] = df["Paid_Subscription"].astype(int).clip(0, 1)
    return df

def drop_id_column(df: pd.DataFrame, keep_for_tracking: bool = True) -> pd.DataFrame:
    if not keep_for_tracking:
        df = df.drop(columns=[config.id])
    return df


def clean_pipeline(raw_path: str = config.raw_path) -> pd.DataFrame:
    df = load_raw_data(raw_path)
    df = fix_dtypes(df)
    return df


if __name__ == "__main__":
    df_clean = clean_pipeline()
    df_clean.to_csv(config.cleaned_path, index=False)