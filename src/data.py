from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ILPD_COLUMNS, TARGET_MAPPING, RANDOM_STATE


def load_raw_ilpd(data_path) -> pd.DataFrame:
    """
    Load the raw ILPD CSV.

    The original UCI file has no header; column names are supplied manually.
    A defensive check removes the header row if it was accidentally written to the file.

    NOTE: Although the UCI dataset page states no missing values, the actual CSV
    contains 4 missing entries in Albumin_and_Globulin_Ratio. These are left as
    NaN here and handled via median imputation inside model pipelines to prevent
    data leakage.
    """
    df = pd.read_csv(data_path, header=None, names=ILPD_COLUMNS)

    first_row = df.iloc[0].astype(str).str.strip().tolist()
    if first_row == ILPD_COLUMNS:
        df = df.iloc[1:].reset_index(drop=True)

    numeric_cols = [col for col in ILPD_COLUMNS if col != "Gender"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_processed_dataset(data_path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def create_basic_clean_dataset(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Minimal cleaning: remove exact duplicates.
    Missing values are intentionally kept for pipeline-based imputation.
    """
    processed = df.copy()
    if drop_duplicates:
        processed = processed.drop_duplicates(keep="first").reset_index(drop=True)
    return processed


def prepare_binary_target(
    df: pd.DataFrame,
    source_col: str = "Selector",
    target_col: str = "target",
) -> pd.DataFrame:
    prepared = df.copy()
    prepared[target_col] = prepared[source_col].map(TARGET_MAPPING)

    if prepared[target_col].isna().any():
        invalid = prepared.loc[prepared[target_col].isna(), source_col].unique()
        raise ValueError(f"Invalid target values in {source_col}: {invalid}")

    prepared = prepared.drop(columns=[source_col])
    return prepared


def split_features_target(df: pd.DataFrame, target_col: str = "target"):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def create_train_test_split(X, y, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_class_distribution(y, prefix: str = "proportion") -> pd.DataFrame:
    counts = y.value_counts().sort_index()
    proportions = y.value_counts(normalize=True).sort_index()
    return pd.DataFrame({"count": counts, prefix: proportions})


def compare_train_test_distribution(y_train, y_test) -> pd.DataFrame:
    train = y_train.value_counts(normalize=True).sort_index().rename("train_proportion")
    test = y_test.value_counts(normalize=True).sort_index().rename("test_proportion")
    return pd.concat([train, test], axis=1)
