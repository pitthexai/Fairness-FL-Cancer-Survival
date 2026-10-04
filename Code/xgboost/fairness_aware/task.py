# xgboost_comprehensive/task.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

LABEL_COL = "survivability"


@dataclass
class SiteSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class SiteSplitDF:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


_SPLITS_CACHE: Dict[str, SiteSplit] = {}
_SPLITS_DF_CACHE: Dict[str, SiteSplitDF] = {}
_SCALERS_CACHE: Dict[str, StandardScaler] = {}


def _read_csv(site_path: str) -> pd.DataFrame:
    df = pd.read_csv(site_path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in {site_path}")

    for col in df.columns:
        if col == LABEL_COL:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(df.median(numeric_only=True))
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)

    return df


def get_feature_columns(site_path: str) -> List[str]:
    df = _read_csv(site_path)
    return [c for c in df.columns if c != LABEL_COL]


def split_site_dataframe(
    df: pd.DataFrame,
    seed: int,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> SiteSplit:
    y = df[LABEL_COL].to_numpy(dtype=int)
    X = df.drop(columns=[LABEL_COL]).to_numpy(dtype=np.float32)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    val_ratio_of_trainval = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio_of_trainval,
        random_state=seed,
        stratify=y_trainval,
    )

    return SiteSplit(
        X_train=X_train.astype(np.float32),
        y_train=y_train,
        X_val=X_val.astype(np.float32),
        y_val=y_val,
        X_test=X_test.astype(np.float32),
        y_test=y_test,
    )


def split_site_dataframe_df(
    df: pd.DataFrame,
    seed: int,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> SiteSplitDF:
    trainval_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[LABEL_COL],
    )

    val_ratio_of_trainval = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_ratio_of_trainval,
        random_state=seed,
        stratify=trainval_df[LABEL_COL],
    )

    return SiteSplitDF(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )


def get_site_split_cached(
    site_path: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> SiteSplit:
    key = f"{site_path}::seed={seed}::test={test_size}::val={val_size}"
    if key in _SPLITS_CACHE:
        return _SPLITS_CACHE[key]

    df = _read_csv(site_path)
    split = split_site_dataframe(df, seed=seed, test_size=test_size, val_size=val_size)
    _SPLITS_CACHE[key] = split
    return split


def get_site_split_df_cached(
    site_path: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> SiteSplitDF:
    key = f"{site_path}::seed={seed}::test={test_size}::val={val_size}::df"
    if key in _SPLITS_DF_CACHE:
        return _SPLITS_DF_CACHE[key]

    df = _read_csv(site_path)
    split_df = split_site_dataframe_df(df, seed=seed, test_size=test_size, val_size=val_size)
    _SPLITS_DF_CACHE[key] = split_df
    return split_df


def get_site_scaler_cached(
    site_path: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> StandardScaler:
    key = f"{site_path}::seed={seed}::test={test_size}::val={val_size}::scaler"
    if key in _SCALERS_CACHE:
        return _SCALERS_CACHE[key]

    split = get_site_split_cached(site_path, seed, test_size, val_size)

    scaler = StandardScaler()
    scaler.fit(split.X_train)

    _SCALERS_CACHE[key] = scaler
    return scaler


def _transform_split_with_scaler(split: SiteSplit, scaler: StandardScaler) -> SiteSplit:
    return SiteSplit(
        X_train=scaler.transform(split.X_train).astype(np.float32),
        y_train=split.y_train,
        X_val=scaler.transform(split.X_val).astype(np.float32),
        y_val=split.y_val,
        X_test=scaler.transform(split.X_test).astype(np.float32),
        y_test=split.y_test,
    )


def get_site_split_scaled_cached(
    site_path: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> SiteSplit:
    split = get_site_split_cached(site_path, seed=seed, test_size=test_size, val_size=val_size)
    scaler = get_site_scaler_cached(site_path, seed=seed, test_size=test_size, val_size=val_size)
    return _transform_split_with_scaler(split, scaler)


def get_kfold_train_val_splits_scaled(
    site_path: str,
    seed: int,
    test_size: float,
    val_size: float,
    k_folds: int = 5,
):
    """
    K-Fold is applied ONLY on the training set.

    Important:
    - unseen test is not used.
    - holdout validation is not used.
    - each fold gets its own scaler fitted only on that fold's training part.
    """
    split = get_site_split_cached(
        site_path=site_path,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
    )

    X_base = split.X_train
    y_base = split.y_train

    skf = StratifiedKFold(
        n_splits=int(k_folds),
        shuffle=True,
        random_state=seed,
    )

    folds = []
    for train_idx, val_idx in skf.split(X_base, y_base):
        X_train_raw = X_base[train_idx]
        y_train_f = y_base[train_idx]

        X_val_raw = X_base[val_idx]
        y_val_f = y_base[val_idx]

        fold_scaler = StandardScaler()
        fold_scaler.fit(X_train_raw)

        X_train_f = fold_scaler.transform(X_train_raw).astype(np.float32)
        X_val_f = fold_scaler.transform(X_val_raw).astype(np.float32)

        folds.append((X_train_f, y_train_f, X_val_f, y_val_f))

    return folds


def to_dmatrix(X: np.ndarray, y: np.ndarray | None = None) -> xgb.DMatrix:
    if y is None:
        return xgb.DMatrix(X)
    return xgb.DMatrix(X, label=y)


def load_data_for_client(
    site_paths: List[str],
    partition_id: int,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[xgb.DMatrix, xgb.DMatrix, int, int]:
    if partition_id < 0 or partition_id >= len(site_paths):
        raise ValueError(f"partition_id={partition_id} out of range for {len(site_paths)} sites")

    split = get_site_split_cached(
        site_path=site_paths[partition_id],
        seed=seed,
        test_size=test_size,
        val_size=val_size,
    )

    dtrain = to_dmatrix(split.X_train, split.y_train)
    dval = to_dmatrix(split.X_val, split.y_val)
    return dtrain, dval, split.X_train.shape[0], split.X_val.shape[0]


def load_data_for_client_scaled(
    site_paths: List[str],
    partition_id: int,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[xgb.DMatrix, xgb.DMatrix, int, int]:
    if partition_id < 0 or partition_id >= len(site_paths):
        raise ValueError(f"partition_id={partition_id} out of range for {len(site_paths)} sites")

    split = get_site_split_scaled_cached(
        site_path=site_paths[partition_id],
        seed=seed,
        test_size=test_size,
        val_size=val_size,
    )

    dtrain = to_dmatrix(split.X_train, split.y_train)
    dval = to_dmatrix(split.X_val, split.y_val)
    return dtrain, dval, split.X_train.shape[0], split.X_val.shape[0]


def load_test_for_site(
    site_path: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    split = get_site_split_cached(site_path, seed=seed, test_size=test_size, val_size=val_size)
    return split.X_test, split.y_test


def load_val_for_site(
    site_path: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    split = get_site_split_cached(site_path, seed=seed, test_size=test_size, val_size=val_size)
    return split.X_val, split.y_val


def load_train_for_site(
    site_path: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    split = get_site_split_cached(site_path, seed=seed, test_size=test_size, val_size=val_size)
    return split.X_train, split.y_train


def load_test_for_site_scaled(
    site_path: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    split = get_site_split_scaled_cached(site_path, seed=seed, test_size=test_size, val_size=val_size)
    return split.X_test, split.y_test


def load_val_for_site_scaled(
    site_path: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    split = get_site_split_scaled_cached(site_path, seed=seed, test_size=test_size, val_size=val_size)
    return split.X_val, split.y_val


def load_train_for_site_scaled(
    site_path: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    split = get_site_split_scaled_cached(site_path, seed=seed, test_size=test_size, val_size=val_size)
    return split.X_train, split.y_train


def load_train_df_for_site(
    site_path: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> pd.DataFrame:
    split_df = get_site_split_df_cached(site_path, seed, test_size, val_size)
    return split_df.train_df.copy()


def load_val_df_for_site(
    site_path: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> pd.DataFrame:
    split_df = get_site_split_df_cached(site_path, seed, test_size, val_size)
    return split_df.val_df.copy()


def load_test_df_for_site(
    site_path: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> pd.DataFrame:
    split_df = get_site_split_df_cached(site_path, seed, test_size, val_size)
    return split_df.test_df.copy()


def replace_keys(input_dict, match="-", target="_"):
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict