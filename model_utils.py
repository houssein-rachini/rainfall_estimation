from __future__ import annotations

from typing import Iterable, List, Tuple

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COL = "GROUND"
DATE_COL = "Date"

NUMERIC_CANDIDATES = [
    "FID",
    "Longitude",
    "Latitude",
    "lon_sin",
    "lon_cos",
    "lat_sin",
    "lat_cos",
    "DEM",
    "Chirps",
    "IMERG(mm/hr)",
    "IMERG(mm/month)",
    "Year",
    "Month",
    "month_sin",
    "month_cos",
]

CATEGORICAL_CANDIDATES = ["Station", "Source", "DEM category", "Location"]


def enrich_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal and spatial engineered columns without mutating input DataFrame."""
    enriched = df.copy()

    # Normalize categorical text values to avoid duplicate labels caused by trailing spaces.
    for col in ["Station", "Source", "DEM category", "Location"]:
        if col in enriched.columns:
            enriched[col] = (
                enriched[col]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
            )

    if DATE_COL in enriched.columns:
        enriched[DATE_COL] = pd.to_datetime(enriched[DATE_COL], errors="coerce")
        enriched["Year"] = enriched[DATE_COL].dt.year
        enriched["Month"] = enriched[DATE_COL].dt.month
        month_vals = pd.to_numeric(enriched["Month"], errors="coerce")
        month_rad = 2.0 * np.pi * month_vals / 12.0
        enriched["month_sin"] = np.sin(month_rad)
        enriched["month_cos"] = np.cos(month_rad)

    if "Longitude" in enriched.columns:
        lon_rad = np.deg2rad(pd.to_numeric(enriched["Longitude"], errors="coerce"))
        enriched["lon_sin"] = np.sin(lon_rad)
        enriched["lon_cos"] = np.cos(lon_rad)

    if "Latitude" in enriched.columns:
        lat_rad = np.deg2rad(pd.to_numeric(enriched["Latitude"], errors="coerce"))
        enriched["lat_sin"] = np.sin(lat_rad)
        enriched["lat_cos"] = np.cos(lat_rad)

    return enriched


def default_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    present_numeric = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    present_categorical = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    all_features = present_numeric + present_categorical
    return all_features, present_numeric, present_categorical


def build_preprocessor(numeric_features: Iterable[str], categorical_features: Iterable[str]) -> ColumnTransformer:
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def prepare_training_frame(df: pd.DataFrame, selected_features: Iterable[str]) -> pd.DataFrame:
    selected_features = [c for c in selected_features if c in df.columns]
    required_cols = list(dict.fromkeys(selected_features + [TARGET_COL]))
    return df[required_cols].dropna(subset=[TARGET_COL]).copy()


def align_features(input_df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    feature_columns = list(feature_columns)
    aligned = input_df.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = pd.NA
    return aligned[feature_columns]


def add_location_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Location values (if present)."""
    if "Location" not in df.columns:
        return df
    encoded = df.copy()
    encoded["Location"] = encoded["Location"].astype("string").str.strip()
    return pd.get_dummies(encoded, columns=["Location"], prefix="Location", dtype=int)


def balanced_group_kfold_splits(
    groups: Iterable,
    n_splits: int,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build leakage-safe group folds with more balanced sample counts than default GroupKFold.
    Entire groups are assigned to a single fold.
    """
    group_series = pd.Series(groups).reset_index(drop=True)
    unique_groups = group_series.dropna().unique()
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique groups, found {len(unique_groups)}."
        )

    group_sizes = group_series.value_counts(dropna=False)
    rng = np.random.default_rng(random_state)
    tie_breaker = {g: float(rng.random()) for g in group_sizes.index}
    ordered_groups = sorted(
        group_sizes.index.tolist(),
        key=lambda g: (-int(group_sizes[g]), tie_breaker[g]),
    )

    fold_groups: List[List] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for g in ordered_groups:
        fold_id = int(np.argmin(fold_sizes))
        fold_groups[fold_id].append(g)
        fold_sizes[fold_id] += int(group_sizes[g])

    all_idx = np.arange(len(group_series))
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_id in range(n_splits):
        val_mask = group_series.isin(fold_groups[fold_id]).to_numpy()
        val_idx = all_idx[val_mask]
        train_idx = all_idx[~val_mask]
        splits.append((train_idx, val_idx))
    return splits
