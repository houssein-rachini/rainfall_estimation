from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from model_utils import balanced_group_kfold_splits


@dataclass
class PipelineData:
    df: pd.DataFrame


def _ensure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("bias_residual_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    return logger


def _to_month_start(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()


def _month_cyclic_features(month_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    month = pd.to_numeric(month_series, errors="coerce")
    radians = 2.0 * np.pi * month / 12.0
    return np.sin(radians), np.cos(radians)


def load_data(
    station_csv_path: str,
    date_col: str = "Date",
    station_col: str = "Station",
    target_col: str = "GROUND",
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    dem_col: str = "DEM",
    imerg_col: str = "IMERG(mm/month)",
    chirps_col: str = "Chirps",
    ndvi_col: str = "NDVI",
    lst_col: str = "LST",
) -> PipelineData:
    df = pd.read_csv(station_csv_path, encoding="utf-8", encoding_errors="replace")
    df.columns = [c.strip() for c in df.columns]

    for c in [station_col, target_col, lat_col, lon_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required station column: {c}")
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col}")

    df[date_col] = _to_month_start(df[date_col])
    df["Year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
    df["Month"] = pd.to_datetime(df[date_col], errors="coerce").dt.month
    df["month_sin"], df["month_cos"] = _month_cyclic_features(df["Month"])
    if "lon_sin" not in df.columns or "lon_cos" not in df.columns:
        lon_rad = np.deg2rad(pd.to_numeric(df[lon_col], errors="coerce"))
        df["lon_sin"] = np.sin(lon_rad)
        df["lon_cos"] = np.cos(lon_rad)
    if "lat_sin" not in df.columns or "lat_cos" not in df.columns:
        lat_rad = np.deg2rad(pd.to_numeric(df[lat_col], errors="coerce"))
        df["lat_sin"] = np.sin(lat_rad)
        df["lat_cos"] = np.cos(lat_rad)

    return PipelineData(df=df)


def _build_bias_features(sensor_col: str, bias_aux_features: List[str]) -> List[str]:
    feats = [sensor_col]
    for f in bias_aux_features:
        if f != sensor_col and f not in feats:
            feats.append(f)
    return feats


def _make_bias_regressor(
    model_type: str, tweedie_power: float, tweedie_alpha: float
) -> Any:
    if model_type == "tweedie":
        return TweedieRegressor(
            power=tweedie_power, alpha=tweedie_alpha, link="log", max_iter=1000
        )
    if model_type == "hist_gbr":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=4,
            max_iter=400,
            min_samples_leaf=20,
            l2_regularization=0.01,
            random_state=42,
        )
    if model_type == "ols":
        return LinearRegression()
    raise ValueError(f"Unsupported bias model type: {model_type}")


def compute_bias_model(
    train_df: pd.DataFrame,
    sensor_col: str,
    target_col: str = "GROUND",
    bias_aux_features: Optional[List[str]] = None,
    model_type: str = "tweedie",
    tweedie_power: float = 1.35,
    tweedie_alpha: float = 0.001,
) -> Tuple[Any, List[str], int]:
    bias_features = _build_bias_features(sensor_col, bias_aux_features or ["month_sin", "month_cos"])
    required = bias_features + [target_col]
    clean = train_df.dropna(subset=required).copy()
    n_fit = len(clean)
    if n_fit == 0:
        raise ValueError(
            f"No rows available to fit {sensor_col} bias model after dropping NA."
        )

    reg = _make_bias_regressor(model_type, tweedie_power, tweedie_alpha)
    y_fit = clean[target_col].to_numpy(dtype="float64")
    if model_type == "tweedie":
        y_fit = np.maximum(y_fit, 0.0) + 1e-6
    reg.fit(clean[bias_features], y_fit)
    return reg, bias_features, n_fit


def _predict_with_na_mask(
    model: Any, df: pd.DataFrame, feature_cols: List[str]
) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        return out
    valid_mask = df[feature_cols].notna().all(axis=1)
    if valid_mask.any():
        preds = model.predict(df.loc[valid_mask, feature_cols])
        out.loc[valid_mask] = np.maximum(np.asarray(preds, dtype="float64"), 0.0)
    return out


def apply_correction(
    df: pd.DataFrame,
    imerg_bias_model: Any,
    imerg_bias_features: List[str],
    chirps_bias_model: Any,
    chirps_bias_features: List[str],
) -> pd.DataFrame:
    out = df.copy()
    out["corrected_IMERG"] = _predict_with_na_mask(
        imerg_bias_model, out, imerg_bias_features
    )
    out["corrected_CHIRPS"] = _predict_with_na_mask(
        chirps_bias_model, out, chirps_bias_features
    )

    both = out[["corrected_IMERG", "corrected_CHIRPS"]].notna().all(axis=1)
    out["corrected_baseline"] = np.nan
    out.loc[both, "corrected_baseline"] = (
        out.loc[both, "corrected_IMERG"] + out.loc[both, "corrected_CHIRPS"]
    ) / 2.0
    return out


def _build_fusion_features(
    optional_features: List[str], df_cols: List[str]
) -> List[str]:
    mandatory = ["corrected_IMERG", "corrected_CHIRPS"]
    final = mandatory.copy()
    for f in optional_features:
        if f in df_cols and f not in final:
            final.append(f)
    return final


def _set_fusion_satellite_inputs(df: pd.DataFrame, satellite_source: str) -> pd.DataFrame:
    out = df.copy()
    if satellite_source == "raw":
        out["corrected_IMERG"] = out["IMERG(mm/month)"]
        out["corrected_CHIRPS"] = out["Chirps"]
    return out


def train_fusion_model(
    train_df: pd.DataFrame,
    fusion_features: List[str],
    objective: str,
    tweedie_variance_power: float,
    xgb_params: Dict[str, Any],
) -> Tuple[XGBRegressor, int]:
    clean = train_df.dropna(subset=["GROUND"] + fusion_features).copy()
    n_fit = len(clean)
    if n_fit == 0:
        raise ValueError("No rows available to fit fusion model after dropping NA.")

    params = {"objective": objective, "random_state": 42, **xgb_params}
    if objective == "reg:tweedie":
        params["tweedie_variance_power"] = tweedie_variance_power

    model = XGBRegressor(**params)
    model.fit(clean[fusion_features], clean["GROUND"])
    return model, n_fit


def predict_fusion_model(
    model: XGBRegressor, df: pd.DataFrame, fusion_features: List[str]
) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    valid_mask = df[fusion_features].notna().all(axis=1)
    if valid_mask.any():
        preds = model.predict(df.loc[valid_mask, fusion_features])
        out.loc[valid_mask] = np.maximum(np.asarray(preds, dtype="float64"), 0.0)
    return out


def _metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "N": 0}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": r2,
        "N": int(len(y_true)),
    }


def evaluate_results(
    df: pd.DataFrame, y_true_col: str, pred_col: str
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    clean = df.dropna(subset=[y_true_col, pred_col, "Station"]).copy()
    if clean.empty:
        return pd.DataFrame(), {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    station_metrics = []
    for st_name, g in clean.groupby("Station"):
        m = _metric_block(g[y_true_col].to_numpy(), g[pred_col].to_numpy())
        station_metrics.append(
            {
                "Station": st_name,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "R2": m["R2"],
                "N": m["N"],
            }
        )

    per_station = pd.DataFrame(station_metrics).sort_values("Station")
    agg_m = _metric_block(clean[y_true_col].to_numpy(), clean[pred_col].to_numpy())
    agg = {"MAE": agg_m["MAE"], "RMSE": agg_m["RMSE"], "R2": agg_m["R2"]}
    return per_station, agg


def compute_baseline_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    def pair(pred_col: str) -> Dict[str, float]:
        clean = df.dropna(subset=["GROUND", pred_col]).copy()
        return _metric_block(clean["GROUND"].to_numpy(), clean[pred_col].to_numpy())

    return {
        "raw_imerg_vs_ground": pair("IMERG(mm/month)"),
        "raw_chirps_vs_ground": pair("Chirps"),
        "corrected_imerg_vs_ground": pair("corrected_IMERG"),
        "corrected_chirps_vs_ground": pair("corrected_CHIRPS"),
        "corrected_baseline_vs_ground": pair("corrected_baseline"),
        "fusion_final_vs_ground": pair("final_prediction"),
    }


def compute_satellite_source_comparison(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for label, col in [
        ("fusion_bias_corrected", "final_prediction_bias_corrected"),
        ("fusion_raw", "final_prediction_raw"),
    ]:
        if col in df.columns:
            clean = df.dropna(subset=["GROUND", col]).copy()
            out[label] = _metric_block(
                clean["GROUND"].to_numpy(dtype="float64"),
                clean[col].to_numpy(dtype="float64"),
            )
    if "fusion_bias_corrected" in out and "fusion_raw" in out:
        out["delta_mae_raw_minus_corrected"] = {
            "value": out["fusion_raw"]["MAE"] - out["fusion_bias_corrected"]["MAE"]
        }
    return out


def _paired_significance_from_predictions(
    df: pd.DataFrame,
    pred_col_a: str,
    pred_col_b: str,
    target_col: str = "GROUND",
) -> Dict[str, float]:
    clean = df.dropna(subset=[target_col, pred_col_a, pred_col_b]).copy()
    n = int(len(clean))
    if n == 0:
        return {
            "mean_error_model_a": np.nan,
            "mean_error_model_b": np.nan,
            "mean_difference": np.nan,
            "paired_ttest_pvalue": np.nan,
            "wilcoxon_pvalue": np.nan,
            "N": 0,
        }

    err_a = np.abs(clean[target_col].to_numpy(dtype="float64") - clean[pred_col_a].to_numpy(dtype="float64"))
    err_b = np.abs(clean[target_col].to_numpy(dtype="float64") - clean[pred_col_b].to_numpy(dtype="float64"))
    diff = err_a - err_b

    ttest_p = np.nan
    wilcoxon_p = np.nan
    if n >= 2:
        from scipy.stats import ttest_rel, wilcoxon

        ttest_p = float(ttest_rel(err_a, err_b, nan_policy="omit").pvalue)
        if np.allclose(diff, 0.0):
            wilcoxon_p = 1.0
        else:
            wilcoxon_p = float(wilcoxon(err_a, err_b).pvalue)

    return {
        "mean_error_model_a": float(np.mean(err_a)),
        "mean_error_model_b": float(np.mean(err_b)),
        "mean_difference": float(np.mean(diff)),
        "paired_ttest_pvalue": ttest_p,
        "wilcoxon_pvalue": wilcoxon_p,
        "N": n,
    }


def compute_significance_tests(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    return {
        "raw_imerg_vs_corrected_imerg": _paired_significance_from_predictions(
            df, "IMERG(mm/month)", "corrected_IMERG"
        ),
        "raw_chirps_vs_corrected_chirps": _paired_significance_from_predictions(
            df, "Chirps", "corrected_CHIRPS"
        ),
        "corrected_baseline_vs_final_prediction": _paired_significance_from_predictions(
            df, "corrected_baseline", "final_prediction"
        ),
    }


def _annotated_scatter(
    ax: Any, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str
) -> None:
    ax.scatter(x, y, s=10, alpha=0.4)
    lim_min = float(min(np.nanmin(x), np.nanmin(y)))
    lim_max = float(max(np.nanmax(x), np.nanmax(y)))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1)

    if len(x) >= 2 and np.nanstd(x) > 0:
        m, b = np.polyfit(x, y, 1)
        y_fit = m * x + b
        ss_res = float(np.sum((y - y_fit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        ax.plot(
            [lim_min, lim_max],
            [m * lim_min + b, m * lim_max + b],
            color="black",
            linewidth=1,
        )
        ax.text(
            0.02,
            -0.22,
            f"fit: y = {m:.3f}x + {b:.3f} | R2 = {r2:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _plot_before_after_correction(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    plots = [
        (
            "IMERG(mm/month)",
            "GROUND",
            "IMERG vs GROUND (Before)",
            "IMERG(mm/month)",
            "GROUND",
        ),
        (
            "corrected_IMERG",
            "GROUND",
            "IMERG vs GROUND (After)",
            "corrected_IMERG",
            "GROUND",
        ),
        ("Chirps", "GROUND", "CHIRPS vs GROUND (Before)", "Chirps", "GROUND"),
        (
            "corrected_CHIRPS",
            "GROUND",
            "CHIRPS vs GROUND (After)",
            "corrected_CHIRPS",
            "GROUND",
        ),
        (
            "corrected_baseline",
            "GROUND",
            "Baseline vs GROUND",
            "corrected_baseline",
            "GROUND",
        ),
        (
            "final_prediction",
            "GROUND",
            "Fusion Final vs GROUND",
            "final_prediction",
            "GROUND",
        ),
    ]

    for ax, (x_col, y_col, title, xlabel, ylabel) in zip(axes.flatten(), plots):
        clean = df.dropna(subset=[x_col, y_col])
        if clean.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No valid rows", ha="center", va="center")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            continue
        _annotated_scatter(
            ax,
            clean[x_col].to_numpy(dtype="float64"),
            clean[y_col].to_numpy(dtype="float64"),
            title,
            xlabel,
            ylabel,
        )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    st.pyplot(fig, width="content")


def run_bias_fusion_cv(
    data: PipelineData,
    out_dir: str,
    n_splits: int = 5,
    group_by_station: bool = True,
    min_rows_per_fold: int = 50,
    bias_model_type: str = "tweedie",
    bias_aux_features: Optional[List[str]] = None,
    tweedie_power: float = 1.35,
    tweedie_alpha: float = 0.001,
    fusion_objective: str = "reg:squarederror",
    fusion_tweedie_variance_power: float = 1.3,
    fusion_satellite_mode: str = "bias_corrected",
    fusion_optional_features: Optional[List[str]] = None,
    fusion_xgb_params: Optional[Dict[str, Any]] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, float],
    Dict[str, Dict[str, float]],
    pd.DataFrame,
]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger = _ensure_logger(out_path / "bias_residual_pipeline.log")
    logger.info("Starting bias-correction + fusion CV pipeline")

    df = data.df.copy()
    selected_bias_aux = bias_aux_features or [
        "month_sin",
        "month_cos",
        "DEM",
        "Slope",
        "Latitude",
        "Longitude",
    ]
    required_cols = ["Date", "Station", "GROUND", "IMERG(mm/month)", "Chirps"] + selected_bias_aux
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    df = df.dropna(subset=["Date", "Station", "GROUND"]).reset_index(drop=True)
    if group_by_station:
        split_iter = balanced_group_kfold_splits(
            df["Station"].astype(str), n_splits=n_splits, random_state=42
        )
    else:
        split_iter = KFold(n_splits=n_splits, shuffle=True, random_state=42).split(
            df, df["GROUND"]
        )

    optional = fusion_optional_features or []
    xgb_params = fusion_xgb_params or {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }

    fold_outputs: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, Any]] = []

    for fold_id, (tr_idx, te_idx) in enumerate(split_iter, start=1):
        train_df = df.iloc[tr_idx].copy()
        test_df = df.iloc[te_idx].copy()
        logger.info("Fold %s | train=%s test=%s", fold_id, len(train_df), len(test_df))

        fold_cfg = {
            "fold": fold_id,
            "bias_model_type": bias_model_type,
            "bias_aux_features": selected_bias_aux,
            "bias_tweedie_power": float(tweedie_power),
            "bias_tweedie_alpha": float(tweedie_alpha),
            "fusion_objective": fusion_objective,
            "fusion_tweedie_variance_power": float(fusion_tweedie_variance_power),
            "fusion_satellite_mode": fusion_satellite_mode,
            "fusion_optional_features": optional,
            "fusion_xgb_params": xgb_params,
            "min_rows_per_fold": int(min_rows_per_fold),
        }

        skip_reason = None
        try:
            imerg_bias_model, imerg_bias_features, n_im_fit = compute_bias_model(
                train_df,
                sensor_col="IMERG(mm/month)",
                target_col="GROUND",
                bias_aux_features=selected_bias_aux,
                model_type=bias_model_type,
                tweedie_power=tweedie_power,
                tweedie_alpha=tweedie_alpha,
            )
            chirps_bias_model, chirps_bias_features, n_ch_fit = compute_bias_model(
                train_df,
                sensor_col="Chirps",
                target_col="GROUND",
                bias_aux_features=selected_bias_aux,
                model_type=bias_model_type,
                tweedie_power=tweedie_power,
                tweedie_alpha=tweedie_alpha,
            )
            logger.info(
                "Fold %s | bias fit rows imerg=%s chirps=%s",
                fold_id,
                n_im_fit,
                n_ch_fit,
            )
            if n_im_fit < min_rows_per_fold or n_ch_fit < min_rows_per_fold:
                skip_reason = (
                    f"Too few bias training rows (IMERG={n_im_fit}, CHIRPS={n_ch_fit})"
                )
                raise ValueError(skip_reason)

            train_corr = apply_correction(
                train_df,
                imerg_bias_model,
                imerg_bias_features,
                chirps_bias_model,
                chirps_bias_features,
            )
            test_corr = apply_correction(
                test_df,
                imerg_bias_model,
                imerg_bias_features,
                chirps_bias_model,
                chirps_bias_features,
            )

            if fusion_satellite_mode not in ["bias_corrected", "raw", "compare_both"]:
                raise ValueError(
                    "fusion_satellite_mode must be one of: bias_corrected, raw, compare_both"
                )

            sat_sources = (
                ["bias_corrected", "raw"]
                if fusion_satellite_mode == "compare_both"
                else [fusion_satellite_mode]
            )

            fusion_models: Dict[str, XGBRegressor] = {}
            fusion_features_used: Dict[str, List[str]] = {}
            fusion_fit_counts: Dict[str, int] = {}
            fusion_pred_counts: Dict[str, int] = {}
            for sat_source in sat_sources:
                train_fusion = _set_fusion_satellite_inputs(train_corr, sat_source)
                test_fusion = _set_fusion_satellite_inputs(test_corr, sat_source)

                fusion_features = _build_fusion_features(optional, list(train_fusion.columns))
                fusion_model, n_fusion_fit = train_fusion_model(
                    train_fusion,
                    fusion_features=fusion_features,
                    objective=fusion_objective,
                    tweedie_variance_power=fusion_tweedie_variance_power,
                    xgb_params=xgb_params,
                )
                logger.info(
                    "Fold %s | fusion(%s) fit rows=%s",
                    fold_id,
                    sat_source,
                    n_fusion_fit,
                )
                if n_fusion_fit < min_rows_per_fold:
                    skip_reason = f"Too few fusion training rows ({n_fusion_fit}) for {sat_source}"
                    raise ValueError(skip_reason)

                pred_col = (
                    "final_prediction"
                    if fusion_satellite_mode != "compare_both"
                    else f"final_prediction_{sat_source}"
                )
                test_corr[pred_col] = predict_fusion_model(
                    fusion_model, test_fusion, fusion_features
                )
                n_fusion_pred = int(test_corr[pred_col].notna().sum())
                logger.info(
                    "Fold %s | fusion(%s) predicted rows=%s",
                    fold_id,
                    sat_source,
                    n_fusion_pred,
                )
                if n_fusion_pred < min_rows_per_fold:
                    logger.warning(
                        "Fold %s | low prediction coverage for %s (%s rows)",
                        fold_id,
                        sat_source,
                        n_fusion_pred,
                    )
                fusion_models[sat_source] = fusion_model
                fusion_features_used[sat_source] = fusion_features
                fusion_fit_counts[sat_source] = int(n_fusion_fit)
                fusion_pred_counts[sat_source] = int(n_fusion_pred)

            if fusion_satellite_mode == "compare_both":
                test_corr["final_prediction"] = test_corr["final_prediction_bias_corrected"]
                n_fusion_fit = fusion_fit_counts["bias_corrected"]
                n_fusion_pred = fusion_pred_counts["bias_corrected"]
            else:
                n_fusion_fit = fusion_fit_counts[sat_sources[0]]
                n_fusion_pred = fusion_pred_counts[sat_sources[0]]

            joblib.dump(
                imerg_bias_model, out_path / f"fold_{fold_id}_bias_imerg_model.pkl"
            )
            joblib.dump(
                chirps_bias_model, out_path / f"fold_{fold_id}_bias_chirps_model.pkl"
            )
            if fusion_satellite_mode == "compare_both":
                for sat_source in sat_sources:
                    joblib.dump(
                        fusion_models[sat_source],
                        out_path / f"fold_{fold_id}_fusion_xgb_model_{sat_source}.pkl",
                    )
                    with open(
                        out_path / f"fold_{fold_id}_fusion_features_{sat_source}.json",
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(fusion_features_used[sat_source], f, indent=2)
            else:
                joblib.dump(
                    fusion_models[sat_sources[0]],
                    out_path / f"fold_{fold_id}_fusion_xgb_model.pkl",
                )
                with open(
                    out_path / f"fold_{fold_id}_fusion_features.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(fusion_features_used[sat_sources[0]], f, indent=2)

            fold_cfg.update(
                {
                    "imerg_bias_features": imerg_bias_features,
                    "chirps_bias_features": chirps_bias_features,
                    "fusion_features": fusion_features_used,
                    "n_imerg_bias_fit": int(n_im_fit),
                    "n_chirps_bias_fit": int(n_ch_fit),
                    "n_fusion_fit": int(n_fusion_fit),
                    "n_fusion_pred": int(n_fusion_pred),
                    "status": "ok",
                }
            )

        except Exception as e:
            if skip_reason is None:
                skip_reason = str(e)
            logger.warning("Fold %s skipped: %s", fold_id, skip_reason)
            test_corr = test_df.copy()
            for col in [
                "corrected_IMERG",
                "corrected_CHIRPS",
                "corrected_baseline",
                "final_prediction",
            ]:
                test_corr[col] = np.nan
            fold_cfg.update({"status": "skipped", "skip_reason": skip_reason})

        with open(out_path / f"fold_{fold_id}_config.json", "w", encoding="utf-8") as f:
            json.dump(fold_cfg, f, indent=2)

        fold_eval = test_corr.dropna(subset=["GROUND", "final_prediction"])
        fm = _metric_block(
            fold_eval["GROUND"].to_numpy(), fold_eval["final_prediction"].to_numpy()
        )
        fm.update({"fold": fold_id, "status": fold_cfg.get("status", "ok")})
        if fold_cfg.get("status") == "skipped":
            fm["skip_reason"] = fold_cfg.get("skip_reason")
        fold_metrics.append(fm)
        fold_outputs.append(test_corr)

    results_df = (
        pd.concat(fold_outputs, ignore_index=True) if fold_outputs else pd.DataFrame()
    )
    fold_metrics_df = (
        pd.DataFrame(fold_metrics).sort_values("fold")
        if fold_metrics
        else pd.DataFrame()
    )
    per_station_df, agg = evaluate_results(results_df, "GROUND", "final_prediction")
    baseline_metrics = (
        compute_baseline_metrics(results_df)
        if not results_df.empty
        else {
            "raw_imerg_vs_ground": {
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "N": 0,
            },
            "raw_chirps_vs_ground": {
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "N": 0,
            },
            "corrected_imerg_vs_ground": {
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "N": 0,
            },
            "corrected_chirps_vs_ground": {
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "N": 0,
            },
            "corrected_baseline_vs_ground": {
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "N": 0,
            },
            "fusion_final_vs_ground": {
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "N": 0,
            },
        }
    )
    significance_tests = (
        compute_significance_tests(results_df)
        if not results_df.empty
        else {
            "raw_imerg_vs_corrected_imerg": {
                "mean_error_model_a": np.nan,
                "mean_error_model_b": np.nan,
                "mean_difference": np.nan,
                "paired_ttest_pvalue": np.nan,
                "wilcoxon_pvalue": np.nan,
                "N": 0,
            },
            "raw_chirps_vs_corrected_chirps": {
                "mean_error_model_a": np.nan,
                "mean_error_model_b": np.nan,
                "mean_difference": np.nan,
                "paired_ttest_pvalue": np.nan,
                "wilcoxon_pvalue": np.nan,
                "N": 0,
            },
            "corrected_baseline_vs_final_prediction": {
                "mean_error_model_a": np.nan,
                "mean_error_model_b": np.nan,
                "mean_difference": np.nan,
                "paired_ttest_pvalue": np.nan,
                "wilcoxon_pvalue": np.nan,
                "N": 0,
            },
        }
    )
    source_comparison = (
        compute_satellite_source_comparison(results_df) if not results_df.empty else {}
    )

    with open(out_path / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    with open(out_path / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, indent=2)
    with open(out_path / "significance_tests.json", "w", encoding="utf-8") as f:
        json.dump(significance_tests, f, indent=2)
    with open(out_path / "satellite_source_comparison.json", "w", encoding="utf-8") as f:
        json.dump(source_comparison, f, indent=2)
    fold_metrics_df.to_csv(out_path / "fold_metrics.csv", index=False)
    per_station_df.to_csv(out_path / "per_station_metrics.csv", index=False)
    results_df.to_csv(out_path / "cv_predictions.csv", index=False)

    logger.info("Finished. Aggregate metrics (fusion final): %s", agg)
    return fold_metrics_df, per_station_df, agg, baseline_metrics, results_df


def show_bias_residual_tab() -> None:
    st.title("Bias Correction + Fusion Learning")
    st.caption(
        "Paper-style pipeline: fold-wise gauge bias-correction + direct fusion XGBoost."
    )

    station_csv = st.text_input(
        "Station CSV path",
        value="final_merged_with_ndvi_imerg_no_leakage.csv",
        key="br_station_csv",
    )
    out_dir = st.text_input(
        "Output directory", value="artifacts_bias_residual", key="br_out_dir"
    )

    st.subheader("CV Settings")
    group_by_station = st.checkbox(
        "GroupKFold by Station (recommended)", value=True, key="br_group_station"
    )
    n_splits = st.slider("Number of folds", 2, 10, 5, key="br_n_splits")
    min_rows_per_fold = st.slider(
        "Minimum rows required per fold step",
        10,
        1000,
        500,
        10,
        key="br_min_rows_per_fold",
    )

    st.subheader("Bias Model")
    bias_model_label = st.selectbox(
        "Bias model",
        options=["Tweedie (recommended for rainfall)", "HistGradientBoosting", "OLS"],
        index=0,
        key="br_bias_model",
    )
    bias_model_type = {
        "Tweedie (recommended for rainfall)": "tweedie",
        "HistGradientBoosting": "hist_gbr",
        "OLS": "ols",
    }[bias_model_label]
    bias_aux_candidates = [
        "month_sin",
        "month_cos",
        "DEM",
        "Slope",
        "Latitude",
        "Longitude",
        "lat_sin",
        "lat_cos",
        "lon_sin",
        "lon_cos",
        "NDVI",
        "LST",
    ]
    bias_aux_features = st.multiselect(
        "Bias auxiliary features (sensor is always included)",
        options=bias_aux_candidates,
        default=["month_sin", "month_cos", "DEM", "Slope", "Latitude", "Longitude"],
        key="br_bias_aux_features",
    )
    tweedie_power = st.slider(
        "Bias Tweedie power (1 < p < 2)",
        min_value=1.05,
        max_value=1.95,
        value=1.35,
        step=0.05,
        key="br_tweedie_power",
        disabled=bias_model_type != "tweedie",
    )
    tweedie_alpha = st.number_input(
        "Bias Tweedie alpha (regularization)",
        min_value=0.0,
        value=0.001,
        step=0.001,
        format="%.4f",
        key="br_tweedie_alpha",
        disabled=bias_model_type != "tweedie",
    )

    st.subheader("Fusion Model")
    fusion_objective = st.selectbox(
        "Fusion objective",
        options=["reg:squarederror", "reg:tweedie"],
        index=0,
        key="br_fusion_objective",
    )
    fusion_satellite_mode = st.selectbox(
        "Fusion satellite inputs",
        options=["bias_corrected", "raw", "compare_both"],
        index=0,
        key="br_fusion_satellite_mode",
        help="compare_both trains two fusion models (one with corrected inputs, one with raw inputs) for direct comparison.",
    )
    fusion_tweedie_variance_power = st.slider(
        "Fusion Tweedie variance power",
        min_value=1.1,
        max_value=1.9,
        value=1.3,
        step=0.1,
        key="br_fusion_tweedie_power",
        disabled=fusion_objective != "reg:tweedie",
    )

    optional_candidates = [
        "NDVI",
        "LST",
        "DEM",
        "Slope",
        "Latitude",
        "Longitude",
        "lat_sin",
        "lat_cos",
        "lon_sin",
        "lon_cos",
        "month_sin",
        "month_cos",
    ]
    default_optional = [
        "DEM",
        "Slope",
        "Latitude",
        "Longitude",
        "lat_sin",
        "lat_cos",
        "lon_sin",
        "lon_cos",
        "month_sin",
        "month_cos",
    ]
    fusion_optional_features = st.multiselect(
        "Optional fusion features (corrected_IMERG and corrected_CHIRPS are always included)",
        options=optional_candidates,
        default=default_optional,
        key="br_fusion_optional_features",
    )

    st.write("XGBoost hyperparameters")
    xgb_n_estimators = st.slider(
        "n_estimators", 50, 1000, 300, 10, key="br_xgb_n_estimators"
    )
    xgb_learning_rate = st.slider(
        "learning_rate", 0.01, 0.5, 0.05, 0.01, key="br_xgb_learning_rate"
    )
    xgb_max_depth = st.slider("max_depth", 2, 12, 3, 1, key="br_xgb_max_depth")
    xgb_min_child_weight = st.slider(
        "min_child_weight", 1, 20, 5, 1, key="br_xgb_min_child_weight"
    )
    xgb_subsample = st.slider("subsample", 0.3, 1.0, 0.8, 0.05, key="br_xgb_subsample")
    xgb_colsample_bytree = st.slider(
        "colsample_bytree", 0.3, 1.0, 0.8, 0.05, key="br_xgb_colsample_bytree"
    )
    xgb_reg_alpha = st.number_input(
        "reg_alpha (L1)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        key="br_xgb_reg_alpha",
    )
    xgb_reg_lambda = st.number_input(
        "reg_lambda (L2)",
        min_value=0.0,
        max_value=20.0,
        value=1.0,
        step=0.1,
        key="br_xgb_reg_lambda",
    )

    if st.button("Run Bias + Fusion Pipeline", key="br_run_btn"):
        fusion_xgb_params = {
            "n_estimators": int(xgb_n_estimators),
            "learning_rate": float(xgb_learning_rate),
            "max_depth": int(xgb_max_depth),
            "min_child_weight": int(xgb_min_child_weight),
            "subsample": float(xgb_subsample),
            "colsample_bytree": float(xgb_colsample_bytree),
            "reg_alpha": float(xgb_reg_alpha),
            "reg_lambda": float(xgb_reg_lambda),
        }

        config = {
            "station_csv_path": station_csv,
            "out_dir": out_dir,
            "n_splits": int(n_splits),
            "group_by_station": bool(group_by_station),
            "min_rows_per_fold": int(min_rows_per_fold),
            "bias_model_type": bias_model_type,
            "bias_aux_features": bias_aux_features,
            "bias_tweedie_power": float(tweedie_power),
            "bias_tweedie_alpha": float(tweedie_alpha),
            "fusion_objective": fusion_objective,
            "fusion_satellite_mode": fusion_satellite_mode,
            "fusion_tweedie_variance_power": float(fusion_tweedie_variance_power),
            "fusion_optional_features": fusion_optional_features,
            "fusion_xgb_params": fusion_xgb_params,
            "na_policy": "No imputation/fill; use complete-case rows per step.",
            "data_source": "CSV only (GEE-derived fields should be precomputed in CSV).",
        }

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        with st.spinner("Running pipeline..."):
            try:
                data = load_data(station_csv_path=station_csv)
                fold_df, station_df, agg, baseline_metrics, results_df = (
                    run_bias_fusion_cv(
                        data=data,
                        out_dir=out_dir,
                        n_splits=n_splits,
                    group_by_station=group_by_station,
                    min_rows_per_fold=min_rows_per_fold,
                        bias_model_type=bias_model_type,
                        bias_aux_features=bias_aux_features,
                        tweedie_power=float(tweedie_power),
                        tweedie_alpha=float(tweedie_alpha),
                        fusion_objective=fusion_objective,
                        fusion_satellite_mode=fusion_satellite_mode,
                        fusion_tweedie_variance_power=float(
                            fusion_tweedie_variance_power
                        ),
                        fusion_optional_features=fusion_optional_features,
                        fusion_xgb_params=fusion_xgb_params,
                    )
                )
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                return

        skipped = (
            fold_df[fold_df.get("status", "") == "skipped"]
            if not fold_df.empty
            else pd.DataFrame()
        )
        if not skipped.empty:
            st.warning(
                f"Skipped folds due to insufficient complete-case rows: {len(skipped)}"
            )

        st.success("Pipeline complete.")
        st.write("### Aggregate Metrics (Fusion Final)")
        st.json(agg)
        st.write("### Baseline Metrics")
        st.json(baseline_metrics)
        st.write("### Statistical Significance Tests")
        st.json(compute_significance_tests(results_df))
        if fusion_satellite_mode == "compare_both":
            st.write("### Raw vs Bias-Corrected Fusion Comparison")
            st.json(compute_satellite_source_comparison(results_df))
        st.write("### Fold Metrics")
        st.dataframe(fold_df, width="stretch")
        st.write("### Per-Station Metrics")
        st.dataframe(station_df, width="stretch")
        st.write("### Scatter Diagnostics")
        _plot_before_after_correction(results_df)
        st.caption(f"Artifacts saved in: {out_dir}")
