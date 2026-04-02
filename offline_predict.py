#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch MPI prediction (no GEE) using a trained Ensemble (DNN + XGB/RF/KNN).
Run from the directory that contains the saved artifacts:
  - ensemble_scaler.pkl
  - trained_ensemble_xgb_model.json + trained_ensemble_xgb_dnn_model.h5
    OR trained_ensemble_rf_model.pkl   + trained_ensemble_rf_dnn_model.h5
    OR trained_ensemble_knn_model.pkl  + trained_ensemble_knn_dnn_model.h5
  - (optional) ensemble_meta.json   # feature order, alpha, base model, etc.

Examples (Windows cmd.exe / PowerShell are the same):
  python predict_ensemble.py --input "your_input.csv" --output "ensemble_predictions.csv" --id-cols Country Region Year

If you DID NOT save ensemble_meta.json, pass the training features explicitly:
  python predict_ensemble.py --input "your_input.csv" --output "ensemble_predictions.csv" --id-cols Country Region Year --feature-cols StdDev_NTL Mean_GPP StdDev_Pop StdDev_LST StdDev_NDVI Mean_NTL StdDev_GPP Mean_Pop Mean_LST Mean_NDVI

Override alpha (optional):
  python predict_ensemble.py --input data.csv --output preds.csv --alpha 0.3
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# xgboost is only needed if your base model is XGBoost
try:
    import xgboost as xgb  # noqa: F401
except Exception:
    xgb = None


# --------------------------- Artifact Loading ---------------------------


def load_artifacts(base_model_hint=None):
    """Load scaler, base model, DNN, and meta (if present) from current dir."""
    # Scaler
    scaler_path = "ensemble_scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Meta (optional)
    meta = None
    if os.path.exists("ensemble_meta.json"):
        with open("ensemble_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

    # Determine base model type
    base_model = (meta and meta.get("base_model")) or base_model_hint
    if base_model is None:
        if os.path.exists("trained_ensemble_xgb_model.json"):
            base_model = "XGBoost"
        elif os.path.exists("trained_ensemble_rf_model.pkl"):
            base_model = "Random Forest"
        elif os.path.exists("trained_ensemble_knn_model.pkl"):
            base_model = "KNN Regressor"
        else:
            raise FileNotFoundError(
                "Could not determine base model type from files or meta. "
                "Pass --base-model {XGBoost|Random Forest|KNN Regressor}."
            )

    # Load base model + DNN
    if base_model == "XGBoost":
        if xgb is None:
            raise ImportError("xgboost is not installed but base model is XGBoost.")
        base = xgb.XGBRegressor()
        base.load_model("trained_ensemble_xgb_model.json")
        dnn_path = "trained_ensemble_xgb_dnn_model.h5"
    elif base_model == "Random Forest":
        base = joblib.load("trained_ensemble_rf_model.pkl")
        dnn_path = "trained_ensemble_rf_dnn_model.h5"
    elif base_model == "KNN Regressor":
        base = joblib.load("trained_ensemble_knn_model.pkl")
        dnn_path = "trained_ensemble_knn_dnn_model.h5"
    else:
        raise ValueError(
            "base_model must be one of: XGBoost, Random Forest, KNN Regressor"
        )

    if not os.path.exists(dnn_path):
        raise FileNotFoundError(f"Missing DNN file: {dnn_path}")
    dnn = tf.keras.models.load_model(dnn_path, compile=False)

    return scaler, base, dnn, meta, base_model


def resolve_feature_order(df, scaler, meta, feature_cols=None):
    """
    Return the exact feature column order expected by scaler/model.
    Priority:
      1) meta['feature_names']
      2) scaler.feature_names_in_ (sklearn >=1.0)
      3) feature_cols (CLI)
    """
    if meta and isinstance(meta.get("feature_names"), list) and meta["feature_names"]:
        needed = meta["feature_names"]
    elif hasattr(scaler, "feature_names_in_"):
        needed = list(scaler.feature_names_in_)
    elif feature_cols:
        needed = list(feature_cols)
    else:
        raise ValueError(
            "Cannot determine feature order. Provide --feature-cols OR save ensemble_meta.json during training."
        )

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV is missing required feature columns: {missing}")

    return needed


# --------------------------- Prediction Core ---------------------------


def predict_ensemble(
    df, id_cols, feature_cols, base_model_hint, override_alpha, clip_to_unit
):
    scaler, base, dnn, meta, base_model = load_artifacts(base_model_hint)

    # Decide alpha
    if override_alpha is not None:
        alpha = float(override_alpha)
    else:
        alpha = float(meta["alpha"]) if (meta and "alpha" in meta) else 0.4

    # Feature order
    needed = resolve_feature_order(df, scaler, meta, feature_cols)

    # Prepare X
    X = df[needed].copy()
    X_scaled = scaler.transform(X)

    # Predict
    y_dnn = dnn.predict(X_scaled, verbose=0).flatten()
    y_base = base.predict(X_scaled)
    y_pred = alpha * y_dnn + (1.0 - alpha) * y_base
    if clip_to_unit:
        y_pred = np.clip(y_pred, 0.0, 1.0)

    # Build output (ID cols first, then Actual_MPI if present, then predictions)
    out_df = pd.DataFrame()

    # Include id_cols if provided
    if id_cols:
        missing_ids = [c for c in id_cols if c not in df.columns]
        if missing_ids:
            raise KeyError(f"Requested id-cols not found in input CSV: {missing_ids}")
        out_df = pd.concat([out_df, df[id_cols].reset_index(drop=True)], axis=1)

    # ✅ Always copy Actual MPI if present
    if "MPI" in df.columns:
        out_df["Actual_MPI"] = df["MPI"].values

    # Predictions
    out_df["Ensemble_Predicted_MPI"] = y_pred

    info = {
        "alpha": alpha,
        "base_model": base_model,
        "rows": len(df),
        "features_used": needed,
    }
    return out_df, info


# --------------------------- CLI ---------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch predict MPI using a saved Ensemble (no GEE)."
    )
    p.add_argument("--input", required=True, help="Input CSV with feature columns.")
    p.add_argument("--output", required=True, help="Output CSV path for predictions.")
    p.add_argument(
        "--id-cols",
        nargs="*",
        default=None,
        help="Optional columns to carry into the output (e.g., Country Governorate Year).",
    )
    p.add_argument(
        "--feature-cols",
        nargs="*",
        default=None,
        help=(
            "Feature columns used in training (order not required). "
            "Only needed if you don't have ensemble_meta.json nor scaler.feature_names_in_."
        ),
    )
    p.add_argument(
        "--base-model",
        choices=["XGBoost", "Random Forest", "KNN Regressor"],
        default=None,
        help="Hint for base model type if meta file is missing.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override alpha (DNN weight). If omitted, use meta['alpha'] else 0.4.",
    )
    p.add_argument(
        "--no-clip",
        action="store_true",
        help="Do NOT clip predictions to [0,1]. Default clips to [0,1].",
    )
    p.add_argument(
        "--dropna",
        action="store_true",
        help="Drop rows with any NaNs in required feature columns before predicting.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)

    # Early feature check (optional hinting)
    try:
        scaler, _, _, meta, _ = load_artifacts(args.base_model)
        _ = resolve_feature_order(df, scaler, meta, args.feature_cols)
    except Exception as e:
        print(f"[Info] Feature order check: {e}")

    # Optionally drop NaNs in needed features
    if args.dropna:
        feature_list = None
        try:
            scaler, _, _, meta, _ = load_artifacts(args.base_model)
            feature_list = resolve_feature_order(df, scaler, meta, args.feature_cols)
        except Exception:
            pass
        if feature_list:
            before = len(df)
            df = df.dropna(subset=feature_list)
            after = len(df)
            print(f"[Info] Dropped {before - after} rows with NaNs in feature columns.")
        else:
            before = len(df)
            df = df.dropna()
            after = len(df)
            print(f"[Info] Dropped {before - after} rows with any NaNs.")

    preds_df, info = predict_ensemble(
        df=df,
        id_cols=args.id_cols,
        feature_cols=args.feature_cols,
        base_model_hint=args.base_model,
        override_alpha=args.alpha,
        clip_to_unit=not args.no_clip,
    )

    preds_df.to_csv(args.output, index=False)
    print(
        f"✅ Saved predictions to {args.output} | rows={info['rows']} "
        f"| base={info['base_model']} | alpha={info['alpha']}"
    )
    print("Features used (in order):")
    for c in info["features_used"]:
        print(f" - {c}")


if __name__ == "__main__":
    main()
