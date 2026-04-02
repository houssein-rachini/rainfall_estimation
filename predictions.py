import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
import xgboost as xgb
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.models import load_model

from model_utils import enrich_datetime_columns

TARGET_COL = "GROUND"

MODEL_PATHS = {
    "DNN": "trained_dnn_model.h5",
    "ML": "trained_ml_model.pkl",
    "DNN+RF": "trained_ensemble_rf_dnn_model.h5",
    "DNN+XGBoost": "trained_ensemble_xgb_dnn_model.h5",
    "DNN+KNN": "trained_ensemble_knn_dnn_model.h5",
}

SCALER_PATHS = {
    "DNN": "dnn_scaler.pkl",
    "ML": "ml_scaler.pkl",
    "Ensemble": "ensemble_scaler.pkl",
}

PRETRAINED_MODELS_PATHS = {
    "DNN": "models/global/trained_dnn_model.h5",
    "ML": "models/global/trained_ml_model.pkl",
    "DNN+RF": "models/global/trained_ensemble_rf_dnn_model.h5",
    "DNN+XGBoost": "models/global/trained_ensemble_xgb_dnn_model.h5",
    "DNN+KNN": "models/global/trained_ensemble_knn_dnn_model.h5",
    "XGBoost": "models/global/trained_ensemble_xgb_model.json",
    "RF": "models/global/trained_ensemble_rf_model.pkl",
    "KNN": "models/global/trained_ensemble_knn_model.pkl",
}

PRETRAINED_SCALERS_PATHS = {
    "DNN": "models/global/dnn_scaler.pkl",
    "ML": "models/global/ml_scaler.pkl",
    "Ensemble": "models/global/ensemble_scaler.pkl",
}

TARGET_TRANSFORM_META = {
    "ML": "trained_ml_model.meta.json",
    "DNN": "trained_dnn_model.meta.json",
    "Ensemble": "trained_ensemble.meta.json",
}


def _load_target_transform(meta_path: str) -> str:
    if not os.path.exists(meta_path):
        return "none"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("target_transform", "none")
    except Exception:
        return "none"


def _apply_target_inverse(preds: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        preds = np.expm1(preds)
    return np.maximum(preds, 0)


def _load_meta(meta_path: str) -> dict:
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _apply_two_stage_gate(preds: np.ndarray, prob_wet: np.ndarray, meta: dict) -> np.ndarray:
    mode = meta.get("two_stage_gate_mode", "hard_threshold")
    prob_threshold = float(meta.get("prob_threshold", 0.5))
    gamma = float(meta.get("soft_gate_gamma", 1.0))
    p = np.clip(np.asarray(prob_wet, dtype=float), 0.0, 1.0)
    preds = np.maximum(np.asarray(preds, dtype=float), 0.0)
    if mode == "soft_probability":
        return preds * np.power(p, max(gamma, 1e-6))
    return preds * (p >= prob_threshold)


def _maybe_add_location_dummies(test_data: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    expects_location = any(col.startswith("Location_") for col in feature_names)
    if not expects_location:
        return test_data

    if "Location" in test_data.columns:
        df = test_data.copy()
        df["Location"] = df["Location"].astype("string").str.strip()
        return pd.get_dummies(df, columns=["Location"], prefix="Location", dtype=int)

    has_location_dummies = any(col.startswith("Location_") for col in test_data.columns)
    if has_location_dummies:
        return test_data

    raise ValueError(
        "Missing required input column: Location (needed for Location one-hot features)."
    )


@st.cache_resource
def load_dnn_model(use_pretrained):
    model_path = PRETRAINED_MODELS_PATHS["DNN"] if use_pretrained else MODEL_PATHS["DNN"]
    return load_model(
        model_path,
        custom_objects={
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "rmse": tf.keras.metrics.RootMeanSquaredError(),
        },
    )


@st.cache_resource
def load_dnn_scaler(use_pretrained):
    scaler_path = PRETRAINED_SCALERS_PATHS["DNN"] if use_pretrained else SCALER_PATHS["DNN"]
    return joblib.load(scaler_path)


@st.cache_resource
def load_ml_model(use_pretrained):
    model_path = PRETRAINED_MODELS_PATHS["ML"] if use_pretrained else MODEL_PATHS["ML"]
    return joblib.load(model_path)


@st.cache_resource
def load_ml_scaler(use_pretrained):
    scaler_path = PRETRAINED_SCALERS_PATHS["ML"] if use_pretrained else SCALER_PATHS["ML"]
    return joblib.load(scaler_path)


@st.cache_resource
def load_ensemble_scaler(use_pretrained):
    scaler_path = (
        PRETRAINED_SCALERS_PATHS["Ensemble"] if use_pretrained else SCALER_PATHS["Ensemble"]
    )
    return joblib.load(scaler_path)


@st.cache_resource
def load_ensemble_models(model_type, use_pretrained):
    dnn_path = PRETRAINED_MODELS_PATHS[model_type] if use_pretrained else MODEL_PATHS[model_type]
    dnn_model = load_model(
        dnn_path,
        custom_objects={
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "rmse": tf.keras.metrics.RootMeanSquaredError(),
        },
    )

    if model_type == "DNN+XGBoost":
        base_path = (
            PRETRAINED_MODELS_PATHS["XGBoost"] if use_pretrained else "trained_ensemble_xgb_model.json"
        )
        base_model = xgb.XGBRegressor()
        base_model.load_model(base_path)
    elif model_type == "DNN+RF":
        base_path = PRETRAINED_MODELS_PATHS["RF"] if use_pretrained else "trained_ensemble_rf_model.pkl"
        base_model = joblib.load(base_path)
    elif model_type == "DNN+KNN":
        base_path = PRETRAINED_MODELS_PATHS["KNN"] if use_pretrained else "trained_ensemble_knn_model.pkl"
        base_model = joblib.load(base_path)
    else:
        raise ValueError(f"Unsupported ensemble model type: {model_type}")

    return dnn_model, base_model


def predict_dnn_fast(test_data, dnn_model, scaler):
    try:
        test_data_scaled = preprocess_data(test_data.copy(), scaler)
    except ValueError as e:
        st.error(str(e))
        return None
    predictions = dnn_model.predict(test_data_scaled).flatten()
    meta = _load_meta(TARGET_TRANSFORM_META["DNN"])
    transform = meta.get("target_transform", "none")
    preds = _apply_target_inverse(predictions, transform)
    if meta.get("two_stage"):
        clf_path = meta.get("classifier_path") or "trained_dnn_classifier.pkl"
        if not os.path.exists(clf_path):
            st.error("Classifier model file not found for two-stage DNN.")
            return None
        clf = joblib.load(clf_path)
        prob_wet = clf.predict_proba(test_data_scaled)[:, 1]
        preds = _apply_two_stage_gate(preds, prob_wet, meta)
    return preds


def predict_ml_fast(test_data, ml_model, scaler):
    try:
        test_data_scaled = preprocess_data(test_data.copy(), scaler)
    except ValueError as e:
        st.error(str(e))
        return None
    meta = _load_meta(TARGET_TRANSFORM_META["ML"])
    transform = meta.get("target_transform", "none")
    if meta.get("two_stage"):
        clf_path = meta.get("classifier_path") or "trained_ml_classifier.pkl"
        if not os.path.exists(clf_path):
            st.error("Classifier model file not found for two-stage ML.")
            return None
        clf = joblib.load(clf_path)
        prob_wet = clf.predict_proba(test_data_scaled)[:, 1]
        predictions = ml_model.predict(test_data_scaled)
        preds = _apply_target_inverse(predictions, transform)
        preds = _apply_two_stage_gate(preds, prob_wet, meta)
        return preds
    predictions = ml_model.predict(test_data_scaled)
    return _apply_target_inverse(predictions, transform)


def predict_ensemble_fast(test_data, dnn_model, base_model, scaler, alpha):
    try:
        test_data_scaled = preprocess_data(test_data.copy(), scaler)
    except ValueError as e:
        st.error(str(e))
        return None
    y_pred_dnn = dnn_model.predict(test_data_scaled).flatten()
    y_pred_base = base_model.predict(test_data_scaled)
    y_pred_ensemble = alpha * y_pred_dnn + (1 - alpha) * y_pred_base
    meta = _load_meta(TARGET_TRANSFORM_META["Ensemble"])
    transform = meta.get("target_transform", "none")
    preds = _apply_target_inverse(y_pred_ensemble, transform)
    if meta.get("two_stage"):
        clf_path = meta.get("classifier_path") or "trained_ensemble_classifier.pkl"
        if not os.path.exists(clf_path):
            st.error("Classifier model file not found for two-stage Ensemble.")
            return None
        clf = joblib.load(clf_path)
        prob_wet = clf.predict_proba(test_data_scaled)[:, 1]
        preds = _apply_two_stage_gate(preds, prob_wet, meta)
    return preds


def preprocess_data(test_data, scaler):
    feature_names = list(scaler.feature_names_in_)
    test_data = _maybe_add_location_dummies(test_data, feature_names)

    missing_columns = [col for col in feature_names if col not in test_data.columns]
    missing_required = []
    for col in missing_columns:
        if col.startswith("Location_"):
            test_data[col] = 0
        else:
            missing_required.append(col)
    if missing_required:
        missing_text = ", ".join(missing_required)
        raise ValueError(
            f"Missing required input columns for prediction: {missing_text}"
        )

    test_data_selected = test_data[feature_names]

    nan_counts = test_data_selected.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if not cols_with_nan.empty:
        cols_text = ", ".join([f"{col} ({int(cnt)})" for col, cnt in cols_with_nan.items()])
        missing_rows = test_data_selected.isna().any(axis=1)
        sample_rows = test_data_selected.index[missing_rows].tolist()[:10]
        raise ValueError(
            "Missing input data found in required columns: "
            f"{cols_text}. Rows with missing values: {int(missing_rows.sum())}. "
            f"Example row indices: {sample_rows}"
        )

    return scaler.transform(test_data_selected)


def predict_dnn(test_data):
    if not os.path.exists(MODEL_PATHS["DNN"]):
        st.error("DNN model file not found. Please train the model first.")
        return None
    if not os.path.exists(SCALER_PATHS["DNN"]):
        st.error("DNN scaler file not found. Please train the model first.")
        return None

    dnn_model = load_model(
        MODEL_PATHS["DNN"],
        custom_objects={
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "rmse": tf.keras.metrics.RootMeanSquaredError(),
        },
    )
    scaler = joblib.load(SCALER_PATHS["DNN"])
    try:
        test_data_scaled = preprocess_data(test_data.copy(), scaler)
    except ValueError as e:
        st.error(str(e))
        return None
    predictions = dnn_model.predict(test_data_scaled).flatten()
    meta = _load_meta(TARGET_TRANSFORM_META["DNN"])
    transform = meta.get("target_transform", "none")
    preds = _apply_target_inverse(predictions, transform)
    if meta.get("two_stage"):
        clf_path = meta.get("classifier_path") or "trained_dnn_classifier.pkl"
        if not os.path.exists(clf_path):
            st.error("Classifier model file not found for two-stage DNN.")
            return None
        clf = joblib.load(clf_path)
        prob_wet = clf.predict_proba(test_data_scaled)[:, 1]
        preds = _apply_two_stage_gate(preds, prob_wet, meta)
    return preds


def predict_ml(test_data):
    if not os.path.exists(MODEL_PATHS["ML"]):
        st.error("ML model file not found. Please'D train/upload model first.")
        return None
    if not os.path.exists(SCALER_PATHS["ML"]):
        st.error("ML scaler file not found. Please train/upload scaler first.")
        return None

    ml_model = joblib.load(MODEL_PATHS["ML"])
    scaler = joblib.load(SCALER_PATHS["ML"])
    try:
        test_data_scaled = preprocess_data(test_data.copy(), scaler)
    except ValueError as e:
        st.error(str(e))
        return None
    meta = _load_meta(TARGET_TRANSFORM_META["ML"])
    transform = meta.get("target_transform", "none")
    if meta.get("two_stage"):
        clf_path = meta.get("classifier_path") or "trained_ml_classifier.pkl"
        if not os.path.exists(clf_path):
            st.error("Classifier model file not found for two-stage ML.")
            return None
        clf = joblib.load(clf_path)
        prob_wet = clf.predict_proba(test_data_scaled)[:, 1]
        predictions = ml_model.predict(test_data_scaled)
        preds = _apply_target_inverse(predictions, transform)
        preds = _apply_two_stage_gate(preds, prob_wet, meta)
        return preds
    predictions = ml_model.predict(test_data_scaled)
    return _apply_target_inverse(predictions, transform)


def predict_ensemble(test_data, model_type, alpha):
    if not os.path.exists(MODEL_PATHS[model_type]):
        st.error(f"DNN model file for '{model_type}' not found.")
        return None
    if not os.path.exists(SCALER_PATHS["Ensemble"]):
        st.error("Ensemble scaler file not found.")
        return None

    dnn_model = load_model(
        MODEL_PATHS[model_type],
        custom_objects={
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "rmse": tf.keras.metrics.RootMeanSquaredError(),
        },
    )

    scaler = joblib.load(SCALER_PATHS["Ensemble"])
    try:
        test_data_scaled = preprocess_data(test_data.copy(), scaler)
    except ValueError as e:
        st.error(str(e))
        return None

    base_model = None
    if model_type == "DNN+XGBoost":
        base_model_path = "trained_ensemble_xgb_model.json"
        if not os.path.exists(base_model_path):
            st.error("XGBoost base model file not found.")
            return None
        base_model = xgb.XGBRegressor()
        base_model.load_model(base_model_path)

    elif model_type == "DNN+RF":
        base_model_path = "trained_ensemble_rf_model.pkl"
        if not os.path.exists(base_model_path):
            st.error("Random Forest base model file not found.")
            return None
        base_model = joblib.load(base_model_path)

    elif model_type == "DNN+KNN":
        base_model_path = "trained_ensemble_knn_model.pkl"
        if not os.path.exists(base_model_path):
            st.error("KNN base model file not found.")
            return None
        base_model = joblib.load(base_model_path)

    y_pred_dnn = dnn_model.predict(test_data_scaled).flatten()
    y_pred_base = base_model.predict(test_data_scaled)
    y_pred_ensemble = alpha * y_pred_dnn + (1 - alpha) * y_pred_base
    meta = _load_meta(TARGET_TRANSFORM_META["Ensemble"])
    transform = meta.get("target_transform", "none")
    preds = _apply_target_inverse(y_pred_ensemble, transform)
    if meta.get("two_stage"):
        clf_path = meta.get("classifier_path") or "trained_ensemble_classifier.pkl"
        if not os.path.exists(clf_path):
            st.error("Classifier model file not found for two-stage Ensemble.")
            return None
        clf = joblib.load(clf_path)
        prob_wet = clf.predict_proba(test_data_scaled)[:, 1]
        preds = _apply_two_stage_gate(preds, prob_wet, meta)
    return preds


def plot_results(test_data):
    st.subheader("Predictions vs Actual GROUND")
    if TARGET_COL in test_data.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            x=test_data[TARGET_COL], y=test_data["Predicted_GROUND"], alpha=0.6, ax=ax
        )
        ax.set_xlabel("Actual GROUND")
        ax.set_ylabel("Predicted GROUND")
        ax.set_title("Actual vs Predicted GROUND")
        ax.axline((0, 0), slope=1, color="red", linestyle="--")
        st.pyplot(fig, width="content")
    else:
        st.warning(
            f"No '{TARGET_COL}' column found in test data. Skipping Actual vs Predicted plot."
        )

    st.subheader("Distribution of Predicted GROUND")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(test_data["Predicted_GROUND"], bins=30, kde=True, color="blue", ax=ax)
    ax.set_xlabel("Predicted GROUND")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Predicted GROUND")
    st.pyplot(fig, width="content")


def show_predictions_tab(df=None):
    st.title("GROUND Prediction")

    model_choice = st.selectbox(
        "Select a model for prediction:",
        ["DNN", "ML", "DNN+RF", "DNN+XGBoost", "DNN+KNN"],
        key="pred_model_choice",
    )

    alpha = 0.15
    if model_choice in ["DNN+RF", "DNN+XGBoost", "DNN+KNN"]:
        alpha = st.slider(
            "Ensemble Weight (DNN Contribution)",
            0.0,
            1.0,
            0.15,
            key="pred_alpha",
        )

    uploaded_file = st.file_uploader(
        "Upload a CSV file for prediction", type="csv", key="pred_upload"
    )

    if uploaded_file:
        test_data = pd.read_csv(uploaded_file, encoding="utf-8", encoding_errors="replace")
        test_data = enrich_datetime_columns(test_data)
        st.write("### Test Data Preview:")
        st.dataframe(test_data.head())

        if st.button("Predict GROUND for All Available Rows", key="pred_run"):
            with st.spinner("Generating predictions..."):
                if model_choice == "DNN":
                    predictions = predict_dnn(test_data)
                    output_file = "test_results_dnn.csv"
                elif model_choice == "ML":
                    predictions = predict_ml(test_data)
                    output_file = "test_results_ml.csv"
                elif model_choice == "DNN+RF":
                    predictions = predict_ensemble(test_data, "DNN+RF", alpha)
                    output_file = "test_results_ensemble_rf.csv"
                elif model_choice == "DNN+XGBoost":
                    predictions = predict_ensemble(test_data, "DNN+XGBoost", alpha)
                    output_file = "test_results_ensemble_xgb.csv"
                else:
                    predictions = predict_ensemble(test_data, "DNN+KNN", alpha)
                    output_file = "test_results_ensemble_knn.csv"

                if predictions is not None:
                    test_data["Predicted_GROUND"] = predictions
                    test_data.to_csv(output_file, index=False)
                    st.success(f"Predictions saved to {output_file}")

                    st.download_button(
                        label="Download Predictions CSV",
                        data=test_data.to_csv(index=False),
                        file_name=output_file,
                        mime="text/csv",
                    )

                    plot_results(test_data)
                else:
                    st.error("Prediction failed. Please check the error messages.")
