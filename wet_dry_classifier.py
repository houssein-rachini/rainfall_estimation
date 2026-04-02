import json
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model_utils import (
    TARGET_COL,
    add_location_dummies,
    balanced_group_kfold_splits,
    enrich_datetime_columns,
)


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]
) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_wet": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_wet": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_wet": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity_dry": float(specificity) if np.isfinite(specificity) else np.nan,
        "N": int(len(y_true)),
        "wet_rate_true": float(np.mean(y_true)),
        "wet_rate_pred": float(np.mean(y_pred)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = np.nan
    return out


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred Dry", "Pred Wet"],
        yticklabels=["True Dry", "True Wet"],
        ax=ax,
    )
    ax.set_title(title)
    st.pyplot(fig, width="content")


def _build_classifier(
    model_name: str,
    random_state: int,
    use_scaler: bool,
    xgb_estimators: int,
    xgb_lr: float,
    xgb_depth: int,
    rf_estimators: int,
    rf_depth: int,
    xgb_scale_pos_weight: float = 1.0,
    xgb_eval_metric: str = "logloss",
) -> object:
    if model_name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=xgb_estimators,
            learning_rate=xgb_lr,
            max_depth=xgb_depth,
            min_child_weight=1,
            objective="binary:logistic",
            eval_metric=xgb_eval_metric,
            scale_pos_weight=xgb_scale_pos_weight,
            random_state=random_state,
        )
    if model_name == "Random Forest":
        max_depth = None if rf_depth <= 0 else rf_depth
        return RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
    if use_scaler:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=random_state),
        )
    return LogisticRegression(max_iter=2000, random_state=random_state)


def _get_splits(
    clean: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_type: str,
    n_splits: int,
    stratify_holdout: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if cv_type == "GroupKFold (Station)":
        return balanced_group_kfold_splits(
            clean["Station"].astype(str), n_splits=n_splits, random_state=42
        )
    if cv_type == "GroupKFold (Year)":
        return balanced_group_kfold_splits(
            clean["Year"].astype(str), n_splits=n_splits, random_state=42
        )
    if cv_type == "TimeSeriesSplit":
        return list(TimeSeriesSplit(n_splits=n_splits).split(X, y))
    if cv_type == "KFold":
        return list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X, y))
    stratify_arg = y if stratify_holdout else None
    tr_idx, te_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.2,
        random_state=42,
        stratify=stratify_arg,
        shuffle=True,
    )
    return [(tr_idx, te_idx)]


def _oof_metrics_for_config(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str,
    use_scaler: bool,
    prob_threshold: float,
    params: Dict[str, float],
    xgb_auto_scale_pos_weight: bool = True,
    xgb_eval_metric: str = "logloss",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    y_oof = np.full(len(X), -1, dtype=int)
    p_oof = np.full(len(X), np.nan, dtype=float)
    for tr_idx, va_idx in splits:
        y_tr = y[tr_idx]
        pos = float(np.sum(y_tr == 1))
        neg = float(np.sum(y_tr == 0))
        spw = (neg / pos) if (xgb_auto_scale_pos_weight and pos > 0) else 1.0
        model_fold = _build_classifier(
            model_name=model_name,
            random_state=42,
            use_scaler=use_scaler,
            xgb_estimators=int(params.get("xgb_estimators", 250)),
            xgb_lr=float(params.get("xgb_lr", 0.08)),
            xgb_depth=int(params.get("xgb_depth", 5)),
            rf_estimators=int(params.get("rf_estimators", 250)),
            rf_depth=int(params.get("rf_depth", 0)),
            xgb_scale_pos_weight=spw,
            xgb_eval_metric=xgb_eval_metric,
        )
        model_fold.fit(X.iloc[tr_idx], y_tr)
        p_va = model_fold.predict_proba(X.iloc[va_idx])[:, 1]
        y_va = (p_va >= prob_threshold).astype(int)
        y_oof[va_idx] = y_va
        p_oof[va_idx] = p_va
    valid = y_oof >= 0
    return _compute_metrics(y[valid], y_oof[valid], p_oof[valid]), y_oof, p_oof


def _auto_search_best_config(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str,
    use_scaler: bool,
    n_trials: int,
    objective_name: str,
    xgb_auto_scale_pos_weight: bool = True,
    xgb_eval_metric: str = "logloss",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    rng = np.random.default_rng(42)
    best_params: Dict[str, float] = {}
    best_metrics: Dict[str, float] = {}
    best_score = -np.inf

    for _ in range(n_trials):
        params: Dict[str, float] = {
            "dry_threshold": float(rng.uniform(0.0, 8.0)),
            "prob_threshold": float(rng.uniform(0.4, 0.9)),
        }
        if model_name == "XGBoost":
            params.update(
                {
                    "xgb_estimators": int(rng.integers(80, 601)),
                    "xgb_lr": float(rng.uniform(0.01, 0.3)),
                    "xgb_depth": int(rng.integers(2, 11)),
                }
            )
        elif model_name == "Random Forest":
            params.update(
                {
                    "rf_estimators": int(rng.integers(80, 601)),
                    "rf_depth": int(rng.integers(0, 21)),
                }
            )

        y_trial = (y > params["dry_threshold"]).astype(int)
        if len(np.unique(y_trial)) < 2:
            continue
        metrics, _, _ = _oof_metrics_for_config(
            X=X,
            y=y_trial,
            splits=splits,
            model_name=model_name,
            use_scaler=use_scaler,
            prob_threshold=float(params["prob_threshold"]),
            params=params,
            xgb_auto_scale_pos_weight=xgb_auto_scale_pos_weight,
            xgb_eval_metric=xgb_eval_metric,
        )
        if objective_name == "f1_wet":
            score = metrics["f1_wet"]
        elif objective_name == "accuracy":
            score = metrics["accuracy"]
        else:
            score = 0.5 * metrics["f1_wet"] + 0.5 * metrics["specificity_dry"]

        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    if len(best_params) == 0:
        raise ValueError("Auto-search failed to find a valid configuration.")
    return best_params, best_metrics


def _auto_search_best_config_optuna(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str,
    use_scaler: bool,
    n_trials: int,
    objective_name: str,
    xgb_auto_scale_pos_weight: bool = True,
    xgb_eval_metric: str = "logloss",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    try:
        import optuna
    except Exception as e:
        raise RuntimeError(
            "Optuna is not installed in this environment. Install with: pip install optuna"
        ) from e

    def score_from_metrics(metrics: Dict[str, float]) -> float:
        if objective_name == "f1_wet":
            return float(metrics["f1_wet"])
        if objective_name == "accuracy":
            return float(metrics["accuracy"])
        return float(0.5 * metrics["f1_wet"] + 0.5 * metrics["specificity_dry"])

    def objective(trial: "optuna.Trial") -> float:
        params: Dict[str, float] = {
            "dry_threshold": trial.suggest_float("dry_threshold", 0.0, 8.0),
            "prob_threshold": trial.suggest_float("prob_threshold", 0.4, 0.9),
        }
        if model_name == "XGBoost":
            params.update(
                {
                    "xgb_estimators": int(trial.suggest_int("xgb_estimators", 80, 600)),
                    "xgb_lr": float(trial.suggest_float("xgb_lr", 0.01, 0.3, log=True)),
                    "xgb_depth": int(trial.suggest_int("xgb_depth", 2, 10)),
                }
            )
        elif model_name == "Random Forest":
            params.update(
                {
                    "rf_estimators": int(trial.suggest_int("rf_estimators", 80, 600)),
                    "rf_depth": int(trial.suggest_int("rf_depth", 0, 20)),
                }
            )

        y_trial = (y > params["dry_threshold"]).astype(int)
        if len(np.unique(y_trial)) < 2:
            return -1.0

        metrics, _, _ = _oof_metrics_for_config(
            X=X,
            y=y_trial,
            splits=splits,
            model_name=model_name,
            use_scaler=use_scaler,
            prob_threshold=float(params["prob_threshold"]),
            params=params,
            xgb_auto_scale_pos_weight=xgb_auto_scale_pos_weight,
            xgb_eval_metric=xgb_eval_metric,
        )
        return score_from_metrics(metrics)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if study.best_trial is None:
        raise ValueError("Optuna did not find a valid trial.")

    best_params = dict(study.best_trial.params)
    if "xgb_estimators" in best_params:
        best_params["xgb_estimators"] = int(best_params["xgb_estimators"])
    if "xgb_depth" in best_params:
        best_params["xgb_depth"] = int(best_params["xgb_depth"])
    if "rf_estimators" in best_params:
        best_params["rf_estimators"] = int(best_params["rf_estimators"])
    if "rf_depth" in best_params:
        best_params["rf_depth"] = int(best_params["rf_depth"])

    y_best = (y > float(best_params["dry_threshold"])).astype(int)
    best_metrics, _, _ = _oof_metrics_for_config(
        X=X,
        y=y_best,
        splits=splits,
        model_name=model_name,
        use_scaler=use_scaler,
        prob_threshold=float(best_params["prob_threshold"]),
        params=best_params,
        xgb_auto_scale_pos_weight=xgb_auto_scale_pos_weight,
        xgb_eval_metric=xgb_eval_metric,
    )
    return best_params, best_metrics


def show_wet_dry_classifier_tab(df: pd.DataFrame) -> None:
    df = enrich_datetime_columns(df)
    st.title("Wet/Dry Classifier")
    st.caption("Standalone binary classification: dry (0) vs wet (1).")

    if TARGET_COL not in df.columns:
        st.error(f"{TARGET_COL} column not found.")
        return

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)

    default_cols = [
        c
        for c in [
            "Chirps",
            "IMERG(mm/month)",
            "DEM",
            "NDVI",
            "Slope",
            "Longitude",
            "Latitude",
            # "lon_sin",
            # "lon_cos",
            # "lat_sin",
            # "lat_cos",
            # "Month",
            "month_sin",
            "month_cos",
            # "Year",
        ]
        if c in numeric_cols
    ]
    selected_features = st.multiselect(
        "Features",
        options=numeric_cols,
        default=default_cols,
        key="wd_features",
    )
    if len(selected_features) == 0:
        st.warning("Select at least one feature.")
        return

    use_location = False
    if "Location" in df.columns:
        use_location = st.checkbox(
            "Use Location one-hot",
            value=True,
            key="wd_use_location",
        )

    dry_threshold = st.slider(
        "Dry threshold (mm)", 0.0, 20.0, 4.5, 0.5, key="wd_dry_threshold"
    )
    prob_threshold = st.slider(
        "Wet probability threshold", 0.0, 1.0, 0.6, 0.05, key="wd_prob_threshold"
    )

    cv_type = st.selectbox(
        "Split strategy",
        [
            "Train/Test Split",
            "GroupKFold (Station)",
            "GroupKFold (Year)",
            "TimeSeriesSplit",
            "KFold",
        ],
        key="wd_cv_type",
    )
    n_splits = st.slider("CV folds/splits", 2, 10, 5, key="wd_n_splits")

    model_name = st.selectbox(
        "Classifier",
        ["XGBoost", "Random Forest", "Logistic Regression"],
        key="wd_model",
    )
    use_scaler = st.checkbox(
        "Use StandardScaler (for Logistic Regression)",
        value=True,
        key="wd_use_scaler",
    )
    xgb_estimators = st.slider("XGB n_estimators", 50, 600, 520, 10, key="wd_xgb_n")
    xgb_lr = st.slider("XGB learning_rate", 0.01, 0.5, 0.04, 0.01, key="wd_xgb_lr")
    xgb_depth = st.slider("XGB max_depth", 2, 10, 2, 1, key="wd_xgb_depth")
    xgb_eval_metric = st.selectbox(
        "XGB eval_metric",
        ["aucpr", "logloss"],
        index=0,
        key="wd_xgb_eval_metric",
    )
    xgb_auto_scale_pos_weight = st.checkbox(
        "Auto class weighting for XGB (scale_pos_weight per split)",
        value=True,
        key="wd_xgb_auto_spw",
    )
    rf_estimators = st.slider("RF n_estimators", 50, 600, 250, 10, key="wd_rf_n")
    rf_depth = st.slider("RF max_depth (0=None)", 0, 30, 0, 1, key="wd_rf_depth")
    st.subheader("Auto Search")
    auto_engine = st.selectbox(
        "Auto-search engine",
        ["Optuna", "Random search"],
        index=0,
        key="wd_auto_engine",
    )
    auto_trials = st.slider("Auto-search trials", 20, 300, 80, 10, key="wd_auto_trials")
    auto_objective = st.selectbox(
        "Auto-search objective",
        ["f1_wet", "balanced_f1_specificity", "accuracy"],
        index=1,
        key="wd_auto_objective",
    )

    feature_cols = selected_features + (["Location"] if use_location else [])
    clean = df.dropna(subset=feature_cols + [TARGET_COL]).copy()
    if clean.empty:
        st.error("No rows after dropping NA for selected features/target.")
        return
    if "Date" in clean.columns:
        clean = clean.dropna(subset=["Date"]).sort_values("Date")

    y_rain = clean[TARGET_COL].to_numpy(dtype="float64")
    y = (y_rain > float(dry_threshold)).astype(int)
    X = clean[feature_cols].copy()
    if use_location:
        X = add_location_dummies(X)

    if len(np.unique(y)) < 2:
        st.error("Only one class found after thresholding. Change dry threshold.")
        return

    if st.button("Auto-search Best Config", key="wd_auto_btn"):
        if cv_type == "GroupKFold (Station)" and "Station" not in clean.columns:
            st.error("Station column is required.")
            return
        if cv_type == "GroupKFold (Year)" and "Year" not in clean.columns:
            st.error("Year column is required.")
            return
        with st.spinner("Running auto-search..."):
            # Important: for auto-search holdout mode, avoid stratifying by a fixed UI threshold.
            splits = _get_splits(
                clean=clean,
                X=X,
                y=y,
                cv_type=cv_type,
                n_splits=n_splits,
                stratify_holdout=False,
            )
            try:
                if auto_engine == "Optuna":
                    best_params, best_metrics = _auto_search_best_config_optuna(
                        X=X,
                        y=y_rain,
                        splits=splits,
                        model_name=model_name,
                        use_scaler=use_scaler,
                        n_trials=int(auto_trials),
                        objective_name=auto_objective,
                        xgb_auto_scale_pos_weight=xgb_auto_scale_pos_weight,
                        xgb_eval_metric=xgb_eval_metric,
                    )
                else:
                    best_params, best_metrics = _auto_search_best_config(
                        X=X,
                        y=y_rain,
                        splits=splits,
                        model_name=model_name,
                        use_scaler=use_scaler,
                        n_trials=int(auto_trials),
                        objective_name=auto_objective,
                        xgb_auto_scale_pos_weight=xgb_auto_scale_pos_weight,
                        xgb_eval_metric=xgb_eval_metric,
                    )
            except Exception as e:
                st.error(f"Auto-search failed: {e}")
                return
        st.write("### Best Auto-search Params")
        st.json(best_params)
        st.write("### Best Auto-search Metrics")
        st.json(best_metrics)

    if st.button("Train Wet/Dry Classifier", key="wd_train_btn"):
        if cv_type == "Train/Test Split":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            pos = float(np.sum(y_train == 1))
            neg = float(np.sum(y_train == 0))
            spw = (neg / pos) if (xgb_auto_scale_pos_weight and pos > 0) else 1.0
            model = _build_classifier(
                model_name=model_name,
                random_state=42,
                use_scaler=use_scaler,
                xgb_estimators=xgb_estimators,
                xgb_lr=xgb_lr,
                xgb_depth=xgb_depth,
                rf_estimators=rf_estimators,
                rf_depth=rf_depth,
                xgb_scale_pos_weight=spw,
                xgb_eval_metric=xgb_eval_metric,
            )
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= prob_threshold).astype(int)
            metrics = _compute_metrics(y_test, y_pred, y_prob)
            st.write("### Holdout Metrics")
            st.json(metrics)
            _plot_confusion(y_test, y_pred, "Confusion Matrix (Holdout)")
        else:
            if cv_type == "GroupKFold (Station)" and "Station" not in clean.columns:
                st.error("Station column is required.")
                return
            if cv_type == "GroupKFold (Year)" and "Year" not in clean.columns:
                st.error("Year column is required.")
                return
            splits = _get_splits(clean=clean, X=X, y=y, cv_type=cv_type, n_splits=n_splits)

            y_oof = np.full(len(X), -1, dtype=int)
            p_oof = np.full(len(X), np.nan, dtype=float)
            for tr_idx, va_idx in splits:
                X_tr = X.iloc[tr_idx]
                X_va = X.iloc[va_idx]
                y_tr = y[tr_idx]
                pos = float(np.sum(y_tr == 1))
                neg = float(np.sum(y_tr == 0))
                spw = (neg / pos) if (xgb_auto_scale_pos_weight and pos > 0) else 1.0
                model_fold = _build_classifier(
                    model_name=model_name,
                    random_state=42,
                    use_scaler=use_scaler,
                    xgb_estimators=xgb_estimators,
                    xgb_lr=xgb_lr,
                    xgb_depth=xgb_depth,
                    rf_estimators=rf_estimators,
                    rf_depth=rf_depth,
                    xgb_scale_pos_weight=spw,
                    xgb_eval_metric=xgb_eval_metric,
                )
                model_fold.fit(X_tr, y_tr)
                p_va = model_fold.predict_proba(X_va)[:, 1]
                y_va = (p_va >= prob_threshold).astype(int)
                y_oof[va_idx] = y_va
                p_oof[va_idx] = p_va

            valid = y_oof >= 0
            metrics = _compute_metrics(y[valid], y_oof[valid], p_oof[valid])
            st.write("### Out-of-Fold Metrics")
            st.json(metrics)
            _plot_confusion(y[valid], y_oof[valid], "Confusion Matrix (OOF)")

        out = {
            "model": model_name,
            "dry_threshold_mm": float(dry_threshold),
            "prob_threshold": float(prob_threshold),
            "split_strategy": cv_type,
            "n_splits": int(n_splits),
            "features": list(X.columns),
            "use_scaler": bool(use_scaler),
            "xgb_eval_metric": xgb_eval_metric,
            "xgb_auto_scale_pos_weight": bool(xgb_auto_scale_pos_weight),
        }
        st.download_button(
            "Download Run Config (JSON)",
            data=json.dumps(out, indent=2),
            file_name="wet_dry_classifier_config.json",
            mime="application/json",
            key="wd_download_config",
        )
