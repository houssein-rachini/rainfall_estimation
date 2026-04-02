import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    TimeSeriesSplit,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

TARGET_COL = "GROUND"
from model_utils import add_location_dummies, enrich_datetime_columns
from model_utils import balanced_group_kfold_splits


def display_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
    st.write(f"**R2 Score:** {r2:.4f}")


def plot_predictions(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r")
    ax.set_xlabel("Actual GROUND")
    ax.set_ylabel("Predicted GROUND")
    ax.set_title("Predicted vs Actual GROUND")
    st.pyplot(fig, width="content")


def plot_learning_curve(
    model,
    X,
    y,
    cv,
    title="Learning Curve",
    groups=None,
    use_scaler=True,
):
    estimator = make_pipeline(StandardScaler(), model) if use_scaler else model
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 10),
    )
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_scores_mean, label="Training Loss", marker="o")
    ax.plot(train_sizes, test_scores_mean, label="Validation Loss", marker="s")
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig, width="content")


def plot_residuals(y_val, y_pred):
    residuals = y_val - y_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=y_val, y=residuals, alpha=0.7, ax=ax)
    ax.axhline(y=0, color="red", linestyle="--")
    ax.set_xlabel("Actual GROUND")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Plot (Error Analysis)")
    st.pyplot(fig, width="content")


def compute_wet_dry_diagnostics(
    y_true,
    prob_wet,
    dry_threshold: float,
    prob_threshold: float,
):
    y_true_arr = np.asarray(y_true, dtype=float)
    prob_arr = np.asarray(prob_wet, dtype=float)
    true_wet = y_true_arr > dry_threshold
    pred_wet = prob_arr >= prob_threshold

    tp = int(np.sum(true_wet & pred_wet))
    tn = int(np.sum((~true_wet) & (~pred_wet)))
    fp = int(np.sum((~true_wet) & pred_wet))
    fn = int(np.sum(true_wet & (~pred_wet)))
    n = int(len(y_true_arr))

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else np.nan
    )
    accuracy = (tp + tn) / n if n > 0 else np.nan

    return {
        "N": n,
        "TP_wet_as_wet": tp,
        "TN_dry_as_dry": tn,
        "FP_dry_as_wet": fp,
        "FN_wet_as_dry": fn,
        "precision_wet": float(precision) if np.isfinite(precision) else np.nan,
        "recall_wet": float(recall) if np.isfinite(recall) else np.nan,
        "specificity_dry": float(specificity) if np.isfinite(specificity) else np.nan,
        "f1_wet": float(f1) if np.isfinite(f1) else np.nan,
        "accuracy": float(accuracy) if np.isfinite(accuracy) else np.nan,
        "predicted_wet_rate": float(np.mean(pred_wet)) if n > 0 else np.nan,
        "actual_wet_rate": float(np.mean(true_wet)) if n > 0 else np.nan,
        "dry_threshold_mm": float(dry_threshold),
        "prob_threshold": float(prob_threshold),
    }


def _apply_two_stage_gate(
    y_pred: np.ndarray,
    prob_wet: np.ndarray,
    prob_threshold: float,
    gate_mode: str,
    soft_gate_gamma: float,
) -> np.ndarray:
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    p = np.clip(np.asarray(prob_wet, dtype=float), 0.0, 1.0)
    if gate_mode == "soft_probability":
        return y_pred * np.power(p, max(float(soft_gate_gamma), 1e-6))
    return y_pred * (p >= float(prob_threshold))


def display_wet_dry_diagnostics(metrics: dict):
    st.write("### Wet/Dry Confusion-Style Diagnostics")
    st.json(metrics)


def show_ml_training_tab(df: pd.DataFrame):
    df = enrich_datetime_columns(df)
    st.title("Machine Learning Training")

    if "ml_results" in st.session_state:
        st.subheader("Previous Training Results")
        results = st.session_state["ml_results"]
        st.write(f"**Model:** {results['model']}")
        if results.get("cv_mae_mean") is not None:
            st.write(
                f"**{results['n_splits']}-Fold CV Mean MAE:** {results['cv_mae_mean']:.4f}"
            )
        if results.get("cv_rmse_mean") is not None:
            st.write(
                f"**{results['n_splits']}-Fold CV Mean RMSE:** {results['cv_rmse_mean']:.4f}"
            )
        if results.get("cv_r2_mean") is not None:
            st.write(
                f"**{results['n_splits']}-Fold CV Mean R2:** {results['cv_r2_mean']:.4f}"
            )
        display_metrics(results["y_test"], results["y_pred"])
        plot_predictions(results["y_test"], results["y_pred"])
        plot_residuals(results["y_test"], results["y_pred"])
        if results.get("wet_dry_diagnostics") is not None:
            display_wet_dry_diagnostics(results["wet_dry_diagnostics"])

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)

    default_cols = [
        col
        for col in [
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
        if col in numeric_cols
    ]

    selected_features = st.multiselect(
        "Select features for training:",
        numeric_cols,
        default=default_cols,
        key="ml_features",
    )
    use_location_feature = False
    if "Location" in df.columns:
        use_location_feature = st.checkbox(
            "Use Location type (one-hot feature)",
            value=True,
            key="ml_use_location_feature",
            help="Adds one-hot columns for Location types (Coastal/Inland/etc.).",
        )
    use_log1p = st.checkbox(
        "Train on log1p(GROUND)",
        value=False,
        key="ml_log1p",
        help="Reduces skew and downweights extreme rainfall values.",
    )
    use_two_stage = st.checkbox(
        "Two-stage model (classify dry vs wet, then regress)",
        value=False,
        key="ml_two_stage",
        help="Helps reduce overestimation at low/zero rainfall.",
    )
    dry_threshold = st.slider(
        "Dry threshold (mm)",
        0.0,
        20.0,
        4.5,
        step=0.5,
        key="ml_dry_threshold",
        help="Values ≤ this are treated as dry (0) in the classifier.",
    )
    prob_threshold = st.slider(
        "Wet probability threshold",
        0.0,
        1.0,
        0.6,
        step=0.05,
        key="ml_prob_threshold",
        help="Classifier probability threshold to output a non-zero prediction.",
    )
    two_stage_gate_mode_ui = st.selectbox(
        "Two-stage output gate",
        ["Hard threshold (zero dry)", "Soft probability (expected rainfall)"],
        index=1,
        key="ml_two_stage_gate_mode",
        help="Soft gate scales rainfall by wet probability instead of forcing dry to zero.",
    )
    two_stage_gate_mode = (
        "soft_probability"
        if two_stage_gate_mode_ui == "Soft probability (expected rainfall)"
        else "hard_threshold"
    )
    soft_gate_gamma = st.slider(
        "Soft gate gamma",
        0.5,
        4.0,
        1.0,
        step=0.1,
        key="ml_soft_gate_gamma",
        disabled=two_stage_gate_mode != "soft_probability",
        help="Pred = wet_prob^gamma * regressor_pred. Higher gamma is more conservative.",
    )

    train_df = df
    if "Location" in df.columns:
        location_options = sorted(df["Location"].dropna().astype(str).unique().tolist())
        selected_locations = st.multiselect(
            "Select Location types for train/test:",
            options=location_options,
            default=location_options,
            key="ml_location_types",
        )
        if len(selected_locations) == 0:
            st.warning("Please select at least one Location type.")
            return
        train_df = df[df["Location"].astype(str).isin(selected_locations)]

    if TARGET_COL not in df.columns:
        st.error(f"{TARGET_COL} column not found in dataset.")
        return

    if len(selected_features) == 0 and not use_location_feature:
        st.warning("Please select at least one feature.")
        return

    cv_type = st.selectbox(
        "Cross-validation strategy:",
        [
            "KFold",
            "GroupKFold (Station)",
            "GroupKFold (Year)",
            "TimeSeriesSplit",
            "No CV (train on full dataset)",
        ],
        key="ml_cv_type",
    )
    do_cv = cv_type != "No CV (train on full dataset)"

    feature_cols = selected_features + (["Location"] if use_location_feature else [])
    df_clean = train_df.dropna(subset=[TARGET_COL] + feature_cols)
    if "Date" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["Date"]).sort_values("Date")

    X = df_clean[feature_cols]
    if use_location_feature:
        X = add_location_dummies(X)
    y_raw = df_clean[TARGET_COL]
    station_groups = (
        df_clean["Station"].astype(str).reset_index(drop=True)
        if "Station" in df_clean.columns
        else None
    )
    year_groups = (
        df_clean["Year"].astype(int).reset_index(drop=True)
        if "Year" in df_clean.columns
        else None
    )

    if cv_type == "No CV (train on full dataset)":
        X_train, X_test = X, X
        y_train_raw, y_test_raw = y_raw, y_raw
        groups_train = None
        st.info(
            "Training on the full dataset without cross-validation or holdout. "
            "Reported metrics will be optimistic."
        )
    elif cv_type == "GroupKFold (Station)":
        if station_groups is None:
            st.error("Station column is required for GroupKFold (Station).")
            return
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y_raw, groups=station_groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_raw, y_test_raw = y_raw.iloc[train_idx], y_raw.iloc[test_idx]
        groups_train = station_groups.iloc[train_idx]
    elif cv_type == "GroupKFold (Year)":
        if year_groups is None:
            st.error("Year column is required for GroupKFold (Year).")
            return
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y_raw, groups=year_groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_raw, y_test_raw = y_raw.iloc[train_idx], y_raw.iloc[test_idx]
        groups_train = year_groups.iloc[train_idx]
    elif cv_type == "TimeSeriesSplit":
        n_test = max(1, int(len(X) * 0.2))
        X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
        y_train_raw, y_test_raw = y_raw.iloc[:-n_test], y_raw.iloc[-n_test:]
        groups_train = None
    else:
        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
            X, y_raw, test_size=0.2, random_state=42
        )
        groups_train = None

    if use_log1p:
        y_train = np.log1p(y_train_raw)
        y_test = np.log1p(y_test_raw)
    else:
        y_train = y_train_raw
        y_test = y_test_raw

    model_options = [
        "XGBoost",
        "Random Forest",
        "Support Vector Regression",
        "KNN Regressor",
    ]
    selected_model = st.selectbox("Select an ML model:", model_options, key="ml_model")
    use_scaler = st.checkbox(
        "Use StandardScaler",
        value=True,
        key="ml_use_scaler",
        help="Recommended for SVR/KNN. Optional for tree-based models.",
    )

    scaler = (
        StandardScaler()
        if use_scaler
        else StandardScaler(with_mean=False, with_std=False)
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = None
    params = {}
    use_huber = False
    huber_slope = None

    if selected_model == "XGBoost":
        loss_choice = st.selectbox(
            "Loss (XGBoost)",
            ["Huber", "MSE", "MAE", "Quantile"],
            key="ml_xgb_loss_choice",
        )
        use_huber = loss_choice == "Huber"
        quantile_alpha = None
        if loss_choice == "Quantile":
            quantile_alpha = st.slider(
                "Quantile alpha",
                0.1,
                0.9,
                0.5,
                step=0.05,
                key="ml_xgb_quantile_alpha",
            )
        if use_huber:
            huber_slope = st.slider(
                "Huber slope (XGBoost)",
                0.1,
                30.0,
                1.0,
                key="ml_xgb_huber_slope",
            )
        params["n_estimators"] = st.slider(
            "Number of Trees (n_estimators)", 50, 1000, 520, key="ml_xgb_n_estimators"
        )
        params["learning_rate"] = st.slider(
            "Learning Rate", 0.01, 0.5, 0.04, key="ml_xgb_lr"
        )
        params["max_depth"] = st.slider("Max Depth", 2, 10, 2, key="ml_xgb_depth")
        params["min_child_weight"] = st.slider(
            "Min Child Weight", 1, 10, 1, key="ml_xgb_min_child_weight"
        )
        if loss_choice == "MAE":
            objective = "reg:absoluteerror"
        elif loss_choice == "Quantile":
            objective = "reg:quantileerror"
        elif loss_choice == "Huber":
            objective = "reg:pseudohubererror"
        else:
            objective = "reg:squarederror"

        model = xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
            random_state=42,
            objective=objective,
            huber_slope=huber_slope if loss_choice == "Huber" else None,
            quantile_alpha=quantile_alpha if loss_choice == "Quantile" else None,
        )

    elif selected_model == "Random Forest":
        params["n_estimators"] = st.slider(
            "Number of Trees (n_estimators)", 50, 300, 150, key="ml_rf_n_estimators"
        )
        params["min_samples_split"] = st.slider(
            "Min Samples Split", 2, 10, 2, key="ml_rf_min_samples_split"
        )
        params["min_samples_leaf"] = st.slider(
            "Min Samples Leaf", 1, 10, 1, key="ml_rf_min_samples_leaf"
        )
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=42,
        )

    elif selected_model == "Support Vector Regression":
        params["C"] = st.slider(
            "Regularization Parameter (C)", 1, 500, 100, key="ml_svr_c"
        )
        params["gamma"] = st.slider(
            "Kernel Coefficient (gamma)", 0.001, 1.0, 0.1, key="ml_svr_gamma"
        )
        model = SVR(kernel="rbf", C=params["C"], gamma=params["gamma"])

    elif selected_model == "KNN Regressor":
        params["n_neighbors"] = st.slider(
            "Number of Neighbors (n_neighbors)", 1, 20, 5, key="ml_knn_neighbors"
        )
        params["metric"] = st.selectbox(
            "Distance Metric",
            ["manhattan", "euclidean", "minkowski"],
            key="ml_knn_metric",
        )
        model = KNeighborsRegressor(
            n_neighbors=params["n_neighbors"], metric=params["metric"]
        )

    n_splits = None
    if do_cv:
        n_splits = st.slider(
            "Select number of folds for Cross Validation (K-Fold)",
            2,
            10,
            5,
            key="ml_n_splits",
        )

    def make_regressor():
        if selected_model == "XGBoost":
            if loss_choice == "MAE":
                objective = "reg:absoluteerror"
            elif loss_choice == "Quantile":
                objective = "reg:quantileerror"
            elif loss_choice == "Huber":
                objective = "reg:pseudohubererror"
            else:
                objective = "reg:squarederror"
            return xgb.XGBRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                min_child_weight=params["min_child_weight"],
                random_state=42,
                objective=objective,
                huber_slope=huber_slope if loss_choice == "Huber" else None,
                quantile_alpha=quantile_alpha if loss_choice == "Quantile" else None,
            )
        if selected_model == "Random Forest":
            return RandomForestRegressor(
                n_estimators=params["n_estimators"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=42,
            )
        if selected_model == "Support Vector Regression":
            return SVR(kernel="rbf", C=params["C"], gamma=params["gamma"])
        return KNeighborsRegressor(
            n_neighbors=params["n_neighbors"], metric=params["metric"]
        )

    classifier_choice = None
    clf_xgb_n_estimators = 520
    clf_xgb_lr = 0.04
    clf_xgb_depth = 2
    clf_xgb_eval_metric = "aucpr"
    clf_xgb_auto_scale_pos_weight = True
    clf_rf_n_estimators = 250
    clf_rf_depth = 0
    if use_two_stage:
        classifier_choice = st.selectbox(
            "Classifier model",
            ["XGBoost", "Random Forest", "Logistic Regression"],
            key="ml_classifier_choice",
        )
        clf_xgb_n_estimators = st.slider(
            "Classifier XGB n_estimators",
            50,
            1000,
            520,
            10,
            key="ml_clf_xgb_n_estimators",
            disabled=classifier_choice != "XGBoost",
        )
        clf_xgb_lr = st.slider(
            "Classifier XGB learning_rate",
            0.01,
            0.5,
            0.04,
            0.01,
            key="ml_clf_xgb_lr",
            disabled=classifier_choice != "XGBoost",
        )
        clf_xgb_depth = st.slider(
            "Classifier XGB max_depth",
            2,
            10,
            2,
            1,
            key="ml_clf_xgb_depth",
            disabled=classifier_choice != "XGBoost",
        )
        clf_xgb_eval_metric = st.selectbox(
            "Classifier XGB eval_metric",
            ["aucpr", "logloss"],
            index=0,
            key="ml_clf_xgb_eval_metric",
            disabled=classifier_choice != "XGBoost",
        )
        clf_xgb_auto_scale_pos_weight = st.checkbox(
            "Classifier XGB auto scale_pos_weight (per split)",
            value=True,
            key="ml_clf_xgb_auto_spw",
            disabled=classifier_choice != "XGBoost",
        )
        clf_rf_n_estimators = st.slider(
            "Classifier RF n_estimators",
            50,
            600,
            250,
            10,
            key="ml_clf_rf_n_estimators",
            disabled=classifier_choice != "Random Forest",
        )
        clf_rf_depth = st.slider(
            "Classifier RF max_depth (0=None)",
            0,
            30,
            0,
            1,
            key="ml_clf_rf_depth",
            disabled=classifier_choice != "Random Forest",
        )

    def make_classifier(y_fit=None):
        y_fit_arr = np.asarray(y_fit, dtype=int) if y_fit is not None else None
        if classifier_choice == "XGBoost":
            spw = 1.0
            if clf_xgb_auto_scale_pos_weight and y_fit_arr is not None:
                pos = float(np.sum(y_fit_arr == 1))
                neg = float(np.sum(y_fit_arr == 0))
                if pos > 0:
                    spw = neg / pos
            return xgb.XGBClassifier(
                n_estimators=int(clf_xgb_n_estimators),
                learning_rate=float(clf_xgb_lr),
                max_depth=int(clf_xgb_depth),
                min_child_weight=1,
                random_state=42,
                objective="binary:logistic",
                eval_metric=clf_xgb_eval_metric,
                scale_pos_weight=spw,
            )
        if classifier_choice == "Random Forest":
            max_depth = None if int(clf_rf_depth) <= 0 else int(clf_rf_depth)
            return RandomForestClassifier(
                n_estimators=int(clf_rf_n_estimators),
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
            )
        return LogisticRegression(max_iter=2000, random_state=42)

    if st.button("Train ML Model", key="ml_train_button"):
        with st.spinner("Training in progress..."):
            if (
                use_two_stage
                and use_log1p
                and selected_model == "XGBoost"
                and use_huber
            ):
                st.warning(
                    "Two-stage + log1p with XGBoost Huber can collapse to zeros. "
                    "Disabling Huber for this run."
                )
                use_huber = False
                huber_slope = None
            cv_mae_mean = None
            cv_rmse_mean = None
            cv_r2_mean = None
            kfold = None
            group_splits = None
            if do_cv:
                if cv_type == "GroupKFold (Station)":
                    if groups_train.nunique() < n_splits:
                        st.error(
                            f"Need at least {n_splits} unique stations for GroupKFold, found {groups_train.nunique()}."
                        )
                        return
                    group_splits = list(
                        balanced_group_kfold_splits(
                            groups_train, n_splits=n_splits, random_state=42
                        )
                    )
                    split_iter = group_splits
                elif cv_type == "GroupKFold (Year)":
                    if groups_train.nunique() < n_splits:
                        st.error(
                            f"Need at least {n_splits} unique years for GroupKFold, found {groups_train.nunique()}."
                        )
                        return
                    group_splits = list(
                        balanced_group_kfold_splits(
                            groups_train, n_splits=n_splits, random_state=42
                        )
                    )
                    split_iter = group_splits
                elif cv_type == "TimeSeriesSplit":
                    kfold = TimeSeriesSplit(n_splits=n_splits)
                    split_iter = kfold.split(X_train, y_train)
                else:
                    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    split_iter = kfold.split(X_train, y_train)

                cv_mae_scores = []
                cv_rmse_scores = []
                cv_r2_scores = []

                for tr_idx, va_idx in split_iter:
                    X_tr = X_train.iloc[tr_idx]
                    X_va = X_train.iloc[va_idx]
                    y_tr_raw = y_train_raw.iloc[tr_idx]
                    y_va_raw = y_train_raw.iloc[va_idx]

                    scaler_cv = (
                        StandardScaler()
                        if use_scaler
                        else StandardScaler(with_mean=False, with_std=False)
                    )
                    X_tr_scaled = scaler_cv.fit_transform(X_tr)
                    X_va_scaled = scaler_cv.transform(X_va)

                    if use_two_stage:
                        y_tr_wet = y_tr_raw > dry_threshold
                        if y_tr_wet.sum() == 0:
                            st.error(
                                "No wet samples in a CV fold. Adjust dry threshold."
                            )
                            return

                        clf = make_classifier(y_tr_wet)
                        clf.fit(X_tr_scaled, y_tr_wet)

                        reg = make_regressor()
                        y_reg_tr = y_tr_raw[y_tr_wet]
                        X_reg_tr = X_tr_scaled[y_tr_wet.to_numpy()]
                        if use_log1p:
                            y_reg_tr = np.log1p(y_reg_tr)
                        reg.fit(X_reg_tr, y_reg_tr)

                        prob_wet = clf.predict_proba(X_va_scaled)[:, 1]
                        y_pred_fold = reg.predict(X_va_scaled)
                        if use_log1p:
                            y_pred_fold = np.expm1(y_pred_fold)
                        y_pred_fold = _apply_two_stage_gate(
                            y_pred=y_pred_fold,
                            prob_wet=prob_wet,
                            prob_threshold=prob_threshold,
                            gate_mode=two_stage_gate_mode,
                            soft_gate_gamma=soft_gate_gamma,
                        )
                    else:
                        reg = make_regressor()
                        y_tr = np.log1p(y_tr_raw) if use_log1p else y_tr_raw
                        reg.fit(X_tr_scaled, y_tr)
                        y_pred_fold = reg.predict(X_va_scaled)
                        if use_log1p:
                            y_pred_fold = np.expm1(y_pred_fold)
                        y_pred_fold = np.maximum(y_pred_fold, 0)

                    cv_mae_scores.append(mean_absolute_error(y_va_raw, y_pred_fold))
                    cv_rmse_scores.append(
                        float(np.sqrt(mean_squared_error(y_va_raw, y_pred_fold)))
                    )
                    cv_r2_scores.append(r2_score(y_va_raw, y_pred_fold))

                cv_mae_mean = float(np.mean(cv_mae_scores))
                cv_rmse_mean = float(np.mean(cv_rmse_scores))
                cv_r2_mean = float(np.mean(cv_r2_scores))

            if use_two_stage:
                y_train_wet = y_train_raw > dry_threshold
                if y_train_wet.sum() == 0:
                    st.error("No wet samples to train regressor. Adjust dry threshold.")
                    return
                clf = make_classifier(y_train_wet)
                clf.fit(X_train_scaled, y_train_wet)

                reg = make_regressor()
                X_reg_tr = X_train_scaled[y_train_wet.to_numpy()]
                y_reg_tr = y_train_raw[y_train_wet]
                if use_log1p:
                    y_reg_tr = np.log1p(y_reg_tr)
                reg.fit(X_reg_tr, y_reg_tr)

                prob_wet = clf.predict_proba(X_test_scaled)[:, 1]
                y_pred = reg.predict(X_test_scaled)
                if use_log1p:
                    y_pred = np.expm1(y_pred)
                y_pred = _apply_two_stage_gate(
                    y_pred=y_pred,
                    prob_wet=prob_wet,
                    prob_threshold=prob_threshold,
                    gate_mode=two_stage_gate_mode,
                    soft_gate_gamma=soft_gate_gamma,
                )
                wet_dry_diagnostics = compute_wet_dry_diagnostics(
                    y_test_raw,
                    prob_wet,
                    dry_threshold=dry_threshold,
                    prob_threshold=prob_threshold,
                )
            else:
                reg = make_regressor()
                reg.fit(X_train_scaled, y_train)
                y_pred = reg.predict(X_test_scaled)
                if use_log1p:
                    y_pred = np.expm1(y_pred)
                y_pred = np.maximum(y_pred, 0)
                wet_dry_diagnostics = None

            if use_two_stage:
                joblib.dump(clf, "trained_ml_classifier.pkl")
                joblib.dump(reg, "trained_ml_model.pkl")
            else:
                joblib.dump(reg, "trained_ml_model.pkl")
            joblib.dump(scaler, "ml_scaler.pkl")
            st.write(
                "Model and scaler saved to 'trained_ml_model.pkl' and 'ml_scaler.pkl'"
            )

        st.subheader("Model Performance")
        st.session_state["ml_results"] = {
            "y_test": y_test_raw,
            "y_pred": y_pred,
            "n_splits": n_splits,
            "model": selected_model,
            "cv_mae_mean": cv_mae_mean,
            "cv_rmse_mean": cv_rmse_mean,
            "cv_r2_mean": cv_r2_mean,
            "wet_dry_diagnostics": wet_dry_diagnostics,
        }

        display_metrics(y_test_raw, y_pred)
        if do_cv:
            st.write(f"**{n_splits}-Fold CV Mean MAE:** {cv_mae_mean:.4f}")
            st.write(f"**{n_splits}-Fold CV Mean RMSE:** {cv_rmse_mean:.4f}")
            st.write(f"**{n_splits}-Fold CV Mean R2:** {cv_r2_mean:.4f}")

        st.subheader("Predictions vs Actual Values")
        plot_predictions(y_test_raw, y_pred)

        if do_cv and (kfold is not None or group_splits is not None):
            st.subheader("Learning Curve")
            cv_for_learning_curve = (
                group_splits if group_splits is not None else kfold
            )
            groups_for_learning_curve = (
                groups_train if group_splits is not None else None
            )
            plot_learning_curve(
                make_regressor(),
                X_train,
                y_train,
                cv_for_learning_curve,
                title=f"Learning Curve ({selected_model})",
                groups=groups_for_learning_curve,
                use_scaler=use_scaler,
            )

        st.subheader("Residual Plot (Error Analysis)")
        plot_residuals(y_test_raw, y_pred)
        if wet_dry_diagnostics is not None:
            display_wet_dry_diagnostics(wet_dry_diagnostics)

        meta_path = "trained_ml_model.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target_transform": "log1p" if use_log1p else "none",
                    "two_stage": bool(use_two_stage),
                    "dry_threshold": float(dry_threshold),
                    "prob_threshold": float(prob_threshold),
                    "two_stage_gate_mode": two_stage_gate_mode,
                    "soft_gate_gamma": float(soft_gate_gamma),
                    "classifier_model": classifier_choice if use_two_stage else None,
                    "classifier_xgb_n_estimators": int(clf_xgb_n_estimators),
                    "classifier_xgb_learning_rate": float(clf_xgb_lr),
                    "classifier_xgb_max_depth": int(clf_xgb_depth),
                    "classifier_xgb_eval_metric": clf_xgb_eval_metric,
                    "classifier_xgb_auto_scale_pos_weight": bool(
                        clf_xgb_auto_scale_pos_weight
                    ),
                    "classifier_rf_n_estimators": int(clf_rf_n_estimators),
                    "classifier_rf_max_depth": int(clf_rf_depth),
                    "use_scaler": bool(use_scaler),
                    "classifier_path": (
                        "trained_ml_classifier.pkl" if use_two_stage else None
                    ),
                },
                f,
            )
