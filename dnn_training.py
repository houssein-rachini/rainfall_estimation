import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay

TARGET_COL = "GROUND"
from model_utils import add_location_dummies, enrich_datetime_columns
from model_utils import balanced_group_kfold_splits


def create_dnn_model(
    input_dim,
    layers_config,
    initial_learning_rate,
    weight_decay,
    optimizer_choice,
    loss_function_choice,
    huber_delta,
):
    lr_schedule = CosineDecay(
        initial_learning_rate=initial_learning_rate, decay_steps=10000, alpha=0.0005
    )
    model = Sequential()

    for i, layer in enumerate(layers_config):
        if layer["type"] == "Dense":
            model.add(
                Dense(
                    layer["units"],
                    activation=layer["activation"],
                    input_shape=(input_dim,) if i == 0 else (),
                )
            )
        elif layer["type"] == "BatchNormalization":
            model.add(BatchNormalization())
        elif layer["type"] == "Dropout":
            model.add(Dropout(layer["rate"]))

    if optimizer_choice == "AdamW":
        optimizer = AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
    elif optimizer_choice == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    elif optimizer_choice == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

    if loss_function_choice == "Huber":
        loss = Huber(delta=huber_delta)
    elif loss_function_choice == "Mean Squared Error":
        loss = "mse"
    else:
        loss = "mae"

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def train_dnn_model(
    X_train,
    X_val,
    y_train,
    y_val,
    epochs,
    initial_learning_rate,
    batch_size,
    early_stopping_patience,
    layers_config,
    weight_decay,
    optimizer_choice,
    loss_function_choice,
    huber_delta,
    scaler_choice,
    save_artifacts=True,
    verbose=1,
    target_transform="none",
    use_two_stage=False,
    dry_threshold=1.0,
    prob_threshold=0.5,
    two_stage_gate_mode="soft_probability",
    soft_gate_gamma=1.0,
    classifier_choice="XGBoost",
    classifier_path="trained_dnn_classifier.pkl",
):
    if scaler_choice == "StandardScaler":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    dnn_model = create_dnn_model(
        X_train_scaled.shape[1],
        layers_config,
        initial_learning_rate,
        weight_decay,
        optimizer_choice,
        loss_function_choice,
        huber_delta,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )

    if target_transform == "log1p":
        y_train_raw = np.expm1(y_train)
        y_val_raw = np.expm1(y_val)
    else:
        y_train_raw = y_train
        y_val_raw = y_val

    clf = None
    if use_two_stage:
        y_train_wet = np.asarray(y_train_raw > dry_threshold)
        if y_train_wet.sum() == 0:
            raise ValueError("No wet samples to train regressor. Adjust dry threshold.")

        if classifier_choice == "XGBoost":
            clf = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=1,
                random_state=42,
                objective="binary:logistic",
                eval_metric="logloss",
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=42,
            )
        clf.fit(X_train_scaled, y_train_wet)

        X_train_reg = X_train_scaled[y_train_wet]
        y_train_reg = np.asarray(y_train)[y_train_wet]
        history = dnn_model.fit(
            X_train_reg,
            y_train_reg,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_scaled, y_val),
            verbose=verbose,
            callbacks=[early_stopping],
        )

        prob_wet = clf.predict_proba(X_val_scaled)[:, 1]
        y_pred_dnn = dnn_model.predict(X_val_scaled).flatten()
        if target_transform == "log1p":
            y_pred_out = np.expm1(y_pred_dnn)
            y_val_out = y_val_raw
        else:
            y_pred_out = y_pred_dnn
            y_val_out = y_val_raw
        y_pred_out = np.maximum(y_pred_out, 0)
        prob_wet = np.clip(np.asarray(prob_wet, dtype=float), 0.0, 1.0)
        if two_stage_gate_mode == "soft_probability":
            y_pred_out = y_pred_out * np.power(prob_wet, max(float(soft_gate_gamma), 1e-6))
        else:
            y_pred_out = y_pred_out * (prob_wet >= prob_threshold)
    else:
        history = dnn_model.fit(
            X_train_scaled,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_scaled, y_val),
            verbose=verbose,
            callbacks=[early_stopping],
        )

        y_pred_dnn = dnn_model.predict(X_val_scaled).flatten()
        if target_transform == "log1p":
            y_pred_out = np.expm1(y_pred_dnn)
            y_val_out = y_val_raw
        else:
            y_pred_out = y_pred_dnn
            y_val_out = y_val_raw
        y_pred_out = np.maximum(y_pred_out, 0)

    mae = mean_absolute_error(y_val_out, y_pred_out)
    rmse = np.sqrt(mean_squared_error(y_val_out, y_pred_out))
    r2 = r2_score(y_val_out, y_pred_out)

    if save_artifacts:
        joblib.dump(scaler, "dnn_scaler.pkl")
        dnn_model.save("trained_dnn_model.h5")
        if use_two_stage and clf is not None:
            joblib.dump(clf, classifier_path)
        st.write(
            "Model and scaler saved to 'trained_dnn_model.h5' and 'dnn_scaler.pkl'"
        )

    return y_val_out, y_pred_out, history.history, mae, rmse, r2


def plot_loss_curve(history):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["loss"], label="Training Loss", color="red")
    ax.plot(history["val_loss"], label="Validation Loss", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Curve")
    ax.legend()
    st.pyplot(fig, width="content")


def plot_results(y_val, y_pred):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=y_val, y=y_pred, alpha=0.7, ax=ax)
    ax.axline((0, 0), slope=1, color="red", linestyle="--")
    ax.set_xlabel("Actual GROUND")
    ax.set_ylabel("Predicted GROUND")
    ax.set_title("Actual vs Predicted GROUND (DNN Model)")
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


def show_dnn_training_tab(df: pd.DataFrame):
    df = enrich_datetime_columns(df)
    st.title("Deep Learning Model Training")

    if "dnn_results" in st.session_state:
        results = st.session_state["dnn_results"]
        st.subheader("Previous Training Results")
        st.write(f"**Mean Absolute Error (MAE):** {results['mae']:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {results['rmse']:.4f}")
        st.write(f"**R2 Score:** {results['r2']:.4f}")
        if results.get("cv_mae_mean") is not None:
            st.write(
                f"**{results['n_splits']}-Fold CV Mean MAE:** {results['cv_mae_mean']:.4f}"
            )
            st.write(
                f"**{results['n_splits']}-Fold CV Mean RMSE:** {results['cv_rmse_mean']:.4f}"
            )
            st.write(
                f"**{results['n_splits']}-Fold CV Mean R2:** {results['cv_r2_mean']:.4f}"
            )
        st.write("### Epochs")
        st.write(pd.DataFrame(results["history"]))
        plot_loss_curve(results["history"])
        plot_results(results["y_val"], results["y_pred"])
        plot_residuals(results["y_val"], results["y_pred"])

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
        key="dnn_features",
    )
    use_location_feature = False
    if "Location" in df.columns:
        use_location_feature = st.checkbox(
            "Use Location type (one-hot feature)",
            value=True,
            key="dnn_use_location_feature",
            help="Adds one-hot columns for Location types (Coastal/Inland/etc.).",
        )

    use_log1p = st.checkbox(
        "Train on log1p(GROUND)",
        value=False,
        key="dnn_log1p",
        help="Reduces skew and downweights extreme rainfall values.",
    )

    use_two_stage = st.checkbox(
        "Two-stage model (classify dry vs wet, then regress)",
        value=False,
        key="dnn_two_stage",
        help="Helps reduce overestimation at low/zero rainfall.",
    )
    dry_threshold = st.slider(
        "Dry threshold (mm)",
        0.0,
        20.0,
        1.0,
        step=0.5,
        key="dnn_dry_threshold",
        help="Values ≤ this are treated as dry (0) in the classifier.",
    )
    prob_threshold = st.slider(
        "Wet probability threshold",
        0.0,
        1.0,
        0.5,
        step=0.05,
        key="dnn_prob_threshold",
        help="Classifier probability threshold to output a non-zero prediction.",
    )
    two_stage_gate_mode_ui = st.selectbox(
        "Two-stage output gate",
        ["Hard threshold (zero dry)", "Soft probability (expected rainfall)"],
        index=1,
        key="dnn_two_stage_gate_mode",
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
        0.1,
        key="dnn_soft_gate_gamma",
        disabled=two_stage_gate_mode != "soft_probability",
        help="Pred = wet_prob^gamma * regressor_pred. Higher gamma is more conservative.",
    )
    classifier_choice = "XGBoost"
    if use_two_stage:
        classifier_choice = st.selectbox(
            "Classifier model",
            ["XGBoost", "Random Forest"],
            key="dnn_classifier_choice",
        )

    train_df = df
    if "Location" in df.columns:
        location_options = sorted(df["Location"].dropna().astype(str).unique().tolist())
        selected_locations = st.multiselect(
            "Select Location types for train/test:",
            options=location_options,
            default=location_options,
            key="dnn_location_types",
        )
        if len(selected_locations) == 0:
            st.warning("Please select at least one Location type.")
            return
        train_df = df[df["Location"].astype(str).isin(selected_locations)]

    if len(selected_features) == 0 and not use_location_feature:
        st.warning("Please select at least one feature.")
        return

    feature_cols = selected_features + (["Location"] if use_location_feature else [])
    df_selected = train_df[feature_cols + [TARGET_COL]].dropna()
    if "Date" in train_df.columns:
        df_selected = train_df[feature_cols + [TARGET_COL, "Date"]].dropna(
            subset=feature_cols + [TARGET_COL]
        )
        df_selected = df_selected.sort_values("Date")
    X = df_selected[feature_cols]
    if use_location_feature:
        X = add_location_dummies(X)
    y_raw = np.maximum(df_selected[TARGET_COL], 0)
    y = np.log1p(y_raw) if use_log1p else y_raw
    cv_type = st.selectbox(
        "Cross-validation strategy:",
        [
            "KFold",
            "GroupKFold (Station)",
            "GroupKFold (Year)",
            "TimeSeriesSplit",
            "No CV (train on full dataset)",
        ],
        key="dnn_cv_type",
    )
    do_cv = cv_type != "No CV (train on full dataset)"
    n_splits = None
    if do_cv:
        n_splits = st.slider("Number of CV folds/splits", 2, 10, 5, key="dnn_n_splits")

    station_series = (
        train_df.loc[df_selected.index, "Station"].astype(str)
        if "Station" in train_df.columns
        else None
    )
    year_series = (
        pd.to_numeric(train_df.loc[df_selected.index, "Year"], errors="coerce")
        .fillna(-1)
        .astype(int)
        if "Year" in train_df.columns
        else None
    )

    if cv_type == "No CV (train on full dataset)":
        X_train, X_val = X, X
        y_train, y_val = y, y
        groups_train = None
        st.info(
            "Training on the full dataset without cross-validation or holdout. "
            "Reported metrics will be optimistic."
        )
    elif cv_type == "GroupKFold (Station)":
        if station_series is None:
            st.error("Station column is required for GroupKFold (Station).")
            return
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups=station_series))
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        groups_train = station_series.iloc[train_idx]
    elif cv_type == "GroupKFold (Year)":
        if year_series is None:
            st.error("Year column is required for GroupKFold (Year).")
            return
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups=year_series))
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        groups_train = year_series.iloc[train_idx]
    elif cv_type == "TimeSeriesSplit":
        n_test = max(1, int(len(X) * 0.2))
        X_train, X_val = X.iloc[:-n_test], X.iloc[-n_test:]
        y_train, y_val = y.iloc[:-n_test], y.iloc[-n_test:]
        groups_train = None
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        groups_train = None

    epochs = st.slider("Number of Epochs", 10, 500, 200, key="dnn_epochs")
    optimizer_choice = st.selectbox(
        "Select Optimizer", ["AdamW", "Adam", "SGD", "RMSprop"], key="dnn_optimizer"
    )
    scaler_choice = "StandardScaler"

    initial_learning_rate = st.number_input(
        "Initial Learning Rate",
        min_value=1e-7,
        max_value=0.1,
        value=0.001,
        step=0.0001,
        format="%.6f",
        key="dnn_lr",
    )

    weight_decay = 0.0
    if optimizer_choice == "AdamW":
        weight_decay = st.number_input(
            "Weight Decay (for AdamW)",
            min_value=0.0,
            max_value=1e-2,
            value=1e-5,
            step=1e-6,
            format="%.6f",
            key="dnn_wd",
        )

    loss_function_choice = st.selectbox(
        "Select Loss Function",
        ["Huber", "Mean Squared Error", "Mean Absolute Error"],
        key="dnn_loss_function",
    )

    huber_delta = None
    if loss_function_choice == "Huber":
        huber_delta = st.number_input(
            "Huber Loss Delta",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.1f",
            key="dnn_huber_delta",
        )

    batch_size = st.slider("Batch Size", 8, 1024, 128, key="dnn_batch_size")
    early_stopping_patience = st.slider(
        "Early Stopping Patience", 5, 1000, 10, key="dnn_patience"
    )

    st.subheader("Neural Network Architecture")
    layers = []
    num_layers = st.number_input(
        "Number of Layers", 1, 20, 3, step=1, key="dnn_num_layers"
    )

    for i in range(num_layers):
        col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
        layer_type = col1.selectbox(
            f"Layer {i+1} Type",
            ["Dense", "BatchNormalization", "Dropout"],
            key=f"dnn_type_{i}",
        )
        if layer_type == "Dense":
            units = col2.slider(f"Units {i+1}", 1, 512, 128, key=f"dnn_units_{i}")
            activation = col3.selectbox(
                f"Activation {i+1}",
                ["relu", "tanh", "sigmoid", "linear"],
                key=f"dnn_activation_{i}",
            )
            layers.append({"type": "Dense", "units": units, "activation": activation})
        elif layer_type == "Dropout":
            rate = col2.slider(
                f"Dropout Rate {i+1}", 0.0, 0.5, 0.1, key=f"dnn_dropout_{i}"
            )
            layers.append({"type": "Dropout", "rate": rate})
        else:
            layers.append({"type": "BatchNormalization"})

    if st.button("Train Model", key="dnn_train_button"):
        with st.spinner("Training the model..."):
            cv_mae_mean = None
            cv_rmse_mean = None
            cv_r2_mean = None
            if do_cv:
                if cv_type == "GroupKFold (Station)":
                    if groups_train.nunique() < n_splits:
                        st.error(
                            f"Need at least {n_splits} unique stations, found {groups_train.nunique()}."
                        )
                        return
                    split_iter = balanced_group_kfold_splits(
                        groups_train, n_splits=n_splits, random_state=42
                    )
                elif cv_type == "GroupKFold (Year)":
                    if groups_train.nunique() < n_splits:
                        st.error(
                            f"Need at least {n_splits} unique years, found {groups_train.nunique()}."
                        )
                        return
                    split_iter = balanced_group_kfold_splits(
                        groups_train, n_splits=n_splits, random_state=42
                    )
                elif cv_type == "TimeSeriesSplit":
                    split_iter = TimeSeriesSplit(n_splits=n_splits).split(
                        X_train, y_train
                    )
                else:
                    split_iter = KFold(
                        n_splits=n_splits, shuffle=True, random_state=42
                    ).split(X_train, y_train)

                cv_mae_scores = []
                cv_rmse_scores = []
                cv_r2_scores = []
                for fold_train_idx, fold_val_idx in split_iter:
                    X_tr = X_train.iloc[fold_train_idx]
                    X_va = X_train.iloc[fold_val_idx]
                    y_tr = y_train.iloc[fold_train_idx]
                    y_va = y_train.iloc[fold_val_idx]
                    _, _, _, fold_mae, fold_rmse, fold_r2 = train_dnn_model(
                        X_tr,
                        X_va,
                        y_tr,
                        y_va,
                        epochs,
                        initial_learning_rate,
                        batch_size,
                        early_stopping_patience,
                        layers,
                        weight_decay,
                        optimizer_choice,
                        loss_function_choice,
                        huber_delta,
                        scaler_choice,
                        save_artifacts=False,
                        verbose=0,
                        target_transform="log1p" if use_log1p else "none",
                        use_two_stage=use_two_stage,
                        dry_threshold=dry_threshold,
                        prob_threshold=prob_threshold,
                        two_stage_gate_mode=two_stage_gate_mode,
                        soft_gate_gamma=soft_gate_gamma,
                        classifier_choice=classifier_choice,
                    )
                    cv_mae_scores.append(float(fold_mae))
                    cv_rmse_scores.append(float(fold_rmse))
                    cv_r2_scores.append(float(fold_r2))

                cv_mae_mean = float(np.mean(cv_mae_scores))
                cv_rmse_mean = float(np.mean(cv_rmse_scores))
                cv_r2_mean = float(np.mean(cv_r2_scores))

            y_val_out, y_pred_dnn, history, mae, rmse, r2 = train_dnn_model(
                X_train,
                X_val,
                y_train,
                y_val,
                epochs,
                initial_learning_rate,
                batch_size,
                early_stopping_patience,
                layers,
                weight_decay,
                optimizer_choice,
                loss_function_choice,
                huber_delta,
                scaler_choice,
                save_artifacts=True,
                verbose=1,
                target_transform="log1p" if use_log1p else "none",
                use_two_stage=use_two_stage,
                dry_threshold=dry_threshold,
                prob_threshold=prob_threshold,
                two_stage_gate_mode=two_stage_gate_mode,
                soft_gate_gamma=soft_gate_gamma,
                classifier_choice=classifier_choice,
            )

        st.success("Training completed!")
        st.session_state["dnn_results"] = {
            "y_val": y_val_out,
            "y_pred": y_pred_dnn,
            "history": history,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "n_splits": n_splits,
            "cv_mae_mean": cv_mae_mean,
            "cv_rmse_mean": cv_rmse_mean,
            "cv_r2_mean": cv_r2_mean,
        }

        st.subheader("Model Performance")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.write(f"**R2 Score:** {r2:.4f}")
        if do_cv:
            st.write(f"**{n_splits}-Fold CV Mean MAE:** {cv_mae_mean:.4f}")
            st.write(f"**{n_splits}-Fold CV Mean RMSE:** {cv_rmse_mean:.4f}")
            st.write(f"**{n_splits}-Fold CV Mean R2:** {cv_r2_mean:.4f}")

        meta_path = "trained_dnn_model.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target_transform": "log1p" if use_log1p else "none",
                    "two_stage": bool(use_two_stage),
                    "dry_threshold": float(dry_threshold),
                    "prob_threshold": float(prob_threshold),
                    "two_stage_gate_mode": two_stage_gate_mode,
                    "soft_gate_gamma": float(soft_gate_gamma),
                    "classifier_path": (
                        "trained_dnn_classifier.pkl" if use_two_stage else None
                    ),
                },
                f,
            )

        st.write("### Epochs")
        st.write(pd.DataFrame(history))

        st.subheader("Training and Validation Loss Curve")
        plot_loss_curve(history)

        st.subheader("Predictions vs Actual Values")
        plot_results(y_val_out, y_pred_dnn)

        st.subheader("Residual Plot (Error Analysis)")
        plot_residuals(y_val_out, y_pred_dnn)
