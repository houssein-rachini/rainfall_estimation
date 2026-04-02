import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay

TARGET_COL = "GROUND"
from model_utils import add_location_dummies, enrich_datetime_columns
from model_utils import balanced_group_kfold_splits

DEFAULT_LAYERS = [
    {"type": "Dense", "units": 256, "activation": "relu"},
    {"type": "BatchNormalization"},
    {"type": "Dropout", "rate": 0.15},
    {"type": "Dense", "units": 128, "activation": "relu"},
    {"type": "BatchNormalization"},
    {"type": "Dropout", "rate": 0.1},
    {"type": "Dense", "units": 64, "activation": "relu"},
    {"type": "Dense", "units": 1, "activation": "relu"},
]


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
        initial_learning_rate=initial_learning_rate, decay_steps=10000, alpha=0.0001
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


def plot_loss_curve(history):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["loss"], label="DNN Training Loss")
    ax.plot(history["val_loss"], label="DNN Validation Loss")
    ax.plot(history["ensemble_train_loss"], label="Ensemble Training Loss")
    ax.plot(history["ensemble_val_loss"], label="Ensemble Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Curve")
    ax.legend()
    st.pyplot(fig, width="content")


def plot_results(y_val, y_pred):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=y_val, y=y_pred, alpha=0.7, ax=ax)
    ax.axline((0, 0), slope=1, linestyle="--")
    ax.set_xlabel("Actual GROUND")
    ax.set_ylabel("Predicted GROUND")
    ax.set_title("Actual vs Predicted GROUND (Ensemble)")
    st.pyplot(fig, width="content")


def plot_residuals(y_val, y_pred):
    residuals = y_val - y_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=y_val, y=residuals, alpha=0.7, ax=ax)
    ax.axhline(y=0, linestyle="--")
    ax.set_xlabel("Actual GROUND")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Plot")
    st.pyplot(fig, width="content")


def train_ensemble_model(
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
    alpha,
    base_model,
    base_model_params,
    scaler_choice,
    save_artifacts=True,
    persist_session=True,
    target_transform="none",
    use_two_stage=False,
    dry_threshold=1.0,
    prob_threshold=0.5,
    two_stage_gate_mode="soft_probability",
    soft_gate_gamma=1.0,
    classifier_choice="XGBoost",
    classifier_path="trained_ensemble_classifier.pkl",
):
    if scaler_choice == "StandardScaler":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if base_model == "XGBoost":
        base_model_instance = xgb.XGBRegressor(**base_model_params)
    elif base_model == "Random Forest":
        base_model_instance = RandomForestRegressor(**base_model_params)
    else:
        base_model_instance = KNeighborsRegressor(**base_model_params)

    if target_transform == "log1p":
        y_train_raw = np.expm1(y_train)
        y_val_raw = np.expm1(y_val)
    else:
        y_train_raw = y_train
        y_val_raw = y_val

    clf = None
    train_mask = np.ones(len(y_train), dtype=bool)
    if use_two_stage:
        y_train_wet = np.asarray(y_train_raw > dry_threshold)
        if y_train_wet.sum() == 0:
            raise ValueError("No wet samples to train regressor. Adjust dry threshold.")
        train_mask = y_train_wet

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

    base_model_instance.fit(X_train_scaled[train_mask], np.asarray(y_train)[train_mask])

    dnn_model = create_dnn_model(
        X_train_scaled.shape[1],
        layers_config,
        initial_learning_rate,
        weight_decay,
        optimizer_choice,
        loss_function_choice,
        huber_delta,
    )

    history = {
        "loss": [],
        "val_loss": [],
        "ensemble_train_loss": [],
        "ensemble_val_loss": [],
    }

    patience_counter = 0
    best_val_loss = float("inf")
    best_weights = dnn_model.get_weights()

    if loss_function_choice == "Huber":
        loss_fn = tf.keras.losses.Huber(delta=huber_delta)
    elif loss_function_choice == "Mean Squared Error":
        loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        loss_fn = tf.keras.losses.MeanAbsoluteError()

    for _ in range(epochs):
        dnn_model.fit(
            X_train_scaled[train_mask],
            np.asarray(y_train)[train_mask],
            epochs=1,
            batch_size=batch_size,
            validation_data=(X_val_scaled, y_val),
            verbose=0,
        )

        y_pred_dnn_train = dnn_model.predict(X_train_scaled, verbose=0).flatten()
        y_pred_dnn_val = dnn_model.predict(X_val_scaled, verbose=0).flatten()
        y_pred_base_train = base_model_instance.predict(X_train_scaled)
        y_pred_base_val = base_model_instance.predict(X_val_scaled)

        y_pred_ens_train = alpha * y_pred_dnn_train + (1 - alpha) * y_pred_base_train
        y_pred_ens_val = alpha * y_pred_dnn_val + (1 - alpha) * y_pred_base_val

        dnn_train_loss = float(loss_fn(y_train, y_pred_dnn_train).numpy())
        dnn_val_loss = float(loss_fn(y_val, y_pred_dnn_val).numpy())
        ens_train_loss = float(loss_fn(y_train, y_pred_ens_train).numpy())
        ens_val_loss = float(loss_fn(y_val, y_pred_ens_val).numpy())

        history["loss"].append(dnn_train_loss)
        history["val_loss"].append(dnn_val_loss)
        history["ensemble_train_loss"].append(ens_train_loss)
        history["ensemble_val_loss"].append(ens_val_loss)

        if ens_val_loss < best_val_loss:
            best_val_loss = ens_val_loss
            best_weights = dnn_model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    dnn_model.set_weights(best_weights)

    y_pred_ensemble = alpha * dnn_model.predict(X_val_scaled, verbose=0).flatten() + (
        1 - alpha
    ) * base_model_instance.predict(X_val_scaled)

    if target_transform == "log1p":
        y_pred_out = np.expm1(y_pred_ensemble)
        y_val_out = y_val_raw
    else:
        y_pred_out = y_pred_ensemble
        y_val_out = y_val_raw

    y_pred_out = np.maximum(y_pred_out, 0)
    if use_two_stage and clf is not None:
        prob_wet = clf.predict_proba(X_val_scaled)[:, 1]
        prob_wet = np.clip(np.asarray(prob_wet, dtype=float), 0.0, 1.0)
        if two_stage_gate_mode == "soft_probability":
            y_pred_out = y_pred_out * np.power(prob_wet, max(float(soft_gate_gamma), 1e-6))
        else:
            y_pred_out = y_pred_out * (prob_wet >= prob_threshold)

    mae = mean_absolute_error(y_val_out, y_pred_out)
    rmse = np.sqrt(mean_squared_error(y_val_out, y_pred_out))
    r2 = r2_score(y_val_out, y_pred_out)

    if save_artifacts:
        joblib.dump(scaler, "ensemble_scaler.pkl")
        if base_model == "XGBoost":
            base_model_instance.save_model("trained_ensemble_xgb_model.json")
            dnn_model.save("trained_ensemble_xgb_dnn_model.h5")
        elif base_model == "Random Forest":
            joblib.dump(base_model_instance, "trained_ensemble_rf_model.pkl")
            dnn_model.save("trained_ensemble_rf_dnn_model.h5")
        else:
            joblib.dump(base_model_instance, "trained_ensemble_knn_model.pkl")
            dnn_model.save("trained_ensemble_knn_dnn_model.h5")
        if use_two_stage and clf is not None:
            joblib.dump(clf, classifier_path)

    if persist_session:
        st.session_state["ensemble_results"] = {
            "y_val": y_val_out,
            "y_pred": y_pred_out,
            "history": history,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    if save_artifacts:
        val_results = pd.DataFrame(
            {"Actual_GROUND": y_val_out, "Predicted_GROUND": y_pred_out}
        )
        val_results.to_csv("ensemble_validation_results.csv", index=False)

    return y_val_out, y_pred_out, history, mae, rmse, r2


def show_ensemble_training_tab(df: pd.DataFrame):
    df = enrich_datetime_columns(df)
    st.title("Ensemble Model Training")

    if "ensemble_results" in st.session_state:
        results = st.session_state["ensemble_results"]
        st.subheader("Previous Training Results")
        st.write(f"**MAE:** {results['mae']:.4f}")
        st.write(f"**RMSE:** {results['rmse']:.4f}")
        st.write(f"**R2:** {results['r2']:.4f}")
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
        key="ensemble_features",
    )
    use_location_feature = False
    if "Location" in df.columns:
        use_location_feature = st.checkbox(
            "Use Location type (one-hot feature)",
            value=True,
            key="ensemble_use_location_feature",
            help="Adds one-hot columns for Location types (Coastal/Inland/etc.).",
        )

    use_log1p = st.checkbox(
        "Train on log1p(GROUND)",
        value=False,
        key="ensemble_log1p",
        help="Reduces skew and downweights extreme rainfall values.",
    )

    use_two_stage = st.checkbox(
        "Two-stage model (classify dry vs wet, then regress)",
        value=False,
        key="ensemble_two_stage",
        help="Helps reduce overestimation at low/zero rainfall.",
    )
    dry_threshold = st.slider(
        "Dry threshold (mm)",
        0.0,
        20.0,
        1.0,
        step=0.5,
        key="ensemble_dry_threshold",
        help="Values ≤ this are treated as dry (0) in the classifier.",
    )
    prob_threshold = st.slider(
        "Wet probability threshold",
        0.0,
        1.0,
        0.5,
        step=0.05,
        key="ensemble_prob_threshold",
        help="Classifier probability threshold to output a non-zero prediction.",
    )
    two_stage_gate_mode_ui = st.selectbox(
        "Two-stage output gate",
        ["Hard threshold (zero dry)", "Soft probability (expected rainfall)"],
        index=1,
        key="ensemble_two_stage_gate_mode",
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
        key="ensemble_soft_gate_gamma",
        disabled=two_stage_gate_mode != "soft_probability",
        help="Pred = wet_prob^gamma * regressor_pred. Higher gamma is more conservative.",
    )
    classifier_choice = "XGBoost"
    if use_two_stage:
        classifier_choice = st.selectbox(
            "Classifier model",
            ["XGBoost", "Random Forest"],
            key="ensemble_classifier_choice",
        )

    train_df = df
    if "Location" in df.columns:
        location_options = sorted(df["Location"].dropna().astype(str).unique().tolist())
        selected_locations = st.multiselect(
            "Select Location types for train/test:",
            options=location_options,
            default=location_options,
            key="ensemble_location_types",
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
        key="ensemble_cv_type",
    )
    do_cv = cv_type != "No CV (train on full dataset)"
    n_splits = None
    if do_cv:
        n_splits = st.slider(
            "Number of CV folds/splits", 2, 10, 5, key="ensemble_n_splits"
        )

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

    alpha = st.slider(
        "Ensemble Weight (DNN contribution)", 0.0, 1.0, 0.4, key="ensemble_alpha"
    )
    epochs = st.slider("Number of Epochs", 10, 600, 300, key="ensemble_epochs")
    optimizer_choice = st.selectbox(
        "Select Optimizer",
        ["AdamW", "Adam", "SGD", "RMSprop"],
        key="ensemble_optimizer",
    )
    scaler_choice = "StandardScaler"

    initial_learning_rate = st.number_input(
        "Initial Learning Rate",
        min_value=1e-7,
        max_value=0.1,
        value=0.0005,
        step=0.0001,
        format="%.6f",
        key="ensemble_lr",
    )

    weight_decay = 0.0
    if optimizer_choice == "AdamW":
        weight_decay = st.number_input(
            "Weight Decay (AdamW)",
            min_value=0.0,
            max_value=1e-2,
            value=1e-6,
            step=1e-6,
            format="%.6f",
            key="ensemble_wd",
        )

    loss_function_choice = st.selectbox(
        "Select Loss Function",
        ["Huber", "Mean Squared Error", "Mean Absolute Error"],
        key="ensemble_loss",
    )

    huber_delta = 1.0
    if loss_function_choice == "Huber":
        huber_delta = st.number_input(
            "Huber Loss Delta",
            min_value=0.001,
            max_value=10.0,
            value=1.0,
            step=0.001,
            format="%.3f",
            key="ensemble_huber_delta",
        )

    batch_size = st.slider("Batch Size", 8, 1024, 128, key="ensemble_batch_size")
    early_stopping_patience = st.slider(
        "Early Stopping Patience", 5, 50, 20, key="ensemble_patience"
    )

    st.subheader("Neural Network Architecture")
    if "layers_config" not in st.session_state:
        st.session_state["layers_config"] = DEFAULT_LAYERS.copy()

    layers = []
    num_layers = st.number_input(
        "Number of Layers",
        1,
        20,
        len(st.session_state["layers_config"]),
        step=1,
        key="ensemble_num_layers",
    )

    default_layers = st.session_state["layers_config"]
    layer_types = ["Dense", "BatchNormalization", "Dropout"]
    activation_choices = ["relu", "tanh", "sigmoid", "linear", "softplus"]

    for i in range(int(num_layers)):
        layer_defaults = (
            default_layers[i] if i < len(default_layers) else {"type": "Dense"}
        )
        col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
        layer_type = col1.selectbox(
            f"Layer {i+1} Type",
            layer_types,
            index=(
                layer_types.index(layer_defaults.get("type", "Dense"))
                if layer_defaults.get("type", "Dense") in layer_types
                else 0
            ),
            key=f"ensemble_type_{i}",
        )
        if layer_type == "Dense":
            units = col2.slider(
                f"Units {i+1}",
                1,
                512,
                int(layer_defaults.get("units", 128)),
                key=f"ensemble_units_{i}",
            )
            activation = col3.selectbox(
                f"Activation {i+1}",
                activation_choices,
                index=(
                    activation_choices.index(layer_defaults.get("activation", "relu"))
                    if layer_defaults.get("activation", "relu") in activation_choices
                    else 0
                ),
                key=f"ensemble_activation_{i}",
            )
            layers.append({"type": "Dense", "units": units, "activation": activation})
        elif layer_type == "Dropout":
            rate = col2.slider(
                f"Dropout Rate {i+1}",
                0.0,
                0.5,
                float(layer_defaults.get("rate", 0.1)),
                key=f"ensemble_dropout_{i}",
            )
            layers.append({"type": "Dropout", "rate": rate})
        else:
            layers.append({"type": "BatchNormalization"})

    st.session_state["layers_config"] = layers

    st.subheader("Base Model")
    base_model = st.selectbox(
        "Select Base Model",
        ["XGBoost", "Random Forest", "KNN Regressor"],
        key="ensemble_base_model",
    )

    if base_model == "XGBoost":
        base_model_params = {
            "learning_rate": st.slider(
                "XGB Learning Rate", 0.01, 0.5, 0.05, key="ensemble_xgb_learning_rate"
            ),
            "max_depth": st.slider(
                "XGB Max Depth", 3, 10, 6, key="ensemble_xgb_max_depth"
            ),
            "n_estimators": st.slider(
                "XGB Trees", 50, 500, 200, key="ensemble_xgb_n_estimators"
            ),
            "min_child_weight": st.slider(
                "XGB Min Child Weight", 1, 10, 2, key="ensemble_xgb_min_child_weight"
            ),
            "random_state": 42,
            "objective": "reg:squarederror",
        }
    elif base_model == "Random Forest":
        base_model_params = {
            "n_estimators": st.slider(
                "RF Trees", 50, 300, 150, key="ensemble_rf_n_estimators"
            ),
            "min_samples_split": st.slider(
                "RF Min Samples Split", 2, 10, 2, key="ensemble_rf_min_samples_split"
            ),
            "min_samples_leaf": st.slider(
                "RF Min Samples Leaf", 1, 10, 1, key="ensemble_rf_min_samples_leaf"
            ),
            "random_state": 42,
        }
    else:
        base_model_params = {
            "n_neighbors": st.slider(
                "KNN Neighbors", 1, 20, 4, key="ensemble_knn_neighbors"
            ),
            "metric": st.selectbox(
                "KNN Distance Metric",
                ["manhattan", "euclidean", "minkowski"],
                key="ensemble_knn_metric",
            ),
        }

    if st.button("Train Model", key="ensemble_train_button"):
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
                    _, _, _, fold_mae, fold_rmse, fold_r2 = train_ensemble_model(
                        X_tr,
                        X_va,
                        y_tr,
                        y_va,
                        epochs,
                        initial_learning_rate,
                        batch_size,
                        early_stopping_patience,
                        st.session_state["layers_config"],
                        weight_decay,
                        optimizer_choice,
                        loss_function_choice,
                        huber_delta,
                        alpha,
                        base_model,
                        base_model_params,
                        scaler_choice,
                        save_artifacts=False,
                        persist_session=False,
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

            y_val_out, y_pred_ensemble, history, mae, rmse, r2 = train_ensemble_model(
                X_train,
                X_val,
                y_train,
                y_val,
                epochs,
                initial_learning_rate,
                batch_size,
                early_stopping_patience,
                st.session_state["layers_config"],
                weight_decay,
                optimizer_choice,
                loss_function_choice,
                huber_delta,
                alpha,
                base_model,
                base_model_params,
                scaler_choice,
                save_artifacts=True,
                persist_session=True,
                target_transform="log1p" if use_log1p else "none",
                use_two_stage=use_two_stage,
                dry_threshold=dry_threshold,
                prob_threshold=prob_threshold,
                two_stage_gate_mode=two_stage_gate_mode,
                soft_gate_gamma=soft_gate_gamma,
                classifier_choice=classifier_choice,
            )

        st.success("Training completed!")
        st.session_state["ensemble_results"]["n_splits"] = n_splits
        st.session_state["ensemble_results"]["cv_mae_mean"] = cv_mae_mean
        st.session_state["ensemble_results"]["cv_rmse_mean"] = cv_rmse_mean
        st.session_state["ensemble_results"]["cv_r2_mean"] = cv_r2_mean
        st.subheader("Model Performance")
        st.write(f"**MAE:** {mae:.4f}")
        st.write(f"**RMSE:** {rmse:.4f}")
        st.write(f"**R2:** {r2:.4f}")
        if do_cv:
            st.write(f"**{n_splits}-Fold CV Mean MAE:** {cv_mae_mean:.4f}")
            st.write(f"**{n_splits}-Fold CV Mean RMSE:** {cv_rmse_mean:.4f}")
            st.write(f"**{n_splits}-Fold CV Mean R2:** {cv_r2_mean:.4f}")

        meta_path = "trained_ensemble.meta.json"
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
                        "trained_ensemble_classifier.pkl" if use_two_stage else None
                    ),
                },
                f,
            )
        st.write("### Epochs")
        st.write(pd.DataFrame(history))

        st.subheader("Training and Validation Loss Curve")
        plot_loss_curve(history)

        st.subheader("Actual vs Predicted Scatter Plot")
        plot_results(y_val_out, y_pred_ensemble)

        st.subheader("Residual Plot (Error Analysis)")
        plot_residuals(y_val_out, y_pred_ensemble)
