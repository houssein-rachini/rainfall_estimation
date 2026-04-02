import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, KFold, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from model_utils import TARGET_COL, balanced_group_kfold_splits, enrich_datetime_columns


def _build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
    group_by_station: bool,
    require_continuous_monthly: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    station_list = []
    year_list = []
    time_list = []

    if group_by_station and "Station" in df.columns:
        grouped = [g for _, g in df.groupby("Station", dropna=True)]
    else:
        grouped = [df]

    for g in grouped:
        cols = feature_cols + [target_col]
        if "Date" in g.columns:
            cols = cols + ["Date"]
        if "Station" in g.columns:
            cols = cols + ["Station"]
        if "Year" in g.columns:
            cols = cols + ["Year"]

        g_sorted = g.sort_values("Date").dropna(subset=feature_cols + [target_col]).copy()
        if len(g_sorted) <= seq_len:
            continue

        x_vals = g_sorted[feature_cols].to_numpy(dtype="float32")
        y_vals = g_sorted[target_col].to_numpy(dtype="float32")
        station_vals = (
            g_sorted["Station"].astype(str).to_numpy() if "Station" in g_sorted.columns else np.array(["all"] * len(g_sorted))
        )
        year_vals = (
            pd.to_numeric(g_sorted["Year"], errors="coerce").fillna(-1).astype(int).to_numpy()
            if "Year" in g_sorted.columns
            else np.array([-1] * len(g_sorted))
        )
        if "Date" in g_sorted.columns:
            time_vals = pd.to_datetime(g_sorted["Date"], errors="coerce").to_numpy()
        else:
            time_vals = np.arange(len(g_sorted))

        month_ord = None
        if require_continuous_monthly and "Date" in g_sorted.columns:
            month_ord = pd.to_datetime(g_sorted["Date"], errors="coerce").dt.to_period("M").astype(int).to_numpy()

        for i in range(seq_len, len(g_sorted)):
            if month_ord is not None:
                start = i - seq_len
                end = i
                if not np.all(np.diff(month_ord[start : end + 1]) == 1):
                    continue
            X_list.append(x_vals[i - seq_len : i, :])
            y_list.append(y_vals[i])
            station_list.append(station_vals[i])
            year_list.append(year_vals[i])
            time_list.append(time_vals[i])

    if not X_list:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype="float32"),
            np.empty((0,), dtype="float32"),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    X = np.asarray(X_list, dtype="float32")
    y = np.asarray(y_list, dtype="float32")
    stations = np.asarray(station_list, dtype=object)
    years = np.asarray(year_list, dtype=int)
    times = np.asarray(time_list)

    if np.issubdtype(times.dtype, np.datetime64):
        order = np.argsort(times.astype("datetime64[ns]"))
    else:
        order = np.argsort(times.astype(float))
    return X[order], y[order], stations[order], years[order], times[order]


def _display_metrics(prefix: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    st.write(f"### {prefix}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
    st.write(f"**R2 Score:** {r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _fit_transform_sequences(
    X_train: np.ndarray,
    X_val: np.ndarray,
    use_scaler: bool,
) -> tuple[np.ndarray, np.ndarray, StandardScaler | None]:
    if not use_scaler:
        return X_train, X_val, None
    scaler = StandardScaler()
    n_features = X_train.shape[2]
    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)
    X_train_s = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_s = scaler.transform(X_val_2d).reshape(X_val.shape)
    return X_train_s, X_val_s, scaler


def _make_lstm_model(
    input_shape: tuple[int, int],
    units: int,
    dropout: float,
    learning_rate: float,
    loss_name: str,
):
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            LSTM(int(units), input_shape=input_shape),
            Dropout(float(dropout)),
            Dense(1),
        ]
    )
    loss_map = {
        "mse": "mse",
        "mae": "mae",
        "huber": Huber(),
    }
    model.compile(optimizer=Adam(learning_rate=float(learning_rate)), loss=loss_map[loss_name])
    return model


def _make_holdout_split(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    station_seq: np.ndarray,
    year_seq: np.ndarray,
    cv_type: str,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx_all = np.arange(len(X_seq))
    if cv_type == "GroupKFold (Station)":
        gss = GroupShuffleSplit(n_splits=1, test_size=float(test_ratio), random_state=42)
        train_idx, test_idx = next(gss.split(X_seq, y_seq, groups=station_seq.astype(str)))
    elif cv_type == "GroupKFold (Year)":
        gss = GroupShuffleSplit(n_splits=1, test_size=float(test_ratio), random_state=42)
        train_idx, test_idx = next(gss.split(X_seq, y_seq, groups=year_seq.astype(str)))
    elif cv_type == "TimeSeriesSplit":
        n_test = max(1, int(len(X_seq) * float(test_ratio)))
        train_idx = idx_all[:-n_test]
        test_idx = idx_all[-n_test:]
    else:
        train_idx, test_idx = train_test_split(
            idx_all,
            test_size=float(test_ratio),
            random_state=42,
            shuffle=True,
        )
    return X_seq[train_idx], X_seq[test_idx], y_seq[train_idx], y_seq[test_idx], train_idx, test_idx


def show_lstm_training_tab(df: pd.DataFrame) -> None:
    try:
        import tensorflow as tf  # noqa: F401
    except Exception:
        st.error("TensorFlow is required for LSTM tab. Install tensorflow in this environment.")
        return

    df = enrich_datetime_columns(df)
    st.title("LSTM Training")
    st.caption("Train/test LSTM for monthly rainfall with optional cross-validation.")

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
            "month_sin",
            "month_cos",
        ]
        if c in numeric_cols
    ]
    feature_cols = st.multiselect(
        "Features",
        options=numeric_cols,
        default=default_cols,
        key="lstm_features",
    )
    if not feature_cols:
        st.warning("Select at least one feature.")
        return

    seq_len = st.slider("Sequence length (months)", 3, 24, 12, 1, key="lstm_seq_len")
    test_ratio = st.slider("Test ratio", 0.1, 0.4, 0.2, 0.05, key="lstm_test_ratio")
    group_by_station = st.checkbox(
        "Build sequences independently per Station",
        value=True,
        key="lstm_group_station",
    )
    require_continuous_monthly = st.checkbox(
        "Require continuous monthly sequences",
        value=True,
        key="lstm_require_continuous_monthly",
        help="If enabled, each window must be consecutive monthly timesteps with no calendar gaps.",
    )
    use_scaler = st.checkbox("Use StandardScaler", value=True, key="lstm_use_scaler")

    cv_type = st.selectbox(
        "Cross-validation strategy",
        [
            "KFold",
            "GroupKFold (Station)",
            "GroupKFold (Year)",
            "TimeSeriesSplit",
            "No CV (train on holdout split only)",
        ],
        key="lstm_cv_type",
    )
    do_cv = cv_type != "No CV (train on holdout split only)"
    n_splits = 5
    if do_cv:
        n_splits = st.slider("Number of CV folds/splits", 2, 10, 5, 1, key="lstm_n_splits")

    units = st.slider("LSTM units", 16, 256, 64, 16, key="lstm_units")
    dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05, key="lstm_dropout")
    learning_rate = st.slider(
        "Learning rate",
        0.0001,
        0.01,
        0.001,
        0.0001,
        format="%.4f",
        key="lstm_lr",
    )
    epochs = st.slider("Epochs", 5, 300, 60, 5, key="lstm_epochs")
    batch_size = st.slider("Batch size", 8, 256, 32, 8, key="lstm_batch")
    loss_ui = st.selectbox("Loss", ["MSE", "MAE", "Huber"], index=0, key="lstm_loss")
    loss_name = {"MSE": "mse", "MAE": "mae", "Huber": "huber"}[loss_ui]
    use_early_stopping = st.checkbox("Use EarlyStopping", value=True, key="lstm_use_early_stopping")
    early_stopping_patience = st.slider(
        "EarlyStopping patience",
        2,
        30,
        10,
        1,
        key="lstm_early_stopping_patience",
        disabled=not use_early_stopping,
    )
    verbose_logs = st.checkbox("Print epoch logs", value=False, key="lstm_verbose_logs")

    base = df.dropna(subset=[TARGET_COL] + feature_cols).copy()
    if "Date" in base.columns:
        base = base.dropna(subset=["Date"]).sort_values("Date")
    if base.empty:
        st.error("No rows after dropping NA.")
        return

    X_seq, y_seq, station_seq, year_seq, _ = _build_sequences(
        base,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        seq_len=int(seq_len),
        group_by_station=bool(group_by_station),
        require_continuous_monthly=bool(require_continuous_monthly),
    )
    if len(y_seq) < 20:
        st.error("Too few sequences. Increase data or reduce sequence length.")
        return

    if st.button("Train LSTM", key="lstm_train_btn"):
        st.write(f"Prepared sequences: **{len(y_seq)}**")
        X_train, X_test, y_train, y_test, train_idx, _ = _make_holdout_split(
            X_seq=X_seq,
            y_seq=y_seq,
            station_seq=station_seq,
            year_seq=year_seq,
            cv_type=cv_type,
            test_ratio=float(test_ratio),
        )
        st.write(f"Holdout split | train={len(X_train)} test={len(X_test)}")
        if len(X_train) < 10 or len(X_test) < 1:
            st.error("Split produced too few samples. Adjust sequence length or test ratio.")
            return

        progress_bar = st.progress(0)
        progress_text = st.empty()

        def _fit_one(model, X_tr_local, y_tr_local, fold_label: str):
            callbacks = []
            if use_early_stopping:
                from tensorflow.keras.callbacks import EarlyStopping

                callbacks.append(
                    EarlyStopping(
                        monitor="val_loss",
                        patience=int(early_stopping_patience),
                        restore_best_weights=True,
                    )
                )
            if verbose_logs:
                from tensorflow.keras.callbacks import Callback

                class StreamlitEpochLogger(Callback):
                    def __init__(self, fold_name: str, total_epochs: int):
                        super().__init__()
                        self.fold_name = fold_name
                        self.total_epochs = total_epochs

                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        loss_val = float(logs.get("loss", np.nan))
                        val_loss_val = float(logs.get("val_loss", np.nan))
                        progress_text.write(
                            f"{self.fold_name} | epoch {epoch + 1}/{self.total_epochs} "
                            f"| loss={loss_val:.4f} val_loss={val_loss_val:.4f}"
                        )

                callbacks.append(StreamlitEpochLogger(fold_label, int(epochs)))
            return model.fit(
                X_tr_local,
                y_tr_local,
                validation_split=0.1,
                epochs=int(epochs),
                batch_size=int(batch_size),
                verbose=0,
                callbacks=callbacks,
            )

        cv_mae = []
        cv_rmse = []
        cv_r2 = []
        if do_cv:
            groups_train_station = (
                station_seq[train_idx] if cv_type == "GroupKFold (Station)" else None
            )
            groups_train_year = (
                year_seq[train_idx] if cv_type == "GroupKFold (Year)" else None
            )
            if cv_type == "GroupKFold (Station)":
                unique_n = len(np.unique(groups_train_station.astype(str)))
                if unique_n < n_splits:
                    st.error(f"Need at least {n_splits} unique stations for CV, found {unique_n}.")
                    return
                split_iter = list(
                    balanced_group_kfold_splits(
                        groups_train_station.astype(str),
                        n_splits=n_splits,
                        random_state=42,
                    )
                )
            elif cv_type == "GroupKFold (Year)":
                unique_n = len(np.unique(groups_train_year.astype(str)))
                if unique_n < n_splits:
                    st.error(f"Need at least {n_splits} unique years for CV, found {unique_n}.")
                    return
                split_iter = list(
                    balanced_group_kfold_splits(
                        groups_train_year.astype(str),
                        n_splits=n_splits,
                        random_state=42,
                    )
                )
            elif cv_type == "TimeSeriesSplit":
                split_iter = list(TimeSeriesSplit(n_splits=n_splits).split(X_train, y_train))
            else:
                split_iter = list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X_train, y_train))

            for fold_id, (tr_idx, va_idx) in enumerate(split_iter, start=1):
                st.write(
                    f"Training CV fold **{fold_id}/{len(split_iter)}** | "
                    f"train={len(tr_idx)} val={len(va_idx)}"
                )
                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                X_tr, X_va, _ = _fit_transform_sequences(X_tr, X_va, use_scaler=bool(use_scaler))
                model_cv = _make_lstm_model(
                    input_shape=(X_tr.shape[1], X_tr.shape[2]),
                    units=int(units),
                    dropout=float(dropout),
                    learning_rate=float(learning_rate),
                    loss_name=loss_name,
                )
                hist = _fit_one(model_cv, X_tr, y_tr, fold_label=f"CV fold {fold_id}")
                y_pred_fold = np.maximum(model_cv.predict(X_va, verbose=0).flatten(), 0.0)
                fold_mae = float(mean_absolute_error(y_va, y_pred_fold))
                fold_rmse = float(np.sqrt(mean_squared_error(y_va, y_pred_fold)))
                fold_r2 = float(r2_score(y_va, y_pred_fold))
                st.write(f"CV fold {fold_id} metrics | MAE={fold_mae:.4f} RMSE={fold_rmse:.4f} R2={fold_r2:.4f}")
                cv_mae.append(fold_mae)
                cv_rmse.append(fold_rmse)
                cv_r2.append(fold_r2)
                progress_bar.progress(min(95, int(95 * fold_id / max(len(split_iter), 1))))

            st.write("### Cross-Validation Mean Metrics")
            st.write(f"**{n_splits}-Fold CV Mean MAE:** {float(np.mean(cv_mae)):.4f}")
            st.write(f"**{n_splits}-Fold CV Mean RMSE:** {float(np.mean(cv_rmse)):.4f}")
            st.write(f"**{n_splits}-Fold CV Mean R2:** {float(np.mean(cv_r2)):.4f}")

        X_train_s, X_test_s, scaler_final = _fit_transform_sequences(X_train, X_test, use_scaler=bool(use_scaler))
        st.write("Training final LSTM on full training split...")
        model_final = _make_lstm_model(
            input_shape=(X_train_s.shape[1], X_train_s.shape[2]),
            units=int(units),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
            loss_name=loss_name,
        )
        final_hist = _fit_one(model_final, X_train_s, y_train, fold_label="Final model")
        y_pred_test = np.maximum(model_final.predict(X_test_s, verbose=0).flatten(), 0.0)
        progress_bar.progress(100)
        progress_text.write("Final model training complete.")

        st.success("LSTM training complete.")
        _display_metrics("Holdout Test Metrics", y_test, y_pred_test)

        model_final.save("trained_lstm_model.h5")
        if scaler_final is not None:
            joblib.dump(scaler_final, "lstm_scaler.pkl")

        fig1, ax1 = plt.subplots(figsize=(7, 4))
        sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6, ax=ax1)
        lo = float(np.min(y_test))
        hi = float(np.max(y_test))
        ax1.plot([lo, hi], [lo, hi], "r--")
        ax1.set_xlabel("Actual GROUND")
        ax1.set_ylabel("Predicted GROUND")
        ax1.set_title("LSTM: Predicted vs Actual")
        st.pyplot(fig1, width="content")

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(final_hist.history.get("loss", []), label="train_loss")
        ax2.plot(final_hist.history.get("val_loss", []), label="val_loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE Loss")
        ax2.set_title("LSTM Loss Curve (final model)")
        ax2.legend()
        st.pyplot(fig2, width="content")
