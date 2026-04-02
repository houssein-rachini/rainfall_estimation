import streamlit as st
import folium
import ee
import numpy as np
import pandas as pd
from streamlit_folium import st_folium

from predictions import (
    load_dnn_model,
    load_dnn_scaler,
    load_ml_model,
    load_ml_scaler,
    load_ensemble_models,
    load_ensemble_scaler,
    predict_dnn_fast,
    predict_ml_fast,
    predict_ensemble_fast,
)


ee.Initialize()


def _fetch_point_features_ee(lon, lat, year, month):
    start = ee.Date.fromYMD(int(year), int(month), 1)
    end = start.advance(1, "month")
    pt = ee.Geometry.Point([float(lon), float(lat)])

    imerg_ic = ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07").filterDate(
        start, end
    )
    imerg_img = ee.Image(imerg_ic.first()).select("precipitation")
    imerg_mm_hr = imerg_img.reduceRegion(ee.Reducer.first(), pt, 10000).get(
        "precipitation"
    )

    chirps_ic = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end)
    chirps_month = chirps_ic.sum()
    chirps_mm = chirps_month.reduceRegion(ee.Reducer.first(), pt, 5000).get(
        "precipitation"
    )

    dem_img = ee.Image("CGIAR/SRTM90_V4").select("elevation")
    dem_val = dem_img.reduceRegion(ee.Reducer.first(), pt, 90).get("elevation")

    vals = ee.Dictionary(
        {
            "IMERG_mm_hr": imerg_mm_hr,
            "Chirps_mm": chirps_mm,
            "DEM": dem_val,
        }
    ).getInfo()

    if vals is None:
        return None

    imerg_mm_hr_val = vals.get("IMERG_mm_hr", None)
    chirps_mm_val = vals.get("Chirps_mm", None)
    dem_val_num = vals.get("DEM", None)

    if imerg_mm_hr_val is None or chirps_mm_val is None or dem_val_num is None:
        return None

    days_in_month = pd.Timestamp(year=int(year), month=int(month), day=1).days_in_month
    imerg_mm_month = float(imerg_mm_hr_val) * days_in_month * 24

    feature_row = {
        "Longitude": float(lon),
        "Latitude": float(lat),
        "lon_sin": float(np.sin(np.deg2rad(float(lon)))),
        "lon_cos": float(np.cos(np.deg2rad(float(lon)))),
        "lat_sin": float(np.sin(np.deg2rad(float(lat)))),
        "lat_cos": float(np.cos(np.deg2rad(float(lat)))),
        "Year": int(year),
        "Month": int(month),
        "Chirps": float(chirps_mm_val),
        "IMERG(mm/hr)": float(imerg_mm_hr_val),
        "IMERG(mm/month)": float(imerg_mm_month),
        "DEM": float(dem_val_num),
    }
    return feature_row


def _show_point_model_prediction():
    st.subheader("Point Model Prediction (Map)")

    lon_default = float(st.session_state.get("point_lon", 35.7523))
    lat_default = float(st.session_state.get("point_lat", 33.9001))

    m = folium.Map(
        location=[lat_default, lon_default], zoom_start=7, tiles="OpenStreetMap"
    )
    m.add_child(folium.LatLngPopup())
    map_out = st_folium(m, width=700, height=420)

    if map_out and map_out.get("last_clicked"):
        st.session_state["point_lat"] = float(map_out["last_clicked"]["lat"])
        st.session_state["point_lon"] = float(map_out["last_clicked"]["lng"])

    q_lon_default = st.session_state.get("point_lon", lon_default)
    q_lat_default = st.session_state.get("point_lat", lat_default)

    col1, col2, col3, col4 = st.columns(4)
    col1.number_input("Longitude", value=q_lon_default, format="%.6f", key="point_lon")
    col2.number_input("Latitude", value=q_lat_default, format="%.6f", key="point_lat")
    q_lon = float(st.session_state.get("point_lon", q_lon_default))
    q_lat = float(st.session_state.get("point_lat", q_lat_default))

    years_text = col3.text_input(
        "Years (comma-separated)",
        value="",
        key="point_years_text",
        help="Example: 2015, 2018, 2022",
    )
    q_month = col4.selectbox("Month", list(range(1, 13)), index=0, key="point_month")

    location_value = st.selectbox(
        "Location type",
        ["Unknown", "Coastal", "Inland", "Mountainous"],
        index=0,
        key="point_location_type",
        help="Used only if the selected model was trained with Location one-hot features.",
    )

    model_choice = st.selectbox(
        "Model",
        ["ML", "DNN", "DNN+RF", "DNN+XGBoost", "DNN+KNN"],
        key="point_model_choice",
    )
    alpha = 0.4
    if model_choice in ["DNN+RF", "DNN+XGBoost", "DNN+KNN"]:
        alpha = st.slider(
            "Ensemble Weight (DNN Contribution)",
            0.0,
            1.0,
            0.4,
            key="point_alpha",
        )

    use_pretrained = st.checkbox(
        "Use pre-trained model files", value=False, key="point_use_pretrained"
    )

    if st.button("Run Point Prediction", key="point_predict_button"):
        try:
            if model_choice == "DNN":
                dnn_model = load_dnn_model(use_pretrained)
                scaler = load_dnn_scaler(use_pretrained)
            elif model_choice == "ML":
                ml_model = load_ml_model(use_pretrained)
                scaler = load_ml_scaler(use_pretrained)
            else:
                dnn_model, base_model = load_ensemble_models(
                    model_choice, use_pretrained
                )
                scaler = load_ensemble_scaler(use_pretrained)
        except Exception as e:
            st.error(f"Failed to load model/scaler: {e}")
            return

        try:
            years = sorted({int(y.strip()) for y in years_text.split(",") if y.strip()})
        except ValueError:
            st.error("Invalid years format. Use comma-separated integers.")
            return

        if not years:
            st.error("Please enter at least one year.")
            return

        results = []
        for y in years:
            try:
                feature_row = _fetch_point_features_ee(q_lon, q_lat, y, q_month)
            except Exception as e:
                st.error(f"Earth Engine fetch failed for {y}: {e}")
                return

            if feature_row is None:
                st.error(
                    f"Earth Engine did not return complete features for {y}-{q_month:02d}."
                )
                return

            feature_row["Location"] = str(location_value)

            df_input = pd.DataFrame([feature_row])
            try:
                if model_choice == "DNN":
                    pred = predict_dnn_fast(df_input, dnn_model, scaler)
                elif model_choice == "ML":
                    pred = predict_ml_fast(df_input, ml_model, scaler)
                else:
                    pred = predict_ensemble_fast(
                        df_input, dnn_model, base_model, scaler, alpha
                    )
            except Exception as e:
                st.error(f"Prediction failed for {y}: {e}")
                return

            if pred is None:
                st.error(f"Prediction failed due to missing inputs for {y}.")
                return

            results.append(
                {
                    "Year": y,
                    "Month": int(q_month),
                    "Predicted_GROUND": float(pred[0]),
                    **feature_row,
                }
            )

        st.success("Live features fetched from Earth Engine.")
        result_df = pd.DataFrame(results)
        st.dataframe(result_df, width="stretch")


def show_helper_tab(df_actual=None):
    st.title("Rainfall Point Prediction")
    _show_point_model_prediction()
