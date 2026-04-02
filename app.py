import pandas as pd
import streamlit as st

from data_explorer import show_data_explorer_tab
from bias_residual_pipeline import show_bias_residual_tab
from dnn_training import show_dnn_training_tab
from ensemble_training import show_ensemble_training_tab
from lstm_training import show_lstm_training_tab
from ml_training import show_ml_training_tab
from model_utils import TARGET_COL, enrich_datetime_columns
from predictions import show_predictions_tab
from updated_predictions import show_helper_tab
from visualization import show_visualization_tab
from wet_dry_classifier import show_wet_dry_classifier_tab

st.set_page_config(page_title="Rainfall Estimation", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    # df = pd.read_csv("final_merged_with_ndvi_imerg.csv")
    df = pd.read_csv("new_final_merged_with_ndvi_imerg_no_leakage_005.csv")
    return enrich_datetime_columns(df)


def main() -> None:
    df = load_data()

    st.title("Monthly Rainfall Estimation")
    st.caption(f"Target variable: {TARGET_COL} (monthly precipitation)")

    pages = [
        "Visualization",
        "Data Explorer",
        "ML Training",
        "DNN Training",
        "LSTM Training",
        "Ensemble Training",
        "Bias Residual",
        "Wet/Dry Classifier",
        "Predictions",
        "Updated Predictions",
    ]
    selected_page = st.radio("Navigation", pages, horizontal=True)

    if selected_page == "Visualization":
        show_visualization_tab(df)
    elif selected_page == "Data Explorer":
        show_data_explorer_tab(df)
    elif selected_page == "ML Training":
        show_ml_training_tab(df)
    elif selected_page == "DNN Training":
        show_dnn_training_tab(df)
    elif selected_page == "LSTM Training":
        show_lstm_training_tab(df)
    elif selected_page == "Ensemble Training":
        show_ensemble_training_tab(df)
    elif selected_page == "Bias Residual":
        show_bias_residual_tab()
    elif selected_page == "Wet/Dry Classifier":
        show_wet_dry_classifier_tab(df)
    elif selected_page == "Predictions":
        show_predictions_tab(df)
    elif selected_page == "Updated Predictions":
        show_helper_tab(df)


if __name__ == "__main__":
    main()
