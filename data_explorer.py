import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from model_utils import TARGET_COL, add_location_dummies, enrich_datetime_columns


def show_data_explorer_tab(df: pd.DataFrame) -> None:
    st.subheader("Dataset Explorer")

    filtered = df.copy()

    if "Station" in filtered.columns:
        station_opts = ["All"] + sorted(filtered["Station"].dropna().unique().tolist())
        selected_station = st.selectbox("Station", station_opts, key="explorer_station")
        if selected_station != "All":
            filtered = filtered[filtered["Station"] == selected_station]

    if "Source" in filtered.columns:
        source_opts = ["All"] + sorted(filtered["Source"].dropna().unique().tolist())
        selected_source = st.selectbox("Source", source_opts, key="explorer_source")
        if selected_source != "All":
            filtered = filtered[filtered["Source"] == selected_source]

    if "Location" in filtered.columns:
        location_opts = ["All"] + sorted(
            filtered["Location"].dropna().unique().tolist()
        )
        selected_location = st.selectbox(
            "Location", location_opts, key="explorer_location"
        )
        if selected_location != "All":
            filtered = filtered[filtered["Location"] == selected_location]

    if "Year" in filtered.columns:
        years = sorted(filtered["Year"].dropna().unique().astype(int).tolist())
        year_opts = ["All"] + years
        selected_year = st.selectbox("Year", year_opts, key="explorer_year")
        if selected_year != "All":
            filtered = filtered[filtered["Year"] == selected_year]

    show_engineered = st.checkbox(
        "Show engineered features (Year/Month + month sin/cos, lon/lat sin/cos, Location one-hot)",
        value=False,
        key="explorer_engineered",
    )

    display_df = filtered
    if show_engineered:
        display_df = enrich_datetime_columns(display_df)
        display_df = add_location_dummies(display_df)

    st.write("### Filtered rows")
    st.dataframe(display_df, width="stretch")

    st.write("### Summary statistics")
    numeric_only = display_df.select_dtypes(include=["number", "bool"])
    if not numeric_only.empty:
        summary = numeric_only.describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        ).transpose()
        summary["missing_count"] = numeric_only.isna().sum()
        summary["missing_pct"] = (summary["missing_count"] / len(filtered) * 100).round(
            2
        )
        st.dataframe(summary, width="stretch")
    else:
        st.info("No numeric columns available for summary statistics.")

    st.write("### Correlation matrix")
    numeric_cols = display_df.select_dtypes(include=["number", "bool"]).columns.tolist()
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
            # "Year",
            TARGET_COL,
        ]
        if c in numeric_cols
    ]
    selected_cols = st.multiselect(
        "Numeric columns",
        options=numeric_cols,
        default=(
            default_cols
            if len(default_cols) >= 2
            else numeric_cols[: min(5, len(numeric_cols))]
        ),
        key="explorer_numeric_cols",
    )

    if len(selected_cols) >= 2:
        corr = display_df[selected_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig, width="content")
    else:
        st.info("Select at least two numeric columns for correlation.")
