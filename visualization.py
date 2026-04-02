import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import r2_score

from model_utils import TARGET_COL


def show_visualization_tab(df: pd.DataFrame) -> None:
    st.subheader("Rainfall Visualization")

    working = df.copy()

    selected_station = "All Stations"
    if "Station" in working.columns:
        station_options = ["All Stations"] + sorted(
            working["Station"].dropna().astype(str).unique().tolist()
        )
        selected_station = st.selectbox(
            "Station",
            station_options,
            key="viz_station_selector",
        )
        if selected_station != "All Stations":
            working = working[working["Station"].astype(str) == selected_station]

    if "Date" in working.columns:
        min_date = working["Date"].min()
        max_date = working["Date"].max()
        if pd.notna(min_date) and pd.notna(max_date):
            selected_range = st.slider(
                "Date range",
                min_value=min_date.to_pydatetime(),
                max_value=max_date.to_pydatetime(),
                value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
                key="viz_date_range",
            )
            working = working[
                (working["Date"] >= pd.Timestamp(selected_range[0]))
                & (working["Date"] <= pd.Timestamp(selected_range[1]))
            ]

    st.write(
        f"### Monthly mean precipitation ({selected_station})"
        if selected_station != "All Stations"
        else "### Monthly mean precipitation (all stations combined)"
    )
    if "Date" in working.columns and TARGET_COL in working.columns:
        monthly = (
            working.dropna(subset=["Date", TARGET_COL])
            .set_index("Date")
            .resample("MS")
            .mean(numeric_only=True)
            .reset_index()
        )

        keep_cols = [c for c in [TARGET_COL, "Chirps", "IMERG(mm/month)"] if c in monthly.columns]
        long_df = monthly.melt(id_vars="Date", value_vars=keep_cols, var_name="Series", value_name="Value")
        chart = (
            alt.Chart(long_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("yearmonth(Date):T", axis=alt.Axis(format="%Y-%m", labelAngle=-45)),
                y="Value:Q",
                color="Series:N",
                tooltip=[
                    alt.Tooltip("yearmonth(Date):T", title="Month", format="%Y-%m"),
                    "Series:N",
                    alt.Tooltip("Value:Q", format=".2f"),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)

    st.write("### GROUND vs satellite estimates")
    compare_cols = [c for c in ["Chirps", "IMERG(mm/month)"] if c in working.columns]
    if TARGET_COL in working.columns and compare_cols:
        pick = st.selectbox("Compare with", compare_cols, key="viz_compare_with")
        scatter = working.dropna(subset=[TARGET_COL, pick])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=scatter, x=pick, y=TARGET_COL, alpha=0.6, ax=ax)

        if len(scatter) >= 2:
            x = scatter[pick].to_numpy()
            y = scatter[TARGET_COL].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            y_fit = slope * x + intercept
            order = np.argsort(x)
            ax.plot(x[order], y_fit[order], color="red", linewidth=2, label="Trendline")
            r2 = r2_score(y, y_fit)
            st.caption(
                f"Trendline: `{TARGET_COL} = {slope:.4f} * {pick} + {intercept:.4f}` | `R² = {r2:.4f}`"
            )
            ax.legend()

        ax.set_xlabel(pick)
        ax.set_ylabel(TARGET_COL)
        ax.set_title(f"{TARGET_COL} vs {pick}")
        st.pyplot(fig, width="content")

    st.write("### Seasonality by month")
    if "Month" in working.columns and TARGET_COL in working.columns:
        seasonal = working.dropna(subset=["Month", TARGET_COL])
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.boxplot(data=seasonal, x="Month", y=TARGET_COL, ax=ax)
        ax.set_title("Monthly distribution of GROUND precipitation")
        st.pyplot(fig, width="content")

    st.write("### Distribution and skewness")
    if TARGET_COL in working.columns:
        series = working[TARGET_COL].dropna()
        if not series.empty:
            stats = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurt()),
            }
            st.write(
                f"Count: `{stats['count']}` | Mean: `{stats['mean']:.2f}` | Median: `{stats['median']:.2f}` | "
                f"Std: `{stats['std']:.2f}` | Skewness: `{stats['skewness']:.2f}` | Kurtosis: `{stats['kurtosis']:.2f}`"
            )

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(series, bins=40, kde=True, ax=ax)
                ax.set_title("GROUND distribution")
                ax.set_xlabel(TARGET_COL)
                st.pyplot(fig, width="content")

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                log_series = np.log1p(series)
                sns.histplot(log_series, bins=40, kde=True, ax=ax)
                ax.set_title("log1p(GROUND) distribution")
                ax.set_xlabel(f"log1p({TARGET_COL})")
                st.pyplot(fig, width="content")
        else:
            st.info("No data available for distribution plots.")
