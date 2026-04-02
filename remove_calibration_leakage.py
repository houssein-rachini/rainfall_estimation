import argparse
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove rows from main dataset that match calibration-station usage "
            "by same Date + nearby Latitude/Longitude."
        )
    )
    parser.add_argument(
        "--main",
        default="final_merged_with_ndvi_imerg.csv",
        help="Main training dataset CSV path.",
    )
    parser.add_argument(
        "--calibration",
        default="lebanon_israel_all.csv",
        help="Calibration stations CSV path.",
    )
    parser.add_argument(
        "--out",
        default="final_merged_with_ndvi_imerg_no_leakage.csv",
        help="Output filtered CSV path.",
    )
    parser.add_argument(
        "--tol-deg",
        type=float,
        default=0.07,
        help="Coordinate distance threshold in degrees on same Date.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    main_df = pd.read_csv(args.main)
    cal_df = pd.read_csv(args.calibration)

    need_main = {"Date", "Latitude", "Longitude"}
    need_cal = {"date", "latitude", "longitude"}
    if not need_main.issubset(main_df.columns):
        raise ValueError(f"Main file missing required columns: {sorted(need_main)}")
    if not need_cal.issubset(cal_df.columns):
        raise ValueError(f"Calibration file missing required columns: {sorted(need_cal)}")

    main_df = main_df.copy()
    cal_df = cal_df.copy()
    main_df["Date"] = pd.to_datetime(main_df["Date"], errors="coerce")
    cal_df["Date"] = pd.to_datetime(cal_df["date"], errors="coerce")

    main_df["_row_id"] = np.arange(len(main_df), dtype=np.int64)
    left = main_df[["_row_id", "Date", "Latitude", "Longitude"]]
    right = cal_df[["Date", "latitude", "longitude"]]

    merged = left.merge(right, on="Date", how="inner")
    merged["dist_deg"] = np.sqrt(
        (merged["Latitude"] - merged["latitude"]) ** 2
        + (merged["Longitude"] - merged["longitude"]) ** 2
    )

    leak_row_ids = set(
        merged.loc[merged["dist_deg"] <= float(args.tol_deg), "_row_id"].tolist()
    )
    before_n = len(main_df)
    filtered_df = main_df.loc[~main_df["_row_id"].isin(leak_row_ids)].drop(
        columns=["_row_id"]
    )
    after_n = len(filtered_df)

    filtered_df.to_csv(args.out, index=False)

    print(f"Input rows: {before_n}")
    print(f"Removed rows: {before_n - after_n}")
    print(f"Output rows: {after_n}")
    if "Station" in main_df.columns and leak_row_ids:
        removed_stations = (
            main_df.loc[main_df["_row_id"].isin(leak_row_ids), "Station"]
            .astype(str)
            .value_counts()
        )
        print("Top removed stations:")
        print(removed_stations.head(15).to_string())


if __name__ == "__main__":
    main()
