import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class PointRow:
    row_idx: int
    key: str
    lat: float
    lon: float


def init_ee(authenticate: bool = False) -> None:
    import ee

    try:
        ee.Initialize()
    except Exception:
        if not authenticate:
            raise
        ee.Authenticate()
        ee.Initialize()


def month_bounds(ts: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=ts.year, month=ts.month, day=1)
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def fetch_monthly_ndvi(
    month: pd.Timestamp,
    points: List[PointRow],
    dataset: str,
    band: str,
    scale_m: int,
    scale_factor: float,
) -> Dict[str, float]:
    import ee

    start, end = month_bounds(month)
    ic = ee.ImageCollection(dataset).filterDate(
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )

    if ic.size().getInfo() == 0:
        return {p.key: float("nan") for p in points}

    image = ee.Image(ic.first()).select([band]).multiply(scale_factor)

    fc = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point([p.lon, p.lat]), {"key": p.key}) for p in points]
    )
    reduced = image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),
        scale=scale_m,
    )

    info = reduced.getInfo()
    out: Dict[str, float] = {p.key: float("nan") for p in points}
    for f in info.get("features", []):
        props = f.get("properties", {})
        key = props.get("key")
        val = props.get("first")
        if key is not None:
            out[key] = float(val) if val is not None else float("nan")
    return out


def fill_ndvi_csv(
    input_csv: str,
    output_csv: str,
    dataset: str,
    band: str,
    scale_m: int,
    scale_factor: float,
    ndvi_col: str,
    overwrite_existing: bool,
) -> None:
    df = pd.read_csv(input_csv, encoding="utf-8", encoding_errors="replace")
    df.columns = [c.strip() for c in df.columns]

    required_cols = ["Date", "Latitude", "Longitude"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{input_csv}: missing required columns {missing_cols}")

    if ndvi_col not in df.columns:
        df[ndvi_col] = pd.NA

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    target_mask = df["Date"].notna() & df["Latitude"].notna() & df["Longitude"].notna()
    if not overwrite_existing:
        target_mask &= df[ndvi_col].isna()

    target_idx = df.index[target_mask]
    if len(target_idx) == 0:
        print(f"{input_csv}: no eligible rows to fill.")
        df.to_csv(output_csv, index=False)
        return

    points_by_month: Dict[pd.Timestamp, List[PointRow]] = defaultdict(list)
    row_meta: Dict[int, str] = {}

    for i in target_idx:
        d = df.at[i, "Date"]
        lat = float(df.at[i, "Latitude"])
        lon = float(df.at[i, "Longitude"])
        month = pd.Timestamp(year=d.year, month=d.month, day=1)
        key = f"{month.strftime('%Y-%m')}_{lat:.6f}_{lon:.6f}"
        points_by_month[month].append(PointRow(row_idx=int(i), key=key, lat=lat, lon=lon))
        row_meta[int(i)] = key

    key_to_ndvi: Dict[str, float] = {}
    months = sorted(points_by_month.keys())
    print(f"{input_csv}: fetching NDVI for {len(months)} months across {len(target_idx)} rows...")
    for idx, m in enumerate(months, start=1):
        month_points = points_by_month[m]
        unique = {p.key: p for p in month_points}
        month_unique_points = list(unique.values())
        print(f"[{idx}/{len(months)}] {m.strftime('%Y-%m')} points={len(month_unique_points)}")
        month_vals = fetch_monthly_ndvi(
            month=m,
            points=month_unique_points,
            dataset=dataset,
            band=band,
            scale_m=scale_m,
            scale_factor=scale_factor,
        )
        key_to_ndvi.update(month_vals)

    filled = 0
    for row_idx, key in row_meta.items():
        ndvi = key_to_ndvi.get(key, float("nan"))
        if pd.isna(ndvi):
            continue
        if overwrite_existing or pd.isna(df.at[row_idx, ndvi_col]):
            df.at[row_idx, ndvi_col] = float(ndvi)
            filled += 1

    df.to_csv(output_csv, index=False)
    print(f"{input_csv}: filled {ndvi_col}={filled}")
    print(f"{input_csv}: saved {output_csv}")


def merge_csvs(input_a: str, input_b: str, output_csv: str) -> None:
    df_a = pd.read_csv(input_a, encoding="utf-8", encoding_errors="replace")
    df_b = pd.read_csv(input_b, encoding="utf-8", encoding_errors="replace")
    df_a.columns = [c.strip() for c in df_a.columns]
    df_b.columns = [c.strip() for c in df_b.columns]

    # Harmonize IMERG columns before concatenation:
    # treat IMERG(mm/hr) and IMERG_V07(mm/hr) as the same signal, same for monthly.
    for df in (df_a, df_b):
        if "IMERG(mm/hr)" not in df.columns:
            df["IMERG(mm/hr)"] = pd.NA
        if "IMERG(mm/month)" not in df.columns:
            df["IMERG(mm/month)"] = pd.NA

        if "IMERG_V07(mm/hr)" in df.columns:
            df["IMERG(mm/hr)"] = df["IMERG(mm/hr)"].fillna(df["IMERG_V07(mm/hr)"])
        if "IMERG_V07(mm/month)" in df.columns:
            df["IMERG(mm/month)"] = df["IMERG(mm/month)"].fillna(
                df["IMERG_V07(mm/month)"]
            )

        # Drop duplicate-source columns after coalescing.
        drop_cols = [c for c in ["IMERG_V07(mm/hr)", "IMERG_V07(mm/month)"] if c in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

    merged = pd.concat([df_a, df_b], ignore_index=True, sort=False)
    merged.to_csv(output_csv, index=False)
    print(f"Merged rows={len(merged)} -> {output_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill monthly NDVI into two CSV files and merge them."
    )
    p.add_argument("--input-a", default="All_stations_IMS_imerg_chirps_filled.csv")
    p.add_argument("--input-b", default="AllData_29-01-2019_imerg_v07.csv")
    p.add_argument("--output-a", default="All_stations_IMS_imerg_chirps_filled_ndvi.csv")
    p.add_argument("--output-b", default="AllData_29-01-2019_imerg_v07_ndvi.csv")
    p.add_argument("--merged-output", default="final_merged_with_ndvi.csv")
    p.add_argument("--dataset", default="MODIS/061/MOD13A3")
    p.add_argument("--band", default="NDVI")
    p.add_argument("--scale", type=int, default=1000)
    p.add_argument("--scale-factor", type=float, default=0.0001)
    p.add_argument("--ndvi-col", default="NDVI_MOD13A3")
    p.add_argument("--overwrite-existing", action="store_true")
    p.add_argument("--authenticate", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_ee(authenticate=args.authenticate)

    fill_ndvi_csv(
        input_csv=args.input_a,
        output_csv=args.output_a,
        dataset=args.dataset,
        band=args.band,
        scale_m=args.scale,
        scale_factor=args.scale_factor,
        ndvi_col=args.ndvi_col,
        overwrite_existing=args.overwrite_existing,
    )
    fill_ndvi_csv(
        input_csv=args.input_b,
        output_csv=args.output_b,
        dataset=args.dataset,
        band=args.band,
        scale_m=args.scale,
        scale_factor=args.scale_factor,
        ndvi_col=args.ndvi_col,
        overwrite_existing=args.overwrite_existing,
    )
    merge_csvs(args.output_a, args.output_b, args.merged_output)


if __name__ == "__main__":
    main()
