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


def fetch_monthly_imerg(
    month: pd.Timestamp,
    points: List[PointRow],
    dataset: str,
    band: str,
    scale_m: int,
) -> Dict[str, float]:
    import ee

    start, end = month_bounds(month)
    ic = ee.ImageCollection(dataset).filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    if ic.size().getInfo() == 0:
        return {p.key: float("nan") for p in points}

    image = ee.Image(ic.first()).select([band])

    fc = ee.FeatureCollection(
        [
            ee.Feature(ee.Geometry.Point([p.lon, p.lat]), {"key": p.key})
            for p in points
        ]
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


def fetch_monthly_chirps(
    month: pd.Timestamp,
    points: List[PointRow],
    dataset: str,
    band: str,
    scale_m: int,
) -> Dict[str, float]:
    import ee

    start, end = month_bounds(month)
    ic = ee.ImageCollection(dataset).filterDate(
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )

    if ic.size().getInfo() == 0:
        return {p.key: float("nan") for p in points}

    # Monthly CHIRPS total from daily precipitation.
    image = ee.Image(ic.sum()).select([band])

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


def backfill_imerg(
    input_csv: str,
    output_csv: str,
    dataset: str,
    band: str,
    scale_m: int,
    authenticate: bool,
    overwrite_existing: bool,
    out_hr_col: str,
    out_month_col: str,
    fill_chirps: bool,
    chirps_dataset: str,
    chirps_band: str,
    chirps_scale_m: int,
    chirps_col: str,
) -> None:
    df = pd.read_csv(input_csv)

    required_cols = ["Date", "Latitude", "Longitude"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if out_hr_col not in df.columns:
        df[out_hr_col] = pd.NA
    if out_month_col not in df.columns:
        df[out_month_col] = pd.NA
    if fill_chirps and chirps_col not in df.columns:
        df[chirps_col] = pd.NA

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    target_mask = df["Date"].notna() & df["Latitude"].notna() & df["Longitude"].notna()
    if not overwrite_existing:
        target_mask &= (
            df[out_hr_col].isna()
            | df[out_month_col].isna()
            | (fill_chirps & df[chirps_col].isna())
        )

    target_idx = df.index[target_mask]
    if len(target_idx) == 0:
        print("No eligible rows to backfill.")
        df.to_csv(output_csv, index=False)
        print(f"Saved unchanged file to: {output_csv}")
        return

    points_by_month: Dict[pd.Timestamp, List[PointRow]] = defaultdict(list)
    row_meta: Dict[int, Tuple[str, pd.Timestamp]] = {}

    for i in target_idx:
        d = df.at[i, "Date"]
        lat = float(df.at[i, "Latitude"])
        lon = float(df.at[i, "Longitude"])
        month = pd.Timestamp(year=d.year, month=d.month, day=1)
        key = f"{month.strftime('%Y-%m')}_{lat:.6f}_{lon:.6f}"
        points_by_month[month].append(PointRow(row_idx=int(i), key=key, lat=lat, lon=lon))
        row_meta[int(i)] = (key, d)

    init_ee(authenticate=authenticate)

    key_to_hr: Dict[str, float] = {}
    key_to_chirps: Dict[str, float] = {}
    months = sorted(points_by_month.keys())
    print(
        f"Fetching IMERG{' + CHIRPS' if fill_chirps else ''} "
        f"for {len(months)} months across {len(target_idx)} rows..."
    )

    for idx, m in enumerate(months, start=1):
        month_points = points_by_month[m]
        unique = {}
        for p in month_points:
            unique[p.key] = p
        month_unique_points = list(unique.values())
        print(f"[{idx}/{len(months)}] {m.strftime('%Y-%m')} points={len(month_unique_points)}")
        month_vals = fetch_monthly_imerg(
            month=m,
            points=month_unique_points,
            dataset=dataset,
            band=band,
            scale_m=scale_m,
        )
        key_to_hr.update(month_vals)
        if fill_chirps:
            chirps_vals = fetch_monthly_chirps(
                month=m,
                points=month_unique_points,
                dataset=chirps_dataset,
                band=chirps_band,
                scale_m=chirps_scale_m,
            )
            key_to_chirps.update(chirps_vals)

    filled_hr = 0
    filled_month = 0
    filled_chirps = 0

    for row_idx, (key, d) in row_meta.items():
        hr = key_to_hr.get(key, float("nan"))
        if pd.isna(hr):
            continue

        month_mm = float(hr * d.days_in_month * 24)

        if overwrite_existing or pd.isna(df.at[row_idx, out_hr_col]):
            df.at[row_idx, out_hr_col] = hr
            filled_hr += 1

        if overwrite_existing or pd.isna(df.at[row_idx, out_month_col]):
            df.at[row_idx, out_month_col] = month_mm
            filled_month += 1

        if fill_chirps:
            chirps_mm = key_to_chirps.get(key, float("nan"))
            if not pd.isna(chirps_mm):
                if overwrite_existing or pd.isna(df.at[row_idx, chirps_col]):
                    df.at[row_idx, chirps_col] = chirps_mm
                    filled_chirps += 1

    df.to_csv(output_csv, index=False)

    print("Done.")
    print(f"Filled {out_hr_col}: {filled_hr}")
    print(f"Filled {out_month_col}: {filled_month}")
    if fill_chirps:
        print(f"Filled {chirps_col}: {filled_chirps}")
    print(f"Saved: {output_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill missing IMERG values in AllData CSV using Google Earth Engine.")
    p.add_argument("--input", default="AllData_29-01-2019.csv", help="Input CSV path")
    p.add_argument("--output", default="AllData_29-01-2019_imerg_filled.csv", help="Output CSV path")
    p.add_argument("--dataset", default="NASA/GPM_L3/IMERG_MONTHLY_V07", help="Earth Engine IMERG monthly collection")
    p.add_argument("--band", default="precipitation", help="Band name to sample (typically 'precipitation')")
    p.add_argument("--scale", type=int, default=10000, help="Sampling scale in meters")
    p.add_argument("--authenticate", action="store_true", help="Run ee.Authenticate() if ee.Initialize() fails")
    p.add_argument("--overwrite-existing", action="store_true", help="Overwrite existing IMERG values too")
    p.add_argument("--out-hr-col", default="IMERG_V07(mm/hr)", help="Output column for IMERG hourly-rate")
    p.add_argument("--out-month-col", default="IMERG_V07(mm/month)", help="Output column for IMERG monthly total")
    p.add_argument("--fill-chirps", action="store_true", help="Also backfill monthly CHIRPS totals")
    p.add_argument("--chirps-dataset", default="UCSB-CHG/CHIRPS/DAILY", help="Earth Engine CHIRPS daily collection")
    p.add_argument("--chirps-band", default="precipitation", help="CHIRPS band name")
    p.add_argument("--chirps-scale", type=int, default=5000, help="CHIRPS sampling scale in meters")
    p.add_argument("--chirps-col", default="Chirps", help="Output column for monthly CHIRPS (mm/month)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    backfill_imerg(
        input_csv=args.input,
        output_csv=args.output,
        dataset=args.dataset,
        band=args.band,
        scale_m=args.scale,
        authenticate=args.authenticate,
        overwrite_existing=args.overwrite_existing,
        out_hr_col=args.out_hr_col,
        out_month_col=args.out_month_col,
        fill_chirps=args.fill_chirps,
        chirps_dataset=args.chirps_dataset,
        chirps_band=args.chirps_band,
        chirps_scale_m=args.chirps_scale,
        chirps_col=args.chirps_col,
    )


if __name__ == "__main__":
    main()
