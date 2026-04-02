import argparse
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class Point:
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


def fetch_slope_for_points(
    points: List[Point],
    dem_dataset: str,
    dem_band: str,
    scale_m: int,
) -> Dict[str, float]:
    import ee

    if not points:
        return {}

    dem = ee.Image(dem_dataset).select([dem_band])
    slope = ee.Terrain.slope(dem)

    fc = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point([p.lon, p.lat]), {"key": p.key}) for p in points]
    )
    reduced = slope.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=scale_m)
    info = reduced.getInfo()

    out: Dict[str, float] = {}
    for feat in info.get("features", []):
        props = feat.get("properties", {})
        key = props.get("key")
        val = props.get("first")
        if key is not None:
            out[key] = float(val) if val is not None else float("nan")
    return out


def add_slope_column(
    input_csv: str,
    output_csv: str,
    latitude_col: str,
    longitude_col: str,
    slope_col: str,
    dem_dataset: str,
    dem_band: str,
    scale_m: int,
    authenticate: bool,
    overwrite_existing: bool,
) -> None:
    df = pd.read_csv(input_csv, encoding="utf-8", encoding_errors="replace")

    missing = [c for c in [latitude_col, longitude_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if slope_col not in df.columns:
        df[slope_col] = pd.NA

    mask = df[latitude_col].notna() & df[longitude_col].notna()
    if not overwrite_existing:
        mask &= df[slope_col].isna()

    target_idx = df.index[mask]
    if len(target_idx) == 0:
        print("No eligible rows to backfill Slope.")
        df.to_csv(output_csv, index=False)
        print(f"Saved unchanged file: {output_csv}")
        return

    key_to_rows: Dict[str, List[int]] = {}
    unique_points: Dict[str, Point] = {}
    for i in target_idx:
        lat = float(df.at[i, latitude_col])
        lon = float(df.at[i, longitude_col])
        key = f"{lat:.6f}_{lon:.6f}"
        key_to_rows.setdefault(key, []).append(int(i))
        unique_points[key] = Point(key=key, lat=lat, lon=lon)

    print(f"Unique locations to query: {len(unique_points)}")
    init_ee(authenticate=authenticate)
    slope_values = fetch_slope_for_points(
        points=list(unique_points.values()),
        dem_dataset=dem_dataset,
        dem_band=dem_band,
        scale_m=scale_m,
    )

    filled = 0
    for key, rows in key_to_rows.items():
        val = slope_values.get(key, float("nan"))
        if pd.isna(val):
            continue
        for r in rows:
            df.at[r, slope_col] = val
            filled += 1

    df.to_csv(output_csv, index=False)
    print(f"Filled {slope_col}: {filled}")
    print(f"Saved: {output_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add Slope column using ee.Terrain.slope(DEM) sampled at station lat/lon."
    )
    p.add_argument("--input", default="final_merged_with_ndvi_imerg.csv")
    p.add_argument("--output", default="final_merged_with_ndvi_imerg.csv")
    p.add_argument("--latitude-col", default="Latitude")
    p.add_argument("--longitude-col", default="Longitude")
    p.add_argument("--slope-col", default="Slope")
    p.add_argument("--dem-dataset", default="CGIAR/SRTM90_V4")
    p.add_argument("--dem-band", default="elevation")
    p.add_argument("--scale", type=int, default=90)
    p.add_argument("--authenticate", action="store_true")
    p.add_argument("--overwrite-existing", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    add_slope_column(
        input_csv=args.input,
        output_csv=args.output,
        latitude_col=args.latitude_col,
        longitude_col=args.longitude_col,
        slope_col=args.slope_col,
        dem_dataset=args.dem_dataset,
        dem_band=args.dem_band,
        scale_m=args.scale,
        authenticate=args.authenticate,
        overwrite_existing=args.overwrite_existing,
    )
