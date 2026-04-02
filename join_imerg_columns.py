import argparse
from typing import List

import pandas as pd


def _find_candidates(columns: List[str], token: str) -> List[str]:
    cols = [c for c in columns if "IMERG" in c and token in c]
    # Prefer canonical names first, then V07, then anything else.
    preferred_order = [
        f"IMERG({token})",
        f"IMERG(mm/{token.split('/')[-1]})" if "/" in token else "",
        f"IMERG_V07({token})",
    ]
    ordered = []
    for p in preferred_order:
        if p and p in cols and p not in ordered:
            ordered.append(p)
    for c in cols:
        if c not in ordered:
            ordered.append(c)
    return ordered


def coalesce_imerg_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = list(out.columns)

    hr_candidates = _find_candidates(cols, "mm/hr")
    month_candidates = _find_candidates(cols, "mm/month")

    if not hr_candidates and not month_candidates:
        print("No IMERG columns found to merge.")
        return out

    if hr_candidates:
        out["IMERG(mm/hr)"] = out[hr_candidates].bfill(axis=1).iloc[:, 0]
    if month_candidates:
        out["IMERG(mm/month)"] = out[month_candidates].bfill(axis=1).iloc[:, 0]

    drop_cols = set(hr_candidates + month_candidates) - {
        "IMERG(mm/hr)",
        "IMERG(mm/month)",
    }
    if drop_cols:
        out = out.drop(columns=sorted(drop_cols))

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge duplicate IMERG columns into canonical hourly/monthly columns."
    )
    p.add_argument(
        "--input",
        default="final_merged_with_ndvi.csv",
        help="Input CSV path",
    )
    p.add_argument(
        "--output",
        default="final_merged_with_ndvi_imerg_unified.csv",
        help="Output CSV path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, encoding="utf-8", encoding_errors="replace")
    cleaned = coalesce_imerg_columns(df)
    cleaned.to_csv(args.output, index=False)
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(cleaned)}")
    print(f"Saved: {args.output}")
    print(f"Columns: {list(cleaned.columns)}")


if __name__ == "__main__":
    main()
