import os
import re
import pandas as pd

INPUT_DIR = "chirpsstations"
OUTPUT_FILE = "lebanon_israel_syria_all.csv"

FNAME_RE = re.compile(
    r"(extra|global)\.stationsUsed\.(\d{4})\.(\d{2})\.csv$", re.IGNORECASE
)
TARGETS = {"lebanon", "israel", "syria"}


def read_loose(path: str) -> pd.DataFrame:
    """
    Try reading as comma-separated first (CSV).
    If it produces 1 column or fails badly, try tab-separated.
    Always skip malformed lines.
    """
    # 1) try comma
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        if df.shape[1] >= 4:  # looks reasonable
            return df
    except Exception:
        pass

    # 2) try tab
    df = pd.read_csv(path, sep="\t", engine="python", on_bad_lines="skip")
    return df


def pick_country_col(cols):
    for c in cols:
        if c.strip().lower() == "country_name":
            return c
    for c in cols:
        if "country" in c.lower():
            return c
    return None


all_parts = []
files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".csv")])

parsed = 0
matched_rows = 0

for fname in files:
    m = FNAME_RE.match(fname)
    if not m:
        continue

    year = int(m.group(2))
    month = int(m.group(3))
    date_str = f"{month}/1/{year}"

    path = os.path.join(INPUT_DIR, fname)

    df = read_loose(path)
    parsed += 1

    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    country_col = pick_country_col(df.columns)

    if country_col is None:
        continue

    mask = df[country_col].astype(str).str.strip().str.lower().isin(TARGETS)

    sub = df.loc[mask].copy()
    if sub.empty:
        continue

    sub["date"] = date_str
    sub["source_file"] = fname

    matched_rows += len(sub)
    all_parts.append(sub)

print(f"Parsed files: {parsed}")
print(f"Matched rows: {matched_rows}")

if not all_parts:
    # quick debug: print a few unique country values from one file
    # to reveal if the country strings differ from 'Lebanon'/'Israel'
    for fname in files:
        m = FNAME_RE.match(fname)
        if not m:
            continue
        df = read_loose(os.path.join(INPUT_DIR, fname))
        df.columns = [c.strip() for c in df.columns]
        cc = pick_country_col(df.columns)
        if cc:
            vals = df[cc].dropna().astype(str).str.strip().unique()[:30]
            print("\nSample country values found:", vals)
            break
    raise SystemExit(
        "No Lebanon/Israel rows found. (Maybe country_name strings differ.)"
    )

out = pd.concat(all_parts, ignore_index=True)

# date first
cols = ["date"] + [c for c in out.columns if c != "date"]
out = out[cols]

# sort by date
out["_dt"] = pd.to_datetime(out["date"], format="%m/%d/%Y", errors="coerce")
out = out.sort_values("_dt").drop(columns=["_dt"])

out.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved: {OUTPUT_FILE} ({len(out)} rows)")
