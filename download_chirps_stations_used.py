import os
import re
import sys
from urllib.parse import urljoin
import requests

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/diagnostics/list_of_stations_used/monthly/"
OUT_DIR = "chirpsstations"

# Matches:
# extra.stationsUsed.2024.10.csv
# global.stationsUsed.2024.10.csv
PATTERN = re.compile(r"(extra|global)\.stationsUsed\.(\d{4})\.(\d{2})\.csv$")


def fetch_html(url: str) -> str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def parse_links(html: str):
    # Grab href="..."
    return re.findall(r'href="([^"]+)"', html)


def download_file(url: str, out_path: str):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp_path = out_path + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    html = fetch_html(BASE_URL)
    links = parse_links(html)

    files = []
    for href in links:
        name = href.split("/")[-1]
        m = PATTERN.match(name)
        if not m:
            continue

        year = int(m.group(2))
        month = int(m.group(3))
        if year < 2000:
            continue
        if not (1 <= month <= 12):
            continue

        files.append(name)

    files = sorted(set(files))
    if not files:
        print(
            "No matching CSV files found. The directory listing format may have changed."
        )
        sys.exit(1)

    print(f"Found {len(files)} matching CSV files (year >= 2000).")

    for i, name in enumerate(files, 1):
        url = urljoin(BASE_URL, name)
        out_path = os.path.join(OUT_DIR, name)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[{i}/{len(files)}] Skip (exists): {name}")
            continue

        print(f"[{i}/{len(files)}] Download: {name}")
        download_file(url, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
