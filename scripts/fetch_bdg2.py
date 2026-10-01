"""Fetch BDG2 dataset via Kaggle CLI.

Requires ~/.kaggle/kaggle.json (Kaggle API token).
Without credentials: prints setup instructions and exits 1.
With credentials: downloads BDG2 electricity meters + metadata, then
preprocesses to per-building hourly CSVs in ../data_bdg2/.
"""
from __future__ import annotations

import os
import subprocess
import sys

KAGGLE_DATASET = "awsaf49/bdg2-dataset"  # community mirror of BDG2
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_bdg2")
KAGGLE_JSON = os.path.expanduser("~/.kaggle/kaggle.json")

HELP = """
[Kaggle credentials not found] BDG2 실데이터 다운로드를 위해 아래를 수행하세요:

1. https://www.kaggle.com/settings -> API -> "Create New Token" 클릭
   -> kaggle.json 다운로드
2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
3. chmod 600 ~/.kaggle/kaggle.json
4. pip install kaggle  (공용 venv에 설치 필요 시 연구자 승인 후)
5. 재실행: python scripts/fetch_bdg2.py

참고: 원본은 https://github.com/buds-lab/building-data-genome-project-2
(GitHub release로도 받을 수 있음: electricity_cleaned.csv.gz 등)
"""


def have_credentials() -> bool:
    return os.path.exists(KAGGLE_JSON)


def preprocess(raw_dir: str, out_dir: str, max_buildings: int = 50):
    """Resample electricity meters to hourly, save per-building CSVs
    with the same schema as the synthetic data (timestamp, load)."""
    import glob

    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    import numpy as np  # noqa: F401
    # BDG2 community mirror layout: electricity.csv, metadata.csv, weather.csv
    elec_p = os.path.join(raw_dir, "electricity.csv")
    meta_p = os.path.join(raw_dir, "metadata.csv")
    if not os.path.exists(elec_p):
        cands = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
        print("CSV files found:", cands[:20])
        sys.exit(f"electricity.csv not found under {raw_dir}")
    elec = pd.read_csv(elec_p, index_col=0, parse_dates=True)
    meta = pd.read_csv(meta_p)
    hourly = elec.resample("h").mean()
    weather_cols = ["airTemperature"]  # site weather if available
    w_p = os.path.join(raw_dir, "weather.csv")
    weather = None
    if os.path.exists(w_p):
        weather = pd.read_csv(w_p, index_col=0, parse_dates=True).resample("h").mean()
    hour = hourly.index.hour.to_numpy()
    dow = hourly.index.dayofweek.to_numpy()
    cal = pd.DataFrame(index=hourly.index, data={
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7)})
    n = 0
    for bid in hourly.columns[:max_buildings]:
        df = pd.DataFrame({"timestamp": hourly.index, "load": hourly[bid].values})
        btype = meta.loc[meta["building_id"] == bid, "primaryspaceusage"]
        df["btype"] = btype.iloc[0] if len(btype) else "unknown"
        df["building_id"] = bid
        if weather is not None and len(weather):
            df["temp"] = weather[weather_cols[0]].reindex(hourly.index).values \
                if weather_cols[0] in weather else 15.0
        else:
            df["temp"] = 15.0
        df = pd.concat([df.reset_index(drop=True), cal.reset_index(drop=True)], axis=1)
        df = df.dropna(subset=["load"])
        df.to_csv(os.path.join(out_dir, f"{bid}.csv"), index=False)
        n += 1
    print(f"Preprocessed {n} buildings -> {out_dir}")


def main():
    if not have_credentials():
        print(HELP)
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
                    "-p", OUT_DIR, "--unzip"], check=True)
    preprocess(OUT_DIR, os.path.join(OUT_DIR, "buildings"))


if __name__ == "__main__":
    main()
