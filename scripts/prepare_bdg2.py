"""BDG2 real-data preparation: building selection, timezone normalization,
gap interpolation, calendar + climatological temperature-proxy features.

Writes data/bdg2/selected/{building_id}.csv with the same schema as the
synthetic dataset (timestamp, load, temp, hour_sin, ..., btype, building_id)
so the existing pipeline (ZScaler, inject_all, models) is reused unchanged.

Selection criteria:
  - >= 95% complete over the full 2016 local-year (8784 h, leap year)
  - 8 Office (seen / train) + 2 Education + 2 Retail (unseen / transfer)
  - non-degenerate series (std > 5% of mean)

Notes:
  - BDG2 electricity timestamps are UTC; converted per-building to local time
    using the metadata timezone (DST handled by pandas tz_convert).
  - BDG2 weather files were not part of the secured download; outdoor
    temperature is replaced by a deterministic climatological proxy
    (latitude-conditioned annual + diurnal sine), documented in the paper.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META = os.path.join(ROOT, "data", "bdg2", "metadata.csv")
ELEC = os.path.join(ROOT, "data", "bdg2", "electricity_cleaned.csv")
OUT_DIR = os.path.join(ROOT, "data", "bdg2", "selected")

N_OFFICE = 8
N_EDU = 2
N_RETAIL = 2
MIN_COMPLETE = 0.95


def climatological_temp(ts: pd.DatetimeIndex, lat: float) -> np.ndarray:
    """Latitude-conditioned annual + diurnal sine proxy (deg C)."""
    doy = ts.dayofyear.to_numpy() + ts.hour.to_numpy() / 24.0
    hod = ts.hour.to_numpy()
    mean = 28.0 - 0.55 * abs(lat)
    amp = np.clip(0.30 * abs(lat) + 4.0, 4.0, 30.0)
    annual = mean + amp * np.sin(2 * np.pi * (doy - 100) / 365.25)
    diurnal = 0.15 * amp * np.sin(2 * np.pi * (hod - 5) / 24.0)
    return annual + diurnal


def main():
    print("loading metadata + electricity (175MB wide csv)...")
    meta = pd.read_csv(META)
    elec = pd.read_csv(ELEC, parse_dates=["timestamp"])
    elec = elec.set_index("timestamp").sort_index()
    n_hours = len(elec)
    print(f"electricity: {elec.shape[1]} buildings x {n_hours} rows "
          f"({elec.index[0]} .. {elec.index[-1]})")

    elec16 = elec[(elec.index >= "2016-01-01") & (elec.index < "2017-01-01")]
    completeness = elec16.notna().mean()
    quality = pd.DataFrame({"completeness": completeness})
    quality = quality.join(meta.set_index("building_id")[["primaryspaceusage",
                                                          "lat", "timezone"]])
    picked = {}
    for usage, n_want, tag in [("Office", N_OFFICE, "office"),
                               ("Education", N_EDU, "education"),
                               ("Retail", N_RETAIL, "retail")]:
        cand = quality[(quality["primaryspaceusage"] == usage)
                       & (quality["completeness"] >= MIN_COMPLETE)]
        # deterministic choice: best completeness, then alphabetical
        cand = cand.sort_values(["completeness"], ascending=False)
        chosen = []
        for bid, row in cand.iterrows():
            s = elec[bid]
            if s.std() < 0.05 * max(s.mean(), 1e-9):  # flat/degenerate meter
                continue
            chosen.append(bid)
            if len(chosen) == n_want:
                break
        picked[tag] = chosen
        print(f"{usage}: {len(cand)} candidates >= {MIN_COMPLETE:.0%} complete; "
              f"picked {chosen}")

    os.makedirs(OUT_DIR, exist_ok=True)
    all_ids = sum(picked.values(), [])
    for tag, bids in picked.items():
        for bid in bids:
            row = quality.loc[bid]
            s = elec[bid]
            s = s[(s.index >= "2016-01-01") & (s.index < "2017-01-01")]
            s = s.tz_localize("UTC").tz_convert(row["timezone"])
            s.index = s.index.tz_localize(None)  # naive local time
            s = s[~s.index.duplicated(keep="first")]  # DST fall-back hour
            local_idx = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
            s = s.reindex(local_idx)
            n_missing = int(s.isna().sum())
            s = s.interpolate(limit=6).ffill().bfill()
            ts = s.index
            hour, dow = ts.hour.to_numpy(), ts.dayofweek.to_numpy()
            df = pd.DataFrame({
                "timestamp": ts,
                "load": s.to_numpy(),
                "temp": climatological_temp(ts, float(row["lat"])),
                "btype": tag,
                "building_id": bid,
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "dow_sin": np.sin(2 * np.pi * dow / 7),
                "dow_cos": np.cos(2 * np.pi * dow / 7),
            })
            df.to_csv(os.path.join(OUT_DIR, f"{bid}.csv"), index=False)
            print(f"saved {bid} ({tag}) — interpolated {n_missing} missing hours "
                  f"({n_missing / len(s):.1%})")
    with open(os.path.join(OUT_DIR, "_selection.txt"), "w") as f:
        for tag, bids in picked.items():
            for b in bids:
                f.write(f"{b}\t{tag}\t{quality.loc[b, 'completeness']:.4f}\n")
    print("done.")


if __name__ == "__main__":
    main()
