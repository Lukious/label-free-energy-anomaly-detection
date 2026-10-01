"""Revision prep: larger building set + REAL measured weather (M6).

Writes data/bdg2/selected_v2/{building_id}.csv, same schema as before but with
`temp` = measured BDG2 airTemperature (site-matched, local time), falling back
to the climatological proxy only for gaps.

Selection (revision, supersedes the 12-building set):
  - >= 95% complete over 2016 local year, non-degenerate meter (std > 5% mean)
  - 24 Office (seen/train) + 12 Education + all available Retail (5) unseen
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META = os.path.join(ROOT, "data", "bdg2", "metadata.csv")
ELEC = os.path.join(ROOT, "data", "bdg2", "electricity_cleaned.csv")
WEATHER = os.path.join(ROOT, "data", "bdg2", "weather.csv")
OUT_DIR = os.path.join(ROOT, "data", "bdg2", "selected_v2")

N_OFFICE, N_EDU, N_RETAIL = 24, 12, 5
MIN_COMPLETE = 0.95


def climatological_temp(ts: pd.DatetimeIndex, lat: float) -> np.ndarray:
    doy = ts.dayofyear.to_numpy() + ts.hour.to_numpy() / 24.0
    hod = ts.hour.to_numpy()
    mean = 28.0 - 0.55 * abs(lat)
    amp = np.clip(0.30 * abs(lat) + 4.0, 4.0, 30.0)
    annual = mean + amp * np.sin(2 * np.pi * (doy - 100) / 365.25)
    diurnal = 0.15 * amp * np.sin(2 * np.pi * (hod - 5) / 24.0)
    return annual + diurnal


def main():
    print("loading metadata + electricity...")
    meta = pd.read_csv(META)
    elec = pd.read_csv(ELEC, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    wx = pd.read_csv(WEATHER, parse_dates=["timestamp"])
    wx = wx[["timestamp", "site_id", "airTemperature"]]

    elec16 = elec[(elec.index >= "2016-01-01") & (elec.index < "2017-01-01")]
    quality = pd.DataFrame({"completeness": elec16.notna().mean()})
    quality = quality.join(meta.set_index("building_id")[
        ["primaryspaceusage", "lat", "timezone", "site_id"]])

    picked = {}
    for usage, n_want, tag in [("Office", N_OFFICE, "office"),
                               ("Education", N_EDU, "education"),
                               ("Retail", N_RETAIL, "retail")]:
        cand = quality[(quality["primaryspaceusage"] == usage)
                       & (quality["completeness"] >= MIN_COMPLETE)]
        cand = cand.sort_values("completeness", ascending=False)
        chosen = []
        for bid, row in cand.iterrows():
            s = elec[bid]
            if s.std() < 0.05 * max(s.mean(), 1e-9):
                continue
            chosen.append(bid)
            if len(chosen) == n_want:
                break
        picked[tag] = chosen
        print(f"{usage}: picked {len(chosen)} / {len(cand)} candidates")

    # site-level weather: UTC -> site-local naive hourly
    site_wx = {}
    tz_by_site = meta.dropna(subset=["site_id"]).groupby("site_id")["timezone"].first()
    for site, g in wx.groupby("site_id"):
        g = g.dropna(subset=["airTemperature"]).drop_duplicates("timestamp") \
              .set_index("timestamp")["airTemperature"].sort_index()
        tz = tz_by_site.get(site)
        if pd.isna(tz):
            continue
        g = g.tz_localize("UTC").tz_convert(tz)
        g.index = g.index.tz_localize(None)
        g = g[~g.index.duplicated(keep="first")]
        site_wx[site] = g

    os.makedirs(OUT_DIR, exist_ok=True)
    local_idx = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
    n_proxy_hours_total = 0
    for tag, bids in picked.items():
        for bid in bids:
            row = quality.loc[bid]
            s = elec[bid]
            s = s[(s.index >= "2016-01-01") & (s.index < "2017-01-01")]
            s = s.tz_localize("UTC").tz_convert(row["timezone"])
            s.index = s.index.tz_localize(None)
            s = s[~s.index.duplicated(keep="first")].reindex(local_idx)
            s = s.interpolate(limit=6).ffill().bfill()
            # measured weather, local time
            g = site_wx.get(row["site_id"], pd.Series(dtype=float)).reindex(local_idx)
            n_missing = int(g.isna().sum())
            temp = g.interpolate(limit=6).ffill().bfill().to_numpy().copy()
            proxy = climatological_temp(local_idx, float(row["lat"]))
            used_proxy = np.isnan(temp)
            temp[used_proxy] = proxy[used_proxy]
            n_proxy_hours_total += int(used_proxy.sum())
            ts = local_idx
            hour, dow = ts.hour.to_numpy(), ts.dayofweek.to_numpy()
            df = pd.DataFrame({
                "timestamp": ts, "load": s.to_numpy(), "temp": temp,
                "btype": tag, "building_id": bid,
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "dow_sin": np.sin(2 * np.pi * dow / 7),
                "dow_cos": np.cos(2 * np.pi * dow / 7),
            })
            df.to_csv(os.path.join(OUT_DIR, f"{bid}.csv"), index=False)
    with open(os.path.join(OUT_DIR, "_selection.txt"), "w") as f:
        for tag, bids in picked.items():
            for b in bids:
                f.write(f"{b}\t{tag}\t{quality.loc[b, 'completeness']:.4f}\n")
    n = sum(len(v) for v in picked.values())
    print(f"saved {n} buildings to {OUT_DIR}; weather missing hours filled by proxy: "
          f"{n_proxy_hours_total} ({n_proxy_hours_total / (n * 8784):.2%})")


if __name__ == "__main__":
    main()
