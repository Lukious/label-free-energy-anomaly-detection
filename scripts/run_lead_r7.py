"""Round-7 (v11) LEAD 1.0-small out-of-domain check under the CURRENT protocol.

Training-free HOW-profile detector exactly as on BDG2: 60/15/25 chronological
split per building, slot medians on the train span, HoW-conditional robust-z
with pre-test (train+val) frozen statistics, causal long scale, the canonical
fusion weight transferred from BDG2 validation (LEAD has no injection
validation set), and the strictly-prior trailing flagger. LEAD's labels are
used only for evaluation; labelled hours inside the pre-test span are part of
the robust statistics (MAD is robust to them).

Inputs : data/lead/lead1.0-small.csv (public: github.com/samy101/lead-dataset)
Outputs: results/final_r7/lead_eval.csv        (per building)
         results/final_r7/lead_by_duration.csv (recovery by label-run length)
         results/final_r7/lead_label_durations.csv
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import scripts.run_final_r7 as R  # noqa: E402
from scripts.run_revision import _runs  # noqa: E402

LEAD = os.path.join(ROOT, "data", "lead", "lead1.0-small.csv")
BINS = [(1, 2), (3, 6), (7, 24), (25, 10 ** 6)]


def features(g):
    h, dow = g["timestamp"].dt.hour, g["timestamp"].dt.dayofweek
    X = np.zeros((len(g), 6))
    X[:, 2], X[:, 3] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    X[:, 4], X[:, 5] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    return X


def main():
    warnings.filterwarnings("ignore")
    df = pd.read_csv(LEAD, parse_dates=["timestamp"])
    sel = pd.read_csv(os.path.join(R.OUT, "selection_log.csv"))
    ws = (sel[sel["what"] == "fusion_w"].sort_values(["seed", "val_obj"], ascending=[True, False],
                                                     kind="stable").groupby("seed")["option"].first())
    w = float(pd.to_numeric(ws).median())
    rows, dur_rows, lab_rows = [], [], []
    for bid, g in df.groupby("building_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        lab_all = g["anomaly"].astype(bool).to_numpy()
        for s, e in _runs(lab_all):
            lab_rows.append({"building_id": bid, "duration_h": e - s})
        if len(g) < 2000 or g["meter_reading"].notna().sum() < 0.5 * len(g):
            continue
        n = len(g)
        a, n_tv = int(n * R.FR[0]), int(n * (R.FR[0] + R.FR[1]))
        y = g["meter_reading"].ffill().bfill().to_numpy()
        how = (g["timestamp"].dt.dayofweek * 24 + g["timestamp"].dt.hour).to_numpy()
        prof = pd.Series(y[:a]).groupby(how[:a]).median()
        sd = float(np.std(y[:a])) or 1.0
        r = (y - prof.reindex(how).to_numpy()) / sd
        r = pd.Series(r).ffill().bfill().to_numpy()
        X = features(g)
        score = R.fuse(w, *R.score_pair(r, r, X, n_tv, "HoW-cond"))
        f = R.causal_flags(score)[n_tv:]
        lab = lab_all[n_tv:]
        runs = _runs(f)
        rows.append({
            "building_id": bid, "n_test_hours": len(f), "flagged_hours": int(f.sum()),
            "flagged_on_labeled": int((f & lab).sum()), "labeled_hours": int(lab.sum()),
            "alarm_rate": float(f.mean()),
            "alarm_rate_unlabeled": float(f[~lab].mean()) if (~lab).any() else np.nan,
            "run_precision": (sum(lab[s:e].any() for s, e in runs) / len(runs)) if runs else np.nan,
            "median_load": float(np.median(y[:a])),
            "label_share_test": float(lab.mean()),
        })
        for s, e in _runs(lab):
            dur_rows.append({"building_id": bid, "duration_h": e - s,
                             "detected": bool(f[s:e].any()), "covered": float(f[s:e].mean())})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(R.OUT, "lead_eval.csv"), index=False)
    d = pd.DataFrame(dur_rows)
    d["bin"] = pd.cut(d["duration_h"], [0, 2, 6, 24, 10 ** 6], labels=["1-2 h", "3-6 h", "7-24 h", ">24 h"])
    d.to_csv(os.path.join(R.OUT, "lead_by_duration.csv"), index=False)
    pd.DataFrame(lab_rows).to_csv(os.path.join(R.OUT, "lead_label_durations.csv"), index=False)
    print(f"w={w}; buildings={len(out)}; flagged-on-label share="
          f"{out['flagged_on_labeled'].sum() / out['flagged_hours'].sum():.3f}; "
          f"recovered labelled hours={out['flagged_on_labeled'].sum() / out['labeled_hours'].sum():.3f}; "
          f"run precision={out['run_precision'].mean():.3f}; alarm rate={out['alarm_rate'].mean():.4f}")
    print(d.groupby("bin")[["detected", "covered"]].mean().round(3), d.groupby("bin").size())


if __name__ == "__main__":
    main()
