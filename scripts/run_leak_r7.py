"""Round-7 fix: transductive-leakage contrast. In run_final_r7.py the
transductive long scale was NaN everywhere (robust z over a series whose first
23 trailing-mean values are NaN), so the variant raised no flags. Here the
seed-coupled encoder is retrained (deterministic -> identical to the canonical
run; asserted below via the canonical PatchTST(HoW) F1) and the contrast is
recomputed with NaN-safe full-series statistics.
Output: results/final_r7/leakage.csv (overwrites the broken file)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import scripts.run_final_r7 as R  # noqa: E402
from scripts.run_revision import FLAG_WINDOW  # noqa: E402
from scripts.run_round2 import how_of  # noqa: E402
from src.evaluate import robust_z_flags  # noqa: E402


def nan_robust_z(x):
    ok = ~np.isnan(x)
    med = np.median(x[ok])
    mad = np.median(np.abs(x[ok] - med)) * 1.4826
    return np.nan_to_num((x - med) / (mad + 1e-9), nan=0.0)


def main():
    ds = R.load_dataset(R.DATA_DIR)
    n_tv = {b: int(len(df) * (R.FR[0] + R.FR[1])) for b, df in ds.items()}
    off = {b: int(len(df) * R.FR[0]) for b, df in ds.items()}
    tb = sorted(b for b in ds if ds[b]["btype"].iloc[0] == "office")
    tr = {b: ds[b].iloc[:off[b]].reset_index(drop=True) for b in ds}
    va = {b: ds[b].iloc[off[b]:n_tv[b]].reset_index(drop=True) for b in ds}
    te = {b: ds[b].iloc[n_tv[b]:].reset_index(drop=True) for b in ds}
    sc = {b: R.ZScaler().fit(tr[b]) for b in ds}
    X = {b: sc[b].transform(ds[b]) for b in ds}
    sel = pd.read_csv(os.path.join(R.OUT, "selection_log.csv"))
    canon = pd.read_csv(os.path.join(R.OUT, "detection_metrics.csv"))
    rows = []
    for seed in R.SEEDS:
        cor, ev = R.inject_all(te, n_events=R.N_EVENTS, seed=seed, ref=tr)
        cfg = R.PatchTSTConfig(); cfg.seed = seed; cfg.train_mask = "full"
        ssl = R.PatchTSTAD(cfg).fit([sc[b].transform(tr[b]) for b in tb],
                                    val_arrays=[sc[b].transform(va[b]) for b in tb])
        w = float(sel.query("what == 'fusion_w' and seed == @seed")
                  .sort_values("val_obj", ascending=False, kind="stable")["option"].iloc[0])
        chk = []
        for b in ds:
            r = ssl.point_residuals(X[b])
            sd = sc[b].sd_["load"]
            d = np.zeros(len(ds[b]))
            d[n_tv[b]:] = (cor[b]["load"].to_numpy() - ds[b]["load"].to_numpy()[n_tv[b]:]) / sd
            ra = r + d
            e = ev[ev["building_id"] == b]
            # determinism check: canonical forward-only arm
            f0 = R.causal_flags(R.fuse(w, *R.score_pair(ra, r, X[b], n_tv[b], "HoW-cond"))[n_tv[b]:])
            chk.append(R.evaluate_building(f0.astype(float), f0, e, len(f0))["f1"])
            # transductive: full-series statistics (test included) + centred flagger
            z_td = R.cond_z_ref(ra, ra, X[b], len(ra), 168, how_of)
            zl_td = nan_robust_z(R._trail_sq(ra))
            s_td = R.fuse(w, z_td, zl_td)[n_tv[b]:]
            ft = robust_z_flags(s_td, window=FLAG_WINDOW)
            m = R.evaluate_building(s_td, ft, e, len(s_td))
            m.update(model="PatchTST-transductive", building_id=b, seed=seed,
                     btype=ds[b]["btype"].iloc[0])
            rows.append(m)
        ref = canon.query("model == 'PatchTST-SSL(HoW-cond)' and seed == @seed")["f1"].mean()
        print(f"seed {seed}: recomputed canonical F1 {np.nanmean(chk):.4f} vs master {ref:.4f}", flush=True)
        assert abs(np.nanmean(chk) - ref) < 1e-6, "non-deterministic retraining"
        del ssl; R.free_mps()
    pd.DataFrame(rows).to_csv(os.path.join(R.OUT, "leakage.csv"), index=False)
    print("leakage rewritten")


if __name__ == "__main__":
    main()
