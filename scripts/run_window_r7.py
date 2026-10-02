"""Round-7 SM: trailing-flagger window sensitivity under the canonical r7
protocol (pre-test frozen stats, seed-coupled training -> identical encoder to
run_final_r7.py). Output: results/final_r7/window_sensitivity.csv"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import scripts.run_final_r7 as R  # noqa: E402

WINDOWS = (168, 336, 672)


def main():
    ds = R.load_dataset(R.DATA_DIR)
    n_tv = {b: int(len(df) * (R.FR[0] + R.FR[1])) for b, df in ds.items()}
    off = {b: int(len(df) * R.FR[0]) for b, df in ds.items()}
    tb = sorted(b for b in ds if ds[b]["btype"].iloc[0] == "office")
    rows = []
    for seed in R.SEEDS:
        tr = {b: ds[b].iloc[:off[b]].reset_index(drop=True) for b in ds}
        va = {b: ds[b].iloc[off[b]:n_tv[b]].reset_index(drop=True) for b in ds}
        te = {b: ds[b].iloc[n_tv[b]:].reset_index(drop=True) for b in ds}
        cor, ev = R.inject_all(te, n_events=R.N_EVENTS, seed=seed, ref=tr)
        sc = {b: R.ZScaler().fit(tr[b]) for b in ds}
        cfg = R.PatchTSTConfig(); cfg.seed = seed; cfg.train_mask = "full"
        ssl = R.PatchTSTAD(cfg).fit([sc[b].transform(tr[b]) for b in tb],
                                    val_arrays=[sc[b].transform(va[b]) for b in tb])
        w = float(pd.read_csv(os.path.join(R.OUT, "selection_log.csv"))
                  .query("what == 'fusion_w' and seed == @seed")
                  .sort_values("val_obj", ascending=False, kind="stable")["option"].iloc[0])
        for b in ds:
            X = sc[b].transform(ds[b])
            sd = sc[b].sd_["load"]
            d = np.zeros(len(ds[b]))
            d[n_tv[b]:] = (cor[b]["load"].to_numpy() - ds[b]["load"].to_numpy()[n_tv[b]:]) / sd
            r0 = {"PatchTST-SSL(HoW-cond)": ssl.point_residuals(X),
                  "HOW-profile(HoW-cond)": R.how_backbone_residuals(tr[b], ds[b], sd)}
            e = ev[ev["building_id"] == b]
            for name, r in r0.items():
                s = R.fuse(w, *R.score_pair(r + d, r, X, n_tv[b], "HoW-cond"))[n_tv[b]:]
                for W in WINDOWS:
                    f = R.causal_flags(s, window=W)
                    m = R.evaluate_building(f.astype(float), f, e, len(f))
                    m.update(model=name, building_id=b, seed=seed, window_h=W)
                    rows.append(m)
        print(f"seed {seed} done (w={w})", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(R.OUT, "window_sensitivity.csv"), index=False)


if __name__ == "__main__":
    main()
