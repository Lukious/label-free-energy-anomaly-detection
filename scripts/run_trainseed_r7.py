"""Round-7 post-hoc diagnostic (3.9 + attribution of the v9 -> v10 change):
training-seed sensitivity of the backbone effect. PatchTST is retrained with
training seeds TRAIN_SEEDS (7 = the single fixed seed of v9) and evaluated, with
HoW scoring and the canonical w, on each injection seed 123-125 of the
canonical run; the HOW control is recomputed identically.
The clean training/validation spans do not depend on the injection seed, so one
model per training seed serves all injection seeds.
Output: results/final_r7/trainseed_sensitivity.csv (building x model x seed)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import scripts.run_final_r7 as R  # noqa: E402

TRAIN_SEEDS = (7, 1, 2, 3, 4)


def main():
    out_p = os.path.join(R.OUT, "trainseed_sensitivity.csv")
    done = pd.read_csv(out_p) if os.path.exists(out_p) else pd.DataFrame()
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
    inj = {s: R.inject_all(te, n_events=R.N_EVENTS, seed=s, ref=tr) for s in R.SEEDS}
    rows = done.to_dict("records")
    for ts in TRAIN_SEEDS:
        if len(done) and (done["train_seed"] == ts).any():
            print(f"skip train seed {ts}"); continue
        cfg = R.PatchTSTConfig(); cfg.seed = ts; cfg.train_mask = "full"
        ssl = R.PatchTSTAD(cfg).fit([sc[b].transform(tr[b]) for b in tb],
                                    val_arrays=[sc[b].transform(va[b]) for b in tb])
        r_pt = {b: ssl.point_residuals(X[b]) for b in ds}
        del ssl; R.free_mps()
        for s in R.SEEDS:
            cor, ev = inj[s]
            w = float(sel.query("what == 'fusion_w' and seed == @s")
                      .sort_values("val_obj", ascending=False, kind="stable")["option"].iloc[0])
            for b in ds:
                sd = sc[b].sd_["load"]
                d = np.zeros(len(ds[b]))
                d[n_tv[b]:] = (cor[b]["load"].to_numpy() - ds[b]["load"].to_numpy()[n_tv[b]:]) / sd
                e = ev[ev["building_id"] == b]
                for name, r in [("PatchTST-SSL(HoW-cond)", r_pt[b]),
                                ("HOW-profile(HoW-cond)", R.how_backbone_residuals(tr[b], ds[b], sd))]:
                    f = R.causal_flags(R.fuse(w, *R.score_pair(r + d, r, X[b], n_tv[b], "HoW-cond"))[n_tv[b]:])
                    m = R.evaluate_building(f.astype(float), f, e, len(f))
                    m.update(model=name, building_id=b, seed=s, train_seed=ts)
                    rows.append(m)
        pd.DataFrame(rows).to_csv(out_p, index=False)
        g = pd.DataFrame(rows).query("train_seed == @ts").pivot_table(
            index="building_id", columns="model", values="f1")
        print(f"train seed {ts}: PatchTST-HOW = "
              f"{(g['PatchTST-SSL(HoW-cond)'] - g['HOW-profile(HoW-cond)']).mean():+.4f}", flush=True)


if __name__ == "__main__":
    main()
