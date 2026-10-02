"""Round-7 H4 sensitivity: additive schedule injection + relative rejection.

Same protocol, seeds, encoders (seed-coupled training is deterministic, so the
retrained encoder is the canonical one), fusion weight and flagger as the
canonical run. Only the injector changes:
  * schedule faults ADD frac x (on-hour mean - off-hour mean of the training
    span) to off-hours (frac per severity as in Table 1), instead of lifting
    off-hours TO a fraction of median load;
  * a draw is rejected if its peak increment is below 1% of the building's
    training-span median load (instead of an absolute 0.5 kW).
Validation injections are not used (w is taken from the canonical run).
Outputs: results/final_r7/schedule_sens/{detection_metrics,event_detail,
         injection,factorial_stats}.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import scripts.run_final_r7 as R  # noqa: E402
from scripts.run_round2 import range_based  # noqa: E402

OUT = os.path.join(R.OUT, "schedule_sens")
ARMS = {"PatchTST-SSL(HoW-cond)": ("PatchTST", "HoW-cond"),
        "PatchTST-SSL(global-MAD)": ("PatchTST", "global-MAD"),
        "PatchTST-SSL(HoD-cond)": ("PatchTST", "HoD-cond"),
        "HOW-profile(HoW-cond)": ("HOW", "HoW-cond"),
        "HOW-profile(global-MAD)": ("HOW", "global-MAD"),
        "HOW-profile(HoD-cond)": ("HOW", "HoD-cond"),
        "TOWT(HoW-cond)": ("TOWT", "HoW-cond")}


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = R.load_dataset(R.DATA_DIR)
    n_tv = {b: int(len(df) * (R.FR[0] + R.FR[1])) for b, df in ds.items()}
    off = {b: int(len(df) * R.FR[0]) for b, df in ds.items()}
    tb = sorted(b for b in ds if ds[b]["btype"].iloc[0] == "office")
    tr = {b: ds[b].iloc[:off[b]].reset_index(drop=True) for b in ds}
    va = {b: ds[b].iloc[off[b]:n_tv[b]].reset_index(drop=True) for b in ds}
    te = {b: ds[b].iloc[n_tv[b]:].reset_index(drop=True) for b in ds}
    sc = {b: R.ZScaler().fit(tr[b]) for b in ds}
    X = {b: sc[b].transform(ds[b]) for b in ds}
    sd = {b: sc[b].sd_["load"] for b in ds}
    sel = pd.read_csv(os.path.join(R.OUT, "selection_log.csv"))
    det, evt, inj = [], [], []
    r_how = {b: R.how_backbone_residuals(tr[b], ds[b], sd[b]) for b in ds}
    r_towt = {b: R.howtowt_fit_predict(tr[b], ds[b]) / sd[b] for b in ds}
    for seed in R.SEEDS:
        cor, ev = R.inject_all(te, n_events=R.N_EVENTS, seed=seed, ref=tr,
                               schedule_mode="additive", min_peak_frac=0.01)
        cfg = R.PatchTSTConfig(); cfg.seed = seed; cfg.train_mask = "full"
        ssl = R.PatchTSTAD(cfg).fit([sc[b].transform(tr[b]) for b in tb],
                                    val_arrays=[sc[b].transform(va[b]) for b in tb])
        r0 = {"PatchTST": {b: ssl.point_residuals(X[b]) for b in ds}, "HOW": r_how, "TOWT": r_towt}
        del ssl; R.free_mps()
        w = float(pd.to_numeric(sel.query("what == 'fusion_w' and seed == @seed")
                                .sort_values("val_obj", ascending=False, kind="stable")["option"]).iloc[0])
        for b in ds:
            d = np.zeros(len(ds[b]))
            d[n_tv[b]:] = (cor[b]["load"].to_numpy() - ds[b]["load"].to_numpy()[n_tv[b]:]) / sd[b]
            e = ev[ev["building_id"] == b]
            # injection magnitude vs natural residual scatter (HOW residual, pre-test)
            ref = r_how[b][:n_tv[b]] * sd[b]
            rsd = np.median(np.abs(ref - np.median(ref))) * 1.4826
            for _, x in e.iterrows():
                s_, t_ = int(x["start"]), int(x["end"])
                add = (d[n_tv[b]:] * sd[b])[s_:t_]
                on = add > 0
                inj.append({"seed": seed, "building_id": b, "type": x["type"],
                            "severity": int(x["severity"]),
                            "mean_add_over_resid_sd": float(add[on].mean() / rsd) if on.any() and rsd > 0 else np.nan,
                            "excess_over_test_energy": float(x["injected_excess_kwh"] / te[b]["load"].sum())})
            for name, (bb, scoring) in ARMS.items():
                ra = r0[bb][b] + d
                s = R.fuse(w, *R.score_pair(ra, r0[bb][b], X[b], n_tv[b], scoring))[n_tv[b]:]
                f = R.causal_flags(s)
                m = R.evaluate_building(f.astype(float), f, e, len(f))
                rP, rR = range_based(f.astype(bool), e, len(f))
                m.update(range_precision=rP, range_recall=rR,
                         range_f1=2 * rP * rR / (rP + rR) if rP + rR else 0.0,
                         model=name, building_id=b, seed=seed, site=b.split("_")[0])
                det.append(m)
                evt += R.event_rows(name, b, seed, f, e, ra[n_tv[b]:] * sd[b])
        print(f"seed {seed} done", flush=True)
    det = pd.DataFrame(det); det.to_csv(os.path.join(OUT, "detection_metrics.csv"), index=False)
    pd.DataFrame(evt).to_csv(os.path.join(OUT, "event_detail.csv"), index=False)
    pd.DataFrame(inj).to_csv(os.path.join(OUT, "injection.csv"), index=False)
    import scripts.stats_r7 as S
    rows = []
    for metric in ("f1", "range_f1"):
        r, cells = S.factorial(det, metric)
        rows += r
        for k, (mu, sdv) in cells.items():
            rows.append({"metric": metric, "family": "cell", "contrast": k, "estimate": mu})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "factorial_stats.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
