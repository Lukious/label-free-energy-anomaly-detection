"""R9: synthetic-data experiment re-run under the NEW protocol
(causal normalization + validation-tuned baselines), replacing the old
transductive/untuned synthetic numbers.

Minimal scope (budget): 8 synthetic buildings (6 office seen + 2 retail
unseen), 1 data seed, injection seeds {123,124,125}, causal flagger,
train-frozen scale stats, baselines tuned on validation injection events
(same grid as the real-data round-1 protocol).
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.run_revision import (  # noqa: E402
    _runs, causal_flags, causal_scale_z, fuse, hour_of, val_event_f1,
    val_test_len, IsolationForestAD_tunable, OCSVMAD_tunable,
    DenseAEAD_tunable, LSTMAEAD_tunable,
)
from src.anomaly_inject import inject_all
from src.data import ZScaler, generate_synthetic_dataset
from src.evaluate import evaluate_building
from src.models.patchtst import PatchTSTAD, PatchTSTConfig

RESULTS_DIR = os.path.join(ROOT, "results")
FR = (0.60, 0.15, 0.25)
N_EVENTS = 6
DATA_SEED = 42
SEEDS = [123, 124, 125]


def main():
    warnings.filterwarnings("ignore")
    t0 = time.time()
    dataset = generate_synthetic_dataset(seed=DATA_SEED)
    n_tv = {b: int(len(df) * (FR[0] + FR[1])) for b, df in dataset.items()}
    train_bids = [b for b in sorted(dataset) if dataset[b]["btype"].iloc[0] != "retail"]
    rows = []
    for seed in SEEDS:
        train, val, test = {}, {}, {}
        for bid, df in dataset.items():
            n = len(df)
            a, b = int(n * FR[0]), int(n * (FR[0] + FR[1]))
            train[bid] = df.iloc[:a].reset_index(drop=True)
            val[bid] = df.iloc[a:b].reset_index(drop=True)
            test[bid] = df.iloc[b:].reset_index(drop=True)
            val_test_len[bid] = b - a
        corrupted, events = inject_all(test, n_events=N_EVENTS, seed=seed)
        val_off = {b: int(len(df) * FR[0]) for b, df in dataset.items()}
        val_c, val_ev = inject_all({b: val[b] for b in train_bids},
                                   n_events=N_EVENTS, seed=seed + 333)
        scalers = {b: ZScaler().fit(train[b]) for b in train}
        tr_arr = [scalers[b].transform(train[b]) for b in train_bids]
        va_arr = [scalers[b].transform(val[b]) for b in train_bids]

        ssl = PatchTSTAD(PatchTSTConfig())
        ssl.fit(tr_arr, val_arrays=va_arr)
        full_arr = {b: scalers[b].transform(dataset[b]) for b in dataset}
        r_pt = {b: ssl.point_residuals(full_arr[b]) for b in dataset}
        r_pl = {b: ssl.residuals(full_arr[b]) for b in dataset}
        zs, zl = {}, {}
        for b in dataset:
            zs[b], zl[b] = causal_scale_z(r_pt[b], r_pl[b], full_arr[b], n_tv[b])
        vzs, vzl = {}, {}
        for b in train_bids:
            d = np.zeros(len(dataset[b]))
            vv = val_c[b]
            d[val_off[b]:val_off[b] + len(vv)] = \
                (vv["load"].to_numpy() - dataset[b]["load"].to_numpy()[val_off[b]:val_off[b] + len(vv)]) \
                / scalers[b].sd_["load"]
            vzs[b], vzl[b] = causal_scale_z(r_pt[b] + d, r_pl[b], full_arr[b], n_tv[b])
        w_best, _ = val_event_f1(vzs, vzl, val_off, val_ev, train_bids, [0.1 * i for i in range(11)])

        fl = {}
        for b in corrupted:
            d = np.zeros(len(dataset[b]))
            d[n_tv[b]:] = (corrupted[b]["load"].to_numpy()
                           - dataset[b]["load"].to_numpy()[n_tv[b]:]) \
                / scalers[b].sd_["load"]
            z_s, z_l = causal_scale_z(r_pt[b] + d, r_pl[b], full_arr[b], n_tv[b])
            fl[b] = causal_flags(fuse(w_best, z_s[n_tv[b]:], z_l[n_tv[b]:]))
        for b, f in fl.items():
            ev = events[events["building_id"] == b]
            m = evaluate_building(f.astype(float), f, ev, len(f))
            m.update(model="PatchTST-SSL", building_id=b, seed=seed,
                     seen=(b in train_bids))
            rows.append(m)

        # HOW-profile+z (same protocol)
        for b in corrupted:
            how_tr = train[b]["timestamp"].dt.dayofweek * 24 + train[b]["timestamp"].dt.hour
            how_te = corrupted[b]["timestamp"].dt.dayofweek * 24 + corrupted[b]["timestamp"].dt.hour
            med = train[b].groupby(how_tr)["load"].median()
            mad = train[b].groupby(how_tr)["load"].apply(
                lambda x: np.median(np.abs(x - np.median(x))) * 1.4826)
            z = (corrupted[b]["load"].to_numpy() - med.reindex(how_te).to_numpy()) \
                / (mad.reindex(how_te).to_numpy() + 1e-9)
            f = causal_flags(z)
            ev = events[events["building_id"] == b]
            m = evaluate_building(z, f, ev, len(z))
            m.update(model="HOW-profile+z", building_id=b, seed=seed,
                     seen=(b in train_bids))
            rows.append(m)

        # tuned point baselines
        tr_pts = np.concatenate(tr_arr, axis=0)

        def tune(name, factory, cfgs):
            best = (None, -1)
            for cfg in cfgs:
                mdl = factory(**cfg)
                mdl.fit(tr_pts)
                tp, nev, nrun, hit = 0, 0, 0, 0
                for b in train_bids:
                    Xv = scalers[b].transform(val_c[b])
                    f = causal_flags(mdl.score(Xv))
                    ev = val_ev[val_ev["building_id"] == b]
                    m = evaluate_building(mdl.score(Xv), f, ev, len(Xv))
                    tp += m["tp_events"]; nev += m["n_events"]
                    anom = np.zeros(len(Xv), dtype=bool)
                    for _, e_ in ev.iterrows():
                        anom[int(e_["start"]):int(e_["end"])] = True
                    runs = _runs(f)
                    nrun += len(runs)
                    hit += sum(1 for s, e in runs if anom[s:e].any())
                P = hit / max(nrun, 1); R = tp / max(nev, 1)
                F = 2 * P * R / max(P + R, 1e-9)
                if F > best[1]:
                    best = (cfg, F)
            mdl = factory(**best[0]); mdl.fit(tr_pts)
            return name, mdl

        for name, mdl in [
            tune("IF(t)", lambda **c: IsolationForestAD_tunable(seed=0, **c),
                 [{"contamination": c} for c in (0.01, 0.05, 0.1)]),
            tune("OC-SVM(t)", lambda **c: OCSVMAD_tunable(seed=0, **c),
                 [{"nu": n, "gamma": g} for n in (0.02, 0.05, 0.1)
                  for g in ("scale", 0.2)]),
            tune("AE(t)", lambda **c: DenseAEAD_tunable(seed=0, **c),
                 [{"hidden": h} for h in ((32, 16, 32), (64, 32, 64))]),
        ]:
            for b in corrupted:
                Xt = scalers[b].transform(corrupted[b])
                f = causal_flags(mdl.score(Xt))
                ev = events[events["building_id"] == b]
                m = evaluate_building(mdl.score(Xt), f, ev, len(Xt))
                m.update(model=name, building_id=b, seed=seed,
                         seen=(b in train_bids))
                rows.append(m)
        agg = pd.DataFrame(rows).groupby("model")["f1"].mean()
        print(f"[seed {seed}] w={w_best:.1f} " +
              ", ".join(f"{m}={v:.3f}" for m, v in agg.items()))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics_round2_synthetic.csv"), index=False)
    ps = df.groupby(["model", "seed"])[["precision", "recall", "f1"]].mean().reset_index()
    L = ["# R9 — Synthetic data re-run under the NEW (causal, tuned) protocol",
         "",
         f"8 synthetic buildings (seed {DATA_SEED}), injection seeds {SEEDS}.",
         "",
         "| Model | Precision | Recall | F1 | seen | unseen |", "|---|---|---|---|---|---|"]
    for m in ps["model"].unique():
        g = ps[ps["model"] == m]
        tr_ = df[df["model"] == m].groupby(["seed", "seen"])["f1"].mean().unstack()
        L.append(f"| {m} | {g['precision'].mean():.2f}±{g['precision'].std():.2f} | "
                 f"{g['recall'].mean():.2f}±{g['recall'].std():.2f} | "
                 f"{g['f1'].mean():.2f}±{g['f1'].std():.2f} | "
                 f"{tr_[True].mean():.2f} | {tr_[False].mean():.2f} |")
    L.append(f"\nTotal runtime: {time.time() - t0:.0f}s")
    with open(os.path.join(RESULTS_DIR, "round2_synthetic.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
