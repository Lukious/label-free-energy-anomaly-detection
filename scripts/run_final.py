"""Final experiment: 5 independent seeds (data + events + model re-generated per
seed), full iter3 pipeline, paper-grade tables + paired statistics.

One-step rerun:  python scripts/run_final.py

Per seed s in SEEDS:
  - synthetic dataset regenerated (generate_synthetic_dataset(seed=s))
  - test events injected with seed s; calibration events seed = s+333 (disjoint)
  - PatchTST-SSL retrained; fusion weight w calibrated on validation spans
  - all models evaluated with the identical rolling robust-z flagger @1% FPR

Outputs:
  results/metrics_final.csv   (per building x model x seed)
  results/final_results.md    (mean +/- std tables, type/severity, seen/unseen,
                               waste, delay/FAR, Wilcoxon paired test vs OC-SVM)
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

from src.anomaly_inject import inject_all
from src.data import ZScaler, generate_synthetic_dataset, train_test_split
from src.evaluate import (aggregate, evaluate_building, per_type_recall,
                          per_type_severity_recall, robust_z_flags,
                          waste_metrics)
from src.models.baselines import (DenseAEAD, IsolationForestAD, LSTMAEAD,
                                  OCSVMAD, DEVICE)
from src.models.patchtst import PatchTSTAD, PatchTSTConfig

RESULTS_DIR = os.path.join(ROOT, "results")
FPR_TARGET = 0.01
N_EVENTS_PER_BUILDING = 6
SEEDS = [123, 124, 125, 126, 127]

MODEL_REGISTRY = {
    "IsolationForest": lambda: IsolationForestAD(seed=0),
    "OC-SVM": lambda: OCSVMAD(seed=0),
    "Autoencoder": lambda: DenseAEAD(seed=0),
    "LSTM-AE": lambda: LSTMAEAD(seed=0),
}


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    out, s = [], None
    for i, f in enumerate(flags):
        if f and s is None:
            s = i
        elif not f and s is not None:
            out.append((s, i)); s = None
    if s is not None:
        out.append((s, len(flags)))
    return out


def calibrate_fusion(model, val_corrupted, val_events, scalers, grid):
    scales = {bid: model.score_scales(scalers[bid].transform(df))
              for bid, df in val_corrupted.items()}
    best = None
    for w in grid:
        tp, nev, nrun, hit = 0, 0, 0, 0
        for bid, df in val_corrupted.items():
            z_s, z_l = scales[bid]
            sc = np.maximum(z_s, z_l) if w is None else w * z_s + (1 - w) * z_l
            flags = robust_z_flags(sc, FPR_TARGET)
            ev = val_events[val_events["building_id"] == bid]
            m = evaluate_building(sc, flags, ev, len(df))
            tp += m["tp_events"]; nev += m["n_events"]
            anom = np.zeros(len(df), dtype=bool)
            for _, e_ in ev.iterrows():
                anom[int(e_["start"]):int(e_["end"])] = True
            runs = _runs(flags)
            nrun += len(runs)
            hit += sum(1 for s, e in runs if anom[s:e].any())
        P = hit / max(nrun, 1); R = tp / max(nev, 1)
        F = 2 * P * R / max(P + R, 1e-9)
        if best is None or F > best[1]:
            best = (w, F)
    return best[0]


def run_seed(seed: int) -> tuple[pd.DataFrame, list[dict], list[dict], float]:
    dataset = generate_synthetic_dataset(seed=seed)
    train, val, test = train_test_split(dataset)
    corrupted, events = inject_all(test, n_events=N_EVENTS_PER_BUILDING, seed=seed)

    scalers = {bid: ZScaler().fit(df) for bid, df in train.items()}
    train_bids = sorted(train.keys())[:6]
    tr_arr = [scalers[bid].transform(train[bid]) for bid in train_bids]
    te_arr = {bid: scalers[bid].transform(df) for bid, df in corrupted.items()}

    ssl = PatchTSTAD(PatchTSTConfig())
    ssl.fit(tr_arr, val_arrays=[scalers[b].transform(val[b]) for b in train_bids])

    val_c, val_ev = inject_all({b: val[b] for b in train_bids},
                               n_events=N_EVENTS_PER_BUILDING, seed=seed + 333)
    grid = [round(w, 2) for w in np.arange(0.0, 1.01, 0.1)]
    w_best = calibrate_fusion(ssl, val_c, val_ev, scalers, grid)

    def ssl_score(X):
        z_s, z_l = ssl.score_scales(X)
        return w_best * z_s + (1 - w_best) * z_l

    rows, type_rows, ts_rows = [], [], []
    for name, factory in MODEL_REGISTRY.items():
        model = factory()
        model.fit(np.concatenate(tr_arr, axis=0))
        score_fn = model.score
        _eval(name, score_fn, model is None, corrupted, te_arr, events, dataset,
              train_bids, rows, type_rows, ts_rows, seed, ssl=None)
    _eval("PatchTST-SSL", ssl_score, True, corrupted, te_arr, events, dataset,
          train_bids, rows, type_rows, ts_rows, seed, ssl=ssl, scalers=scalers)
    return pd.DataFrame(rows), type_rows, ts_rows, w_best


def _eval(name, score_fn, _, corrupted, te_arr, events, dataset, train_bids,
          rows, type_rows, ts_rows, seed, ssl=None, scalers=None):
    for bid in corrupted:
        n = len(te_arr[bid])
        sc = score_fn(te_arr[bid])
        flags = robust_z_flags(sc, FPR_TARGET)
        ev = events[events["building_id"] == bid]
        m = evaluate_building(sc, flags, ev, n)
        m.update(model=name, building_id=bid, seed=seed,
                 btype=dataset[bid]["btype"].iloc[0],
                 seen=(bid in train_bids))
        if ssl is not None:
            sd = float(scalers[bid].sd_["load"])
            res_kwh = ssl.waste_residuals(te_arr[bid]) * sd
            m.update(waste_metrics(res_kwh, flags, ev))
        for etype, d in per_type_recall(flags, ev).items():
            type_rows.append({"model": name, "type": etype, "seed": seed, **d})
        for (etype, sev), d in per_type_severity_recall(flags, ev).items():
            d = dict(d); d.pop("severity", None)
            ts_rows.append({"model": name, "type": etype, "severity": sev,
                            "seed": seed, **d})
        rows.append(m)


def ms(x):
    return f"{np.nanmean(x):.2f} ± {np.nanstd(x, ddof=1):.2f}"


def main():
    warnings.filterwarnings("ignore")
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows, all_type, all_ts = [], [], []
    w_chosen = {}
    for seed in SEEDS:
        ts_ = time.time()
        df, tr, tsr, w = run_seed(seed)
        w_chosen[seed] = w
        all_rows.append(df); all_type += tr; all_ts += tsr
        agg = df.groupby("model")[["precision", "recall", "f1"]].mean()
        print(f"[seed {seed}] w={w:.1f} F1: " +
              ", ".join(f"{m}={agg.loc[m, 'f1']:.3f}" for m in agg.index) +
              f" ({time.time() - ts_:.0f}s)")
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics_final.csv"), index=False)
    tdf, tsdf = pd.DataFrame(all_type), pd.DataFrame(all_ts)

    models = list(MODEL_REGISTRY) + ["PatchTST-SSL"]
    # per-seed overall metrics for mean±std & pairing
    per_seed = (df.groupby(["model", "seed"])[
        ["precision", "recall", "f1", "detection_delay_h", "false_alarm_rate"]]
        .mean().reset_index())

    L = []
    A = L.append
    A("# Final Results (5-seed) — SSL Energy Anomaly Detection (P4)")
    A("")
    A(f"- Date: 2026-09-28 | Device: {DEVICE} | FPR target: 1% | "
      f"seeds {SEEDS} (data + events + model independent per seed)")
    A(f"- Chosen fusion weights per seed: "
      + ", ".join(f"s{s}: w={w_chosen[s]:.1f}" for s in SEEDS))
    A("")
    A("## (a) Overall comparison, mean ± std over 5 seeds (event-level)")
    A("")
    A("| Model | Precision | Recall | F1 | Delay(h) | FAR | Waste err(total) |")
    A("|---|---|---|---|---|---|---|")
    for m in models:
        g = per_seed[per_seed["model"] == m]
        wg = df[df["model"] == m]["waste_error_total"].dropna()
        we = ms(wg) if len(wg) else "n/a"
        A(f"| {m} | {ms(g['precision'])} | {ms(g['recall'])} | {ms(g['f1'])} | "
          f"{ms(g['detection_delay_h'])} | {ms(g['false_alarm_rate']*100)}% | {we} |")
    A("")
    A("## (b) Recall by anomaly type (mean ± std, 5 seeds)")
    A("")
    A("| Model | spike | drift | schedule |")
    A("|---|---|---|---|")
    for m in models:
        cells = []
        for et in ["spike", "drift", "schedule"]:
            q = tdf[(tdf["model"] == m) & (tdf["type"] == et)].groupby("seed")["recall"].mean()
            cells.append(ms(q) if len(q) else "n/a")
        A(f"| {m} | " + " | ".join(cells) + " |")
    A("")
    A("## (b') Recall by type × severity (mean over 5 seeds)")
    A("")
    A("| Model | type | subtle(1) | moderate(2) | obvious(3) |")
    A("|---|---|---|---|---|")
    for m in models:
        sub = tsdf[tsdf["model"] == m]
        for et in ["spike", "drift", "schedule"]:
            cells = []
            for sev in [1, 2, 3]:
                q = sub[(sub["type"] == et) & (sub["severity"] == sev)].groupby("seed")["recall"].mean()
                cells.append(f"{np.nanmean(q):.2f}" if len(q) else "n/a")
            A(f"| {m} | {et} | " + " | ".join(cells) + " |")
    A("")
    A("## (c) Seen (office) vs Unseen (retail) transfer, F1 (mean ± std)")
    A("")
    A("| Model | seen (office) | unseen (retail) |")
    A("|---|---|---|")
    for m in models:
        g = df[df["model"] == m].groupby(["seed", "seen"])["f1"].mean().unstack()
        A(f"| {m} | {ms(g[True])} | {ms(g[False])} |")
    A("")
    A("## (d) Waste estimation error, PatchTST-SSL (mean ± std over buildings×seeds)")
    A("")
    wg = df[df["model"] == "PatchTST-SSL"]
    A(f"- per-event mean error: {ms(wg['waste_error_mean'])}")
    A(f"- total error: {ms(wg['waste_error_total'])}")
    A("")
    A("## (e) Detection delay & FAR (from table a; per-seed detail)")
    A("")
    A("| seed | PatchTST delay | PatchTST FAR | OC-SVM delay | OC-SVM FAR |")
    A("|---|---|---|---|---|")
    for s in SEEDS:
        p = per_seed[(per_seed["model"] == "PatchTST-SSL") & (per_seed["seed"] == s)].iloc[0]
        o = per_seed[(per_seed["model"] == "OC-SVM") & (per_seed["seed"] == s)].iloc[0]
        A(f"| {s} | {p['detection_delay_h']:.1f}h | {p['false_alarm_rate']:.2%} | "
          f"{o['detection_delay_h']:.1f}h | {o['false_alarm_rate']:.2%} |")
    A("")
    A("## (f) Statistical test — PatchTST-SSL vs strongest baseline (OC-SVM)")
    A("")
    pv = per_seed.pivot(index="seed", columns="model", values="f1")
    pairs = pv["PatchTST-SSL"].values, pv["OC-SVM"].values
    try:
        stat_w, p_w = stats.wilcoxon(*pairs)
        A(f"- Wilcoxon signed-rank on per-seed F1 (n=5): "
          f"stat={stat_w:.1f}, p={p_w:.4f}")
    except Exception as e:
        p_w = float("nan")
        A(f"- Wilcoxon failed: {e}")
    stat_t, p_t = stats.ttest_rel(*pairs)
    A(f"- Paired t-test on per-seed F1: t={stat_t:.3f}, p={p_t:.4f}")
    A(f"- Per-seed F1 PatchTST-SSL: {np.round(pv['PatchTST-SSL'].values, 3).tolist()}")
    A(f"- Per-seed F1 OC-SVM:       {np.round(pv['OC-SVM'].values, 3).tolist()}")
    A(f"- Mean difference: {np.mean(pairs[0] - pairs[1]):+.3f}")
    A("")
    A(f"Total runtime: {time.time() - t0:.0f}s")
    with open(os.path.join(RESULTS_DIR, "final_results.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"[done] ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
