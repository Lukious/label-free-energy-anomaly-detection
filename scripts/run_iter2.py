"""Iteration 2: multi-scale scoring + stronger PatchTST + deeper evaluation.

One-step rerun:  python scripts/run_iter2.py
Changes vs run_smoke.py (iter1):
  1. PatchTST-SSL score is MULTI-SCALE: max( robust-z(point residual),
     robust-z(squared pooled residual) ) — fixes the spike dilution that
     caused iter1 spike recall 0.07 (see src/models/patchtst.py::score).
  2. PatchTST trained with more epochs (80, early stop patience 8 on clean
     validation spans) and a larger model (d_model 128, 3 layers, ff 256).
  3. Waste estimation uses the point-scale (undiluted) residual.
  4. Evaluation adds a type x severity (3-level) recall breakdown and a
     1st-vs-2nd iteration comparison table (iter1 read from results/metrics.csv).
Baselines are unchanged (same seeds) so the comparison isolates our changes.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src.anomaly_inject import inject_all
from src.data import ZScaler, load_dataset, train_test_split
from src.evaluate import (aggregate, evaluate_building, per_type_recall,
                          per_type_severity_recall, robust_z_flags,
                          waste_metrics)
from src.models.baselines import DEVICE, DenseAEAD, IsolationForestAD, LSTMAEAD, OCSVMAD
from src.models.patchtst import PatchTSTAD, PatchTSTConfig

DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FPR_TARGET = 0.01
N_EVENTS_PER_BUILDING = 6
SEED = 123

MODEL_REGISTRY = {
    "IsolationForest": lambda: IsolationForestAD(seed=0),
    "OC-SVM": lambda: OCSVMAD(seed=0),
    "Autoencoder": lambda: DenseAEAD(seed=0),
    "LSTM-AE": lambda: LSTMAEAD(seed=0),
    "PatchTST-SSL": lambda: PatchTSTAD(PatchTSTConfig()),
}


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset = load_dataset(DATA_DIR)
    train, val, test = train_test_split(dataset)

    corrupted, events = inject_all(test, n_events=N_EVENTS_PER_BUILDING, seed=SEED)
    events.to_csv(os.path.join(RESULTS_DIR, "events_iter2.csv"), index=False)
    print(f"[inject] {len(events)} events (types: {events['type'].value_counts().to_dict()})")

    scalers = {bid: ZScaler().fit(df) for bid, df in train.items()}
    train_bids = sorted(train.keys())[:6]
    tr_arr = [scalers[bid].transform(train[bid]) for bid in train_bids]
    va_arr = [scalers[bid].transform(val[bid]) for bid in train_bids]  # early-stop val
    te_arr = {bid: scalers[bid].transform(df) for bid, df in corrupted.items()}

    rows, type_rows, ts_rows = [], [], []
    for name, factory in MODEL_REGISTRY.items():
        t = time.time()
        model = factory()
        if name == "PatchTST-SSL":
            model.fit(tr_arr, val_arrays=va_arr)
            hist = model.history
            stopped = next((h["epoch"] for h in hist if h.get("stopped")), hist[-1]["epoch"])
            best = min((h.get("val_loss", float("nan")) for h in hist), default=float("nan"))
            print(f"[train] epochs run {stopped}, best val loss {best:.4f}")
        else:
            model.fit(np.concatenate(tr_arr, axis=0))
        fit_s = time.time() - t

        for bid in corrupted:
            n = len(te_arr[bid])
            sc = model.score(te_arr[bid])
            flags = robust_z_flags(sc, FPR_TARGET)
            ev = events[events["building_id"] == bid]
            m = evaluate_building(sc, flags, ev, n)
            m.update(model=name, building_id=bid,
                     btype=dataset[bid]["btype"].iloc[0],
                     seen=(bid in train_bids), fit_seconds=round(fit_s, 1))
            if name == "PatchTST-SSL":
                sd = float(scalers[bid].sd_["load"])
                res_kwh = model.waste_residuals(te_arr[bid]) * sd  # point-scale
                m.update(waste_metrics(res_kwh, flags, ev))
            for etype, d in per_type_recall(flags, ev).items():
                type_rows.append({"model": name, "type": etype, **d})
            for (etype, sev), d in per_type_severity_recall(flags, ev).items():
                ts_rows.append({"model": name, "type": etype, "severity": sev, **d})
            rows.append(m)
        print(f"[model] {name}: fit {fit_s:.1f}s done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics_iter2.csv"), index=False)
    tdf, tsdf = pd.DataFrame(type_rows), pd.DataFrame(ts_rows)

    # ---- aggregates ----
    aggs = {}
    for name in MODEL_REGISTRY:
        sub = df[df["model"] == name]
        aggs[name] = aggregate({bid: r.to_dict() for bid, r in
                                sub.set_index("building_id").iterrows()})

    # ---- iter1 comparison (overall F1, type recall, FAR) ----
    comp_lines = ["| Model | F1 (iter1 -> iter2) | FAR (iter1 -> iter2) | spike R | drift R | schedule R |",
                  "|---|---|---|---|---|---|"]
    try:
        it1 = pd.read_csv(os.path.join(RESULTS_DIR, "metrics.csv"))
        it1_f1 = it1.groupby("model")[["f1", "false_alarm_rate"]].mean()
        # iter1 per-type recalls are in the smoke_results table; recompute
        # nothing — parse from results/metrics.csv events (same seed) is not
        # stored, so use the documented smoke numbers for type recall.
        it1_type = {"IsolationForest": (0.96, 1.00, 1.00),
                    "OC-SVM": (0.96, 0.83, 0.29),
                    "Autoencoder": (1.00, 1.00, 1.00),
                    "LSTM-AE": (0.29, 0.33, 0.50),
                    "PatchTST-SSL": (0.07, 0.73, 0.43)}
    except FileNotFoundError:
        it1, it1_f1, it1_type = None, None, {}
    for name in MODEL_REGISTRY:
        a = aggs[name]
        s = tdf[tdf["model"] == name].groupby("type")["recall"].mean()
        if it1_f1 is not None and name in it1_f1.index:
            f1c = f"{it1_f1.loc[name, 'f1']:.2f} -> {a.get('f1', 0):.2f}"
            farc = (f"{it1_f1.loc[name, 'false_alarm_rate']:.2%} -> "
                    f"{a.get('false_alarm_rate', 0):.2%}")
        else:
            f1c, farc = f"{a.get('f1', 0):.2f}", f"{a.get('false_alarm_rate', 0):.2%}"
        sp, dr, sc_ = it1_type.get(name, (float("nan"),) * 3)
        comp_lines.append(
            f"| {name} | {f1c} | {farc} | {sp:.2f} -> {s.get('spike', float('nan')):.2f} "
            f"| {dr:.2f} -> {s.get('drift', float('nan')):.2f} "
            f"| {sc_:.2f} -> {s.get('schedule', float('nan')):.2f} |")

    lines = ["# Iteration 2 Results — SSL Energy Anomaly Detection (P4)",
             "",
             f"- Date: 2026-09-28 | Device: {DEVICE} | FPR target: {FPR_TARGET:.0%}",
             "- Same data / events / seeds / flagger (rolling 2-week robust z) as iter1; "
             "only PatchTST-SSL changed (multi-scale score, stronger training) — "
             "baseline rows verify reproducibility.",
             "",
             "## 1st vs 2nd iteration (key metrics)",
             "", *comp_lines,
             "",
             "## Overall (all 8 buildings, event-level)",
             "",
             "| Model | P | R | F1 | Delay(h) | FAR | Waste err(total) |",
             "|---|---|---|---|---|---|---|"]
    for name in MODEL_REGISTRY:
        a = aggs[name]
        we = f"{a.get('waste_error_total', float('nan')):.1%}" if "waste_error_total" in a else "n/a"
        lines.append(
            f"| {name} | {a.get('precision', 0):.2f} | {a.get('recall', 0):.2f} | "
            f"{a.get('f1', 0):.2f} | {a.get('detection_delay_h', float('nan')):.1f} | "
            f"{a.get('false_alarm_rate', 0):.2%} | {we} |")

    lines += ["", "## Seen (office, 6) vs Unseen (retail, 2)", "",
              "| Model | F1 seen | F1 unseen | Degradation |", "|---|---|---|---|"]
    for name in MODEL_REGISTRY:
        sub = df[df["model"] == name]
        f1s, f1u = sub[sub["seen"]]["f1"].mean(), sub[~sub["seen"]]["f1"].mean()
        lines.append(f"| {name} | {f1s:.2f} | {f1u:.2f} | {f1s - f1u:+.2f} |")

    lines += ["", "## By anomaly type (event recall)", "",
              "| Model | spike | drift | schedule |", "|---|---|---|---|"]
    for name in MODEL_REGISTRY:
        s = tdf[tdf["model"] == name].groupby("type")["recall"].mean()
        lines.append(f"| {name} | {s.get('spike', float('nan')):.2f} | "
                     f"{s.get('drift', float('nan')):.2f} | "
                     f"{s.get('schedule', float('nan')):.2f} |")

    lines += ["", "## By type x severity (event recall) — iter2 new", "",
              "| Model | type | subtle(1) | moderate(2) | obvious(3) |", "|---|---|---|---|---|"]
    for name in MODEL_REGISTRY:
        sub = tsdf[tsdf["model"] == name]
        for etype in ["spike", "drift", "schedule"]:
            cells = []
            for sev in [1, 2, 3]:
                q = sub[(sub["type"] == etype) & (sub["severity"] == sev)]["recall"]
                cells.append(f"{q.mean():.2f} (n={int(q.count())})" if len(q) else "n=0")
            lines.append(f"| {name} | {etype} | " + " | ".join(cells) + " |")

    lines += ["", "## Interpretation", "",
              "- Multi-scale score design: SHORT scale = stride-1 window residual at "
              "the last hour (full 168h history, no pooling -> spikes keep full "
              "magnitude); LONG scale = iter1's pooled residual (drift-sensitive). "
              "Each scale robust-z standardized, combined by element-wise MAX so "
              "an hour is anomalous if extreme on ANY scale; the rolling robust-z "
              "flagger re-standardizes the combined score.",
              "- Waste now uses the point-scale residual (iter1's pooled residual "
              "spread a spike's excess across its whole 168h window).",
              "", f"Total runtime: {time.time() - t0:.0f}s"]
    with open(os.path.join(RESULTS_DIR, "iter2_results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[done] results written ({time.time() - t0:.0f}s total)")
    print(df.groupby("model")[["precision", "recall", "f1", "false_alarm_rate"]].mean().round(3))


if __name__ == "__main__":
    main()
