"""Iteration 3: precision-bottleneck attack (false-alarm analysis + calibrated
scale fusion) — same data / events / seeds / flagger as iter1 & iter2.

One-step rerun:  python scripts/run_iter3.py

Changes vs run_iter2.py:
  1. FALSE-POSITIVE ANALYSIS first (diagnosis): every flagged run outside any
     true event is attributed to hour-of-day / weekday / test-span position
     (edge vs interior) / dominant scale (short vs long).
  2. VALIDATION-CALIBRATED SCALE FUSION (fix): iter2 combined the two score
     scales with element-wise max — an uncalibrated choice. iter3 injects a
     SEPARATE calibration event set into the clean validation spans (seed 456,
     disjoint from the test seed 123) and grid-searches the fusion
     score = w * z_short + (1 - w) * z_long  (plus the max reference) for the
     best event-level F1. Only validation data is used for calibration — the
     test protocol (rolling 2-week robust-z flagger @ 1% FPR) is IDENTICAL for
     all models, so numbers remain directly comparable to iter1/iter2.
  3. Protocol hygiene: a min-duration postprocess (drop flag runs < 3h) is
     evaluated as a clearly-labelled UNIFORM VARIANT applied to every model,
     never mixed into the headline protocol (iter2 lesson).
Baselines are unchanged (same seeds).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.anomaly_inject import inject_all
from src.data import ZScaler, load_dataset, train_test_split
from src.evaluate import (aggregate, evaluate_building, per_type_recall,
                          per_type_severity_recall, robust_z_flags,
                          waste_metrics)
from src.models.baselines import DenseAEAD, IsolationForestAD, LSTMAEAD, OCSVMAD, DEVICE
from src.models.patchtst import PatchTSTAD, PatchTSTConfig

DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FPR_TARGET = 0.01
N_EVENTS_PER_BUILDING = 6
SEED = 123          # test events — identical to iter1/iter2
CAL_SEED = 456      # calibration events injected into VALIDATION spans only
MIN_DUR = 3         # uniform postprocess variant: drop flag runs < 3h

MODEL_REGISTRY = {
    "IsolationForest": lambda: IsolationForestAD(seed=0),
    "OC-SVM": lambda: OCSVMAD(seed=0),
    "Autoencoder": lambda: DenseAEAD(seed=0),
    "LSTM-AE": lambda: LSTMAEAD(seed=0),
    "PatchTST-SSL": lambda: PatchTSTAD(PatchTSTConfig()),
}


# ---------------------------------------------------------------- FP analysis
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


def false_alarm_runs(flags, events: pd.DataFrame, n_points: int,
                     timestamps: pd.Series, z_short=None, z_long=None) -> list[dict]:
    """Flagged runs that do NOT overlap any true event, with attribution."""
    anom = np.zeros(n_points, dtype=bool)
    for _, ev in events.iterrows():
        anom[int(ev["start"]):int(ev["end"])] = True
    recs = []
    for s, e in _runs(flags):
        if anom[s:e].any():
            continue
        ts = timestamps.iloc[s]
        rec = {
            "hour": int(ts.hour), "dow": int(ts.dayofweek),
            "len_h": e - s,
            "edge": bool(s < 336 or e > n_points - 336),
            "scale": None,
        }
        if z_short is not None and z_long is not None:
            rec["scale"] = "short" if z_short[s:e].mean() >= z_long[s:e].mean() else "long"
        recs.append(rec)
    return recs


def drop_short_runs(flags: np.ndarray, min_dur: int = MIN_DUR) -> np.ndarray:
    out = flags.copy()
    for s, e in _runs(flags):
        if e - s < min_dur:
            out[s:e] = False
    return out


def summarize_fp(recs: list[dict]) -> dict:
    if not recs:
        return {"n": 0}
    df = pd.DataFrame(recs)
    df["len_hist"] = df["len_h"].clip(upper=6).map(lambda x: str(int(x)) if x < 6 else "6+")
    hod = pd.cut(df["hour"], [-1, 5, 7, 17, 21, 24],
                 labels=["night(0-5)", "ramp(6-7)", "daytime(8-17)",
                         "evening(18-21)", "late(22-23)"], )
    return {
        "n": len(df),
        "mean_len": round(float(df["len_h"].mean()), 1),
        "frac_len<=2": round(float((df["len_h"] <= 2).mean()), 3),
        "by_hod": hod.value_counts().to_dict(),
        "len_hist": df["len_hist"].value_counts().to_dict(),
        "frac_weekend": round(float((df["dow"] >= 5).mean()), 3),
        "frac_edge": round(float(df["edge"].mean()), 3),
        "by_scale": df["scale"].value_counts().to_dict() if "scale" in df else {},
    }


# ------------------------------------------------------------- calibration
def calibrate_fusion(model, val_corrupted, val_events, scalers, grid):
    """Grid-search fusion weight on validation-injected anomalies only."""
    scales = {}
    for bid, df in val_corrupted.items():
        X = scalers[bid].transform(df)
        scales[bid] = model.score_scales(X)
    best = None
    rows = []
    for w in grid:
        tp_runs = {"tp": 0, "runs": 0, "hit": 0, "nev": 0}
        for bid, df in val_corrupted.items():
            z_s, z_l = scales[bid]
            sc = np.maximum(z_s, z_l) if w is None else w * z_s + (1 - w) * z_l
            flags = robust_z_flags(sc, FPR_TARGET)
            ev = val_events[val_events["building_id"] == bid]
            m = evaluate_building(sc, flags, ev, len(df))
            tp_runs["tp"] += m["tp_events"]; tp_runs["nev"] += m["n_events"]
            # precision components
            anom = np.zeros(len(df), dtype=bool)
            for _, e_ in ev.iterrows():
                anom[int(e_["start"]):int(e_["end"])] = True
            runs = _runs(flags)
            tp_runs["runs"] += len(runs)
            tp_runs["hit"] += sum(1 for s, e in runs if anom[s:e].any())
        P = tp_runs["hit"] / max(tp_runs["runs"], 1)
        R = tp_runs["tp"] / max(tp_runs["nev"], 1)
        F = 2 * P * R / max(P + R, 1e-9)
        label = "max" if w is None else f"w={w:.2f}"
        rows.append({"fusion": label, "P": round(P, 3), "R": round(R, 3), "F1": round(F, 3)})
        if best is None or F > best[1]:
            best = (w, F)
    return best[0], pd.DataFrame(rows), scales


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset = load_dataset(DATA_DIR)
    train, val, test = train_test_split(dataset)

    corrupted, events = inject_all(test, n_events=N_EVENTS_PER_BUILDING, seed=SEED)
    events.to_csv(os.path.join(RESULTS_DIR, "events_iter3.csv"), index=False)
    print(f"[inject] {len(events)} test events (seed {SEED}, same as iter2)")

    scalers = {bid: ZScaler().fit(df) for bid, df in train.items()}
    train_bids = sorted(train.keys())[:6]
    tr_arr = [scalers[bid].transform(train[bid]) for bid in train_bids]
    va_arr = [scalers[bid].transform(val[bid]) for bid in train_bids]
    te_arr = {bid: scalers[bid].transform(df) for bid, df in corrupted.items()}

    # ---- train PatchTST (identical config/seed to iter2) ----
    ssl = PatchTSTAD(PatchTSTConfig())
    ssl.fit(tr_arr, val_arrays=[scalers[b].transform(val[b]) for b in train_bids])
    best_val = min(h.get("val_loss", np.nan) for h in ssl.history)
    print(f"[train] PatchTST best val loss {best_val:.4f}")

    # ---- calibration on validation spans with injected anomalies ----
    val_c, val_ev = inject_all({b: val[b] for b in train_bids},
                               n_events=N_EVENTS_PER_BUILDING, seed=CAL_SEED)
    grid = [None] + [round(w, 2) for w in np.arange(0.0, 1.01, 0.1)]
    w_best, cal_df, _ = calibrate_fusion(ssl, val_c, val_ev, scalers, grid)
    print("[calibration] validation grid:")
    print(cal_df.to_string(index=False))
    print(f"[calibration] chosen fusion: {'max' if w_best is None else f'w={w_best}'}")
    cal_df.to_csv(os.path.join(RESULTS_DIR, "calibration_iter3.csv"), index=False)

    def ssl_score(X):
        z_s, z_l = ssl.score_scales(X)
        return np.maximum(z_s, z_l) if w_best is None else w_best * z_s + (1 - w_best) * z_l

    def ssl_scales(X):
        return ssl.score_scales(X)

    # ---- false-positive analysis: iter2 (max) vs iter3 (calibrated) ----
    fp_iter2, fp_iter3 = [], []
    for bid in corrupted:
        ev = events[events["building_id"] == bid]
        ts = corrupted[bid]["timestamp"].reset_index(drop=True)
        z_s, z_l = ssl_scales(te_arr[bid])
        f_max = robust_z_flags(np.maximum(z_s, z_l), FPR_TARGET)
        f_cal = robust_z_flags(ssl_score(te_arr[bid]), FPR_TARGET)
        fp_iter2 += false_alarm_runs(f_max, ev, len(ts), ts, z_s, z_l)
        fp_iter3 += false_alarm_runs(f_cal, ev, len(ts), ts, z_s, z_l)
    s2, s3 = summarize_fp(fp_iter2), summarize_fp(fp_iter3)
    print(f"[fp] iter2 max-score FPs: {s2}")
    print(f"[fp] iter3 calibrated FPs: {s3}")

    # ---- evaluation loop (identical protocol for every model) ----
    rows, type_rows, ts_rows, md_rows = [], [], [], []
    for name, factory in MODEL_REGISTRY.items():
        t = time.time()
        if name == "PatchTST-SSL":
            model, score_fn = ssl, ssl_score
        else:
            model = factory()
            model.fit(np.concatenate(tr_arr, axis=0))
            score_fn = model.score
        fit_s = time.time() - t

        for bid in corrupted:
            n = len(te_arr[bid])
            sc = score_fn(te_arr[bid])
            flags = robust_z_flags(sc, FPR_TARGET)
            ev = events[events["building_id"] == bid]
            m = evaluate_building(sc, flags, ev, n)
            m.update(model=name, building_id=bid,
                     btype=dataset[bid]["btype"].iloc[0],
                     seen=(bid in train_bids), fit_seconds=round(fit_s, 1))
            if name == "PatchTST-SSL":
                sd = float(scalers[bid].sd_["load"])
                res_kwh = ssl.waste_residuals(te_arr[bid]) * sd
                m.update(waste_metrics(res_kwh, flags, ev))
            for etype, d in per_type_recall(flags, ev).items():
                type_rows.append({"model": name, "type": etype, **d})
            for (etype, sev), d in per_type_severity_recall(flags, ev).items():
                ts_rows.append({"model": name, "type": etype, "severity": sev, **d})
            rows.append(m)

            # uniform postprocess VARIANTs (all models, kept out of headline)
            for d in (1, 2):
                flags_md = drop_short_runs(flags, d)
                m2 = evaluate_building(sc, flags_md, ev, n)
                m2.update(model=name, building_id=bid, variant=f"min_dur{d}")
                md_rows.append(m2)
        print(f"[model] {name}: fit {fit_s:.1f}s done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics_iter3.csv"), index=False)
    pd.DataFrame(md_rows).to_csv(os.path.join(RESULTS_DIR, "metrics_iter3_mindur.csv"),
                                 index=False)
    tdf, tsdf = pd.DataFrame(type_rows), pd.DataFrame(ts_rows)

    aggs = {name: aggregate({bid: r.to_dict() for bid, r in
                             df[df["model"] == name].set_index("building_id").iterrows()})
            for name in MODEL_REGISTRY}

    # trend vs iter1/iter2
    it = {1: "metrics.csv", 2: "metrics_iter2.csv"}
    prev = {}
    for i, fn in it.items():
        p = os.path.join(RESULTS_DIR, fn)
        if os.path.exists(p):
            d = pd.read_csv(p)
            prev[i] = d.groupby("model")[["precision", "recall", "f1", "false_alarm_rate"]].mean()

    L = []
    A = L.append
    A("# Iteration 3 Results — SSL Energy Anomaly Detection (P4)")
    A("")
    A(f"- Date: 2026-09-28 | Device: {DEVICE} | FPR target: {FPR_TARGET:.0%} | "
      f"test seed {SEED} (same events as iter1/2)")
    A("- Protocol identical to iter1/2 for every model: rolling 2-week robust-z "
      "flagger. Fusion weight calibrated ONLY on validation spans with a "
      "disjoint calibration event set (seed 456).")
    A("")
    A("## 1. False-alarm analysis (diagnosis, on iter2 max-combined score)")
    A("")
    A("```json")
    A(str(s2))
    A("```")
    A("")
    A("Interpretation of where the false alarms live (see JSON): dominant scale, "
      "hour-of-day band, weekday/weekend, span-edge share, and how many FP runs "
      "are ultra-short (<=2h) — ultra-short noisy runs and short-scale noise "
      "floor are the precision bottleneck candidates.")
    A("")
    A("After calibrated fusion:")
    A("")
    A("```json")
    A(str(s3))
    A("```")
    A("")
    A("## 2. Calibration (validation spans, injected events seed 456)")
    A("")
    A("| fusion | P | R | F1 |")
    A("|---|---|---|---|")
    for _, r in cal_df.iterrows():
        A(f"| {r['fusion']} | {r['P']} | {r['R']} | {r['F1']} |")
    A("")
    A(f"Selected: **{'max' if w_best is None else f'w={w_best}'}** "
      f"(best validation F1). PatchTST-SSL test score = "
      f"{'max(z_short, z_long)' if w_best is None else f'{w_best}*z_short + {round(1-w_best,2)}*z_long'}.")
    A("")
    A("## 3. Iteration trend (1st -> 2nd -> 3rd, overall)")
    A("")
    A("| Model | F1 | Precision | Recall | FAR |")
    A("|---|---|---|---|---|")
    for name in MODEL_REGISTRY:
        a = aggs[name]
        def cell(metric):
            parts = [f"{prev[i].loc[name, metric]:.2f}" for i in (1, 2) if i in prev and name in prev[i].index]
            parts.append(f"{a.get(metric, 0):.2f}")
            return " -> ".join(parts)
        farc = []
        for i in (1, 2):
            if i in prev and name in prev[i].index:
                farc.append(f"{prev[i].loc[name, 'false_alarm_rate']:.2%}")
        farc.append(f"{a.get('false_alarm_rate', 0):.2%}")
        A(f"| {name} | {cell('f1')} | {cell('precision')} | {cell('recall')} | "
          f"{' -> '.join(farc)} |")
    A("")
    A("## 4. Overall (all 8 buildings, event-level, headline protocol)")
    A("")
    A("| Model | P | R | F1 | Delay(h) | FAR | Waste err(total) |")
    A("|---|---|---|---|---|---|---|")
    for name in MODEL_REGISTRY:
        a = aggs[name]
        we = f"{a.get('waste_error_total', float('nan')):.1%}" if "waste_error_total" in a else "n/a"
        A(f"| {name} | {a.get('precision', 0):.2f} | {a.get('recall', 0):.2f} | "
          f"{a.get('f1', 0):.2f} | {a.get('detection_delay_h', float('nan')):.1f} | "
          f"{a.get('false_alarm_rate', 0):.2%} | {we} |")
    A("")
    A("## 5. By anomaly type (event recall)")
    A("")
    A("| Model | spike | drift | schedule |")
    A("|---|---|---|---|")
    for name in MODEL_REGISTRY:
        s = tdf[tdf["model"] == name].groupby("type")["recall"].mean()
        A(f"| {name} | {s.get('spike', np.nan):.2f} | {s.get('drift', np.nan):.2f} | "
          f"{s.get('schedule', np.nan):.2f} |")
    A("")
    A("## 6. By type x severity (event recall)")
    A("")
    A("| Model | type | subtle(1) | moderate(2) | obvious(3) |")
    A("|---|---|---|---|---|")
    for name in MODEL_REGISTRY:
        sub = tsdf[tsdf["model"] == name]
        for etype in ["spike", "drift", "schedule"]:
            cells = []
            for sev in [1, 2, 3]:
                q = sub[(sub["type"] == etype) & (sub["severity"] == sev)]["recall"]
                cells.append(f"{q.mean():.2f} (n={int(q.count())})" if len(q) else "n=0")
            A(f"| {name} | {etype} | " + " | ".join(cells) + " |")
    A("")
    A("## 7. Uniform min-duration postprocess variants — protocol variants, NOT headline")
    A("")
    mdd = pd.DataFrame(md_rows)
    A("| Model | F1 (headline / dur1 / dur2) | FAR (headline / dur1 / dur2) |")
    A("|---|---|---|")
    for name in MODEL_REGISTRY:
        h = df[df["model"] == name]
        cells_f, cells_far = [f"{h['f1'].mean():.2f}"], [f"{h['false_alarm_rate'].mean():.2%}"]
        for d in (1, 2):
            v = mdd[(mdd["model"] == name) & (mdd["variant"] == f"min_dur{d}")]
            cells_f.append(f"{v['f1'].mean():.2f}")
            cells_far.append(f"{v['false_alarm_rate'].mean():.2%}")
        A(f"| {name} | {' / '.join(cells_f)} | {' / '.join(cells_far)} |")
    A("")
    A(f"Total runtime: {time.time() - t0:.0f}s")
    with open(os.path.join(RESULTS_DIR, "iter3_results.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[done] written ({time.time() - t0:.0f}s)")
    print(df.groupby("model")[["precision", "recall", "f1", "false_alarm_rate"]].mean().round(3))


if __name__ == "__main__":
    main()
