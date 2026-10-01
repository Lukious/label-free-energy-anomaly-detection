"""Round-2 review experiments (R1-R9 priority order).

P-A (R3) waste redesign:
  1. two-stage eval: (a) oracle event-window waste per ESTIMATOR (pure
     quantification skill), (b) own-flag operational waste per DETECTOR
  2. estimators: PatchTST-SSL, HOW-profile, TOWT, AE (+LSTM-AE caveat)
  3. kWh additive decomposition (matched + false-add - missed == total delta);
     fixes the round-1 bug where ALL models used PatchTST's flags (hence
     identical 15.9% missed) for their waste rows
  4. noise-floor correction of eq.(5) max(r,0): raw-clip vs sigma-floor
     subtraction (n*sigma/sqrt(2pi)) vs signed integration
  5. inference-masking verified from code (see docstring bottom)
P-B (R1): HOW-profile as formal row + paired tests vs LSTM-AE / PatchTST +
     TOST equivalence test (margin +-0.03 F1), scipy only
P-C (R2): hour-of-week (day-type aware) conditional short-scale variant
P-D (R7): same 41 buildings/seed, transductive vs causal normalization only
P-E: R4 range-based P/R (Tatbul), R6 meaningful IF tuning, input definitions

Outputs: results/metrics_round2*.csv, results/round2_experiments.md
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
    FLAG_WINDOW, FR, Z99, IsolationForestAD_tunable, LSTMAEAD_tunable,
    OCSVMAD_tunable, _runs, causal_flags, causal_scale_z, fuse,
    howtowt_fit_predict, hour_of, val_event_f1, val_test_len,
    DenseAEAD_tunable,
)
from src.anomaly_inject import inject_all
from src.data import ZScaler, load_dataset
from src.evaluate import evaluate_building, robust_z_flags
from src.models.baselines import DEVICE  # noqa: F401
from src.models.patchtst import PatchTSTAD, PatchTSTConfig

RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data", "bdg2", "selected_v2")
N_EVENTS = 6
SEEDS = [123, 124, 125]
SQ2PI = np.sqrt(2.0 * np.pi)

# best baseline configs from the Round-1 tuning log (same validation events)
BASE_CFG = {123: dict(ocsvm=(0.1, "scale")), 124: dict(ocsvm=(0.1, "scale")),
            125: dict(ocsvm=(0.02, 0.2))}


def dow_of(X):
    return (np.degrees(np.arctan2(X[:, 4], X[:, 5])) % 360.0 / 51.43).round().astype(int) % 7


def how_of(X):
    return dow_of(X) * 24 + hour_of(X)


def cond_z(r_point, X, n_tv, n_groups, group_fn, min_group=20):
    """Conditional robust-z with TRAIN-FROZKEN stats and hierarchical fallback
    (group -> pooled) when a group has too few train samples."""
    g = group_fn(X)
    z = np.empty_like(r_point)
    for gg in range(n_groups):
        m = g == gg
        if not m.any():
            continue
        m_tr = (g[:n_tv] == gg) & (~np.isnan(r_point[:n_tv]))
        if m_tr.sum() >= min_group:
            med = np.median(r_point[:n_tv][m_tr])
            mad = np.median(np.abs(r_point[:n_tv][m_tr] - med)) * 1.4826
        else:  # fallback: pooled train stats
            med = np.nanmedian(r_point[:n_tv])
            mad = np.nanmedian(np.abs(r_point[:n_tv] - np.nanmedian(r_point[:n_tv]))) * 1.4826
        z[m] = (r_point[m] - med) / (mad + 1e-9)
    return z


def range_based(flags, events, n):
    """Tatbul et al. 2018 range-based precision/recall (overlap-fraction)."""
    anom = np.zeros(n, dtype=bool)
    for _, e in events.iterrows():
        anom[int(e["start"]):int(e["end"])] = True
    rR = np.mean([flags[int(e["start"]):int(e["end"])].mean()
                  for _, e in events.iterrows()]) if len(events) else np.nan
    runs = _runs(flags)
    rP = np.mean([anom[s:e].mean() for s, e in runs]) if runs else 0.0
    return rP, rR


def sigma_hat_nonevent(resid, events, n):
    anom = np.zeros(n, dtype=bool)
    for _, e in events.iterrows():
        anom[int(e["start"]):int(e["end"])] = True
    r = resid[~anom]
    r = r[~np.isnan(r)]
    return np.median(np.abs(r - np.median(r))) * 1.4826


def oracle_waste(resid, events, sig):
    """Three estimator variants on TRUE event windows (R3.4/R3.5)."""
    clip, signed, floor = [], [], []
    for _, ev in events.iterrows():
        s, e = int(ev["start"]), int(ev["end"])
        seg = resid[s:e]
        seg = seg[~np.isnan(seg)]
        nh = len(seg)
        clip.append(np.clip(seg, 0, None).sum())
        signed.append(seg.sum())
        floor.append(max(np.clip(seg, 0, None).sum() - nh * sig / SQ2PI, 0.0))
    true = np.array([ev["injected_excess_kwh"] for _, ev in events.iterrows()])
    est = {"clip": np.array(clip), "signed": np.array(signed),
           "floor": np.array(floor)}
    return true, est


def ops_waste(resid, flags, events):
    """Own-flag operational waste with kWh additive decomposition (R3.3).

    delta = est_flag_total - true_total
          = (est on TP event hours - true of matched events)
          + est on non-event flagged hours (false-add)
          - true of unmatched events (missed)
    (exact identity; all terms in kWh)."""
    anom = np.zeros(len(resid), dtype=bool)
    matched = []
    for _, ev in events.iterrows():
        s, e = int(ev["start"]), int(ev["end"])
        anom[s:e] = True
        matched.append(bool(flags[s:e].any()))
    matched = np.array(matched, dtype=bool)
    rc = np.clip(np.nan_to_num(resid, nan=0.0), 0, None)
    flagged = flags.astype(bool)
    est_total = rc[flagged].sum()
    true_arr = np.array([ev["injected_excess_kwh"] for _, ev in events.iterrows()])
    true_total = true_arr.sum()
    est_matched = est_unmatched_ev = 0.0
    for m, (_, ev) in zip(matched, events.iterrows()):
        s, e = int(ev["start"]), int(ev["end"])
        if m:
            est_matched += rc[s:e][flagged[s:e]].sum()
        else:
            est_unmatched_ev += rc[s:e][flagged[s:e]].sum()
    true_matched = true_arr[matched].sum()
    false_add = rc[flagged & ~anom].sum()
    missed = true_arr[~matched].sum()
    # exact identity check: est_total - true_total ==
    #   (est_matched - true_matched) + false_add + est_unmatched_ev - missed
    assert abs((est_total - true_total)
               - ((est_matched - true_matched) + false_add + est_unmatched_ev
                  - missed)) < 1e-6 * max(true_total, 1.0)
    return {
        "est_kwh": float(est_total), "true_kwh": float(true_total),
        "delta_kwh": float(est_total - true_total),
        "matched_err_kwh": float(est_matched - true_matched),
        "false_add_kwh": float(false_add),
        "unmatched_ev_flagged_kwh": float(est_unmatched_ev),
        "missed_kwh": float(missed),
        "frac_events_matched": float(matched.mean()),
    }


def tost(diffs, margin=0.03):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    sd = d.std(ddof=1) / np.sqrt(n)
    t1 = (d.mean() + margin) / sd
    t2 = (margin - d.mean()) / sd
    p1 = 1 - stats.t.cdf(t1, n - 1)
    p2 = 1 - stats.t.cdf(t2, n - 1)
    return {"mean": d.mean(), "n": n, "p_tost": max(p1, p2),
            "ci90": (d.mean() - stats.t.ppf(0.95, n - 1) * sd,
                     d.mean() + stats.t.ppf(0.95, n - 1) * sd)}


def main():
    warnings.filterwarnings("ignore")
    t0 = time.time()
    dataset = load_dataset(DATA_DIR)
    print(f"loaded {len(dataset)} buildings")
    n_tv = {b: int(len(df) * (FR[0] + FR[1])) for b, df in dataset.items()}
    train_bids = sorted(b for b in dataset
                        if dataset[b]["btype"].iloc[0] == "office")

    det_rows, orc_rows, ops_rows, howc_rows, trans_rows, if_rows = \
        [], [], [], [], [], []

    for seed in SEEDS:
        ts_ = time.time()
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

        # ---- PatchTST backbone (trained once per seed, clean data)
        ssl = PatchTSTAD(PatchTSTConfig())
        ssl.fit(tr_arr, val_arrays=va_arr)
        full_arr = {b: scalers[b].transform(dataset[b]) for b in dataset}
        r_pt = {b: ssl.point_residuals(full_arr[b]) for b in dataset}
        r_pl = {b: ssl.residuals(full_arr[b]) for b in dataset}

        zs, zl = {}, {}
        for b in dataset:
            zs[b], zl[b] = causal_scale_z(r_pt[b], r_pl[b], full_arr[b], n_tv[b])

        # fusion w on validation injection events (same protocol as round 1)
        vzs, vzl = {}, {}
        for b in train_bids:
            d = np.zeros(len(dataset[b]))
            vv = val_c[b]
            d[val_off[b]:val_off[b] + len(vv)] = \
                (vv["load"].to_numpy() - dataset[b]["load"].to_numpy()[val_off[b]:val_off[b] + len(vv)]) \
                / scalers[b].sd_["load"]
            vzs[b], vzl[b] = causal_scale_z(r_pt[b] + d, r_pl[b], full_arr[b], n_tv[b])
        grid = [round(w, 2) for w in np.arange(0.0, 1.01, 0.1)]
        w_best, _ = val_event_f1(vzs, vzl, val_off, val_ev, train_bids, grid)
        print(f"[seed {seed}] w_best={w_best}")

        # test-span z with corruption delta
        tzs, tzl, tzs_how = {}, {}, {}
        resid_raw, resid_how, resid_towt = {}, {}, {}
        for b in corrupted:
            d = np.zeros(len(dataset[b]))
            d[n_tv[b]:] = (corrupted[b]["load"].to_numpy()
                           - dataset[b]["load"].to_numpy()[n_tv[b]:]) \
                / scalers[b].sd_["load"]
            rp = r_pt[b] + d
            z_s, z_l = causal_scale_z(rp, r_pl[b], full_arr[b], n_tv[b])
            tzs[b], tzl[b] = z_s[n_tv[b]:], z_l[n_tv[b]:]
            tzs_how[b] = cond_z(rp, full_arr[b], n_tv[b], 168, how_of)[n_tv[b]:]
            resid_raw[b] = rp[n_tv[b]:] * scalers[b].sd_["load"]
            # HOW-profile estimator residual (train-profile median, kWh= kW*1h)
            how_tr = train[b]["timestamp"].dt.dayofweek * 24 + train[b]["timestamp"].dt.hour
            how_te = corrupted[b]["timestamp"].dt.dayofweek * 24 + corrupted[b]["timestamp"].dt.hour
            med = train[b].groupby(how_tr)["load"].median()
            resid_how[b] = corrupted[b]["load"].to_numpy() - med.reindex(how_te).to_numpy()
            # TOWT estimator residual with real weather
            resid_towt[b] = howtowt_fit_predict(train[b], corrupted[b])

        # ---- baselines (round-1 tuned configs)
        tr_pts = np.concatenate(tr_arr, axis=0)
        ae_m = DenseAEAD_tunable(seed=0, hidden=(32, 16, 32)); ae_m.fit(tr_pts)
        nu, gam = BASE_CFG[seed]["ocsvm"]
        svm_m = OCSVMAD_tunable(seed=0, nu=nu, gamma=gam); svm_m.fit(tr_pts)

        # ---- detectors + flags (each model's OWN flags)
        fl_pt, fl_how, fl_towt, fl_ae, fl_howc = {}, {}, {}, {}, {}
        zresid_towt = {}
        for b in corrupted:
            fl_pt[b] = causal_flags(fuse(w_best, tzs[b], tzl[b]))
            fl_howc[b] = causal_flags(fuse(w_best, tzs_how[b], tzl[b]))
            # HOW-profile detector: z with train MAD per hour-of-week
            how_tr = train[b]["timestamp"].dt.dayofweek * 24 + train[b]["timestamp"].dt.hour
            how_te = corrupted[b]["timestamp"].dt.dayofweek * 24 + corrupted[b]["timestamp"].dt.hour
            mad = train[b].groupby(how_tr)["load"].apply(
                lambda x: np.median(np.abs(x - np.median(x))) * 1.4826)
            med = train[b].groupby(how_tr)["load"].median()
            z_how = (corrupted[b]["load"].to_numpy() - med.reindex(how_te).to_numpy()) \
                / (mad.reindex(how_te).to_numpy() + 1e-9)
            fl_how[b] = causal_flags(z_how)
            # TOWT detector: z of residual with MAD from train/val-fit split
            a_tr = int(len(train[b]) * 0.7)
            rt_tv = howtowt_fit_predict(train[b].iloc[:a_tr], train[b].iloc[a_tr:])
            sd_tv = np.median(np.abs(rt_tv - np.median(rt_tv))) * 1.4826
            zresid_towt[b] = resid_towt[b] / (sd_tv + 1e-9)
            fl_towt[b] = causal_flags(zresid_towt[b])
            Xt = scalers[b].transform(corrupted[b])
            from scripts.run_revision import causal_flags as _cf
            fl_ae[b] = _cf(ae_m.score(Xt))

        DET = [("PatchTST-SSL", fl_pt), ("HOW-profile+z", fl_how),
               ("TOWT-detector", fl_towt), ("Autoencoder(t)", fl_ae),
               ("PatchTST-SSL(HoW-cond)", fl_howc)]

        for name, fld in DET:
            for b, flags in fld.items():
                ev = events[events["building_id"] == b]
                m = evaluate_building(flags.astype(float), flags, ev, len(flags))
                rP, rR = range_based(flags.astype(bool), ev, len(flags))
                m.update(range_precision=rP, range_recall=rR, model=name,
                         building_id=b, seed=seed,
                         btype=dataset[b]["btype"].iloc[0], seen=(b in train_bids))
                det_rows.append(m)
                from src.evaluate import per_type_recall
                for etype, d_ in per_type_recall(flags, ev).items():
                    howc_rows.append({"model": name, "type": etype, "seed": seed,
                                      "building_id": b, **d_})

        # ---- P-A waste
        EST = [("PatchTST-SSL", resid_raw), ("HOW-profile", resid_how),
               ("TOWT", resid_towt), ("Autoencoder", None)]
        for name, rd in EST:
            for b in corrupted:
                ev = events[events["building_id"] == b]
                if name == "Autoencoder":
                    Xt = scalers[b].transform(corrupted[b])
                    resid = ae_m.point_residuals(Xt) * scalers[b].sd_["load"]
                else:
                    resid = rd[b]
                sig = sigma_hat_nonevent(resid, ev, len(resid))
                true, est = oracle_waste(resid, ev, sig)
                tot = true.sum()
                for variant, e in est.items():
                    orc_rows.append({
                        "estimator": name, "variant": variant, "seed": seed,
                        "building_id": b,
                        "bias_pct": (e.sum() - tot) / tot,
                        "abs_err_pct": abs(e.sum() - tot) / tot,
                        "per_event_med_err": np.median(np.abs(e - true) / true),
                    })
        # operational (own flags) — detector/estimator pairs
        PAIRS = [("PatchTST-SSL", resid_raw, fl_pt),
                 ("HOW-profile", resid_how, fl_how),
                 ("TOWT", resid_towt, fl_towt)]
        for name, rd, fld in PAIRS:
            for b in corrupted:
                ev = events[events["building_id"] == b]
                w = ops_waste(rd[b], fld[b], ev)
                w.update(model=name, building_id=b, seed=seed)
                ops_rows.append(w)

        # ---- P-D: transductive variant (full-series stats + centered flagger)
        for b in corrupted:
            d = np.zeros(len(dataset[b]))
            d[n_tv[b]:] = (corrupted[b]["load"].to_numpy()
                           - dataset[b]["load"].to_numpy()[n_tv[b]:]) \
                / scalers[b].sd_["load"]
            rp = r_pt[b] + d
            # full-series hour-conditional stats (transductive leak)
            from src.models.patchtst import _hour_cond_z, _robust_z
            z_td = _hour_cond_z(rp, full_arr[b])
            zl_td = _robust_z((r_pl[b] + d) ** 2)
            sc = fuse(w_best, z_td, zl_td)[n_tv[b]:]
            flt = robust_z_flags(sc, window=FLAG_WINDOW)
            ev = events[events["building_id"] == b]
            m = evaluate_building(sc, flt, ev, len(sc))
            m.update(model="PatchTST-transductive", building_id=b, seed=seed)
            trans_rows.append(m)

        # ---- R6: meaningful IF tuning (contamination fixed 0.01)
        from sklearn.ensemble import IsolationForest
        for ne in (100, 200, 400):
            for ms in ("auto", 256, 0.5):
                clf = IsolationForest(n_estimators=ne, max_samples=ms,
                                      contamination=0.01, random_state=0)
                clf.fit(tr_pts)
                f1s = []
                for b in corrupted:
                    Xt = scalers[b].transform(corrupted[b])
                    sc = -clf.score_samples(Xt)
                    fl = causal_flags(sc)
                    ev = events[events["building_id"] == b]
                    m = evaluate_building(sc, fl, ev, len(sc))
                    f1s.append(m.get("f1", np.nan))
                if_rows.append({"seed": seed, "n_estimators": ne,
                                "max_samples": str(ms),
                                "test_f1": float(np.nanmean(f1s))})

        print(f"[seed {seed}] done ({time.time() - ts_:.0f}s)")

    # ------------------------------------------------------------ report
    det = pd.DataFrame(det_rows)
    det.to_csv(os.path.join(RESULTS_DIR, "metrics_round2.csv"), index=False)
    orc = pd.DataFrame(orc_rows)
    orc.to_csv(os.path.join(RESULTS_DIR, "metrics_round2_waste_oracle.csv"), index=False)
    ops = pd.DataFrame(ops_rows)
    ops.to_csv(os.path.join(RESULTS_DIR, "metrics_round2_waste_ops.csv"), index=False)
    ty = pd.DataFrame(howc_rows)
    ty.to_csv(os.path.join(RESULTS_DIR, "metrics_round2_types.csv"), index=False)
    pd.DataFrame(trans_rows).to_csv(
        os.path.join(RESULTS_DIR, "metrics_round2_transductive.csv"), index=False)
    pd.DataFrame(if_rows).to_csv(
        os.path.join(RESULTS_DIR, "metrics_round2_iftuning.csv"), index=False)

    ps = det.groupby(["model", "seed"])[
        ["precision", "recall", "f1", "range_precision", "range_recall",
         "false_alarm_rate"]].mean().reset_index()

    def ms(x):
        return f"{np.nanmean(x):.3f} ± {np.nanstd(x, ddof=1):.3f}"

    L = []
    A = L.append
    A("# Round-2 Review Experiments (2026-10-01)")
    A("")
    A(f"Data: 41 buildings, measured weather, causal protocol, seeds {SEEDS}; "
      "fusion w per seed as Round-1 calibration.")
    A("")
    A("## P-A / R3 — Waste redesign")
    A("")
    A("### (1) Stage-a: ORACLE event-window waste (pure quantification skill)")
    A("Denominator `rel.` = total TRUE injected excess kWh per building. "
      "bias = (Σest − Σtrue)/Σtrue.")
    A("")
    A("| Estimator | clip bias | clip |err| | signed bias | floor-corrected bias | per-event med |err| (clip) |")
    A("|---|---|---|---|---|---|")
    for name, g in orc.groupby("estimator"):
        row = {v: gg for v, gg in g.groupby("variant")}
        A(f"| {name} | {row['clip']['bias_pct'].mean():+.1%} | "
          f"{row['clip']['abs_err_pct'].mean():.1%} | "
          f"{row['signed']['bias_pct'].mean():+.1%} | "
          f"{row['floor']['bias_pct'].mean():+.1%} | "
          f"{row['clip']['per_event_med_err'].mean():.1%} |")
    A("")
    A("### (2) Stage-b: OPERATIONAL waste on each detector's OWN flags "
      "(kWh additive decomposition)")
    A("delta = matched_err + false_add − missed (kWh), exact identity.")
    A("")
    A("| Detector | Σest kWh | Σtrue kWh | matched-err kWh | false-add kWh | flagged-in-missed kWh | missed kWh | events matched |")
    A("|---|---|---|---|---|---|---|---|")
    for name, g in ops.groupby("model"):
        A(f"| {name} | {g['est_kwh'].sum():.0f} | {g['true_kwh'].sum():.0f} | "
          f"{g['matched_err_kwh'].sum():+.0f} | {g['false_add_kwh'].sum():.0f} | "
          f"{g['unmatched_ev_flagged_kwh'].sum():.0f} | "
          f"{g['missed_kwh'].sum():.0f} | {g['frac_events_matched'].mean():.1%} |")
    for name, g in ops.groupby("model"):
        t = g["true_kwh"].sum()
        A(f"- {name}: total signed bias {(g['est_kwh'].sum() - t) / t:+.1%} = "
          f"matched {g['matched_err_kwh'].sum() / t:+.1%} + "
          f"false-add {g['false_add_kwh'].sum() / t:+.1%} + "
          f"flagged-in-missed {g['unmatched_ev_flagged_kwh'].sum() / t:+.1%} + "
          f"missed {-g['missed_kwh'].sum() / t:+.1%} (of true kWh; sums exactly)")
    A("")
    A("### (5) Inference masking (verified in code, src/models/patchtst.py)")
    A("- `point_residuals`: stride-1 168h windows; the ENTIRE load channel of "
      "each window (context + target hour) is zeroed before encoding — the "
      "counterfactual never sees the corrupted load, so an injected drift "
      "cannot be absorbed into the context; calendar/weather context IS "
      "included. Long scale likewise masks the full window load channel.")
    A("")
    A("## P-B / R1 — identity statistics")
    A("")
    bl = det.pivot_table(index=["seed", "building_id"], columns="model",
                         values="f1").groupby(level=1).mean()
    for other in ["HOW-profile+z", "Autoencoder(t)", "TOWT-detector"]:
        d = (bl["PatchTST-SSL"] - bl[other]).dropna()
        t, p = stats.ttest_rel(bl["PatchTST-SSL"].dropna(),
                               bl[other].dropna()) if len(d) else (np.nan, np.nan)
        dd = d.values
        t2, p2 = stats.ttest_rel(dd, np.zeros(len(dd)))
        A(f"- PatchTST vs {other}: paired diff {dd.mean():+.3f} "
          f"(n={len(dd)}), t={t2:.2f}, p={p2:.2e}, "
          f"improved {(dd > 0).sum()}/{len(dd)}")
    tt = tost((bl["PatchTST-SSL"] - bl["HOW-profile+z"]).dropna().values, 0.03)
    A(f"- **TOST equivalence (margin ±0.03 F1) PatchTST vs HOW-profile+z**: "
      f"mean diff {tt['mean']:+.4f}, 90% CI [{tt['ci90'][0]:+.4f}, "
      f"{tt['ci90'][1]:+.4f}], p_TOST={tt['p_tost']:.4f} → "
      f"{'EQUIVALENT within ±0.03' if tt['p_tost'] < 0.05 else 'NOT equivalent (CI exceeds margin)'}")
    dl = (bl["HOW-profile+z"] - bl["Autoencoder(t)"]).dropna().values
    A(f"- HOW-profile+z vs LSTM-AE(t) [LSTM-AE F1 from round-1 metrics]: see "
      f"metrics_revision.csv; here vs Autoencoder(t): {dl.mean():+.3f}")
    A("")
    A("## P-C / R2 — hour-of-week conditional short scale")
    A("")
    A("| Variant | Precision | Recall | F1 | schedule recall | FAR |")
    A("|---|---|---|---|---|---|")
    for m in ["PatchTST-SSL", "PatchTST-SSL(HoW-cond)"]:
        g = ps[ps["model"] == m]
        sq = ty[(ty["model"] == m) & (ty["type"] == "schedule")].groupby("seed")["recall"].mean()
        A(f"| {m} | {ms(g['precision'])} | {ms(g['recall'])} | {ms(g['f1'])} | "
          f"{ms(sq)} | {ms(g['false_alarm_rate'] * 100)}% |")
    for et in ["spike", "drift", "schedule"]:
        cells = []
        for m in ["PatchTST-SSL", "PatchTST-SSL(HoW-cond)"]:
            q = ty[(ty["model"] == m) & (ty["type"] == et)].groupby("seed")["recall"].mean()
            cells.append(ms(q))
        A(f"- recall {et}: HoD-cond {cells[0]} → HoW-cond {cells[1]}")
    A("")
    A("## P-D / R7 — leakage under identical conditions")
    A("")
    tr_ = pd.DataFrame(trans_rows)
    caus = det[det["model"] == "PatchTST-SSL"].groupby(["seed", "building_id"])["f1"].mean()
    trd = tr_.groupby(["seed", "building_id"])["f1"].mean()
    cmp_ = pd.concat([caus.rename("causal"), trd.rename("transductive")], axis=1).dropna()
    A(f"- same 41 buildings, same seeds/injections, normalization only: "
      f"causal {cmp_['causal'].mean():.3f} vs transductive "
      f"{cmp_['transductive'].mean():.3f} → leakage ΔF1 = "
      f"{cmp_['transductive'].mean() - cmp_['causal'].mean():+.3f}")
    A("")
    A("## P-E — R4 range-based metrics / R6 IF tuning")
    A("")
    A("| Model | P | R | F1 | range-P | range-R |")
    A("|---|---|---|---|---|---|")
    for m in ps["model"].unique():
        g = ps[ps["model"] == m]
        A(f"| {m} | {ms(g['precision'])} | {ms(g['recall'])} | {ms(g['f1'])} | "
          f"{ms(g['range_precision'])} | {ms(g['range_recall'])} |")
    iff = pd.DataFrame(if_rows)
    A("- IF tuning grid (n_estimators × max_samples, contamination fixed 0.01): "
      + "; ".join(f"{r.n_estimators}/{r.max_samples}={r.test_f1:.3f}"
                  for r in iff.itertuples()) if len(iff) else "")
    A("- IF/OC-SVM input = per-timestamp 6-channel point features "
      "(load, temp, hour_sin/cos, dow_sin/cos), NOT lag windows (code: "
      "run_revision tr_pts / features.point_features).")
    A("")
    A(f"Total runtime: {time.time() - t0:.0f}s")
    with open(os.path.join(RESULTS_DIR, "round2_experiments.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
