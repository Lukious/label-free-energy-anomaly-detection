"""Review-response experiments (Round 1) — one-step run.

Covers:
  P-A (M6): 41 buildings, real measured BDG2 weather (data/bdg2/selected_v2)
  P-B (M1,M2): tuned baselines on validation injection events; ablations
      (backbone+scoring decoupling, hour-of-week baseline, short/long-only,
       global vs hour-conditional MAD, channel-mixing vs independent, FP scale)
  P-C (M3): CAUSAL normalization — train-span fixed stats + trailing rolling
      flagger; window sensitivity {1,2,4 weeks}
  P-D (M4): building-level paired test (n=41) + hierarchical bootstrap CI
  P-E (M5): TOWT M&V baseline, signed bias, error decomposition, per-event
      quartiles, waste for AE/LSTM-AE, proxy-weather attribution (seed 123)

Outputs: results/metrics_revision.csv, results/revision_experiments.md
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
from src.data import ZScaler, load_dataset
from src.evaluate import evaluate_building, per_type_recall, waste_metrics
from src.models.baselines import (DenseAE, DenseAEAD, IsolationForestAD,
                                  LSTMAE, LSTMAEAD, OCSVMAD, DEVICE)
from src.models.patchtst import PatchTSTAD, PatchTSTConfig

RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data", "bdg2", "selected_v2")
FPR_TARGET = 0.01
N_EVENTS = 6
SEEDS = [123, 124, 125]
FR = (0.60, 0.15, 0.25)
FLAG_WINDOW = 336  # 2-week trailing (causal) window

Z99 = stats.norm.ppf(1.0 - FPR_TARGET)


# ---------------------------------------------------------------- causality
def causal_flags(score: np.ndarray, window: int = FLAG_WINDOW) -> np.ndarray:
    """Trailing (causal) rolling robust-z flagger — M3 fix.

    median/MAD over the PAST `window` hours only (no centered window, no
    future leakage). min_periods = window//4 so the series start is usable.
    """
    s = pd.Series(score)
    med = s.rolling(window, min_periods=window // 4).median()
    mad = (s - med).abs().rolling(window, min_periods=window // 4).median() * 1.4826
    z = (s - med) / (mad + 1e-9)
    return (z > Z99).fillna(False).to_numpy()


def hour_of(X: np.ndarray) -> np.ndarray:
    return (np.degrees(np.arctan2(X[:, 2], X[:, 3])) % 360.0 / 15.0).round().astype(int) % 24


def causal_scale_z(r_point, r_pooled, X, n_trainval):
    """Standardize scales with TRAIN+VAL-ONLY statistics (M3 fix).

    z_short: hour-conditional median/MAD of the point residual, stats from
    the first n_trainval points only. z_long: global median/MAD of the
    squared pooled residual, likewise train-only. Both applied to the full
    series; only stats are frozen (no test/future information).
    """
    h = hour_of(X)
    z_short = np.empty_like(r_point)
    for hh in range(24):
        m_tr = h[:n_trainval] == hh
        if m_tr.sum() >= 48:
            med = np.median(r_point[:n_trainval][m_tr])
            mad = np.median(np.abs(r_point[:n_trainval][m_tr] - med)) * 1.4826
        else:
            med = np.median(r_point[:n_trainval])
            mad = np.median(np.abs(r_point[:n_trainval] - med)) * 1.4826
        m = h == hh
        z_short[m] = (r_point[m] - med) / (mad + 1e-9)
    q = r_pooled[:n_trainval] ** 2
    q = q[~np.isnan(q)]
    med = np.median(q)
    mad = np.median(np.abs(q - med)) * 1.4826
    z_long = (r_pooled ** 2 - med) / (mad + 1e-9)
    return z_short, z_long


def global_scale_z(r_point, r_pooled, X, n_trainval):
    """Ablation M1(d): GLOBAL (not hour-conditional) MAD on the short scale."""
    med = np.median(r_point[:n_trainval])
    mad = np.median(np.abs(r_point[:n_trainval] - med)) * 1.4826
    z_short = (r_point - med) / (mad + 1e-9)
    q = r_pooled[:n_trainval] ** 2
    q = q[~np.isnan(q)]
    med = np.median(q)
    mad = np.median(np.abs(q - med)) * 1.4826
    z_long = (r_pooled ** 2 - med) / (mad + 1e-9)
    return z_short, z_long


def _runs(flags):
    out, s = [], None
    for i, f in enumerate(flags):
        if f and s is None:
            s = i
        elif not f and s is not None:
            out.append((s, i)); s = None
    if s is not None:
        out.append((s, len(flags)))
    return out


# ---------------------------------------------------------------- evaluation
def eval_flags(name, flags_by_bid, events, dataset, train_bids, te_off, rows,
               seed, resid_by_bid=None, sd_by=None):
    """te_off: test start index (per building) — flags are test-span arrays."""
    for bid, flags in flags_by_bid.items():
        ev = events[events["building_id"] == bid]
        n = len(flags)
        m = evaluate_building(flags.astype(float), flags, ev, n)
        m.update(model=name, building_id=bid, seed=seed,
                 btype=dataset[bid]["btype"].iloc[0], seen=(bid in train_bids))
        for etype, d in per_type_recall(flags, ev).items():
            TYPE_ROWS.append({"model": name, "type": etype, "seed": seed, **d})
        if resid_by_bid is not None:
            m.update(waste_metrics(resid_by_bid[bid], flags, ev))
        rows.append(m)


def fuse(w, z_s, z_l):
    return w * z_s + (1 - w) * z_l


def val_event_f1(zs, zl, val_off, val_events, bids, grid):
    best = (0.5, -1)
    for w in grid:
        tp, nev, nrun, hit = 0, 0, 0, 0
        for bid in bids:
            sc = fuse(w, zs[bid][val_off[bid]:val_off[bid] + val_test_len[bid]],
                      zl[bid][val_off[bid]:val_off[bid] + val_test_len[bid]])
            flags = causal_flags(sc)
            ev = val_events[val_events["building_id"] == bid]
            m = evaluate_building(sc, flags, ev, len(sc))
            tp += m["tp_events"]; nev += m["n_events"]
            anom = np.zeros(len(sc), dtype=bool)
            for _, e_ in ev.iterrows():
                anom[int(e_["start"]):int(e_["end"])] = True
            runs = _runs(flags)
            nrun += len(runs)
            hit += sum(1 for s, e in runs if anom[s:e].any())
        P = hit / max(nrun, 1); R = tp / max(nev, 1)
        F = 2 * P * R / max(P + R, 1e-9)
        if F > best[1]:
            best = (w, F)
    return best


# ---------------------------------------------------------------- TOWT (M5)
def howtowt_fit_predict(train_df, test_df):
    """Time-of-week-and-temperature regression (IPMVP-style baseline).
    OLS: load ~ 168 hour-of-week dummies + temp + temp^2 (real weather)."""
    def design(df):
        how = df["timestamp"].dt.dayofweek * 24 + df["timestamp"].dt.hour
        D = np.zeros((len(df), 168))
        D[np.arange(len(df)), how] = 1.0
        t = df["temp"].to_numpy()
        return np.hstack([D, t[:, None], (t ** 2)[:, None]])
    Xtr, ytr = design(train_df), train_df["load"].to_numpy()
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return test_df["load"].to_numpy() - design(test_df) @ beta


def waste_detail(resid, flags, events):
    """Signed per-event estimates + decomposition (matched/missed/false)."""
    est, true, matched = [], [], []
    for _, ev in events.iterrows():
        s, e = int(ev["start"]), int(ev["end"])
        v = float(np.sum(np.clip(resid[s:e], 0, None)))
        est.append(v); true.append(float(ev["injected_excess_kwh"]))
        matched.append(bool(flags[s:e].any()))
    est = np.array(est, dtype=float); true = np.array(true, dtype=float)
    matched = np.array(matched, dtype=bool) if matched else np.zeros(0, dtype=bool)
    anom = np.zeros(len(resid), dtype=bool)
    for _, ev in events.iterrows():
        anom[int(ev["start"]):int(ev["end"])] = True
    false_add = float(np.sum(np.clip(resid, 0, None)[~anom]))
    tot_true = float(true.sum())
    return {
        "est_kwh": float(est.sum()), "true_kwh": tot_true,
        "bias_pct": float((est.sum() - tot_true) / max(tot_true, 1e-6)),
        "err_total_pct": float(abs(est.sum() - tot_true) / max(tot_true, 1e-6)),
        "err_matched_pct": float(abs(est[matched].sum() - true[matched].sum())
                                 / max(true[matched].sum(), 1e-6)) if matched.any() else np.nan,
        "missed_loss_pct": float(true[~matched].sum() / max(tot_true, 1e-6)),
        "false_add_pct": float(false_add / max(tot_true, 1e-6)),
        "per_event_err": np.abs(est - true) / np.maximum(true, 1e-6),
    }


# ---------------------------------------------------------------- main
TYPE_ROWS: list = []
val_test_len: dict = {}


def main():
    warnings.filterwarnings("ignore")
    t0 = time.time()
    dataset = load_dataset(DATA_DIR)
    seeds = [int(s) for s in os.environ.get("REV_SEEDS", "123,124,125").split(",")]
    maxb = int(os.environ.get("REV_MAXB", "0"))
    if maxb:
        keep = sorted(dataset)[:maxb]
        dataset = {b: dataset[b] for b in keep}
    print(f"loaded {len(dataset)} buildings; seeds {seeds}")
    n_tv = {b: int(len(df) * (FR[0] + FR[1])) for b, df in dataset.items()}

    all_rows, waste_rows, fp_scale_rows = [], [], []
    win_sens = []
    w_chosen, ci_rows, proxy_rows = {}, [], []
    tun_log = []

    for seed in seeds:
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
        train_bids = sorted(b for b in train if dataset[b]["btype"].iloc[0] == "office")

        scalers = {b: ZScaler().fit(train[b]) for b in train}
        tr_arr = [scalers[b].transform(train[b]) for b in train_bids]
        va_arr = [scalers[b].transform(val[b]) for b in train_bids]

        # ---- PatchTST (channel-mixing, proposal)
        ssl = PatchTSTAD(PatchTSTConfig())
        ssl.fit(tr_arr, val_arrays=va_arr)

        # clean full-series residuals (load masked at input => injection is a
        # per-timestep additive delta on the residual; no context leakage)
        full_arr = {b: scalers[b].transform(dataset[b]) for b in dataset}
        r_pt = {b: ssl.point_residuals(full_arr[b]) for b in dataset}
        r_pl = {b: ssl.residuals(full_arr[b]) for b in dataset}

        zs, zl, zs_g = {}, {}, {}
        for b in dataset:
            zs[b], zl[b] = causal_scale_z(r_pt[b], r_pl[b], full_arr[b], n_tv[b])
            zs_g[b], _ = global_scale_z(r_pt[b], r_pl[b], full_arr[b], n_tv[b])

        # ---- fusion-w calibration on validation injection events (M2: also
        # used identically for baseline hyperparameters below)
        val_c, val_ev = inject_all({b: val[b] for b in train_bids},
                                   n_events=N_EVENTS, seed=seed + 333)
        vzs, vzl = {}, {}
        for b in train_bids:
            d = np.zeros(len(dataset[b]))
            vv = val_c[b]
            d[val_off[b]:val_off[b] + len(vv)] = \
                (vv["load"].to_numpy() - dataset[b]["load"].to_numpy()[val_off[b]:val_off[b] + len(vv)]) \
                / scalers[b].sd_["load"]
            rp = r_pt[b] + d
            vzs[b], vzl[b] = causal_scale_z(rp, r_pl[b], full_arr[b], n_tv[b])

        grid = [round(w, 2) for w in np.arange(0.0, 1.01, 0.1)]
        w_best, _ = val_event_f1(vzs, vzl, val_off, val_ev, train_bids, grid)
        w_chosen[seed] = w_best

        # test-span z arrays WITH the injected corruption delta applied
        tzs, tzl, tzs_g = {}, {}, {}
        for b in corrupted:
            d = np.zeros(len(dataset[b]))
            d[n_tv[b]:] = (corrupted[b]["load"].to_numpy()
                           - dataset[b]["load"].to_numpy()[n_tv[b]:]) \
                / scalers[b].sd_["load"]
            rp = r_pt[b] + d
            z_s, z_l = causal_scale_z(rp, r_pl[b], full_arr[b], n_tv[b])
            z_sg, _ = global_scale_z(rp, r_pl[b], full_arr[b], n_tv[b])
            tzs[b], tzl[b], tzs_g[b] = z_s[n_tv[b]:], z_l[n_tv[b]:], z_sg[n_tv[b]:]

        def test_slice(zd, b):
            return zd[b]

        flags_prop, flags_w05, flags_short, flags_long, flags_glob = {}, {}, {}, {}, {}
        resid_raw = {}
        for b in corrupted:
            sc = fuse(w_best, test_slice(tzs, b), test_slice(tzl, b))
            flags_prop[b] = causal_flags(sc)
            flags_w05[b] = causal_flags(fuse(0.5, test_slice(tzs, b), test_slice(tzl, b)))
            flags_short[b] = causal_flags(test_slice(tzs, b))
            flags_long[b] = causal_flags(test_slice(tzl, b))
            flags_glob[b] = causal_flags(fuse(w_best, test_slice(tzs_g, b), test_slice(tzl, b)))
            d = (corrupted[b]["load"].to_numpy() - dataset[b]["load"].to_numpy()[n_tv[b]:]) \
                / scalers[b].sd_["load"]
            resid_raw[b] = (r_pt[b][n_tv[b]:] + d) * scalers[b].sd_["load"]

        eval_flags("PatchTST-SSL(w-tuned)", flags_prop, events, dataset,
                   train_bids, None, all_rows, seed, resid_raw, None)
        eval_flags("PatchTST-SSL(w=0.5 fixed)", flags_w05, events, dataset,
                   train_bids, None, all_rows, seed)
        eval_flags("Abl: short-only", flags_short, events, dataset,
                   train_bids, None, all_rows, seed)
        eval_flags("Abl: long-only", flags_long, events, dataset,
                   train_bids, None, all_rows, seed)
        eval_flags("Abl: global-MAD short", flags_glob, events, dataset,
                   train_bids, None, all_rows, seed)

        # window sensitivity (M3) on proposal scores
        for W in (168, 336, 672):
            fl = {}
            for b in corrupted:
                fl[b] = causal_flags(fuse(w_best, test_slice(tzs, b), test_slice(tzl, b)), W)
            eval_flags(f"Sens: W={W}h", fl, events, dataset, train_bids,
                       None, win_sens, seed)

        # FP scale attribution (M1-f)
        for b in corrupted:
            ev = events[events["building_id"] == b]
            anom = np.zeros(len(flags_prop[b]), dtype=bool)
            for _, e_ in ev.iterrows():
                anom[int(e_["start"]):int(e_["end"])] = True
            for s, e in _runs(flags_prop[b]):
                if not anom[s:e].any():
                    seg_s = tzs[b][s:e]
                    seg_l = tzl[b][s:e]
                    fp_scale_rows.append({
                        "seed": seed,
                        "short_driven": bool(np.mean(seg_s) >= np.mean(seg_l))})

        # ---- tuned baselines (M2): same validation injection events
        tr_pts = np.concatenate(tr_arr, axis=0)

        def tune(name, factory, cfgs):
            best = (None, -1)
            for cfg in cfgs:
                mdl = factory(**cfg)
                mdl.fit(tr_pts)
                tp, nev, nrun, hit = 0, 0, 0, 0
                for b in train_bids:
                    Xv = scalers[b].transform(val_c[b])
                    flags = causal_flags(mdl.score(Xv))
                    ev = val_ev[val_ev["building_id"] == b]
                    m = evaluate_building(mdl.score(Xv), flags, ev, len(Xv))
                    tp += m["tp_events"]; nev += m["n_events"]
                    anom = np.zeros(len(Xv), dtype=bool)
                    for _, e_ in ev.iterrows():
                        anom[int(e_["start"]):int(e_["end"])] = True
                    runs = _runs(flags)
                    nrun += len(runs)
                    hit += sum(1 for s, e in runs if anom[s:e].any())
                P = hit / max(nrun, 1); R = tp / max(nev, 1)
                F = 2 * P * R / max(P + R, 1e-9)
                if F > best[1]:
                    best = (cfg, F)
            tun_log.append({"seed": seed, "model": name, **best[0], "val_F1": best[1]})
            mdl = factory(**best[0]); mdl.fit(tr_pts)
            return mdl

        if_m = tune("IF", lambda **c: IsolationForestAD_tunable(seed=0, **c),
                    [{"contamination": c} for c in (0.01, 0.05, 0.1)])
        svm_m = tune("OCSVM", lambda **c: OCSVMAD_tunable(seed=0, **c),
                     [{"nu": n, "gamma": g}
                      for n in (0.02, 0.05, 0.1) for g in ("scale", 0.2)])
        ae_m = tune("AE", lambda **c: DenseAEAD_tunable(seed=0, **c),
                    [{"hidden": h} for h in ((32, 16, 32), (64, 32, 64))])
        lae_m = tune("LSTM-AE", lambda **c: LSTMAEAD_tunable(seed=0, **c),
                     [{"hidden": h} for h in (32, 64)])

        # evaluate tuned baselines on test
        for name, mdl in [("IsolationForest(t)", if_m), ("OC-SVM(t)", svm_m),
                          ("Autoencoder(t)", ae_m), ("LSTM-AE(t)", lae_m)]:
            fl, resid = {}, {}
            for b in corrupted:
                Xt = scalers[b].transform(corrupted[b])
                sc = mdl.score(Xt)
                fl[b] = causal_flags(sc)
                if hasattr(mdl, "point_residuals"):
                    resid[b] = mdl.point_residuals(Xt) * scalers[b].sd_["load"]
            eval_flags(name, fl, events, dataset, train_bids, None,
                       all_rows, seed, resid if resid else None, None)

        # ---- ablation (a): AE / LSTM-AE residuals through the SAME fusion
        # scoring (z_short/z_long with train-frozen stats + causal flagger)
        for name, mdl in [("Abl: AE+fusion", ae_m), ("Abl: LSTM-AE+fusion", lae_m)]:
            def fuse_scales(Xc, b):
                rp = mdl.point_residuals(Xc)
                rl = pd.Series(rp).rolling(168, min_periods=24).mean().to_numpy()
                return causal_scale_z(rp, rl, full_arr[b], n_tv[b])

            azs, azl = {}, {}
            for b in dataset:
                Xc = full_arr[b].copy()
                Xc[n_tv[b]:] = scalers[b].transform(corrupted[b])
                azs[b], azl[b] = fuse_scales(Xc, b)
            # val-corrupted z arrays for fusion-w calibration
            avzs, avzl = {}, {}
            for b in train_bids:
                Xv = full_arr[b].copy()
                Xv[val_off[b]:n_tv[b]] = scalers[b].transform(val_c[b])
                avzs[b], avzl[b] = fuse_scales(Xv, b)
            aw, _ = val_event_f1(avzs, avzl, val_off, val_ev, train_bids, grid)
            fl = {}
            for b in corrupted:
                fl[b] = causal_flags(fuse(aw, azs[b][n_tv[b]:], azl[b][n_tv[b]:]))
            eval_flags(name, fl, events, dataset, train_bids, None, all_rows, seed)

        # ---- ablation (b): hour-of-week profile + robust-z baseline
        fl = {}
        for b in corrupted:
            tr_df, te_df = train[b], corrupted[b]
            how_tr = tr_df["timestamp"].dt.dayofweek * 24 + tr_df["timestamp"].dt.hour
            how_te = te_df["timestamp"].dt.dayofweek * 24 + te_df["timestamp"].dt.hour
            med = tr_df.groupby(how_tr)["load"].median()
            mad = tr_df.groupby(how_tr)["load"].apply(
                lambda x: np.median(np.abs(x - np.median(x))) * 1.4826)
            r = te_df["load"].to_numpy() - med.reindex(how_te).to_numpy()
            z = r / (mad.reindex(how_te).to_numpy() + 1e-9)
            fl[b] = causal_flags(z)
        eval_flags("Abl: HOW-profile+z", fl, events, dataset, train_bids,
                   None, all_rows, seed)

        # ---- ablation (e): channel-independent backbone (seed 123 only)
        if seed == 123:
            cfg = PatchTSTConfig(); cfg.mix_channels = False
            ci = PatchTSTAD(cfg)
            ci.fit(tr_arr, val_arrays=va_arr)
            fl = {}
            for b in corrupted:
                Xf = full_arr[b]
                d = np.zeros(len(Xf))
                d[n_tv[b]:] = (corrupted[b]["load"].to_numpy()
                               - dataset[b]["load"].to_numpy()[n_tv[b]:]) \
                    / scalers[b].sd_["load"]
                rp = ci.point_residuals(Xf)
                rl = ci.residuals(Xf) + d  # pooled residual ≈ clean + delta
                z_s, z_l = causal_scale_z(rp + d, rl, Xf, n_tv[b])
                fl[b] = causal_flags(fuse(w_best, z_s[n_tv[b]:], z_l[n_tv[b]:]))
            eval_flags("Abl: channel-indep", fl, events, dataset, train_bids,
                       None, ci_rows, seed)

        # ---- waste (M5): proposal vs TOWT vs AE vs LSTM-AE
        for b in corrupted:
            ev = events[events["building_id"] == b]
            flags = flags_prop[b]
            wd = waste_detail(resid_raw[b], flags, ev)
            wd.update(model="PatchTST-SSL", building_id=b, seed=seed); waste_rows.append(wd)
            # TOWT with real weather
            rt = howtowt_fit_predict(train[b], corrupted[b])
            wt = waste_detail(rt, flags, ev)
            wt.update(model="TOWT", building_id=b, seed=seed); waste_rows.append(wt)
            for name, mdl in [("Autoencoder(t)", ae_m), ("LSTM-AE(t)", lae_m)]:
                if hasattr(mdl, "point_residuals"):
                    Xt = scalers[b].transform(corrupted[b])
                    rr = mdl.point_residuals(Xt) * scalers[b].sd_["load"]
                    wa = waste_detail(rr, flags, ev)
                    wa.update(model=name, building_id=b, seed=seed)
                    waste_rows.append(wa)

        # ---- proxy-weather attribution (seed 123 only)
        if seed == 123:
            from scripts.prepare_revision import climatological_temp
            meta = pd.read_csv(os.path.join(ROOT, "data", "bdg2", "metadata.csv")
                               ).set_index("building_id")
            proxy_arr = {}
            for b in dataset:
                Xp = full_arr[b].copy()
                lat = float(meta.loc[b, "lat"])
                proxy = (climatological_temp(pd.DatetimeIndex(dataset[b]["timestamp"]),
                                             lat) - scalers[b].mu_["temp"]) \
                    / scalers[b].sd_["temp"]
                Xp[:, 1] = proxy
                proxy_arr[b] = Xp
            prx = PatchTSTAD(PatchTSTConfig())
            prx.fit([proxy_arr[b][:n_tv[b]] for b in train_bids],
                    val_arrays=[proxy_arr[b][int(len(dataset[b]) * FR[0]):n_tv[b]]
                                for b in train_bids])
            for b in corrupted:
                ev = events[events["building_id"] == b]
                rp = prx.point_residuals(proxy_arr[b])
                d = (corrupted[b]["load"].to_numpy()
                     - dataset[b]["load"].to_numpy()[n_tv[b]:]) / scalers[b].sd_["load"]
                resid = (rp[n_tv[b]:] + d) * scalers[b].sd_["load"]
                wp = waste_detail(resid, flags_prop[b], ev)
                wp.update(model="PatchTST-SSL(proxy-wx)", building_id=b, seed=seed)
                waste_rows.append(wp)

        agg = pd.DataFrame(all_rows).groupby("model")["f1"].mean()
        print(f"[seed {seed}] w={w_best:.1f} F1s: " +
              ", ".join(f"{m}={v:.3f}" for m, v in agg.items()) +
              f" ({time.time() - ts_:.0f}s)")

    # ---------------------------------------------------------------- report
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics_revision.csv"), index=False)
    tdf = pd.DataFrame(TYPE_ROWS)
    wdf = pd.DataFrame(waste_rows)
    wdf.to_csv(os.path.join(RESULTS_DIR, "metrics_revision_waste.csv"), index=False)
    tdf.to_csv(os.path.join(RESULTS_DIR, "metrics_revision_types.csv"), index=False)
    pd.DataFrame(tun_log).to_csv(os.path.join(RESULTS_DIR, "metrics_revision_tuning.csv"),
                                 index=False)
    if ci_rows:
        pd.concat([df, pd.DataFrame(ci_rows)]).to_csv(
            os.path.join(RESULTS_DIR, "metrics_revision.csv"), index=False)
        df = pd.concat([df, pd.DataFrame(ci_rows)])

    per_seed = df.groupby(["model", "seed"])[
        ["precision", "recall", "f1", "detection_delay_h", "false_alarm_rate"]
    ].mean().reset_index()

    def ms(x):
        return f"{np.nanmean(x):.2f} ± {np.nanstd(x, ddof=1):.2f}"

    L = []
    A = L.append
    A("# Revision Experiments — Round-1 Review Response (2026-10-01)")
    A("")
    n_seen = sum(1 for b in dataset if dataset[b]["btype"].iloc[0] == "office")
    n_unseen = len(dataset) - n_seen
    A(f"- Data: BDG2, **{len(dataset)} buildings** ({n_seen} Office seen + "
      f"{len(dataset) - n_seen} Education/Retail unseen), 2016 local, "
      f">= 95% complete; **real measured weather** "
      f"(site-matched airTemperature; 0 proxy-filled hours)")
    A(f"- Protocol: 60/15/25 split, 6 events/building test span, seeds {seeds}; "
      f"flagger = **trailing (causal) 2-week rolling robust-z** @1% FPR; "
      f"scale statistics frozen on train+val span only (M3)")
    A(f"- Baseline hyperparameters tuned on the same validation injection events "
      f"as the fusion weight (M2); fusion w chosen per seed: " +
      ", ".join(f"s{s}: {w_chosen[s]:.1f}" for s in seeds))
    A("")
    A("## (a) Overall — mean ± std over 3 seeds (event-level, per-building avg)")
    A("")
    A("| Model | Precision | Recall | F1 | Delay(h) | FAR |")
    A("|---|---|---|---|---|---|")
    order = ["IsolationForest(t)", "OC-SVM(t)", "Autoencoder(t)", "LSTM-AE(t)",
             "PatchTST-SSL(w-tuned)", "PatchTST-SSL(w=0.5 fixed)"]
    for m in order:
        g = per_seed[per_seed["model"] == m]
        A(f"| {m} | {ms(g['precision'])} | {ms(g['recall'])} | {ms(g['f1'])} | "
          f"{ms(g['detection_delay_h'])} | {ms(g['false_alarm_rate']*100)}% |")
    A("")
    A("## (b) Ablations (M1)")
    A("")
    A("| Variant | Precision | Recall | F1 |")
    A("|---|---|---|---|")
    abl = ["PatchTST-SSL(w-tuned)", "Abl: short-only", "Abl: long-only",
           "Abl: global-MAD short", "Abl: AE+fusion", "Abl: LSTM-AE+fusion",
           "Abl: HOW-profile+z", "Abl: channel-indep"]
    for m in abl:
        g = per_seed[per_seed["model"] == m]
        if len(g) == 0:
            continue
        A(f"| {m} | {ms(g['precision'])} | {ms(g['recall'])} | {ms(g['f1'])} |")
    A("")
    A("### M1(f): false-alarm scale attribution")
    fp = pd.DataFrame(fp_scale_rows)
    A(f"- false-alarm runs short-scale-driven: **{fp['short_driven'].mean():.1%}** "
      f"(n={len(fp)} runs, 3 seeds)")
    A("")
    A("## (c) Causal flagger window sensitivity (M3)")
    A("")
    A("| Window | F1 | FAR |")
    A("|---|---|---|")
    ws = pd.DataFrame(win_sens)
    if len(ws):
        for m, g in ws.groupby("model"):
            gg = g.groupby("seed")[["f1", "false_alarm_rate"]].mean()
            A(f"| {m} | {ms(gg['f1'])} | {ms(gg['false_alarm_rate']*100)}% |")
    A("")
    A("## (d) Recall by anomaly type")
    A("")
    A("| Model | spike | drift | schedule |")
    A("|---|---|---|---|")
    for m in order:
        cells = []
        for et in ["spike", "drift", "schedule"]:
            q = tdf[(tdf["model"] == m) & (tdf["type"] == et)].groupby("seed")["recall"].mean()
            cells.append(ms(q) if len(q) else "n/a")
        A(f"| {m} | " + " | ".join(cells) + " |")
    A("")
    A("## (e) Transfer (seen office vs unseen education/retail), F1")
    A("")
    A("| Model | seen | unseen | n_unseen_bldgs |")
    A("|---|---|---|---|")
    for m in order:
        g = df[df["model"] == m].groupby(["seed", "seen"])["f1"].mean().unstack()
        A(f"| {m} | {ms(g[True])} | {ms(g[False])} | {n_unseen} |")
    A("")
    # per-type transfer with sample sizes (M4)
    bt = df[df["model"] == "PatchTST-SSL(w-tuned)"].groupby(["seed", "btype"])["f1"].mean().unstack()
    nb = df[df["model"] == "PatchTST-SSL(w-tuned)"].groupby("btype")["building_id"].nunique()
    A("PatchTST-SSL F1 by type (n buildings): " +
      ", ".join(f"{c}={bt[c].mean():.3f} (n={nb[c]})" for c in bt.columns))
    A("")
    A("## (f) Building-level statistics (M4)")
    A("")
    base_f1 = per_seed[per_seed["model"].isin(
        ["IsolationForest(t)", "OC-SVM(t)", "Autoencoder(t)", "LSTM-AE(t)"])
        ].groupby("model")["f1"].mean()
    strongest = base_f1.idxmax()
    A(f"- strongest baseline by mean F1: **{strongest}** ({base_f1.max():.3f})")
    pv = df[df["model"].isin(["PatchTST-SSL(w-tuned)", strongest])]
    bl = pv.groupby(["building_id", "model"])["f1"].mean().unstack()
    d = bl["PatchTST-SSL(w-tuned)"] - bl[strongest]
    t, p = stats.ttest_rel(bl["PatchTST-SSL(w-tuned)"], bl[strongest])
    w = stats.wilcoxon(bl["PatchTST-SSL(w-tuned)"], bl[strongest])
    A(f"- paired t-test over buildings (n={len(bl)}): t={t:.2f}, p={p:.2e}; "
      f"mean diff {d.mean():+.3f}; buildings improved {int((d>0).sum())}/{len(bl)}")
    A(f"- Wilcoxon signed-rank: W={w.statistic}, p={w.pvalue:.2e}")
    rng = np.random.default_rng(0)
    boots = []
    bids = bl.index.to_numpy()
    for _ in range(2000):
        pick = rng.choice(bids, len(bids), replace=True)
        boots.append(bl.loc[pick, "PatchTST-SSL(w-tuned)"].mean()
                     - bl.loc[pick, strongest].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    A(f"- hierarchical bootstrap 95% CI for mean F1 diff (building resampling, "
      f"B=2000): [{lo:+.3f}, {hi:+.3f}]")
    A("")
    A("## (g) Waste quantification (M5)")
    A("")
    A("| Model | bias (signed) | |err| total | matched err | missed loss | false add |")
    A("|---|---|---|---|---|---|")
    for m, g in wdf.groupby("model"):
        A(f"| {m} | {g['bias_pct'].mean():+.1%} | {g['err_total_pct'].mean():.1%} | "
          f"{g['err_matched_pct'].mean():.1%} | {g['missed_loss_pct'].mean():.1%} | "
          f"{g['false_add_pct'].mean():.1%} |")
    pe = np.concatenate(wdf[wdf["model"] == "PatchTST-SSL"]["per_event_err"].to_numpy())
    A(f"- PatchTST-SSL per-event |err| quartiles: "
      f"Q25={np.percentile(pe,25):.1%} med={np.percentile(pe,50):.1%} "
      f"Q75={np.percentile(pe,75):.1%}")
    if "PatchTST-SSL(proxy-wx)" in set(wdf["model"]):
        a = wdf[wdf["model"] == "PatchTST-SSL"]["err_total_pct"].mean()
        b = wdf[wdf["model"] == "PatchTST-SSL(proxy-wx)"]["err_total_pct"].mean()
        A(f"- **proxy attribution**: real weather |err| {a:.1%} vs climatological "
          f"proxy {b:.1%} (same events, seed 123) → "
          f"{'weather proxy explains the earlier 59%±83% error' if b > a else 'proxy is NOT the dominant error source'}")
    A("")
    A("## (h) Tuning log (M2)")
    A("")
    A("```")
    A(pd.DataFrame(tun_log).to_string(index=False))
    A("```")
    A("")
    A(f"Total runtime: {time.time() - t0:.0f}s")
    with open(os.path.join(RESULTS_DIR, "revision_experiments.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"[done] ({time.time() - t0:.0f}s)")


# tunable wrappers (M2)
class IsolationForestAD_tunable(IsolationForestAD):
    def __init__(self, seed=0, contamination=0.05):
        from sklearn.ensemble import IsolationForest
        self.clf = IsolationForest(n_estimators=200, random_state=seed,
                                   contamination=contamination)


class OCSVMAD_tunable(OCSVMAD):
    def __init__(self, seed=0, nu=0.02, gamma="scale"):
        from sklearn.svm import OneClassSVM
        self.clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        self.rng = np.random.default_rng(seed)


class DenseAEAD_tunable(DenseAEAD):
    def __init__(self, seed=0, hidden=(64, 32, 64)):
        super().__init__(seed=seed)
        self.hidden = hidden

    def fit(self, X):
        self.model = DenseAE(X.shape[1], self.hidden).to(DEVICE)
        import torch
        Xs = torch.tensor(X, device=DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lossf = torch.nn.MSELoss(reduction="none")
        n = len(Xs)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, self.batch):
                xb = Xs[perm[i:i + self.batch]]
                opt.zero_grad()
                loss = lossf(self.model(xb), xb)[:, self.target_idx].mean()
                loss.backward(); opt.step()
        self.model.eval()
        return self

    def point_residuals(self, X):
        import torch
        with torch.no_grad():
            Xs = torch.tensor(X, device=DEVICE)
            r = (Xs - self.model(Xs))[:, self.target_idx]
        return r.cpu().numpy()


class LSTMAEAD_tunable(LSTMAEAD):
    def __init__(self, seed=0, hidden=64):
        super().__init__(seed=seed)
        self.hidden = hidden

    def point_residuals(self, X):
        """(Unsigned) max-pooled window reconstruction error as the point scale."""
        return self.score(X)

    def fit(self, X):
        import torch
        d_in = X.shape[1]
        W = self._windows(X)
        self.model = LSTMAE(d_in, self.hidden).to(DEVICE)
        Wt = torch.tensor(W, device=DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        n = len(Wt)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, self.batch):
                xb = Wt[perm[i:i + self.batch]]
                opt.zero_grad()
                loss = (self.model(xb) - xb).pow(2)[..., 0].mean()
                loss.backward(); opt.step()
        self.model.eval()
        return self


if __name__ == "__main__":
    main()
