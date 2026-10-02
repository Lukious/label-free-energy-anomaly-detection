"""CANONICAL RUN — Round-7 review response (supersedes run_final_consolidated).

Every number of manuscript v10 derives from results/final_r7/*.csv.

Changes vs. run_final_consolidated.py (see paper/reviews/review_round7.md):
  MC1/X1  true 2x2 factorial: backbone {PatchTST, HOW-profile} x scoring
          {HoW-conditional, global-MAD}; hour-of-day conditioning moved to a
          separate granularity ablation for BOTH backbones.
  MC2/X2  robust-z statistics are estimated on the PRE-TEST span
          (train + validation) of the UNINJECTED residual series and frozen;
          validation-injected series are only scored with those statistics.
  X9      long scale is computed causally for every backbone (trailing 168 h
          mean of the point residual, squared) — the former PatchTST pooled
          residual averaged forward-overlapping windows.
  MC7/X4  window-extension rule chosen on validation injections from
          {none, z6, z12, cusum} by the MEDIAN per-building |relative error|
          (cancellation-resistant), for PatchTST, HOW and TOWT alike.
  3.9     training seed coupled to evaluation seed (PatchTST, AE, LSTM-AE,
          IF, OC-SVM all use `seed`), so +- includes training variance.
  M2      all four generic baselines are grid-tuned IN THIS RUN on the same
          validation injection events (event-F1 through the same flagger).
  MC6     masking ablation: PatchTST trained with 40% random load-patch
          masking, inferred with full-window load masking (canonical model is
          trained AND inferred with full-window load masking).
  MC3/3.4 per-event detail CSV (type, severity, duration, detected, delay,
          range coverage, oracle clip estimate) for type x severity tables.

Usage:  python scripts/run_final_r7.py [real|synth|all]
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import gc

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.run_round2 import (  # noqa: E402
    SEEDS, how_of, ops_waste, oracle_waste, range_based, sigma_hat_nonevent,
)
from scripts.run_revision import (  # noqa: E402
    Z99, FLAG_WINDOW, FR, IsolationForestAD_tunable, LSTMAEAD_tunable,
    OCSVMAD_tunable, fuse, hour_of, howtowt_fit_predict,
    DenseAEAD_tunable, _runs,
)
from scripts.run_round3 import (  # noqa: E402
    how_backbone_residuals, point_residuals_targetmask, prescribe_windows,
)
from src.anomaly_inject import inject_all  # noqa: E402
from src.data import ZScaler, load_dataset, generate_synthetic_dataset  # noqa: E402
from src.evaluate import evaluate_building, robust_z_flags  # noqa: E402
from src.models.patchtst import PatchTSTAD, PatchTSTConfig, _robust_z  # noqa: E402

OUT = os.path.join(ROOT, "results", "final_r7")


def causal_flags(score: np.ndarray, window: int = FLAG_WINDOW) -> np.ndarray:
    """Strictly trailing flagger (v11): hour t is standardized by the median
    and MAD of the score over [t-window, t-1] -- the hour being judged never
    enters its own normalization -- and flagged if z > Phi^-1(0.99)."""
    s = pd.Series(score)
    past = s.shift(1)
    med = past.rolling(window, min_periods=window // 4).median()
    mad = (past - med).abs().rolling(window, min_periods=window // 4).median() * 1.4826
    z = (s - med) / (mad + 1e-9)
    return (z > Z99).fillna(False).to_numpy()
DATA_DIR = os.path.join(ROOT, "data", "bdg2", "selected_v3")
N_EVENTS = 6
W_GRID = [round(0.1 * i, 1) for i in range(11)]
EXT_MODES = ("none", "z6", "z12", "cusum")

FACTORIAL = ["PatchTST-SSL(HoW-cond)", "PatchTST-SSL(global-MAD)",
             "HOW-profile(HoW-cond)", "HOW-profile(global-MAD)"]
GRANULARITY = ["PatchTST-SSL(HoD-cond)", "HOW-profile(HoD-cond)"]
ABLATION = ["PatchTST-SSL(HoW-cond, patch-mask train)"]
OTHERS = ["TOWT(HoW-cond)", "HOW-profile+z(slot)", "Autoencoder(t)",
          "IsolationForest(t)", "OC-SVM(t)", "LSTM-AE(t)"]
MODELS = FACTORIAL + GRANULARITY + ABLATION + OTHERS
WASTE_MODELS = ["PatchTST-SSL(HoW-cond)", "HOW-profile(HoW-cond)",
                "TOWT(HoW-cond)", "PatchTST-SSL(HoW-cond, patch-mask train)"]
EXT_MODELS = ["PatchTST-SSL(HoW-cond)", "HOW-profile(HoW-cond)",
              "TOWT(HoW-cond)"]

BASELINE_GRIDS = {
    # contamination only shifts IF scores by a constant (invisible to the
    # robust-z flagger) -> fixed at 0.01; tune what changes the ranking
    "IsolationForest(t)": [{"n_estimators": n, "max_samples": m}
                           for n in (100, 200, 400) for m in ("auto", 256, 0.5)],
    "OC-SVM(t)": [{"nu": n, "gamma": g} for n in (0.02, 0.05, 0.1)
                  for g in ("scale", 0.2)],
    "Autoencoder(t)": [{"hidden": h} for h in ((32, 16, 32), (64, 32, 64))],
    "LSTM-AE(t)": [{"hidden": h} for h in (32, 64)],
}


# ------------------------------------------------- pre-test frozen scoring
def cond_z_ref(r_apply, r_ref, X, n_tv, n_groups, group_fn, min_group=20):
    """Calendar-conditional robust z. Stats from r_ref[:n_tv] (uninjected,
    pre-test span), applied to r_apply. Pooled fallback if < min_group."""
    g = group_fn(X)
    ref = r_ref[:n_tv]
    gr = g[:n_tv]
    ok = ~np.isnan(ref)
    pm = np.median(ref[ok])
    pmad = np.median(np.abs(ref[ok] - pm)) * 1.4826
    z = np.empty_like(r_apply)
    for gg in range(n_groups):
        m = g == gg
        if not m.any():
            continue
        mt = (gr == gg) & ok
        if mt.sum() >= min_group:
            med = np.median(ref[mt])
            mad = np.median(np.abs(ref[mt] - med)) * 1.4826
        else:
            med, mad = pm, pmad
        z[m] = (r_apply[m] - med) / (mad + 1e-9)
    return z


def global_z_ref(r_apply, r_ref, n_tv):
    ref = r_ref[:n_tv]
    ref = ref[~np.isnan(ref)]
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) * 1.4826
    return (r_apply - med) / (mad + 1e-9)


def _trail_sq(r):
    return pd.Series(r).rolling(168, min_periods=24).mean().to_numpy() ** 2


def z_long_ref(r_apply, r_ref, n_tv):
    """Causal long scale: squared trailing-168 h mean residual, robust z with
    pre-test stats of the uninjected series."""
    q = _trail_sq(r_ref)[:n_tv]
    q = q[~np.isnan(q)]
    med = np.median(q)
    mad = np.median(np.abs(q - med)) * 1.4826
    return (_trail_sq(r_apply) - med) / (mad + 1e-9)


SCORINGS = {
    "HoW-cond": lambda ra, rr, X, n: cond_z_ref(ra, rr, X, n, 168, how_of),
    "HoD-cond": lambda ra, rr, X, n: cond_z_ref(ra, rr, X, n, 24, hour_of),
    "global-MAD": lambda ra, rr, X, n: global_z_ref(ra, rr, n),
}


def score_pair(r_apply, r_ref, X, n_tv, scoring):
    return SCORINGS[scoring](r_apply, r_ref, X, n_tv), z_long_ref(r_apply, r_ref, n_tv)


# ------------------------------------------------- waste / extension
def extension_decomp(resid, z, flags, events, mode):
    """Estimator over flagged hours ('none') or prescribed windows, with the
    exact kWh decomposition  delta = matched_err + false_add - missed."""
    true = np.array([ev["injected_excess_kwh"] for _, ev in events.iterrows()],
                    dtype=float)
    anom = np.zeros(len(resid), dtype=bool)
    matched = []
    for _, ev in events.iterrows():
        s, e = int(ev["start"]), int(ev["end"])
        anom[s:e] = True
        matched.append(bool(flags[s:e].any()))
    matched = np.array(matched, dtype=bool)
    rc = np.clip(np.nan_to_num(resid, nan=0.0), 0, None)
    if mode == "none":
        wmask = flags.astype(bool).copy()
    else:
        wmask = np.zeros(len(resid), dtype=bool)
        for s, e in prescribe_windows(resid, z, flags.astype(bool), mode):
            wmask[s:e] = True
    est = rc[wmask].sum()
    est_ev = rc[wmask & anom].sum()
    false_add = rc[wmask & ~anom].sum()
    missed = true[~matched].sum()
    tt = true.sum()
    m_err = est_ev - true[matched].sum()
    chk = abs((est - tt) - (m_err + false_add - missed))
    assert chk < 1e-6 * max(tt, 1.0), chk
    return {"mode": mode, "est_kwh": float(est), "true_kwh": float(tt),
            "bias": (est - tt) / tt if tt else np.nan,
            "matched_err_kwh": float(m_err), "false_add_kwh": float(false_add),
            "missed_kwh": float(missed),
            "window_cov": float(wmask[anom].mean()) if anom.any() else np.nan}


def select_extension(val_items):
    """val_items: list of (resid, z, flags, events) per validation building.
    Objective: median over buildings of |relative error| (no cancellation
    across buildings). Ties -> earlier mode in EXT_MODES."""
    obj = {}
    for mode in EXT_MODES:
        errs = [abs(extension_decomp(r, z, f, ev, mode)["bias"])
                for r, z, f, ev in val_items if len(ev)]
        obj[mode] = float(np.nanmedian(errs))
    return min(EXT_MODES, key=lambda m: obj[m]), obj


# ------------------------------------------------- tuning helpers
def flag_event_f1(score_by_b, events, bids):
    tp = nev = nrun = hit = 0
    for b in bids:
        sc = score_by_b[b]
        f = causal_flags(sc)
        ev = events[events["building_id"] == b]
        m = evaluate_building(sc, f, ev, len(sc))
        tp += m["tp_events"]; nev += m["n_events"]
        anom = np.zeros(len(sc), dtype=bool)
        for _, e_ in ev.iterrows():
            anom[int(e_["start"]):int(e_["end"])] = True
        runs = _runs(f)
        nrun += len(runs)
        hit += sum(1 for s, e in runs if anom[s:e].any())
    P = hit / max(nrun, 1); R = tp / max(nev, 1)
    return 2 * P * R / max(P + R, 1e-9)


def factory(name, seed, cfg):
    if name == "IsolationForest(t)":
        m = IsolationForestAD_tunable(seed=seed, contamination=0.01)
        m.clf.set_params(**cfg)
        return m
    if name == "OC-SVM(t)":
        return OCSVMAD_tunable(seed=seed, **cfg)
    if name == "Autoencoder(t)":
        return DenseAEAD_tunable(seed=seed, **cfg)
    return LSTMAEAD_tunable(seed=seed, **cfg)


def free_mps():
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _baselines_on_cpu():
    """Generic AE/LSTM-AE baselines are small; training them on CPU avoids an
    MPS graph-cache exhaustion observed after the PatchTST inference passes."""
    import scripts.run_revision as rv
    import src.models.baselines as bl
    bl.DEVICE = "cpu"
    rv.DEVICE = "cpu"


def tune_baselines(tr_pts, val_X, val_ev, bids, seed, log):
    import torch
    _baselines_on_cpu()
    out = {}
    for name, grid in BASELINE_GRIDS.items():
        best = (None, -1.0, None)
        for cfg in grid:
            torch.manual_seed(seed)
            mdl = factory(name, seed, cfg).fit(tr_pts)
            F = flag_event_f1({b: mdl.score(val_X[b]) for b in bids}, val_ev, bids)
            log.append({"seed": seed, "model": name, "cfg": str(cfg), "val_f1": F})
            if F > best[1]:
                best = (cfg, F, mdl)
            del mdl
            free_mps()
        out[name] = best[2]
        print(f"  tuned {name}: {best[0]} val-F1={best[1]:.3f}", flush=True)
    return out


def event_rows(name, b, seed, flags, ev, resid=None):
    rows = []
    rc = None if resid is None else np.clip(np.nan_to_num(resid, nan=0.0), 0, None)
    for _, e_ in ev.iterrows():
        s, e = int(e_["start"]), int(e_["end"])
        hit = np.flatnonzero(flags[s:e])
        r = {"model": name, "building_id": b, "seed": seed,
             "type": e_["type"], "severity": int(e_["severity"]),
             "severity_name": e_["severity_name"], "duration_h": e - s,
             "true_kwh": float(e_["injected_excess_kwh"]),
             "detected": bool(len(hit)),
             "delay_h": float(hit[0]) if len(hit) else np.nan,
             "range_cov": float(flags[s:e].mean())}
        if rc is not None:
            r["oracle_clip_kwh"] = float(rc[s:e].sum())
        rows.append(r)
    return rows


# ================================================================ REAL DATA
def run_real(seeds=SEEDS, out=OUT):
    import torch
    warnings.filterwarnings("ignore")
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    dataset = load_dataset(DATA_DIR)
    print(f"loaded {len(dataset)} buildings", flush=True)
    n_tv = {b: int(len(df) * (FR[0] + FR[1])) for b, df in dataset.items()}
    val_off = {b: int(len(df) * FR[0]) for b, df in dataset.items()}
    train_bids = sorted(b for b in dataset
                        if dataset[b]["btype"].iloc[0] == "office")

    det_rows, evt_rows, leak_rows, orc_rows, ops_rows = [], [], [], [], []
    ext_rows, lev_rows, tune_log, sel_rows = [], [], [], []

    for seed in seeds:
        ts_ = time.time()
        train, val, test = {}, {}, {}
        for bid, df in dataset.items():
            a, b = val_off[bid], n_tv[bid]
            train[bid] = df.iloc[:a].reset_index(drop=True)
            val[bid] = df.iloc[a:b].reset_index(drop=True)
            test[bid] = df.iloc[b:].reset_index(drop=True)
        corrupted, events = inject_all(test, n_events=N_EVENTS, seed=seed, ref=train)
        val_c, val_ev = inject_all({b: val[b] for b in train_bids},
                                   n_events=N_EVENTS, seed=seed + 333, ref=train)

        scalers = {b: ZScaler().fit(train[b]) for b in train}
        sd = {b: scalers[b].sd_["load"] for b in dataset}
        tr_arr = [scalers[b].transform(train[b]) for b in train_bids]
        va_arr = [scalers[b].transform(val[b]) for b in train_bids]
        full_arr = {b: scalers[b].transform(dataset[b]) for b in dataset}

        # injection deltas (standardized), test span and validation span
        d_te, d_va = {}, {}
        for b in dataset:
            d = np.zeros(len(dataset[b]))
            d[n_tv[b]:] = (corrupted[b]["load"].to_numpy()
                           - dataset[b]["load"].to_numpy()[n_tv[b]:]) / sd[b]
            d_te[b] = d
            dv = np.zeros(len(dataset[b]))
            if b in train_bids:
                vv = val_c[b]["load"].to_numpy()
                dv[val_off[b]:n_tv[b]] = (vv - dataset[b]["load"].to_numpy()[val_off[b]:n_tv[b]]) / sd[b]
            d_va[b] = dv

        # ---- backbones (clean residual series r0 over the full year)
        def fit_ssl(mask):
            cfg = PatchTSTConfig()
            cfg.seed = seed
            cfg.train_mask = mask
            m = PatchTSTAD(cfg)
            m.fit(tr_arr, val_arrays=va_arr)
            return m
        ssl = fit_ssl("full")
        ssl_pm = fit_ssl("patch")
        r0 = {"PatchTST": {b: ssl.point_residuals(full_arr[b]) for b in dataset},
              "PatchTST-pm": {b: ssl_pm.point_residuals(full_arr[b]) for b in dataset},
              "HOW": {b: how_backbone_residuals(train[b], dataset[b], sd[b]) for b in dataset},
              "TOWT": {b: howtowt_fit_predict(train[b], dataset[b]) / sd[b] for b in dataset}}
        r_tm = {b: point_residuals_targetmask(ssl, full_arr[b]) for b in dataset}
        del ssl, ssl_pm
        free_mps()
        print(f"[seed {seed}] backbones ready ({time.time() - ts_:.0f}s)", flush=True)

        # ---- fusion weight on validation injections (PatchTST, HoW scoring)
        vz = {b: score_pair(r0["PatchTST"][b] + d_va[b], r0["PatchTST"][b],
                            full_arr[b], n_tv[b], "HoW-cond") for b in train_bids}
        best = (None, -1.0)
        for w in W_GRID:
            F = flag_event_f1({b: fuse(w, *vz[b])[val_off[b]:n_tv[b]] for b in train_bids},
                              val_ev, train_bids)
            sel_rows.append({"seed": seed, "what": "fusion_w", "option": w, "val_obj": F})
            if F > best[1]:
                best = (w, F)
        w_best = best[0]
        print(f"[seed {seed}] w_best={w_best} (val F1 {best[1]:.3f})", flush=True)

        # ---- residual-backbone arms: (key, backbone, scoring)
        ARMS = {"PatchTST-SSL(HoW-cond)": ("PatchTST", "HoW-cond"),
                "PatchTST-SSL(global-MAD)": ("PatchTST", "global-MAD"),
                "HOW-profile(HoW-cond)": ("HOW", "HoW-cond"),
                "HOW-profile(global-MAD)": ("HOW", "global-MAD"),
                "PatchTST-SSL(HoD-cond)": ("PatchTST", "HoD-cond"),
                "HOW-profile(HoD-cond)": ("HOW", "HoD-cond"),
                "PatchTST-SSL(HoW-cond, patch-mask train)": ("PatchTST-pm", "HoW-cond"),
                "TOWT(HoW-cond)": ("TOWT", "HoW-cond")}
        fl, score_te, resid_te = {}, {}, {}
        for name, (bb, sc) in ARMS.items():
            for b in dataset:
                ra = r0[bb][b] + d_te[b]
                s = fuse(w_best, *score_pair(ra, r0[bb][b], full_arr[b], n_tv[b], sc))[n_tv[b]:]
                score_te[(name, b)] = s
                fl[(name, b)] = causal_flags(s)
                resid_te[(name, b)] = ra[n_tv[b]:] * sd[b]

        # ---- extension rule chosen on validation (cancellation-resistant)
        chosen = {}
        for name in EXT_MODELS:
            bb, sc = ARMS[name]
            items = []
            for b in train_bids:
                ra = r0[bb][b] + d_va[b]
                s = fuse(w_best, *score_pair(ra, r0[bb][b], full_arr[b], n_tv[b], sc))
                sv = s[val_off[b]:n_tv[b]]
                items.append((ra[val_off[b]:n_tv[b]] * sd[b], sv, causal_flags(sv),
                              val_ev[val_ev["building_id"] == b]))
            mode, obj = select_extension(items)
            chosen[name] = mode
            for m_, v_ in obj.items():
                sel_rows.append({"seed": seed, "what": f"extension:{name}",
                                 "option": m_, "val_obj": v_, "chosen": m_ == mode})
            print(f"[seed {seed}] {name}: val median|err| "
                  f"{ {k: round(v, 3) for k, v in obj.items()} } -> {mode}", flush=True)

        # ---- generic baselines, tuned here on the same validation events
        tr_pts = np.concatenate(tr_arr, axis=0)
        val_X = {b: scalers[b].transform(val_c[b]) for b in train_bids}
        tuned = tune_baselines(tr_pts, val_X, val_ev, train_bids, seed, tune_log)
        for b in dataset:
            Xt = scalers[b].transform(corrupted[b])
            for name, mdl in tuned.items():
                fl[(name, b)] = causal_flags(mdl.score(Xt))
            # slot-level HOW variant: slot z on raw load, own flagger
            how_te = corrupted[b]["timestamp"].dt.dayofweek * 24 + corrupted[b]["timestamp"].dt.hour
            how_tr = train[b]["timestamp"].dt.dayofweek * 24 + train[b]["timestamp"].dt.hour
            mad = train[b].groupby(how_tr)["load"].apply(
                lambda x: np.median(np.abs(x - np.median(x))) * 1.4826)
            med = train[b].groupby(how_tr)["load"].median()
            fl[("HOW-profile+z(slot)", b)] = causal_flags(
                (corrupted[b]["load"].to_numpy() - med.reindex(how_te).to_numpy())
                / (mad.reindex(how_te).to_numpy() + 1e-9))

            # leakage contrast: full-series stats + centered flagger, else equal
            ra = r0["PatchTST"][b] + d_te[b]
            Xf = full_arr[b]
            z_td = cond_z_ref(ra, ra, Xf, len(ra), 168, how_of)
            zl_td = _robust_z(_trail_sq(ra))
            sc_td = fuse(w_best, z_td, zl_td)[n_tv[b]:]
            flt = robust_z_flags(sc_td, window=FLAG_WINDOW)
            ev = events[events["building_id"] == b]
            m = evaluate_building(sc_td, flt, ev, len(sc_td))
            m.update(model="PatchTST-transductive", building_id=b, seed=seed,
                     btype=dataset[b]["btype"].iloc[0])
            leak_rows.append(m)

        # ---- detection metrics + per-event detail
        for name in MODELS:
            for b in dataset:
                flags = fl[(name, b)]
                ev = events[events["building_id"] == b]
                m = evaluate_building(flags.astype(float), flags, ev, len(flags))
                rP, rR = range_based(flags.astype(bool), ev, len(flags))
                m.update(range_precision=rP, range_recall=rR,
                         range_f1=2 * rP * rR / (rP + rR) if rP + rR else 0.0,
                         model=name, building_id=b, seed=seed,
                         btype=dataset[b]["btype"].iloc[0],
                         site=b.split("_")[0], seen=(b in train_bids))
                det_rows.append(m)
                evt_rows += event_rows(name, b, seed, flags, ev,
                                       resid_te.get((name, b)))

        # ---- waste: oracle, operational, extension, level bias
        for name in WASTE_MODELS:
            for b in dataset:
                ev = events[events["building_id"] == b]
                rd = resid_te[(name, b)]
                sig = sigma_hat_nonevent(rd, ev, len(rd))
                true, est = oracle_waste(rd, ev, sig)
                for i, (_, e_) in enumerate(ev.iterrows()):
                    orc_rows.append({"model": name, "building_id": b, "seed": seed,
                                     "site": b.split("_")[0], "type": e_["type"],
                                     "severity": int(e_["severity"]),
                                     "true_kwh": float(true[i]),
                                     "clip_kwh": float(est["clip"][i]),
                                     "signed_kwh": float(est["signed"][i]),
                                     "floor_kwh": float(est["floor"][i])})
                w_ = ops_waste(rd, fl[(name, b)], ev)
                w_.update(model=name, building_id=b, seed=seed, site=b.split("_")[0])
                ops_rows.append(w_)
                if name in EXT_MODELS:
                    for mode in EXT_MODES:
                        r = extension_decomp(rd, score_te[(name, b)], fl[(name, b)], ev, mode)
                        r.update(model=name, building_id=b, seed=seed,
                                 site=b.split("_")[0], chosen=(mode == chosen[name]))
                        ext_rows.append(r)
        for b in dataset:
            ev = events[events["building_id"] == b]
            for _, e_ in ev.iterrows():
                s, e = int(e_["start"]), int(e_["end"])
                lev_rows.append({
                    "seed": seed, "building_id": b, "type": e_["type"],
                    "true_kwh": float(e_["injected_excess_kwh"]),
                    "patchtst_fullmask_mean": float(np.nanmean(resid_te[("PatchTST-SSL(HoW-cond)", b)][s:e])) / sd[b],
                    "patchtst_targetmask_mean": float(np.nanmean(((r_tm[b] + d_te[b])[n_tv[b]:] * sd[b])[s:e])) / sd[b],
                    "patchtst_patchmasktrain_mean": float(np.nanmean(resid_te[("PatchTST-SSL(HoW-cond, patch-mask train)", b)][s:e])) / sd[b],
                    "how_mean": float(np.nanmean(resid_te[("HOW-profile(HoW-cond)", b)][s:e])) / sd[b],
                })
        del tuned
        free_mps()
        print(f"[seed {seed}] done ({time.time() - ts_:.0f}s)", flush=True)

        # checkpoint after every seed
        for fname, rows in [("detection_metrics", det_rows), ("event_detail", evt_rows),
                            ("leakage", leak_rows), ("waste_oracle", orc_rows),
                            ("waste_ops", ops_rows), ("extension", ext_rows),
                            ("levelbias", lev_rows), ("tuning_log", tune_log),
                            ("selection_log", sel_rows)]:
            pd.DataFrame(rows).to_csv(f"{out}/{fname}.csv", index=False)
    return {"runtime_s": time.time() - t0}


# ================================================================ SYNTHETIC
def run_synth(seeds=SEEDS, out=OUT):
    """Same protocol on the generator data (controlled type x severity)."""
    import torch
    warnings.filterwarnings("ignore")
    t0 = time.time()
    dataset = generate_synthetic_dataset(seed=42)
    n_tv = {b: int(len(df) * (FR[0] + FR[1])) for b, df in dataset.items()}
    val_off = {b: int(len(df) * FR[0]) for b, df in dataset.items()}
    train_bids = [b for b in sorted(dataset) if dataset[b]["btype"].iloc[0] != "retail"]
    rows, evt, tlog = [], [], []
    for seed in seeds:
        train, val, test = {}, {}, {}
        for bid, df in dataset.items():
            train[bid] = df.iloc[:val_off[bid]].reset_index(drop=True)
            val[bid] = df.iloc[val_off[bid]:n_tv[bid]].reset_index(drop=True)
            test[bid] = df.iloc[n_tv[bid]:].reset_index(drop=True)
        corrupted, events = inject_all(test, n_events=N_EVENTS, seed=seed, ref=train)
        val_c, val_ev = inject_all({b: val[b] for b in train_bids},
                                   n_events=N_EVENTS, seed=seed + 333, ref=train)
        scalers = {b: ZScaler().fit(train[b]) for b in train}
        sd = {b: scalers[b].sd_["load"] for b in dataset}
        tr_arr = [scalers[b].transform(train[b]) for b in train_bids]
        va_arr = [scalers[b].transform(val[b]) for b in train_bids]
        full_arr = {b: scalers[b].transform(dataset[b]) for b in dataset}
        cfg = PatchTSTConfig(); cfg.seed = seed
        ssl = PatchTSTAD(cfg).fit(tr_arr, val_arrays=va_arr)
        r0 = {"PatchTST": {b: ssl.point_residuals(full_arr[b]) for b in dataset},
              "HOW": {b: how_backbone_residuals(train[b], dataset[b], sd[b]) for b in dataset},
              "TOWT": {b: howtowt_fit_predict(train[b], dataset[b]) / sd[b] for b in dataset}}
        d_te, d_va = {}, {}
        for b in dataset:
            d = np.zeros(len(dataset[b]))
            d[n_tv[b]:] = (corrupted[b]["load"].to_numpy() - dataset[b]["load"].to_numpy()[n_tv[b]:]) / sd[b]
            d_te[b] = d
            dv = np.zeros(len(dataset[b]))
            if b in train_bids:
                dv[val_off[b]:n_tv[b]] = (val_c[b]["load"].to_numpy()
                                          - dataset[b]["load"].to_numpy()[val_off[b]:n_tv[b]]) / sd[b]
            d_va[b] = dv
        vz = {b: score_pair(r0["PatchTST"][b] + d_va[b], r0["PatchTST"][b],
                            full_arr[b], n_tv[b], "HoW-cond") for b in train_bids}
        w_best = max(W_GRID, key=lambda w: flag_event_f1(
            {b: fuse(w, *vz[b])[val_off[b]:n_tv[b]] for b in train_bids}, val_ev, train_bids))
        tr_pts = np.concatenate(tr_arr, axis=0)
        val_X = {b: scalers[b].transform(val_c[b]) for b in train_bids}
        del ssl
        free_mps()
        tuned = tune_baselines(tr_pts, val_X, val_ev, train_bids, seed, tlog)
        ARMS = {"PatchTST-SSL(HoW-cond)": ("PatchTST", "HoW-cond"),
                "PatchTST-SSL(global-MAD)": ("PatchTST", "global-MAD"),
                "HOW-profile(HoW-cond)": ("HOW", "HoW-cond"),
                "HOW-profile(global-MAD)": ("HOW", "global-MAD"),
                "PatchTST-SSL(HoD-cond)": ("PatchTST", "HoD-cond"),
                "HOW-profile(HoD-cond)": ("HOW", "HoD-cond"),
                "TOWT(HoW-cond)": ("TOWT", "HoW-cond")}
        for b in dataset:
            ev = events[events["building_id"] == b]
            flags = {}
            for name, (bb, sc) in ARMS.items():
                ra = r0[bb][b] + d_te[b]
                flags[name] = causal_flags(fuse(w_best, *score_pair(
                    ra, r0[bb][b], full_arr[b], n_tv[b], sc))[n_tv[b]:])
            Xt = scalers[b].transform(corrupted[b])
            for name, mdl in tuned.items():
                flags[name] = causal_flags(mdl.score(Xt))
            for name, f in flags.items():
                m = evaluate_building(f.astype(float), f, ev, len(f))
                m.update(model=name, building_id=b, seed=seed, seen=(b in train_bids))
                rows.append(m)
                evt += event_rows(name, b, seed, f, ev)
        agg = pd.DataFrame([r for r in rows if r["seed"] == seed]).groupby("model")["f1"].mean()
        print(f"[synth seed {seed}] w={w_best} " +
              ", ".join(f"{m}={v:.3f}" for m, v in agg.items()), flush=True)
    pd.DataFrame(rows).to_csv(f"{out}/synthetic_metrics.csv", index=False)
    pd.DataFrame(evt).to_csv(f"{out}/synthetic_event_detail.csv", index=False)
    pd.DataFrame(tlog).to_csv(f"{out}/synthetic_tuning_log.csv", index=False)
    return {"runtime_s": time.time() - t0}


REAL_FILES = ["detection_metrics", "event_detail", "leakage", "waste_oracle", "waste_ops",
              "extension", "levelbias", "tuning_log", "selection_log"]
SYNTH_FILES = ["synthetic_metrics", "synthetic_event_detail", "synthetic_tuning_log"]


def merge(kind):
    files = REAL_FILES if kind == "real" else SYNTH_FILES
    for f in files:
        parts = [pd.read_csv(os.path.join(OUT, f"{kind}_seed{s}", f"{f}.csv")) for s in SEEDS]
        pd.concat(parts, ignore_index=True).to_csv(os.path.join(OUT, f"{f}.csv"), index=False)
    print(f"merged {kind}: {files}")


if __name__ == "__main__":
    # one process per seed (fresh MPS state), then merge:
    #   python scripts/run_final_r7.py real-seed 123   (x3), merge-real
    #   python scripts/run_final_r7.py synth-seed 123  (x3), merge-synth
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    if cmd in ("real-seed", "synth-seed"):
        sd = int(sys.argv[2])
        kind = cmd.split("-")[0]
        out = os.path.join(OUT, f"{kind}_seed{sd}")
        os.makedirs(out, exist_ok=True)
        fn = run_real if kind == "real" else run_synth
        print(kind.upper(), "DONE", fn(seeds=[sd], out=out))
    elif cmd in ("merge-real", "merge-synth"):
        merge(cmd.split("-")[1])
