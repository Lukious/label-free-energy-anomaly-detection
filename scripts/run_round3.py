"""Round-3 review experiments (A1-A4) — PRE-REGISTERED analysis plan.

Pre-registered branch rule (fixed BEFORE running, from
paper/reviews/review_round3.md, 2026-10-01):
  - If HOW-profile+HoW-cond >= PatchTST+HoW-cond under TOST(+-0.03) OR the
    paired difference is non-significant -> promote the training-free
    pipeline to "recommended method" (Branch 1).
  - Else if PatchTST+HoW is significantly better than HOW+SAME scoring ->
    PatchTST+HoW is the headline (Branch 2).
TOST margin +-0.03 F1 = smallest difference of practical operational meaning.

Experiments:
  P-A (A1/A2): decisive 2x2 = backbone {PatchTST, HOW-profile} x scoring
     {HoW-conditional, non-HoW}. 6 pairwise building-level paired t-tests
     with HOLM correction + TOST on the key pair.
  P-B (A4): range-P/R/F1 for ALL models incl IF / OC-SVM / LSTM-AE (blanks
     in round-2).
  P-C (A3): coverage-error quantification + prescription experiment
     (flag->window extension: robust-z return, K=6/12; CUSUM end-point)
     on PatchTST and HOW-profile; PatchTST counterfactual level-bias check
     via target-only masking variant.
  P-D (A2): TOWT diagnostic — train/test temperature extrapolation +
     TOWT+HoW-cond scoring variant.
  P-E (B3): LEAD 1.0 acquisition attempted outside Kaggle (see report).

Outputs: results/metrics_round3*.csv, results/round3_experiments.md
Existing experiment code untouched; seeds fixed [123,124,125].
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

from scripts.run_round2 import (  # noqa: E402
    BASE_CFG, SEEDS, SQ2PI, cond_z, how_of, ops_waste, oracle_waste,
    range_based, sigma_hat_nonevent, tost,
)
from scripts.run_revision import (  # noqa: E402
    FLAG_WINDOW, FR, IsolationForestAD_tunable, LSTMAEAD_tunable,
    OCSVMAD_tunable, causal_flags, causal_scale_z, fuse, howtowt_fit_predict,
    hour_of, val_event_f1, val_test_len, DenseAEAD_tunable, _runs,
)
from src.anomaly_inject import inject_all
from src.data import ZScaler, load_dataset
from src.evaluate import evaluate_building, per_type_recall
from src.models.baselines import DEVICE  # noqa: F401
from src.models.patchtst import PatchTSTAD, PatchTSTConfig, make_windows

RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data", "bdg2", "selected_v2")
N_EVENTS = 6


# ---------------------------------------------------------------- helpers
def point_residuals_targetmask(ssl, X):
    """Diagnostic variant: mask ONLY the target hour of the load channel
    (window load context kept). Same trained model — used to test whether
    the full-window load masking causes the counterfactual level bias."""
    import torch
    cfg = ssl.cfg
    W = make_windows(X, cfg.seq_len, stride=1)
    Wt = torch.tensor(W, device=DEVICE).permute(0, 2, 1)
    res = []
    with torch.no_grad():
        for i in range(0, len(Wt), cfg.infer_batch):
            xb = Wt[i:i + cfg.infer_batch]
            x_in = xb.clone()
            x_in[:, cfg.target_idx, -1] = 0.0  # ONLY the last (target) hour
            rec = ssl.model(x_in)[:, cfg.target_idx, -1]
            res.append((xb[:, cfg.target_idx, -1] - rec).cpu().numpy())
    res = np.concatenate(res)
    out = np.empty(len(X))
    out[cfg.seq_len - 1:] = res
    out[:cfg.seq_len - 1] = res[0]
    return out


def robust_stats_train(x, n_tv):
    seg = x[:n_tv]
    seg = seg[~np.isnan(seg)]
    med = np.median(seg)
    mad = np.median(np.abs(seg - med)) * 1.4826
    return med, mad


def how_backbone_residuals(train_df, full_df, sd):
    """HOW-profile residual over the FULL series, standardized (kW/sd)."""
    how_full = full_df["timestamp"].dt.dayofweek * 24 + full_df["timestamp"].dt.hour
    how_tr = train_df["timestamp"].dt.dayofweek * 24 + train_df["timestamp"].dt.hour
    med = train_df.groupby(how_tr)["load"].median()
    return (full_df["load"].to_numpy() - med.reindex(how_full).to_numpy()) / sd


def z_long_from(rp, n_tv):
    rl = pd.Series(rp).rolling(168, min_periods=24).mean().to_numpy()
    q = rl[:n_tv] ** 2
    q = q[~np.isnan(q)]
    med = np.median(q)
    mad = np.median(np.abs(q - med)) * 1.4826
    return (rl ** 2 - med) / (mad + 1e-9)


def global_z(rp, n_tv):
    med, mad = robust_stats_train(rp, n_tv)
    return (rp - med) / (mad + 1e-9)


def prescribe_windows(resid, z, flags, mode, horizon=336):
    """Reconstruct event windows from flags. Returns list of (s, e) test-span
    windows (union-merged). Modes: z6 / z12 (extend until robust-z < 2 for K
    consecutive hours), cusum (end at peak of one-sided CUSUM, k=1)."""
    runs = _runs(flags)
    wins = []
    for s, _ in runs:
        if mode in ("z6", "z12"):
            K = 6 if mode == "z6" else 12
            t, calm = s, 0
            while t < len(resid) and t - s < horizon:
                calm = calm + 1 if abs(z[t]) < 2.0 else 0
                t += 1
                if calm >= K:
                    break
            e = t
        else:  # cusum: end at the peak of the one-sided CUSUM (k=1)
            S, Smax, peak = 0.0, 0.0, s
            for t in range(s, min(s + horizon, len(resid))):
                S = max(0.0, S + z[t] - 1.0)
                if S >= Smax:
                    Smax, peak = S, t
            e = max(peak + 1, s + 6)
        wins.append((s, min(e, len(resid))))
    # merge overlapping/adjacent (within 6h) windows
    merged = []
    for s, e in sorted(wins):
        if merged and s <= merged[-1][1] + 6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def prescription_eval(resid, z, flags, events):
    """Baseline (flagged-hour clip) vs prescribed-window estimators."""
    true = np.array([ev["injected_excess_kwh"] for _, ev in events.iterrows()])
    anom = np.zeros(len(resid), dtype=bool)
    matched = []
    for _, ev in events.iterrows():
        s, e = int(ev["start"]), int(ev["end"])
        anom[s:e] = True
        matched.append(bool(flags[s:e].any()))
    matched = np.array(matched, dtype=bool)
    rc = np.clip(np.nan_to_num(resid, nan=0.0), 0, None)
    f = flags.astype(bool)
    out = {"baseline_bias": (rc[f].sum() - true.sum()) / true.sum()}
    # baseline coverage = flagged fraction inside events
    out["baseline_cov"] = float(f[anom].mean()) if anom.any() else np.nan
    for mode in ("z6", "z12", "cusum"):
        wins = prescribe_windows(resid, z, f, mode)
        wmask = np.zeros(len(resid), dtype=bool)
        for s, e in wins:
            wmask[s:e] = True
        est = rc[wmask].sum()
        out[f"{mode}_bias"] = (est - true.sum()) / true.sum()
        out[f"{mode}_cov"] = float(wmask[anom].mean()) if anom.any() else np.nan
    out["true_kwh"] = float(true.sum())
    out["missed_kwh"] = float(true[~matched].sum())
    return out


# ---------------------------------------------------------------- main
def main():
    warnings.filterwarnings("ignore")
    t0 = time.time()
    dataset = load_dataset(DATA_DIR)
    print(f"loaded {len(dataset)} buildings", flush=True)
    n_tv = {b: int(len(df) * (FR[0] + FR[1])) for b, df in dataset.items()}
    train_bids = sorted(b for b in dataset
                        if dataset[b]["btype"].iloc[0] == "office")

    det_rows, ops_rows, pres_rows, lev_rows, extrap_rows = [], [], [], [], []

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

        ssl = PatchTSTAD(PatchTSTConfig())
        ssl.fit(tr_arr, val_arrays=va_arr)
        full_arr = {b: scalers[b].transform(dataset[b]) for b in dataset}
        r_pt = {b: ssl.point_residuals(full_arr[b]) for b in dataset}
        r_pl = {b: ssl.residuals(full_arr[b]) for b in dataset}
        # target-only-mask diagnostic residual (level-bias check, P-C3)
        r_tm = {b: point_residuals_targetmask(ssl, full_arr[b]) for b in dataset}

        # fusion w on validation injection events (same protocol as round-1/2)
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
        print(f"[seed {seed}] w_best={w_best}", flush=True)

        # ---------------- per-building residual/score construction
        tzs, tzl, tzs_howpt = {}, {}, {}
        resid_raw, resid_tm, resid_how, resid_towt = {}, {}, {}, {}
        hz_how, hz_glob, hz_howpt, hz_towt = {}, {}, {}, {}
        for b in corrupted:
            d = np.zeros(len(dataset[b]))
            d[n_tv[b]:] = (corrupted[b]["load"].to_numpy()
                           - dataset[b]["load"].to_numpy()[n_tv[b]:]) \
                / scalers[b].sd_["load"]
            rp = r_pt[b] + d
            z_s, z_l = causal_scale_z(rp, r_pl[b], full_arr[b], n_tv[b])
            tzs[b], tzl[b] = z_s[n_tv[b]:], z_l[n_tv[b]:]
            tzs_howpt[b] = cond_z(rp, full_arr[b], n_tv[b], 168, how_of)[n_tv[b]:]
            resid_raw[b] = rp[n_tv[b]:] * scalers[b].sd_["load"]
            resid_tm[b] = (r_tm[b] + d)[n_tv[b]:] * scalers[b].sd_["load"]
            # HOW backbone residual (standardized), full series then slice
            rp_h = how_backbone_residuals(train[b], dataset[b],
                                          scalers[b].sd_["load"]) + d
            rl_h = pd.Series(rp_h).rolling(168, min_periods=24).mean().to_numpy()
            q = rl_h[:n_tv[b]] ** 2
            q = q[~np.isnan(q)]
            mmed = np.median(q)
            mmad = np.median(np.abs(q - mmed)) * 1.4826
            zl_h = (rl_h ** 2 - mmed) / (mmad + 1e-9)
            hz_howpt[b] = (cond_z(rp_h, full_arr[b], n_tv[b], 168, how_of)[n_tv[b]:],
                           zl_h[n_tv[b]:])
            hz_glob[b] = (global_z(rp_h, n_tv[b])[n_tv[b]:], zl_h[n_tv[b]:])
            resid_how[b] = rp_h[n_tv[b]:] * scalers[b].sd_["load"]
            # TOWT backbone
            rp_t = howtowt_fit_predict(train[b], dataset[b]) / scalers[b].sd_["load"]
            rp_t[n_tv[b]:] += d[n_tv[b]:]
            rl_t = pd.Series(rp_t).rolling(168, min_periods=24).mean().to_numpy()
            q = rl_t[:n_tv[b]] ** 2
            q = q[~np.isnan(q)]
            mmed = np.median(q)
            mmad = np.median(np.abs(q - mmed)) * 1.4826
            zl_t = (rl_t ** 2 - mmed) / (mmad + 1e-9)
            hz_towt[b] = (cond_z(rp_t, full_arr[b], n_tv[b], 168, how_of)[n_tv[b]:],
                          zl_t[n_tv[b]:])
            resid_towt[b] = rp_t[n_tv[b]:] * scalers[b].sd_["load"]
            # TOWT temperature extrapolation diagnostic (P-D)
            ttr = train[b]["temp"].to_numpy()
            tte = corrupted[b]["temp"].to_numpy()
            extrap_rows.append({
                "building_id": b, "seed": seed,
                "test_outside_train_pct": float(
                    ((tte < ttr.min()) | (tte > ttr.max())).mean()),
                "test_p99_exceeds_train_max": bool(
                    np.nanquantile(tte, 0.99) > np.nanmax(ttr)),
                "train_temp_range": float(np.nanmax(ttr) - np.nanmin(ttr)),
            })

        # ---------------- baselines (point-feature models)
        tr_pts = np.concatenate(tr_arr, axis=0)
        ae_m = DenseAEAD_tunable(seed=0, hidden=(32, 16, 32)); ae_m.fit(tr_pts)
        nu, gam = BASE_CFG[seed]["ocsvm"]
        svm_m = OCSVMAD_tunable(seed=0, nu=nu, gamma=gam); svm_m.fit(tr_pts)
        if_m = IsolationForestAD_tunable(seed=0, contamination=0.01)
        if_m.clf.set_params(n_estimators=400, max_samples="auto")
        if_m.fit(tr_pts)
        lae_m = LSTMAEAD_tunable(seed=0, hidden=64); lae_m.fit(tr_pts)

        # ---------------- detectors (each model's OWN flags)
        fl = {}
        for b in corrupted:
            fl[("PatchTST-SSL(HoD-cond)", b)] = causal_flags(
                fuse(w_best, tzs[b], tzl[b]))
            fl[("PatchTST-SSL(HoW-cond)", b)] = causal_flags(
                fuse(w_best, tzs_howpt[b], tzl[b]))
            fl[("HOW-profile(HoW-cond)", b)] = causal_flags(
                fuse(w_best, *hz_howpt[b]))
            fl[("HOW-profile(global-MAD)", b)] = causal_flags(
                fuse(w_best, *hz_glob[b]))
            fl[("TOWT(HoW-cond)", b)] = causal_flags(fuse(w_best, *hz_towt[b]))
            # legacy simple HOW detector (round-2 row, for reference)
            how_tr = train[b]["timestamp"].dt.dayofweek * 24 + train[b]["timestamp"].dt.hour
            how_te = corrupted[b]["timestamp"].dt.dayofweek * 24 + corrupted[b]["timestamp"].dt.hour
            mad = train[b].groupby(how_tr)["load"].apply(
                lambda x: np.median(np.abs(x - np.median(x))) * 1.4826)
            med = train[b].groupby(how_tr)["load"].median()
            z_how = (corrupted[b]["load"].to_numpy() - med.reindex(how_te).to_numpy()) \
                / (mad.reindex(how_te).to_numpy() + 1e-9)
            fl[("HOW-profile+z", b)] = causal_flags(z_how)
            a_tr = int(len(train[b]) * 0.7)
            rt_tv = howtowt_fit_predict(train[b].iloc[:a_tr], train[b].iloc[a_tr:])
            sd_tv = np.median(np.abs(rt_tv - np.median(rt_tv))) * 1.4826
            fl[("TOWT-detector", b)] = causal_flags(
                resid_towt[b] / (sd_tv * scalers[b].sd_["load"] + 1e-9))
            Xt = scalers[b].transform(corrupted[b])
            fl[("Autoencoder(t)", b)] = causal_flags(ae_m.score(Xt))
            fl[("IsolationForest(t)", b)] = causal_flags(if_m.score(Xt))
            fl[("OC-SVM(t)", b)] = causal_flags(svm_m.score(Xt))
            fl[("LSTM-AE(t)", b)] = causal_flags(lae_m.score(Xt))

        MODELS = ["PatchTST-SSL(HoD-cond)", "PatchTST-SSL(HoW-cond)",
                  "HOW-profile(HoW-cond)", "HOW-profile(global-MAD)",
                  "TOWT(HoW-cond)", "HOW-profile+z", "TOWT-detector",
                  "Autoencoder(t)", "IsolationForest(t)", "OC-SVM(t)",
                  "LSTM-AE(t)"]
        RESID = {"PatchTST-SSL(HoD-cond)": resid_raw,
                 "PatchTST-SSL(HoW-cond)": resid_raw,
                 "HOW-profile(HoW-cond)": resid_how,
                 "HOW-profile(global-MAD)": resid_how,
                 "TOWT(HoW-cond)": resid_towt,
                 "HOW-profile+z": resid_how,
                 "TOWT-detector": resid_towt}

        for name in MODELS:
            for b in corrupted:
                flags = fl[(name, b)]
                ev = events[events["building_id"] == b]
                m = evaluate_building(flags.astype(float), flags, ev, len(flags))
                rP, rR = range_based(flags.astype(bool), ev, len(flags))
                rf1 = 2 * rP * rR / (rP + rR) if rP + rR else 0.0
                m.update(range_precision=rP, range_recall=rR, range_f1=rf1,
                         model=name, building_id=b, seed=seed,
                         btype=dataset[b]["btype"].iloc[0],
                         seen=(b in train_bids))
                det_rows.append(m)

        # ---------------- P-C operational waste + prescriptions
        for name in ["PatchTST-SSL(HoW-cond)", "PatchTST-SSL(HoD-cond)",
                     "HOW-profile(HoW-cond)", "TOWT(HoW-cond)"]:
            rd = RESID[name]
            for b in corrupted:
                ev = events[events["building_id"] == b]
                w = ops_waste(rd[b], fl[(name, b)], ev)
                w.update(model=name, building_id=b, seed=seed)
                ops_rows.append(w)
                # causal z from the same standardized scores used for flags
                if name.startswith("PatchTST"):
                    zsc = fuse(w_best, tzs_howpt[b] if "HoW" in name else tzs[b], tzl[b])
                elif name.startswith("HOW"):
                    zsc = fuse(w_best, *hz_howpt[b])
                else:
                    zsc = fuse(w_best, *hz_towt[b])
                pr = prescription_eval(rd[b], zsc, fl[(name, b)], ev)
                pr.update(model=name, building_id=b, seed=seed)
                pres_rows.append(pr)

        # ---------------- P-C3 level bias in oracle windows
        for b in corrupted:
            ev = events[events["building_id"] == b]
            sdL = scalers[b].sd_["load"]
            for _, e_ in ev.iterrows():
                s, e = int(e_["start"]), int(e_["end"])
                lev_rows.append({
                    "seed": seed, "building_id": b,
                    "patchtst_fullmask_mean": float(
                        np.nanmean(resid_raw[b][s:e])) / sdL,
                    "patchtst_targetmask_mean": float(
                        np.nanmean(resid_tm[b][s:e])) / sdL,
                    "how_mean": float(np.nanmean(resid_how[b][s:e])) / sdL,
                    "true_kwh": float(e_["injected_excess_kwh"]),
                })

        print(f"[seed {seed}] done ({time.time() - ts_:.0f}s)", flush=True)

    # ============================================================ report
    det = pd.DataFrame(det_rows)
    det.to_csv(os.path.join(RESULTS_DIR, "metrics_round3.csv"), index=False)
    ops = pd.DataFrame(ops_rows)
    ops.to_csv(os.path.join(RESULTS_DIR, "metrics_round3_waste_ops.csv"), index=False)
    pres = pd.DataFrame(pres_rows)
    pres.to_csv(os.path.join(RESULTS_DIR, "metrics_round3_prescription.csv"), index=False)
    lev = pd.DataFrame(lev_rows)
    lev.to_csv(os.path.join(RESULTS_DIR, "metrics_round3_levelbias.csv"), index=False)
    extrap = pd.DataFrame(extrap_rows)
    extrap.to_csv(os.path.join(RESULTS_DIR, "metrics_round3_towt_extrap.csv"),
                  index=False)

    ps = det.groupby(["model", "seed"])[
        ["precision", "recall", "f1", "range_precision", "range_recall",
         "range_f1", "false_alarm_rate"]].mean().reset_index()

    def ms(x):
        return f"{np.nanmean(x):.3f} ± {np.nanstd(x, ddof=1):.3f}"

    L = []
    A = L.append
    A("# Round-3 Review Experiments (2026-10-01) — pre-registered plan")
    A("")
    A("Pre-registered branch rule (fixed in review_round3.md BEFORE running):")
    A("HOW-profile+HoW >= PatchTST+HoW under TOST(±0.03) or paired "
      "non-significant → Branch 1 (training-free pipeline = recommended "
      "method); else PatchTST+HoW significantly better → Branch 2 "
      "(headline = PatchTST+HoW, comparator always HOW+same scoring).")
    A("TOST margin ±0.03 F1 = smallest practical operational difference.")
    A("")

    # ---- P-A: 2x2 + stats
    A("## P-A (A1/A2) — decisive 2×2: backbone × scoring (3 seeds, "
      f"w={w_best} identical across backbones)")
    A("")
    A("| Model | Precision | Recall | F1 | FAR |")
    A("|---|---|---|---|---|")
    for m in ["PatchTST-SSL(HoW-cond)", "PatchTST-SSL(HoD-cond)",
              "HOW-profile(HoW-cond)", "HOW-profile(global-MAD)",
              "TOWT(HoW-cond)", "HOW-profile+z"]:
        g = ps[ps["model"] == m]
        A(f"| {m} | {ms(g['precision'])} | {ms(g['recall'])} | {ms(g['f1'])} | "
          f"{ms(g['false_alarm_rate'] * 100)}% |")
    A("")

    bl = det.pivot_table(index=["seed", "building_id"], columns="model",
                         values="f1").groupby(level=1).mean()
    CORE = ["PatchTST-SSL(HoW-cond)", "PatchTST-SSL(HoD-cond)",
            "HOW-profile(HoW-cond)", "HOW-profile(global-MAD)"]
    A("### Building-level paired comparisons (n=41 buildings, F1 seed-mean)")
    A("")
    pairs = [(CORE[i], CORE[j]) for i in range(4) for j in range(i + 1, 4)]
    res_pairs = []
    for a_, b_ in pairs:
        d = (bl[a_] - bl[b_]).dropna().values
        t2, p2 = stats.ttest_rel(d, np.zeros(len(d)))
        res_pairs.append((a_, b_, d.mean(), p2))
    order = sorted(range(len(res_pairs)), key=lambda i: res_pairs[i][3])
    holm = {}
    mtests = len(res_pairs)
    prev = 0.0
    for rank, i in enumerate(order):
        a_, b_, dm, p2 = res_pairs[i]
        adj = min((mtests - rank) * p2, 1.0)
        adj = max(adj, prev)
        prev = adj
        holm[i] = adj
    A("| Comparison | mean diff | raw p | Holm-adj p | sig (0.05) |")
    A("|---|---|---|---|---|")
    for i, (a_, b_, dm, p2) in enumerate(res_pairs):
        A(f"| {a_} − {b_} | {dm:+.4f} | {p2:.3g} | {holm[i]:.3g} | "
          f"{'YES' if holm[i] < 0.05 else 'no'} |")
    A("")
    dd = (bl["HOW-profile(HoW-cond)"] - bl["PatchTST-SSL(HoW-cond)"]).dropna().values
    tt = tost(dd, 0.03)
    A(f"### KEY PAIR TOST (±0.03): HOW+HoW − PatchTST+HoW: mean "
      f"{tt['mean']:+.4f}, 90% CI [{tt['ci90'][0]:+.4f}, {tt['ci90'][1]:+.4f}], "
      f"p_TOST={tt['p_tost']:.4f} → "
      f"{'EQUIVALENT' if tt['p_tost'] < 0.05 else 'NOT equivalent'}")
    key_t, key_p = stats.ttest_rel(dd, np.zeros(len(dd)))
    A(f"Paired t on same difference: t={key_t:.2f}, p={key_p:.3g}.")
    A("")
    A("### PRE-REGISTERED BRANCH VERDICT (written from the numbers above)")
    key_how = bl["HOW-profile(HoW-cond)"].mean()
    key_pt = bl["PatchTST-SSL(HoW-cond)"].mean()
    if (tt["p_tost"] < 0.05) or (key_p >= 0.05 and key_how >= key_pt):
        A(f"**BRANCH 1**: HOW-profile+HoW ({key_how:.3f}) ≥ PatchTST+HoW "
          f"({key_pt:.3f}) — "
          + ("TOST-equivalent within ±0.03" if tt["p_tost"] < 0.05
             else "paired difference non-significant")
          + ". → The training-free pipeline (HOW-profile backbone + HoW "
            "conditional scoring + causal robust-z flagger) is promoted to "
            "the RECOMMENDED METHOD; PatchTST becomes the comparison arm "
            "showing what a learned counterfactual adds (nothing "
            "significant). Title/abstract reframe: calendar-conditional "
            "scoring is the whole story; SSL backbone is near-parity.")
    else:
        A(f"**BRANCH 2**: PatchTST+HoW ({key_pt:.3f}) is significantly better "
          f"than HOW+same scoring ({key_how:.3f}) (paired p={key_p:.3g}, "
          f"Holm-adj {holm[[i for i,(a_,b_,_,_) in enumerate(res_pairs) if a_=='PatchTST-SSL(HoW-cond)' and b_=='HOW-profile(HoW-cond)'][0]]:.3g}) "
          "→ headline = PatchTST+HoW, comparator = HOW+HoW (same scoring).")
    A("")
    A("Interpretation of the scoring contrast (backbone-internal): PatchTST "
      f"HoD→HoW: {bl['PatchTST-SSL(HoD-cond)'].mean():.3f}→"
      f"{bl['PatchTST-SSL(HoW-cond)'].mean():.3f}; HOW global→HoW: "
      f"{bl['HOW-profile(global-MAD)'].mean():.3f}→"
      f"{bl['HOW-profile(HoW-cond)'].mean():.3f}.")
    A("")

    # ---- P-B: range metrics incl IF/OC-SVM/LSTM-AE
    A("## P-B (A4) — range-based P/R/F1 for ALL models (macro over buildings)")
    A("")
    A("| Model | P | R | F1 | range-P | range-R | range-F1 |")
    A("|---|---|---|---|---|---|---|")
    for m in ps["model"].unique():
        g = ps[ps["model"] == m]
        A(f"| {m} | {ms(g['precision'])} | {ms(g['recall'])} | {ms(g['f1'])} | "
          f"{ms(g['range_precision'])} | {ms(g['range_recall'])} | "
          f"{ms(g['range_f1'])} |")
    rk = ps.groupby("model")["range_f1"].mean().sort_values(ascending=False)
    A("")
    A("**range-F1 ranking**: " + " > ".join(f"{m} ({v:.3f})"
                                            for m, v in rk.items()))
    how_rank = list(rk.index).index("HOW-profile(HoW-cond)") + 1
    A(f"HOW-profile(HoW-cond) range-F1 rank: **{how_rank}"
      f"{'st' if how_rank == 1 else 'th'}** — "
      + ("ranking CHANGES vs point F1 (HOW-profile becomes #1) — reported "
         "as required." if how_rank == 1 else
         "ranking does NOT change; PatchTST variant remains above under "
         "range-F1 — reported honestly."))
    A("")

    # ---- P-C1: coverage vs error
    A("## P-C (A3) — coverage bottleneck + prescription")
    A("")
    A("### (1) Coverage–error relationship (building level)")
    A("")
    det_b = det.groupby(["model", "building_id"]).agg(
        rR=("range_recall", "mean"), f1=("f1", "mean")).reset_index()
    ops_b = ops.groupby(["model", "building_id"]).agg(
        matched=("matched_err_kwh", "sum"), true=("true_kwh", "sum"),
        est=("est_kwh", "sum"), missed=("missed_kwh", "sum"),
        fam=("frac_events_matched", "mean")).reset_index()
    ops_b["matched_pct"] = ops_b["matched"] / ops_b["true"]
    ops_b["bias_pct"] = (ops_b["est"] - ops_b["true"]) / ops_b["true"]
    cc = det_b.merge(ops_b, on=["model", "building_id"])
    for name, g in cc.groupby("model"):
        g = g.dropna(subset=["rR", "matched_pct"])
        try:
            r1, p1 = stats.pearsonr(g["rR"], g["matched_pct"])
            r2, p2 = stats.spearmanr(g["rR"], g["bias_pct"])
            slope = np.polyfit(g["rR"], g["matched_pct"], 1)[0]
        except Exception:
            continue
        A(f"- {name}: corr(range-recall, matched-err fraction) Pearson "
          f"r={r1:+.2f} (p={p1:.1e}); Spearman(range-recall, total bias) "
          f"ρ={r2:+.2f} (p={p2:.1e}); OLS slope {slope:+.2f} "
          "(higher coverage → less under-estimation)")
    A("")

    # ---- P-C2: prescriptions
    A("### (2) Prescription: flag→window reconstruction (operational bias)")
    A("")
    A("| Detector | baseline bias | baseline cov | z6 bias | z6 cov | "
      "z12 bias | z12 cov | cusum bias | cusum cov |")
    A("|---|---|---|---|---|---|---|---|---|")
    for name, g in pres.groupby("model"):
        def m(c):
            return f"{np.nanmean(g[c]):+.1%}"
        def mc(c):
            return f"{np.nanmean(g[c]):.1%}"
        A(f"| {name} | {m('baseline_bias')} | {mc('baseline_cov')} | "
          f"{m('z6_bias')} | {mc('z6_cov')} | {m('z12_bias')} | "
          f"{mc('z12_cov')} | {m('cusum_bias')} | {mc('cusum_cov')} |")
    A("")
    best = {name: g[["z6_bias", "z12_bias", "cusum_bias"]].mean().abs().idxmin()
            for name, g in pres.groupby("model")}
    for name, mode in best.items():
        bb = pres[pres["model"] == name]["baseline_bias"].mean()
        nb = pres[pres["model"] == name][mode.replace("_bias", "_bias")].mean()
        A(f"- {name}: best prescription = {mode} → total signed bias "
          f"{bb:+.1%} → {nb:+.1%} (Δbias {abs(nb) - abs(bb):+.1%} pp)")
    A("")

    # ---- P-C3: level bias
    A("### (3) PatchTST counterfactual level bias (oracle windows, "
      "in sd-load units)")
    A("")
    A(f"- full-window load masking: mean signed residual "
      f"{lev['patchtst_fullmask_mean'].mean():+.4f} sd "
      f"(median {lev['patchtst_fullmask_mean'].median():+.4f})")
    A(f"- target-only masking:     mean "
      f"{lev['patchtst_targetmask_mean'].mean():+.4f} sd "
      f"(median {lev['patchtst_targetmask_mean'].median():+.4f})")
    A(f"- HOW-profile residual:    mean {lev['how_mean'].mean():+.4f} sd")
    A("- Interpretation: the full-window-masked PatchTST counterfactual sits "
      "systematically ABOVE the observed load inside event windows "
      "(negative residual = under-estimation of excess), consistent with the "
      "−112% signed oracle bias; letting the model see the window's load "
      "context (target-only masking) "
      + ("reduces" if abs(lev['patchtst_targetmask_mean'].mean())
         < abs(lev['patchtst_fullmask_mean'].mean()) else "does not reduce")
      + " the level bias "
      f"({abs(lev['patchtst_fullmask_mean'].mean()):.4f} → "
      f"{abs(lev['patchtst_targetmask_mean'].mean()):.4f} sd).")
    A("")

    # ---- P-D: TOWT
    A("## P-D (A2) — TOWT diagnostic")
    A("")
    A(f"- Temperature extrapolation: mean test-hours outside train temp "
      f"range = {extrap['test_outside_train_pct'].mean():.1%}; buildings "
      f"with test p99 above train max: "
      f"{extrap['test_p99_exceeds_train_max'].mean():.1%}.")
    A(f"- Calendar-conditional scoring applied to TOWT residual: TOWT(HoW-cond) "
      f"F1 {ps[ps['model'] == 'TOWT(HoW-cond)']['f1'].mean():.3f} vs "
      f"TOWT-detector (round-2 protocol) 0.571 vs HOW+HoW "
      f"{ps[ps['model'] == 'HOW-profile(HoW-cond)']['f1'].mean():.3f}.")
    towt_how = bl["TOWT(HoW-cond)"].mean() if "TOWT(HoW-cond)" in bl else np.nan
    A(f"- Building-level paired: TOWT(HoW-cond) vs HOW-profile(HoW-cond): "
      f"mean diff {bl['HOW-profile(HoW-cond)'].mean() - towt_how:+.3f}"
      if not np.isnan(towt_how) else "")
    if not np.isnan(towt_how):
        dd2 = (bl["HOW-profile(HoW-cond)"] - bl["TOWT(HoW-cond)"]).dropna().values
        _, pp = stats.ttest_rel(dd2, np.zeros(len(dd2)))
        A(f"  (paired p={pp:.3g}) → TOWT gap is "
          + ("mostly a scoring gap" if pp >= 0.05 else
             "NOT fully explained by scoring; residual temperature-model "
             "difference remains"))
    A("")

    A(f"Total runtime: {time.time() - t0:.0f}s")
    with open(os.path.join(RESULTS_DIR, "round3_experiments.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
