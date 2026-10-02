"""CANONICAL FINAL RUN (Round-4 B1/B3 response) — single run, single set of
master CSVs, every final number in the paper derives from these files.

Outputs (results/final_consolidated/):
  detection_metrics.csv      (a) all detectors x building x seed:
                                 event-F1, P/R, range-P/R/F1, FAR, delay,
                                 per-type recall, seen/unseen
  paired_stats.csv           (b) 2x2 building-level paired t + HOLM (family
                                 of 6, range stated) + TOST with CI computed
                                 as diff = PatchTST(HoW) - comparator
  leakage.csv                (c) transductive-leak variant under identical
                                 conditions
  waste_oracle.csv           (d) oracle 3 estimator variants (clip/signed/
                                 floor) + per-type + kWh-weighted level bias
  waste_ops.csv              (d) operational waste, kWh additive decomp (Eq.6)
  prescription.csv           (d) rule (z6/z12/cusum) selected on VALIDATION
                                 injections, applied to test; post-extension
                                 Eq.(6) decomposition; per-building |err|
  synthetic_metrics.csv      (e) new-protocol synthetic, same-condition
                                 comparisons (HoW-cond arms paired)
  lead_label_durations.csv   (f) LEAD 1.0 labeled-run duration distribution
  final_consolidation.md     summary of every final number

Design decisions (recorded per review B1/B3/B4):
  * HOW official control = HOW-profile (HoW-cond.): slot = dayofweek x hour
    (168), train-frozen slot median/MAD applied to the HOW-profile RESIDUAL,
    fused short+long causal scoring -- identical conditioning to the 2x2
    scoring factor. "HOW-profile + z (slot-level)" is a SIMPLER VARIANT
    (slot stats on RAW load, own flagger, no fusion) -- both reported.
  * LSTM-AE row = its own tuned flagger (0.51-class), fusion-scored variant
    reported separately.
  * TOST diff order fixed: PatchTST(HoW) - HOW(HoW); CI reported directly.
  * Prescription rule chosen on validation injection events only.

Usage:  python scripts/run_final_consolidated.py [--part real|synth|lead|all]
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
    SEEDS, SQ2PI, cond_z, how_of, ops_waste, oracle_waste, range_based,
    sigma_hat_nonevent, tost,
)
from scripts.run_revision import (  # noqa: E402
    FLAG_WINDOW, FR, IsolationForestAD_tunable, LSTMAEAD_tunable,
    OCSVMAD_tunable, causal_flags, causal_scale_z, fuse, howtowt_fit_predict,
    val_event_f1, val_test_len, DenseAEAD_tunable, _runs,
)
from scripts.run_round3 import (  # noqa: E402
    how_backbone_residuals, point_residuals_targetmask, prescribe_windows,
    robust_stats_train,
)
from src.anomaly_inject import inject_all  # noqa: E402
from src.data import ZScaler, load_dataset, generate_synthetic_dataset  # noqa: E402
from src.evaluate import evaluate_building, robust_z_flags, per_type_recall  # noqa: E402
from src.models.patchtst import (  # noqa: E402
    PatchTSTAD, PatchTSTConfig, _hour_cond_z, _robust_z,
)

OUT = os.path.join(ROOT, "results", "final_consolidated")
DATA_DIR = os.path.join(ROOT, "data", "bdg2", "selected_v2")
LEAD_CSV = "/tmp/lead1.0-small.csv"
N_EVENTS = 6
MODELS = ["PatchTST-SSL(HoW-cond)", "PatchTST-SSL(HoD-cond)",
          "HOW-profile(HoW-cond)", "HOW-profile(global-MAD)",
          "TOWT(HoW-cond)", "HOW-profile+z(slot)", "Autoencoder(t)",
          "IsolationForest(t)", "OC-SVM(t)", "LSTM-AE(t)"]


def z_long(rp, n_tv):
    rl = pd.Series(rp).rolling(168, min_periods=24).mean().to_numpy()
    q = rl[:n_tv] ** 2
    q = q[~np.isnan(q)]
    med = np.median(q)
    mad = np.median(np.abs(q - med)) * 1.4826
    return (rl ** 2 - med) / (mad + 1e-9)


def global_z(rp, n_tv):
    med, mad = robust_stats_train(rp, n_tv)
    return (rp - med) / (mad + 1e-9)


def prescription_decomp(resid, z, flags, events, mode):
    """Chosen-rule estimator + Eq.(6)-style kWh decomposition AFTER extension.

    est = clip-resid summed over prescribed windows.
    delta = est - true_total decomposed as
      (est_on_event_hours_in_windows - true_of_matched_events)  matched err
      + est_on_non-event_hours_in_windows                        false add
      - true_of_unmatched_events                                  missed
    """
    true = np.array([ev["injected_excess_kwh"] for _, ev in events.iterrows()])
    anom = np.zeros(len(resid), dtype=bool)
    matched = []
    for _, ev in events.iterrows():
        s, e = int(ev["start"]), int(ev["end"])
        anom[s:e] = True
        matched.append(bool(flags[s:e].any()))
    matched = np.array(matched, dtype=bool)
    rc = np.clip(np.nan_to_num(resid, nan=0.0), 0, None)
    wins = prescribe_windows(resid, z, flags, mode)
    wmask = np.zeros(len(resid), dtype=bool)
    for s, e in wins:
        wmask[s:e] = True
    est = rc[wmask].sum()
    est_ev = rc[wmask & anom].sum()
    false_add = rc[wmask & ~anom].sum()
    missed = true[~matched].sum()
    true_total = true.sum()
    return {
        "mode": mode, "est_kwh": float(est), "true_kwh": float(true_total),
        "bias": (est - true_total) / true_total if true_total else np.nan,
        "matched_err_kwh": float(est_ev - true[matched].sum()),
        "false_add_kwh": float(false_add),
        "missed_kwh": float(missed),
        "window_cov": float(wmask[anom].mean()) if anom.any() else np.nan,
        "n_windows": len(wins),
        # identity check (Eq. 6 extended to windows)
        "_check": abs((est - true_total)
                      - ((est_ev - true[matched].sum()) + false_add - missed)),
    }


# ================================================================ REAL DATA
def run_real():
    warnings.filterwarnings("ignore")
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    dataset = load_dataset(DATA_DIR)
    print(f"loaded {len(dataset)} buildings", flush=True)
    n_tv = {b: int(len(df) * (FR[0] + FR[1])) for b, df in dataset.items()}
    train_bids = sorted(b for b in dataset
                        if dataset[b]["btype"].iloc[0] == "office")

    det_rows, leak_rows, orc_rows, ops_rows, pres_rows, lev_rows = \
        [], [], [], [], [], []
    val_rule_choice = {}  # (model) -> rule chosen on validation

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
        r_tm = {b: point_residuals_targetmask(ssl, full_arr[b]) for b in dataset}

        # fusion w on validation injection events (identical across backbones)
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

        # ---------------- residual / score construction
        tzs, tzl, tzs_howpt = {}, {}, {}
        resid_raw, resid_tm, resid_how, resid_towt = {}, {}, {}, {}
        hz_how, hz_glob, hz_towt = {}, {}, {}
        # validation-span arrays for prescription-rule selection
        v_arr = {}  # model -> {b: (resid_kwh_val, z_val, flags_val)}
        for b in corrupted:
            sdL = scalers[b].sd_["load"]
            d = np.zeros(len(dataset[b]))
            d[n_tv[b]:] = (corrupted[b]["load"].to_numpy()
                           - dataset[b]["load"].to_numpy()[n_tv[b]:]) / sdL
            rp = r_pt[b] + d
            z_s, z_l = causal_scale_z(rp, r_pl[b], full_arr[b], n_tv[b])
            tzs[b], tzl[b] = z_s[n_tv[b]:], z_l[n_tv[b]:]
            tzs_howpt[b] = cond_z(rp, full_arr[b], n_tv[b], 168, how_of)[n_tv[b]:]
            resid_raw[b] = rp[n_tv[b]:] * sdL
            resid_tm[b] = (r_tm[b] + d)[n_tv[b]:] * sdL
            # HOW-profile backbone residual, standardized
            rp_h = how_backbone_residuals(train[b], dataset[b], sdL) + d
            zl_h_full = z_long(rp_h, n_tv[b])
            hz_how[b] = (cond_z(rp_h, full_arr[b], n_tv[b], 168, how_of)[n_tv[b]:],
                         zl_h_full[n_tv[b]:])
            hz_glob[b] = (global_z(rp_h, n_tv[b])[n_tv[b]:],
                          zl_h_full[n_tv[b]:])
            resid_how[b] = rp_h[n_tv[b]:] * sdL
            # TOWT backbone
            rp_t = howtowt_fit_predict(train[b], dataset[b]) / sdL
            rp_t[n_tv[b]:] += d[n_tv[b]:]
            zl_t_full = z_long(rp_t, n_tv[b])
            hz_towt[b] = (cond_z(rp_t, full_arr[b], n_tv[b], 168, how_of)[n_tv[b]:],
                          zl_t_full[n_tv[b]:])
            resid_towt[b] = rp_t[n_tv[b]:] * sdL

            # --- validation-span construction (same machinery, val offsets)
            dv = np.zeros(len(dataset[b]))
            if b in train_bids:
                vv = val_c[b]
                dv[val_off[b]:val_off[b] + len(vv)] = (vv["load"].to_numpy()
                                                       - dataset[b]["load"].to_numpy()[val_off[b]:val_off[b] + len(vv)]) / sdL
                rp_v = r_pt[b] + dv
                zs_v, zl_v = causal_scale_z(rp_v, r_pl[b], full_arr[b], n_tv[b])
                v_arr.setdefault("PatchTST-SSL(HoW-cond)", {})[b] = (
                    rp_v[val_off[b]:val_off[b] + len(vv)] * sdL,
                    fuse(w_best, zs_v, zl_v)[val_off[b]:val_off[b] + len(vv)],
                    causal_flags(fuse(w_best, zs_v, zl_v)[val_off[b]:val_off[b] + len(vv)]))
                rp_h_full = how_backbone_residuals(train[b], dataset[b], sdL) + dv
                zh_full = cond_z(rp_h_full, full_arr[b], n_tv[b], 168, how_of)
                rl_h = pd.Series(rp_h_full).rolling(168, min_periods=24).mean().to_numpy()
                qq = rl_h[:n_tv[b]] ** 2
                qq = qq[~np.isnan(qq)]
                zlv = (rl_h ** 2 - np.median(qq)) / (np.median(np.abs(qq - np.median(qq))) * 1.4826 + 1e-9)
                sc = fuse(w_best, zh_full, zlv)[val_off[b]:val_off[b] + len(vv)]
                v_arr.setdefault("HOW-profile(HoW-cond)", {})[b] = (
                    rp_h_full[val_off[b]:val_off[b] + len(vv)] * sdL, sc,
                    causal_flags(sc))

        # ---------------- B3 fix: choose prescription rule on VALIDATION
        for model, d_b in v_arr.items():
            biases = {}
            for mode in ("z6", "z12", "cusum"):
                bs = []
                for b, (res, zc, fl) in d_b.items():
                    ev = val_ev[val_ev["building_id"] == b]
                    r = prescription_decomp(res, zc, fl, ev, mode)
                    bs.append(r["bias"])
                biases[mode] = float(np.nanmean(bs))
            chosen = min(biases, key=lambda m: abs(biases[m]))
            val_rule_choice.setdefault(model, {})[seed] = chosen
            print(f"[seed {seed}] {model}: val biases "
                  f"{ {m: round(v, 3) for m, v in biases.items()} } -> {chosen}",
                  flush=True)

        # ---------------- baselines
        tr_pts = np.concatenate(tr_arr, axis=0)
        ae_m = DenseAEAD_tunable(seed=0, hidden=(32, 16, 32)); ae_m.fit(tr_pts)
        nu, gam = 0.05, "scale"
        try:
            from scripts.run_round2 import BASE_CFG
            nu, gam = BASE_CFG[seed]["ocsvm"]
        except Exception:
            pass
        svm_m = OCSVMAD_tunable(seed=0, nu=nu, gamma=gam); svm_m.fit(tr_pts)
        if_m = IsolationForestAD_tunable(seed=0, contamination=0.01)
        if_m.clf.set_params(n_estimators=400, max_samples="auto")
        if_m.fit(tr_pts)
        lae_m = LSTMAEAD_tunable(seed=0, hidden=64); lae_m.fit(tr_pts)

        # ---------------- flags per model
        fl = {}
        for b in corrupted:
            fl[("PatchTST-SSL(HoD-cond)", b)] = causal_flags(fuse(w_best, tzs[b], tzl[b]))
            fl[("PatchTST-SSL(HoW-cond)", b)] = causal_flags(fuse(w_best, tzs_howpt[b], tzl[b]))
            fl[("HOW-profile(HoW-cond)", b)] = causal_flags(fuse(w_best, *hz_how[b]))
            fl[("HOW-profile(global-MAD)", b)] = causal_flags(fuse(w_best, *hz_glob[b]))
            fl[("TOWT(HoW-cond)", b)] = causal_flags(fuse(w_best, *hz_towt[b]))
            # VARIANT: slot-level z on raw load, own flagger (round-2 row)
            how_te = corrupted[b]["timestamp"].dt.dayofweek * 24 + corrupted[b]["timestamp"].dt.hour
            how_tr = train[b]["timestamp"].dt.dayofweek * 24 + train[b]["timestamp"].dt.hour
            mad = train[b].groupby(how_tr)["load"].apply(
                lambda x: np.median(np.abs(x - np.median(x))) * 1.4826)
            med = train[b].groupby(how_tr)["load"].median()
            z_slot = (corrupted[b]["load"].to_numpy() - med.reindex(how_te).to_numpy()) \
                / (mad.reindex(how_te).to_numpy() + 1e-9)
            fl[("HOW-profile+z(slot)", b)] = causal_flags(z_slot)
            Xt = scalers[b].transform(corrupted[b])
            fl[("Autoencoder(t)", b)] = causal_flags(ae_m.score(Xt))
            fl[("IsolationForest(t)", b)] = causal_flags(if_m.score(Xt))
            fl[("OC-SVM(t)", b)] = causal_flags(svm_m.score(Xt))
            fl[("LSTM-AE(t)", b)] = causal_flags(lae_m.score(Xt))
            # (c) leakage: transductive variant under identical conditions
            d = np.zeros(len(dataset[b]))
            d[n_tv[b]:] = (corrupted[b]["load"].to_numpy()
                           - dataset[b]["load"].to_numpy()[n_tv[b]:]) / scalers[b].sd_["load"]
            z_td = _hour_cond_z(r_pt[b] + d, full_arr[b])
            zl_td = _robust_z((r_pl[b] + d) ** 2)
            sc_td = fuse(w_best, z_td, zl_td)[n_tv[b]:]
            flt = robust_z_flags(sc_td, window=FLAG_WINDOW)
            ev = events[events["building_id"] == b]
            m = evaluate_building(sc_td, flt, ev, len(sc_td))
            m.update(model="PatchTST-transductive", building_id=b, seed=seed,
                     btype=dataset[b]["btype"].iloc[0])
            leak_rows.append(m)

        RESID = {"PatchTST-SSL(HoD-cond)": resid_raw,
                 "PatchTST-SSL(HoW-cond)": resid_raw,
                 "HOW-profile(HoW-cond)": resid_how,
                 "HOW-profile(global-MAD)": resid_how,
                 "TOWT(HoW-cond)": resid_towt,
                 "HOW-profile+z(slot)": resid_how}
        SCOREZ = {"PatchTST-SSL(HoW-cond)": lambda b: fuse(w_best, tzs_howpt[b], tzl[b]),
                  "PatchTST-SSL(HoD-cond)": lambda b: fuse(w_best, tzs[b], tzl[b]),
                  "HOW-profile(HoW-cond)": lambda b: fuse(w_best, *hz_how[b]),
                  "TOWT(HoW-cond)": lambda b: fuse(w_best, *hz_towt[b])}

        # ---------------- (a) detection metrics
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

        # ---------------- (d) oracle waste (3 variants) + level bias
        for name in ["PatchTST-SSL(HoW-cond)", "HOW-profile(HoW-cond)",
                     "TOWT(HoW-cond)"]:
            rd = RESID[name]
            for b in corrupted:
                ev = events[events["building_id"] == b]
                sig = sigma_hat_nonevent(rd[b], ev, len(rd[b]))
                true, est = oracle_waste(rd[b], ev, sig)
                for i, (_, e_) in enumerate(ev.iterrows()):
                    orc_rows.append({
                        "model": name, "building_id": b, "seed": seed,
                        "type": e_["type"], "true_kwh": float(true[i]),
                        "clip_kwh": float(est["clip"][i]),
                        "signed_kwh": float(est["signed"][i]),
                        "floor_kwh": float(est["floor"][i]),
                    })
                # operational waste with Eq.(6) decomposition
                w = ops_waste(rd[b], fl[(name, b)], ev)
                w.update(model=name, building_id=b, seed=seed)
                ops_rows.append(w)

        # level bias per event (in sd units) + per type (for B3f)
        for b in corrupted:
            ev = events[events["building_id"] == b]
            sdL = scalers[b].sd_["load"]
            for _, e_ in ev.iterrows():
                s, e = int(e_["start"]), int(e_["end"])
                lev_rows.append({
                    "seed": seed, "building_id": b, "type": e_["type"],
                    "patchtst_fullmask_mean": float(np.nanmean(resid_raw[b][s:e])) / sdL,
                    "patchtst_targetmask_mean": float(np.nanmean(resid_tm[b][s:e])) / sdL,
                    "how_mean": float(np.nanmean(resid_how[b][s:e])) / sdL,
                    "true_kwh": float(e_["injected_excess_kwh"]),
                })

        # ---------------- (d) prescriptions: all rules on test + chosen
        for name in ["PatchTST-SSL(HoW-cond)", "HOW-profile(HoW-cond)",
                     "TOWT(HoW-cond)"]:
            rd, zf = RESID[name], SCOREZ[name]
            for b in corrupted:
                ev = events[events["building_id"] == b]
                for mode in ("z6", "z12", "cusum"):
                    r = prescription_decomp(rd[b], zf(b), fl[(name, b)], ev, mode)
                    r.update(model=name, building_id=b, seed=seed,
                             chosen=(mode == val_rule_choice.get(name, {}).get(seed, "z6")))
                    pres_rows.append(r)
        print(f"[seed {seed}] done ({time.time() - ts_:.0f}s)", flush=True)

    det = pd.DataFrame(det_rows)
    det.to_csv(f"{OUT}/detection_metrics.csv", index=False)
    pd.DataFrame(leak_rows).to_csv(f"{OUT}/leakage.csv", index=False)
    pd.DataFrame(orc_rows).to_csv(f"{OUT}/waste_oracle.csv", index=False)
    pd.DataFrame(ops_rows).to_csv(f"{OUT}/waste_ops.csv", index=False)
    pd.DataFrame(pres_rows).to_csv(f"{OUT}/prescription.csv", index=False)
    pd.DataFrame(lev_rows).to_csv(f"{OUT}/levelbias.csv", index=False)

    # ---------------- (b) paired stats + TOST (diff = PatchTST - comparator)
    bl = det.pivot_table(index=["seed", "building_id"], columns="model",
                         values="f1").groupby(level=1).mean()
    CORE = ["PatchTST-SSL(HoW-cond)", "PatchTST-SSL(HoD-cond)",
            "HOW-profile(HoW-cond)", "HOW-profile(global-MAD)"]
    pairs = [(CORE[i], CORE[j]) for i in range(4) for j in range(i + 1, 4)]
    rows, raw = [], []
    for a_, b_ in pairs:
        d = (bl[a_] - bl[b_]).dropna().values
        t2, p2 = stats.ttest_rel(d, np.zeros(len(d)))
        raw.append(p2)
        rows.append({"comparison": f"{a_} - {b_}", "mean_diff": d.mean(),
                     "n": len(d), "raw_p": p2})
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min((len(raw) - rank) * raw[i], 1.0)
        adj = max(adj, prev)
        prev = adj
        rows[i]["holm_p"] = adj
        rows[i]["holm_mtests"] = len(raw)  # Holm family = 6 pairwise 2x2 contrasts
    # TOST: diff = PatchTST(HoW) - HOW(HoW)  (fixed sign convention)
    for comp_name, comp in [("HOW-profile(HoW-cond)", "HOW-profile(HoW-cond)"),
                            ("HOW-profile+z(slot)", "HOW-profile+z(slot)")]:
        d = (bl["PatchTST-SSL(HoW-cond)"] - bl[comp]).dropna().values
        tt = tost(d, 0.03)
        rows.append({
            "comparison": f"PatchTST-SSL(HoW-cond) - {comp}",
            "mean_diff": tt["mean"], "n": tt["n"],
            "raw_p": np.nan, "holm_p": np.nan, "holm_mtests": np.nan,
            "tost_p": tt["p_tost"], "ci90_lo": tt["ci90"][0],
            "ci90_hi": tt["ci90"][1],
            "margin": 0.03,
            "ci_excludes_margin": bool(tt["ci90"][0] > 0.03 or tt["ci90"][1] < -0.03),
        })
    pd.DataFrame(rows).to_csv(f"{OUT}/paired_stats.csv", index=False)
    return {"runtime_s": time.time() - t0, "w_best": w_best,
            "val_rule_choice": val_rule_choice}


# ================================================================ SYNTHETIC
def run_synth():
    warnings.filterwarnings("ignore")
    t0 = time.time()
    dataset = generate_synthetic_dataset(seed=42)
    n_tv = {b: int(len(df) * (FR[0] + FR[1])) for b, df in dataset.items()}
    train_bids = [b for b in sorted(dataset)
                  if dataset[b]["btype"].iloc[0] != "retail"]
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
        vzs, vzl = {}, {}
        for b in train_bids:
            d = np.zeros(len(dataset[b]))
            vv = val_c[b]
            d[val_off[b]:val_off[b] + len(vv)] = \
                (vv["load"].to_numpy() - dataset[b]["load"].to_numpy()[val_off[b]:val_off[b] + len(vv)]) \
                / scalers[b].sd_["load"]
            vzs[b], vzl[b] = causal_scale_z(r_pt[b] + d, r_pl[b], full_arr[b], n_tv[b])
        w_best, _ = val_event_f1(vzs, vzl, val_off, val_ev, train_bids,
                                 [0.1 * i for i in range(11)])
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

        tuned = [tune("IF(t)", lambda **c: IsolationForestAD_tunable(seed=0, **c),
                      [{"contamination": c} for c in (0.01, 0.05, 0.1)]),
                 tune("OC-SVM(t)", lambda **c: OCSVMAD_tunable(seed=0, **c),
                      [{"nu": n, "gamma": g} for n in (0.02, 0.05, 0.1)
                       for g in ("scale", 0.2)]),
                 tune("AE(t)", lambda **c: DenseAEAD_tunable(seed=0, **c),
                      [{"hidden": h} for h in ((32, 16, 32), (64, 32, 64))]),
                 tune("LSTM-AE(t)", lambda **c: LSTMAEAD_tunable(seed=0, **c),
                      [{"hidden": h} for h in (32, 64)])]

        for b in corrupted:
            sdL = scalers[b].sd_["load"]
            d = np.zeros(len(dataset[b]))
            d[n_tv[b]:] = (corrupted[b]["load"].to_numpy()
                           - dataset[b]["load"].to_numpy()[n_tv[b]:]) / sdL
            scores = {}
            rp = r_pt[b] + d
            z_s, z_l = causal_scale_z(rp, r_pl[b], full_arr[b], n_tv[b])
            scores["PatchTST-SSL(HoD-cond)"] = fuse(w_best, z_s[n_tv[b]:], z_l[n_tv[b]:])
            scores["PatchTST-SSL(HoW-cond)"] = cond_z(rp, full_arr[b], n_tv[b], 168, how_of)[n_tv[b]:]
            # HOW-profile with the SAME residual+cond_z pipeline as real data
            rp_h = how_backbone_residuals(train[b], dataset[b], sdL) + d
            rl_h = pd.Series(rp_h).rolling(168, min_periods=24).mean().to_numpy()
            qq = rl_h[:n_tv[b]] ** 2
            qq = qq[~np.isnan(qq)]
            zl_h = (rl_h ** 2 - np.median(qq)) / (np.median(np.abs(qq - np.median(qq))) * 1.4826 + 1e-9)
            scores["HOW-profile(HoW-cond)"] = fuse(w_best, cond_z(rp_h, full_arr[b], n_tv[b], 168, how_of)[n_tv[b]:], zl_h[n_tv[b]:])
            # slot-level variant (raw load)
            how_te = corrupted[b]["timestamp"].dt.dayofweek * 24 + corrupted[b]["timestamp"].dt.hour
            how_tr = train[b]["timestamp"].dt.dayofweek * 24 + train[b]["timestamp"].dt.hour
            mad = train[b].groupby(how_tr)["load"].apply(
                lambda x: np.median(np.abs(x - np.median(x))) * 1.4826)
            med = train[b].groupby(how_tr)["load"].median()
            scores["HOW-profile+z(slot)"] = (corrupted[b]["load"].to_numpy() - med.reindex(how_te).to_numpy()) \
                / (mad.reindex(how_te).to_numpy() + 1e-9)
            rp_t = howtowt_fit_predict(train[b], dataset[b]) / sdL
            rp_t[n_tv[b]:] += d[n_tv[b]:]
            scores["TOWT(HoW-cond)"] = cond_z(rp_t, full_arr[b], n_tv[b], 168, how_of)[n_tv[b]:]
            Xt = scalers[b].transform(corrupted[b])
            for name, mdl in tuned:
                scores[name] = mdl.score(Xt)
            ev = events[events["building_id"] == b]
            for name, s in scores.items():
                f = causal_flags(s)
                m = evaluate_building(np.asarray(s, dtype=float), f, ev, len(s))
                m.update(model=name, building_id=b, seed=seed, seen=(b in train_bids))
                rows.append(m)
                for _, e_ in ev.iterrows():
                    rows.append({"model": name, "building_id": b, "seed": seed,
                                 "type": e_["type"], "severity": e_["severity_name"],
                                 "type_recall": float(f[int(e_["start"]):int(e_["end"])].any())})
        agg = pd.DataFrame([r for r in rows if "f1" in r and r["seed"] == seed]) \
            .groupby("model")["f1"].mean()
        print(f"[synth seed {seed}] w={w_best:.1f} " +
              ", ".join(f"{m}={v:.3f}" for m, v in agg.items()), flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/synthetic_metrics.csv", index=False)
    return {"runtime_s": time.time() - t0}


# ================================================================ LEAD
def run_lead():
    df = pd.read_csv(LEAD_CSV, parse_dates=["timestamp"])
    rows = []
    for bid, g in df.groupby("building_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        lab = g["anomaly"].astype(bool).to_numpy()
        for s, e in _runs(lab):
            rows.append({"building_id": bid, "duration_h": e - s})
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/lead_label_durations.csv", index=False)
    q = d["duration_h"].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
    return {"n_labels": len(d), "median_h": q[0.5], "q25": q[0.25],
            "q75": q[0.75], "q90": q[0.9], "mean_h": d["duration_h"].mean(),
            "pct_le_6h": (d["duration_h"] <= 6).mean(),
            "pct_le_24h": (d["duration_h"] <= 24).mean()}


if __name__ == "__main__":
    part = sys.argv[-1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    if part in ("real", "all"):
        info = run_real()
        print("REAL DONE", info["runtime_s"], info["val_rule_choice"])
    if part in ("synth", "all"):
        print("SYNTH DONE", run_synth()["runtime_s"])
    if part in ("lead", "all"):
        print("LEAD DONE", run_lead())
