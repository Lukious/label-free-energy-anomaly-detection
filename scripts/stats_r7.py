"""Round-7 statistics (pre-registered in paper/reviews/review_round7.md).

Input : results/final_r7/detection_metrics.csv, waste_ops.csv, extension.csv
Output: results/final_r7/factorial_stats.csv   (main effects, interaction,
                                                simple effects, TOST, for
                                                event-F1 and range-F1)
        results/final_r7/waste_ci.csv          (aggregate bias / gross with
                                                building + site bootstrap CI)

Unit of inference = building (seed-averaged). Sensitivity: sign-flip
permutation (10^4), building bootstrap, site-cluster bootstrap (sites =
building-id prefix), mixed model contrast ~ 1 + (1 | site).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MC = os.path.join(ROOT, "results", "final_r7")
B = 10_000
RNG = np.random.default_rng(20261002)

PT_H, PT_G = "PatchTST-SSL(HoW-cond)", "PatchTST-SSL(global-MAD)"
HW_H, HW_G = "HOW-profile(HoW-cond)", "HOW-profile(global-MAD)"
MARGIN = 0.03


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    prev = 0.0
    for r, i in enumerate(o):
        prev = max(prev, min((len(p) - r) * p[i], 1.0))
        adj[i] = prev
    return adj


def site_of(b):
    return b.split("_")[0]


def perm_p(d):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    s = RNG.choice([-1.0, 1.0], size=(B, len(d)))
    null = np.abs((s * d).mean(axis=1))
    return (1 + (null >= obs - 1e-15).sum()) / (B + 1)


def boot_ci(d):
    d = np.asarray(d, float)
    idx = RNG.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def cluster_boot_ci(d, sites):
    d = np.asarray(d, float)
    sites = np.asarray(sites)
    us = np.unique(sites)
    groups = [d[sites == s] for s in us]
    ms = np.empty(B)
    for k in range(B):
        pick = RNG.integers(0, len(us), size=len(us))
        v = np.concatenate([groups[i] for i in pick])
        ms[k] = v.mean()
    return np.percentile(ms, [2.5, 97.5])


def mixed_p(d, sites):
    import statsmodels.formula.api as smf
    df = pd.DataFrame({"d": d, "site": sites})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = smf.mixedlm("d ~ 1", df, groups=df["site"]).fit(reml=True)
            return float(r.params["Intercept"]), float(r.pvalues["Intercept"])
        except Exception:
            return np.nan, np.nan


def tost(d, margin=MARGIN):
    d = np.asarray(d, float)
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    t_lo = (m + margin) / se
    t_hi = (m - margin) / se
    p = max(1 - stats.t.cdf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
    q = stats.t.ppf(0.95, n - 1)
    return p, m - q * se, m + q * se


def describe(name, family, d, sites, metric):
    t, p = stats.ttest_1samp(d, 0.0)
    q = stats.t.ppf(0.975, len(d) - 1)
    se = d.std(ddof=1) / np.sqrt(len(d))
    bl, bh = boot_ci(d)
    cl, ch = cluster_boot_ci(d, sites)
    mm, mp = mixed_p(d, sites)
    return {"metric": metric, "family": family, "contrast": name, "n": len(d),
            "n_sites": len(set(sites)), "estimate": d.mean(),
            "ci95_lo": d.mean() - q * se, "ci95_hi": d.mean() + q * se,
            "p_t": p, "p_perm": perm_p(d), "boot_lo": bl, "boot_hi": bh,
            "cluster_lo": cl, "cluster_hi": ch,
            "mixed_est": mm, "p_mixed": mp,
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}


def factorial(det, metric):
    bl = det.pivot_table(index=["seed", "building_id"], columns="model",
                         values=metric).groupby(level=1).mean()
    bl = bl[[PT_H, PT_G, HW_H, HW_G]].dropna()
    sites = np.array([site_of(b) for b in bl.index])
    a, b, c, e = (bl[k].to_numpy() for k in (PT_H, PT_G, HW_H, HW_G))
    rows = []
    primary = [("backbone main effect", 0.5 * ((a - c) + (b - e))),
               ("scoring main effect", 0.5 * ((a - b) + (c - e))),
               ("interaction (scoring effect: PatchTST - HOW)", (a - b) - (c - e))]
    for name, d in primary:
        rows.append(describe(name, "primary", d, sites, metric))
    simple = [("PatchTST(HoW) - HOW(HoW)  [backbone | HoW]", a - c),
              ("PatchTST(global) - HOW(global)  [backbone | global]", b - e),
              ("PatchTST(HoW) - PatchTST(global)  [scoring | PatchTST]", a - b),
              ("HOW(HoW) - HOW(global)  [scoring | HOW]", c - e)]
    for name, d in simple:
        rows.append(describe(name, "simple", d, sites, metric))
    for fam in ("primary", "simple"):
        idx = [i for i, r in enumerate(rows) if r["family"] == fam]
        adj = holm([rows[i]["p_t"] for i in idx])
        for i, v in zip(idx, adj):
            rows[i]["holm_p"] = v
    p, lo, hi = tost(a - c)
    rows.append({"metric": metric, "family": "tost",
                 "contrast": "PatchTST(HoW) - HOW(HoW)  TOST +-0.03",
                 "n": len(a), "estimate": (a - c).mean(), "p_tost": p,
                 "ci90_lo": lo, "ci90_hi": hi,
                 "ci_excludes_margin": bool(lo > MARGIN or hi < -MARGIN),
                 "ci_inside_margin": bool(lo > -MARGIN and hi < MARGIN)})
    cells = {k: (bl[k].mean(), bl[k].std(ddof=1)) for k in (PT_H, PT_G, HW_H, HW_G)}
    return rows, cells


def waste_ci(ops, ext):
    """Aggregate signed bias and gross error with building and site bootstrap."""
    out = []
    frames = [("flagged-hours", ops.assign(mode="none-ops"))]
    frames += [(f"ext:{m}", ext[ext["mode"] == m]) for m in ext["mode"].unique()]
    frames += [("ext:chosen", ext[ext["chosen"]])]
    for label, d in frames:
        for model, g in d.groupby("model"):
            pb = g.groupby("building_id")[["est_kwh", "true_kwh", "matched_err_kwh",
                                           "false_add_kwh", "missed_kwh"]].sum()
            pb = pb[pb["true_kwh"] > 0]
            sites = np.array([site_of(b) for b in pb.index])
            arr = pb.to_numpy()

            def stat(a):
                est, tru, mt, fa, mi = a.sum(axis=0)
                agg_gross = abs(mt) + abs(fa) + abs(mi)
                pb_gross = np.abs(a[:, 2:5]).sum()
                rel = np.abs(a[:, 0] - a[:, 1]) / a[:, 1]
                return np.array([(est - tru) / tru, agg_gross / tru, pb_gross / tru,
                                 np.median(rel)])
            base = stat(arr)
            bs = np.array([stat(arr[RNG.integers(0, len(arr), len(arr))]) for _ in range(2000)])
            us = np.unique(sites)
            cs = []
            for _ in range(2000):
                pick = RNG.choice(us, size=len(us))
                cs.append(stat(np.concatenate([arr[sites == s] for s in pick])))
            cs = np.array(cs)
            for j, nm in enumerate(["agg_bias", "agg_gross_rel", "pb_gross_rel", "median_abs_rel_err"]):
                out.append({"estimator": label, "model": model, "stat": nm,
                            "value": base[j],
                            "boot_lo": np.percentile(bs[:, j], 2.5),
                            "boot_hi": np.percentile(bs[:, j], 97.5),
                            "cluster_lo": np.percentile(cs[:, j], 2.5),
                            "cluster_hi": np.percentile(cs[:, j], 97.5),
                            "n_buildings": len(arr), "n_sites": len(us)})
    return pd.DataFrame(out)


def main(mc=MC):
    det = pd.read_csv(os.path.join(mc, "detection_metrics.csv"))
    rows = []
    # NIAR (false_alarm_rate column) is analysed as a secondary metric;
    # lower is better, so a negative backbone effect means fewer alarms.
    for metric in ("f1", "range_f1", "false_alarm_rate"):
        r, cells = factorial(det, metric)
        rows += r
        for k, (m, s) in cells.items():
            rows.append({"metric": metric, "family": "cell", "contrast": k,
                         "estimate": m, "sd_buildings": s})
    fs = pd.DataFrame(rows)
    fs.to_csv(os.path.join(mc, "factorial_stats.csv"), index=False)
    pd.set_option("display.width", 220)
    print(fs[["metric", "family", "contrast", "estimate", "ci95_lo", "ci95_hi", "p_t",
              "holm_p", "p_perm", "cluster_lo", "cluster_hi", "p_mixed"]].round(4).to_string())
    ops = pd.read_csv(os.path.join(mc, "waste_ops.csv"))
    ext = pd.read_csv(os.path.join(mc, "extension.csv"))
    wc = waste_ci(ops, ext)
    wc.to_csv(os.path.join(mc, "waste_ci.csv"), index=False)
    print(wc[wc["estimator"].isin(["flagged-hours", "ext:chosen"])].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else MC)
