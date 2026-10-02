"""Generate LaTeX table fragments + numbers.tex macros DIRECTLY from the
canonical master CSVs in results/final_consolidated/ (Round-4 B1 fix).

The paper \input{}s these fragments; no number in any table is hand-typed.
Reads ONLY results/final_consolidated/*.csv — no legacy result files.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MC = os.path.join(ROOT, "results", "final_consolidated")
TAB = os.path.join(ROOT, "paper", "latex", "tables")
os.makedirs(TAB, exist_ok=True)

MODEL_ORDER = ["PatchTST-SSL(HoW-cond)", "PatchTST-SSL(HoD-cond)",
               "HOW-profile(HoW-cond)", "HOW-profile(global-MAD)",
               "TOWT(HoW-cond)", "HOW-profile+z(slot)",
               "IsolationForest(t)", "OC-SVM(t)", "Autoencoder(t)",
               "LSTM-AE(t)"]
NICE = {
    "PatchTST-SSL(HoW-cond)": r"\textbf{PatchTST-SSL (HoW-cond., ours)}",
    "PatchTST-SSL(HoD-cond)": r"PatchTST-SSL (HoD-cond.)",
    "HOW-profile(HoW-cond)": r"\quad HOW-profile (HoW-cond.)",
    "HOW-profile(global-MAD)": r"\quad HOW-profile (global-MAD)",
    "TOWT(HoW-cond)": r"TOWT $+$ HoW-cond.\ scoring",
    "HOW-profile+z(slot)": r"HOW-profile $+$ $z$ (slot-level variant)",
    # "(t)" = tuned on the same validation injection events; all baselines
    # are flagged by the same causal (trailing) operator -- NOT transductive.
    "IsolationForest(t)": r"Isolation Forest (tuned)",
    "OC-SVM(t)": r"OC-SVM (tuned)",
    "Autoencoder(t)": r"Autoencoder (tuned)",
    "LSTM-AE(t)": r"LSTM-AE (own tuned flagger)",
}


def ms(x, dec=3):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 1:
        return f"{x[0]:.{dec}f}"
    return f"{np.nanmean(x):.{dec}f} $\\pm$ {np.nanstd(x, ddof=1):.{dec}f}"


def load(name):
    return pd.read_csv(os.path.join(MC, name))


# ---------------------------------------------------------------- tables
def table_detection():
    d = load("detection_metrics.csv")
    ps = d.groupby(["model", "seed"])[
        ["precision", "recall", "f1", "range_precision", "range_recall",
         "range_f1", "false_alarm_rate", "detection_delay_h"]].mean().reset_index()
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r"Model & P & R & F1 & range-P & range-R & range-F1 & FAR \\",
             r"\midrule"]
    for m in MODEL_ORDER:
        g = ps[ps["model"] == m]
        if not len(g):
            continue
        far = g["false_alarm_rate"] * 100
        lines.append(
            f"{NICE[m]} & {ms(g['precision'],2)} & {ms(g['recall'],2)} & "
            f"{ms(g['f1'],2)} & {ms(g['range_precision'],2)} & "
            f"{ms(g['range_recall'],2)} & {ms(g['range_f1'],2)} & "
            f"{ms(far,2)}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "table_detection.tex"), "w").write("\n".join(lines))


def table_paired():
    d = load("paired_stats.csv")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Contrast (A $-$ B) & $\Delta$ & $n$ & raw $p$ & Holm $p$ \\",
             r"\midrule"]
    holm_rows = d[d["holm_p"].notna()]
    for _, r in holm_rows.iterrows():
        lines.append(f"{r['comparison']} & {r['mean_diff']:+.4f} & {int(r['n'])} & "
                     f"{r['raw_p']:.3g} & {r['holm_p']:.3g} \\\\")
    lines.append(r"\midrule")
    for _, r in d[d["holm_p"].isna()].iterrows():
        excl = "yes" if r.get("ci_excludes_margin") else "no"
        lines.append(
            f"{r['comparison']} (TOST) & {r['mean_diff']:+.4f} & {int(r['n'])} & "
            f"90\\% CI $[{r['ci90_lo']:+.4f}, {r['ci90_hi']:+.4f}]$ & "
            f"$p_{{\\mathrm{{TOST}}}}={r['tost_p']:.3f}$; CI excl.\\ margin ({excl}) \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("%% Holm family = 6 pairwise contrasts of the 2x2 design.")
    open(os.path.join(TAB, "table_paired_stats.tex"), "w").write("\n".join(lines))


def table_waste_ops():
    d = load("waste_ops.csv")
    g = d.groupby("model").agg(
        true=("true_kwh", "sum"), est=("est_kwh", "sum"),
        matched=("matched_err_kwh", "sum"), fadd=("false_add_kwh", "sum"),
        miss=("missed_kwh", "sum"),
        n=("building_id", lambda x: len(set(x)))).reset_index()
    g["bias"] = (g["est"] - g["true"]) / g["true"]
    g["gross"] = g[["matched", "fadd", "miss"]].abs().sum(axis=1)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Model & total bias & matched err (kWh) & false add (kWh) & missed (kWh) & gross err (kWh) & $n$ bldgs \\",
             r"\midrule"]
    for _, r in g.iterrows():
        lines.append(f"{NICE.get(r['model'], r['model'])} & {r['bias']:+.1%}".replace("%", r"\%") + " & "
                     f"{r['matched']:+.0f} & {r['fadd']:+.0f} & "
                     f"$-${r['miss']:.0f} & {r['gross']:.0f} & {int(r['n'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              "%% Eq.(6): est-true == matched_err + false_add - missed (kWh).",
              "%% gross = |matched| + |false add| + |missed| (no cancellation)."]
    open(os.path.join(TAB, "table_waste_ops.tex"), "w").write("\n".join(lines))


def table_prescription():
    d = load("prescription.csv")
    ch = d[d["chosen"]]
    # per building |err| quartiles for the chosen rule (ours = PatchTST HoW)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Model (rule chosen on validation) & bias & matched err & false add & missed & gross err & window cov. \\",
             r"\midrule"]
    for m, g in ch.groupby("model"):
        rule = g["mode"].iloc[0]
        tot_t, tot_e = g["true_kwh"].sum(), g["est_kwh"].sum()
        gross = g[["matched_err_kwh", "false_add_kwh", "missed_kwh"]].abs().sum().sum()
        lines.append(
            f"{NICE.get(m, m)} [{rule}] & " + f"{(tot_e - tot_t) / tot_t:+.1%}".replace("%", r"\%") + " & "
            f"{g['matched_err_kwh'].sum():+.0f} & {g['false_add_kwh'].sum():+.0f} & "
            f"$-${g['missed_kwh'].sum():.0f} & {gross:.0f} & "
            f"{g['window_cov'].mean():.1%}".replace("%", r"\%") + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              "%% Post-extension Eq.(6) decomposition; rule selected on "
              "validation injection events, never on test.",
              "%% gross = |matched| + |false add| + |missed|: compare with the",
              "%% same column of Table waste-ops (before extension)."]
    open(os.path.join(TAB, "table_prescription.tex"), "w").write("\n".join(lines))


def table_levelbias():
    d = load("levelbias.csv")
    d["w"] = d["true_kwh"]
    rows = []
    # event-simple vs kWh-weighted
    for col, lab in [("patchtst_fullmask_mean", "full-window mask"),
                     ("patchtst_targetmask_mean", "target-only mask"),
                     ("how_mean", "HOW-profile")]:
        simple = d[col].mean()
        weighted = np.average(d[col], weights=d["w"])
        rows.append((lab, simple, weighted))
    per_type = {}
    for t, g in d.groupby("type"):
        per_type[t] = {
            "fullmask_simple": g["patchtst_fullmask_mean"].mean(),
            "fullmask_kwh": np.average(g["patchtst_fullmask_mean"],
                                       weights=g["true_kwh"]),
            "how_simple": g["how_mean"].mean(),
            "how_kwh": np.average(g["how_mean"], weights=g["true_kwh"]),
        }
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Residual & mean (event-simple) & mean (kWh-weighted) \\",
             r"\midrule"]
    for lab, s, w in rows:
        lines.append(f"{lab} & {s:+.3f} $\\sigma$ & {w:+.3f} $\\sigma$ \\\\")
    lines.append(r"\midrule")
    for t, r in per_type.items():
        lines.append(f"\\quad full-window mask, {t} & {r['fullmask_simple']:+.3f} $\\sigma$ & "
                     f"{r['fullmask_kwh']:+.3f} $\\sigma$ \\\\")
    for t, r in per_type.items():
        lines.append(f"\\quad HOW profile, {t} & {r['how_simple']:+.3f} $\\sigma$ & "
                     f"{r['how_kwh']:+.3f} $\\sigma$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "table_levelbias.tex"), "w").write("\n".join(lines))


def table_synthetic():
    if not os.path.exists(os.path.join(MC, "synthetic_metrics.csv")):
        print("synthetic_metrics.csv not yet present; skipping")
        return
    d = load("synthetic_metrics.csv")
    base = d[d["f1"].notna()]
    ps = base.groupby(["model", "seed"])[
        ["precision", "recall", "f1", "false_alarm_rate"]].mean().reset_index()
    ty = d[d["type_recall"].notna()].groupby(["model", "type"])["type_recall"].mean()
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Model & P & R & F1 & FAR \\", r"\midrule"]
    for m in MODEL_ORDER:
        g = ps[ps["model"] == m]
        if not len(g):
            continue
        rec = " / ".join(f"{ty.get((m, t), np.nan):.2f}"
                         for t in ("spike", "drift", "schedule"))
        lines.append(f"{NICE[m]} & {ms(g['precision'],2)} & {ms(g['recall'],2)} & "
                     f"{ms(g['f1'],2)} & {ms(g['false_alarm_rate']*100,1)}\\% \\\\"
                     f" %% recall spike/drift/schedule: {rec}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "table_synthetic.tex"), "w").write("\n".join(lines))


def table_waste_oracle():
    """Stage (a): oracle-window bias by estimator variant, kWh-weighted."""
    d = load("waste_oracle.csv")
    g = d.groupby("model").agg(true=("true_kwh", "sum"),
                               clip=("clip_kwh", "sum"),
                               signed=("signed_kwh", "sum"),
                               floor=("floor_kwh", "sum")).reset_index()
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Estimator & signed & clip & floor-corrected \\",
             r"\midrule"]
    for _, r in g.iterrows():
        b = lambda v: f"{(r[v] - r['true']) / r['true']:+.1%}".replace("%", r"\%")
        lines.append(f"{NICE.get(r['model'], r['model'])} & "
                     f"{b('signed')} & {b('clip')} & {b('floor')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              "%% Bias relative to total true injected excess kWh (kWh-weighted)."]
    open(os.path.join(TAB, "table_waste_oracle.tex"), "w").write("\n".join(lines))


# ---------------------------------------------------------------- numbers
def write_numbers():
    det = load("detection_metrics.csv")
    ps = det.groupby(["model", "seed"])[
        ["f1", "range_f1", "false_alarm_rate", "precision", "recall"]].mean()
    pv = load("paired_stats.csv")
    leak = load("leakage.csv")
    ops = load("waste_ops.csv")
    pres = load("prescription.csv")
    lev = load("levelbias.csv")
    orc = load("waste_oracle.csv")

    def f1(m):
        return ps.loc[m, "f1"].mean()

    bl = det.pivot_table(index=["seed", "building_id"], columns="model",
                         values="f1").groupby(level=1).mean()
    d_how = (bl["PatchTST-SSL(HoW-cond)"] - bl["HOW-profile(HoW-cond)"]).dropna()
    trow = pv[pv["comparison"].str.contains(r"How\)\s-\sHOW-profile\(HoW", regex=True)]
    if not len(trow):
        trow = pv[pv["tost_p"].notna()].iloc[0:0]
    tr = pv[pv["tost_p"].notna()]
    key = tr[tr["comparison"].str.startswith("PatchTST-SSL(HoW-cond) - HOW-profile(HoW-cond)")].iloc[0]
    key_slot = tr[tr["comparison"].str.endswith("HOW-profile+z(slot)")].iloc[0]

    ops_g = ops.groupby("model").agg(est=("est_kwh", "sum"), true=("true_kwh", "sum"))
    ops_bias = ((ops_g["est"] - ops_g["true"]) / ops_g["true"]).to_dict()
    ch = pres[pres["chosen"]]
    pres_g = ch.groupby("model").agg(est=("est_kwh", "sum"), true=("true_kwh", "sum"),
                                     rule=("mode", "first"))
    pres_bias = ((pres_g["est"] - pres_g["true"]) / pres_g["true"]).to_dict()
    pres_rule = pres_g["rule"].to_dict()
    base_bias = ops_bias.copy()

    orc_g = orc.groupby("model").agg(true=("true_kwh", "sum"),
                                     clip=("clip_kwh", "sum"),
                                     signed=("signed_kwh", "sum"),
                                     floor=("floor_kwh", "sum"))
    orc_signed = ((orc_g["signed"] - orc_g["true"]) / orc_g["true"]).to_dict()
    orc_clip = ((orc_g["clip"] - orc_g["true"]) / orc_g["true"]).to_dict()
    orc_floor = ((orc_g["floor"] - orc_g["true"]) / orc_g["true"]).to_dict()

    # per-building |bias| quartiles for the chosen rule (ours = PatchTST HoW)
    ch_pt = ch[(ch["model"] == "PatchTST-SSL(HoW-cond)")]
    pb = (ch_pt.groupby("building_id")
          .apply(lambda g: (g["est_kwh"].sum() - g["true_kwh"].sum())
                 / g["true_kwh"].sum()))
    pb = pb.abs().dropna()

    # gross-error (sum of |decomposition terms|) before/after extension, MWh
    def gross(d):
        g = d.groupby("model")[["matched_err_kwh", "false_add_kwh",
                                "missed_kwh"]].sum()
        return (g.abs().sum(axis=1) / 1e6).to_dict()

    gross_before = gross(ops)
    gross_after = gross(ch[ch["chosen"]])

    # per-building |bias| quartiles BEFORE extension (ours)
    ops_pt = ops[ops["model"] == "PatchTST-SSL(HoW-cond)"]
    pb0 = (ops_pt.groupby("building_id")
           .apply(lambda g: (g["est_kwh"].sum() - g["true_kwh"].sum())
                  / g["true_kwh"].sum()))
    pb0 = pb0.abs().dropna()

    # oracle per-event median |err| (clip form), per model
    orc_med = (orc.assign(e=(orc["clip_kwh"] - orc["true_kwh"]).abs()
                          / orc["true_kwh"])
               .groupby("model")["e"].median()).to_dict()

    caus = det[det["model"] == "PatchTST-SSL(HoD-cond)"].groupby("building_id")["f1"].mean()
    trd = leak.groupby("building_id")["f1"].mean()
    common = caus.index.intersection(trd.index)
    leak_delta = float((trd[common] - caus[common]).mean())

    lead = load("lead_label_durations.csv")
    lev_full = lev["patchtst_fullmask_mean"]
    lev_tm = lev["patchtst_targetmask_mean"]
    lev_w = lev["true_kwh"]

    N = {
        # headline detection (building macro-avg over 3 seeds)
        "PatchTSTHowF": f1("PatchTST-SSL(HoW-cond)"),
        "PatchTSTHoDF": f1("PatchTST-SSL(HoD-cond)"),
        "HOWHowF": f1("HOW-profile(HoW-cond)"),
        "HOWGlobalF": f1("HOW-profile(global-MAD)"),
        "HOWSlotF": f1("HOW-profile+z(slot)"),
        "TOWTHowF": f1("TOWT(HoW-cond)"),
        "IFF": f1("IsolationForest(t)"),
        "OCSVMF": f1("OC-SVM(t)"),
        "AEF": f1("Autoencoder(t)"),
        "LSTMAEF": f1("LSTM-AE(t)"),
        "PatchTSTHowRF": ps.loc["PatchTST-SSL(HoW-cond)", "range_f1"].mean(),
        "HOWHowRF": ps.loc["HOW-profile(HoW-cond)", "range_f1"].mean(),
        "PatchTSTFAR": ps.loc["PatchTST-SSL(HoW-cond)", "false_alarm_rate"].mean(),
        "HOWFAR": ps.loc["HOW-profile(HoW-cond)", "false_alarm_rate"].mean(),
        # paired contrast (diff = PatchTST - HOW, fixed order)
        "PatchTSTHOWDiff": float(key["mean_diff"]),
        "DiffCIlo": float(key["ci90_lo"]),
        "DiffCIhi": float(key["ci90_hi"]),
        "DiffTOSTp": float(key["tost_p"]),
        "DiffHolmP": float(pv[pv["comparison"].str.contains(
            "PatchTST-SSL\(HoW-cond\) - HOW-profile\(HoW-cond\)")]["holm_p"].iloc[0]),
        "DiffSlot": float(key_slot["mean_diff"]),
        "PairedN": int(key["n"]),
        # leakage
        "LeakDeltaF": leak_delta,
        # waste
        "OpsBiasPatchTST": float(ops_bias.get("PatchTST-SSL(HoW-cond)", np.nan)),
        "OpsBiasHOW": float(ops_bias.get("HOW-profile(HoW-cond)", np.nan)),
        "OpsBiasTOWT": float(ops_bias.get("TOWT(HoW-cond)", np.nan)),
        "PresRulePatchTST": pres_rule.get("PatchTST-SSL(HoW-cond)", "z6"),
        "PresRuleHOW": pres_rule.get("HOW-profile(HoW-cond)", "z6"),
        "PresBiasPatchTST": float(pres_bias.get("PatchTST-SSL(HoW-cond)", np.nan)),
        "PresBiasHOW": float(pres_bias.get("HOW-profile(HoW-cond)", np.nan)),
        "PresRuleTOWT": pres_rule.get("TOWT(HoW-cond)", "z6"),
        "PresBiasTOWT": float(pres_bias.get("TOWT(HoW-cond)", np.nan)),
        "DiffSlotCIlo": float(key_slot["ci90_lo"]),
        "DiffSlotCIhi": float(key_slot["ci90_hi"]),
        "DiffSlotTOSTp": float(key_slot["tost_p"]),
        "HOWSlotRF": ps.loc["HOW-profile+z(slot)", "range_f1"].mean(),
        "ErrQOne": float(pb.quantile(0.25)),
        "ErrQMed": float(pb.median()),
        "ErrQThree": float(pb.quantile(0.75)),
        "OracleSignedBiasPatchTST": float(orc_signed.get("PatchTST-SSL(HoW-cond)", np.nan)),
        # percent macros rounded in Python so text and tables cannot diverge
        "OracleClipBiasPatchTSTPct": round(100 * float(orc_clip.get("PatchTST-SSL(HoW-cond)", np.nan)), 1),
        "OracleClipBiasPatchTST": float(orc_clip.get("PatchTST-SSL(HoW-cond)", np.nan)),
        "OracleFloorBiasPatchTST": float(orc_floor.get("PatchTST-SSL(HoW-cond)", np.nan)),
        "OracleSignedBiasHOW": float(orc_signed.get("HOW-profile(HoW-cond)", np.nan)),
        "OracleClipBiasHOW": float(orc_clip.get("HOW-profile(HoW-cond)", np.nan)),
        "OracleFloorBiasHOW": float(orc_floor.get("HOW-profile(HoW-cond)", np.nan)),
        "OracleSignedBiasTOWT": float(orc_signed.get("TOWT(HoW-cond)", np.nan)),
        "OracleClipBiasTOWT": float(orc_clip.get("TOWT(HoW-cond)", np.nan)),
        "OracleFloorBiasTOWT": float(orc_floor.get("TOWT(HoW-cond)", np.nan)),
        "LevBiasFull": float(lev_full.mean()),
        "LevBiasTarget": float(lev_tm.mean()),
        "LevBiasFullKwh": float(np.average(lev_full, weights=lev_w)),
        "LevBiasHOW": float(lev["how_mean"].mean()),
        "LevBiasSpike": float(lev[lev["type"] == "spike"]["patchtst_fullmask_mean"].mean()),
        "LevBiasDrift": float(lev[lev["type"] == "drift"]["patchtst_fullmask_mean"].mean()),
        "LevBiasSchedule": float(lev[lev["type"] == "schedule"]["patchtst_fullmask_mean"].mean()),
        # E3: gross error before/after window extension (TWh-scale kWh -> M)
        "GrossBeforePatchTST": float(gross_before["PatchTST-SSL(HoW-cond)"]),
        "GrossAfterPatchTST": float(gross_after["PatchTST-SSL(HoW-cond)"]),
        "GrossReductionPct": float(100 * (1 - gross_after["PatchTST-SSL(HoW-cond)"]
                                          / gross_before["PatchTST-SSL(HoW-cond)"])),
        "GrossBeforeHOW": float(gross_before["HOW-profile(HoW-cond)"]),
        "GrossAfterHOW": float(gross_after["HOW-profile(HoW-cond)"]),
        "GrossBeforeTOWT": float(gross_before["TOWT(HoW-cond)"]),
        "GrossAfterTOWT": float(gross_after["TOWT(HoW-cond)"]),
        # per-building |bias| quartiles BEFORE extension (ours)
        "ErrQOneBefore": float(pb0.quantile(0.25)),
        "ErrQMedBefore": float(pb0.median()),
        "ErrQThreeBefore": float(pb0.quantile(0.75)),
        # E4: HOW-profile per-type residual levels (sigma)
        "LevBiasHOWSpike": float(lev[lev["type"] == "spike"]["how_mean"].mean()),
        "LevBiasHOWDrift": float(lev[lev["type"] == "drift"]["how_mean"].mean()),
        "LevBiasHOWSchedule": float(lev[lev["type"] == "schedule"]["how_mean"].mean()),
        # E4: kWh-weighted schedule residual (learned backbone)
        "LevBiasScheduleKwh": float(np.average(
            lev[lev["type"] == "schedule"]["patchtst_fullmask_mean"],
            weights=lev[lev["type"] == "schedule"]["true_kwh"])),
        # oracle per-event median |err| (clip), for Limitations
        "OracleMedErrPatchTST": float(orc_med.get("PatchTST-SSL(HoW-cond)", np.nan)),
        "OracleMedErrHOW": float(orc_med.get("HOW-profile(HoW-cond)", np.nan)),
        "OracleMedErrTOWT": float(orc_med.get("TOWT(HoW-cond)", np.nan)),
        # LEAD appendix
        "LeadMedianDur": float(lead["duration_h"].median()),
        "LeadQ九十Dur": float(lead["duration_h"].quantile(0.90)),
        "LeadLabels": int(len(lead)),
        "LeadPctLeTwentyFour": float((lead["duration_h"] <= 24).mean()),
        "NBuildings": int(det["building_id"].nunique()),
        "NEventsPerBuilding": 6,
    }
    # sanitize non-ascii keys
    N["LeadQninetyDur"] = N.pop("LeadQ九十Dur")
    out = ["%% AUTO-GENERATED by scripts/make_tables.py from",
           "%% results/final_consolidated/*.csv (canonical run). Do not edit.",
           "\\makeatletter\\@ifundefined{r@numbersautogen}{}{}\\makeatother"]
    for k, v in N.items():
        if isinstance(v, float):
            out.append(f"\\newcommand{{\\{k}}}{{{v:.4f}}}" if abs(v) < 100 else
                       f"\\newcommand{{\\{k}}}{{{v:.0f}}}")
        else:
            out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    open(os.path.join(ROOT, "paper", "latex", "numbers.tex"), "w").write(
        "\n".join(out) + "\n")
    print(f"wrote {len(N)} macros, tables in {TAB}")


if __name__ == "__main__":
    table_detection()
    table_paired()
    table_waste_ops()
    table_prescription()
    table_levelbias()
    table_waste_oracle()
    table_synthetic()
    write_numbers()
