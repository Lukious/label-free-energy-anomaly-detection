"""Canonical figure pipeline (Round-4 B1): ALL paper figures regenerated
DIRECTLY from results/final_consolidated/*.csv (the single canonical run).
No legacy result files are read; no hard-coded numbers. dpi=300.

Figures (filenames kept stable so paper assets are unchanged):
  fig_revision_overall.png    Fig.1  event-F1 / range-F1 / FAR (2x2 + all)
  fig_round2_leakage.png      Fig.2  causal vs transductive per-building F1
  fig_revision_recall_type.png Fig.3 recall by anomaly type (real data)
  fig_revision_waste.png      Fig.5b operational waste bias + validation-
                              chosen prescription (before/after), from the
                              SAME master CSVs as the tables (B1 fix)
  fig_severity_matrix.png     Fig.5  synthetic type x severity recall
  fig_revision_transfer.png   Fig.6  seen/unseen F1 + range-recall
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MC = os.path.join(ROOT, "results", "final_consolidated")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 300})

C2X2 = {"PatchTST-SSL(HoW-cond)": "#1f77b4", "PatchTST-SSL(HoD-cond)": "#7fb3d5",
        "HOW-profile(HoW-cond)": "#d62728", "HOW-profile(global-MAD)": "#f0a0a0"}


def load(name):
    return pd.read_csv(os.path.join(MC, name))


def fig1():
    d = load("detection_metrics.csv")
    g = d.groupby("model").agg(
        f1=("f1", "mean"), f1s=("f1", "std"),
        rf1=("range_f1", "mean"), rf1s=("range_f1", "std"),
        far=("false_alarm_rate", "mean")).reset_index()
    order = g.sort_values("f1", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, (col, err, ttl) in zip(
            axes, [("f1", "f1s", "Event-level F1"),
                   ("rf1", "rf1s", "Range-based F1"),
                   ("far", None, "False-alarm rate (%)")]):
        o = order if col != "far" else g.assign(_f=g["far"] * 100).sort_values("_f")
        colors = [C2X2.get(m, "#999999") for m in o["model"]]
        y = np.arange(len(o))[::-1]
        vals = o["_f"] if col == "far" else o[col]
        errs = None if (err is None or col == "far") else o[err]
        ax.barh(y, vals, xerr=errs, color=colors, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(o["model"], fontsize=7.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(ttl, fontsize=9)
        if col == "far":
            ax.set_xlabel("false-alarm rate (% of normal hours)", fontsize=8)
        else:
            ax.set_xlim(0, max(0.75, o[col].max() * 1.15))
    axes[0].set_xlabel("building macro-avg F1")
    axes[1].set_xlabel("building macro-avg range-F1")
    fig.suptitle("Backbone x scoring 2x2 (blue=PatchTST, red=HOW-profile; "
                 "dark=HoW-conditional, light=non-HoW) and all baselines",
                 fontsize=8, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_revision_overall.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig2():
    det = load("detection_metrics.csv")
    leak = load("leakage.csv")
    a = det[det["model"] == "PatchTST-SSL(HoD-cond)"].groupby("building_id")["f1"].mean()
    b = leak.groupby("building_id")["f1"].mean()
    common = a.index.intersection(b.index)
    a, b = a[common], b[common]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.scatter(a, b, s=14, alpha=0.7, color="#333")
    lims = [min(a.min(), b.min()) - 0.05, max(a.max(), b.max()) + 0.05]
    ax.plot(lims, lims, "k--", lw=0.8)
    ax.set_xlabel("train-frozen causal scoring: F1")
    ax.set_ylabel("transductive (full-series stats): F1")
    ax.set_title(f"Normalization leakage, same buildings/injections\n"
                 f"mean $\\Delta$F1 = {(b - a).mean():+.3f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_round2_leakage.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig3():
    d = load("detection_metrics.csv")
    if "type_recall" in d.columns:
        ty = d[d["type_recall"].notna()]
    else:
        ty = d.iloc[0:0]
    if len(ty):
        piv = ty.groupby(["model", "type"])["type_recall"].mean().unstack()
    else:
        piv = d.groupby(["model"])["recall"].mean().to_frame("overall")
    keep = [m for m in ["PatchTST-SSL(HoW-cond)", "PatchTST-SSL(HoD-cond)",
                        "HOW-profile(HoW-cond)", "TOWT(HoW-cond)",
                        "IsolationForest(t)", "OC-SVM(t)", "LSTM-AE(t)",
                        "Autoencoder(t)"] if m in piv.index]
    piv = piv.loc[keep]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    x = np.arange(len(piv.columns))
    w = 0.8 / max(len(piv.index), 1)
    for i, m in enumerate(piv.index):
        ax.bar(x + i * w - 0.4 + w / 2, piv.loc[m], width=w, label=m,
               color=C2X2.get(m, "#999999"))
    ax.set_xticks(x)
    ax.set_xticklabels(piv.columns)
    ax.set_ylabel("event recall")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    ax.set_title("Recall by anomaly type (real BDG-Pier, canonical run)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_revision_recall_type.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_waste():
    """Fig.5b — operational waste bias + validation-chosen prescription,
    from the SAME master CSVs as the waste tables (B1 figure-table fix)."""
    ops = load("waste_ops.csv")
    pres = load("prescription.csv")
    g = ops.groupby("model").agg(est=("est_kwh", "sum"), true=("true_kwh", "sum"))
    base = ((g["est"] - g["true"]) / g["true"]) * 100
    ch = pres[pres["chosen"]]
    pg = ch.groupby("model").agg(est=("est_kwh", "sum"), true=("true_kwh", "sum"),
                                 rule=("mode", "first"))
    after = ((pg["est"] - pg["true"]) / pg["true"]) * 100
    order = list(base.index)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(order))
    ax.bar(x - 0.2, [base[m] for m in order], width=0.4, label="flagged-hour clip",
           color="#888")
    ax.bar(x + 0.2, [after.get(m, np.nan) for m in order], width=0.4,
           label="validation-chosen window rule", color="#1f77b4")
    for i, m in enumerate(order):
        if m in pg.index:
            ax.annotate(pg.loc[m, "rule"], (i + 0.2, after[m]),
                        ha="center", va="bottom", fontsize=6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("(", "\n(") for m in order], fontsize=6.5)
    ax.set_ylabel("total waste-estimate bias (%)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Operational waste: baseline vs prescribed windows\n"
                 "(rule selected on validation injections; canonical run)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_revision_waste.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_severity():
    d = load("synthetic_metrics.csv")
    ty = d[d["type_recall"].notna()]
    piv = ty.groupby(["model", "type", "severity"])["type_recall"].mean().unstack(
        level=["type", "severity"])
    keep = [m for m in ["PatchTST-SSL(HoW-cond)", "PatchTST-SSL(HoD-cond)",
                        "HOW-profile(HoW-cond)", "TOWT(HoW-cond)",
                        "IsolationForest(t)", "LSTM-AE(t)"] if m in piv.index]
    piv = piv.loc[keep]
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{a}\n{b}" for a, b in piv.columns], fontsize=6)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=6.5)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center",
                    va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="recall")
    ax.set_title("Synthetic type x severity recall (new causal protocol)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_severity_matrix.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_transfer():
    d = load("detection_metrics.csv")
    g = d.groupby(["model", "seen"]).agg(
        f1=("f1", "mean"), rR=("range_recall", "mean")).reset_index()
    keep = ["PatchTST-SSL(HoW-cond)", "HOW-profile(HoW-cond)", "TOWT(HoW-cond)",
            "LSTM-AE(t)", "IsolationForest(t)"]
    g = g[g["model"].isin(keep)]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    for ax, (col, ttl) in zip(axes, [("f1", "event F1"),
                                     ("rR", "range-recall")]):
        piv = g.pivot(index="model", columns="seen", values=col).loc[
            [m for m in keep if m in g["model"].unique()]]
        x = np.arange(len(piv))
        ax.bar(x - 0.2, piv[True], width=0.4, label="seen (office)", color="#1f77b4")
        ax.bar(x + 0.2, piv[False], width=0.4, label="unseen", color="#f0a0a0")
        ax.set_xticks(x)
        ax.set_xticklabels([m.split("(")[0] for m in piv.index],
                           rotation=20, ha="right", fontsize=6.5)
        ax.set_ylabel(ttl)
        ax.legend(fontsize=6, frameon=False)
    fig.suptitle("Seen vs unseen building types (canonical run)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_revision_transfer.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_lead():
    """Supplementary S.5: LEAD 1.0 labelled-run duration distribution."""
    d = load("lead_label_durations.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.hist(np.minimum(d["duration_h"], 48), bins=48, color="#1f77b4")
    ax.axvline(d["duration_h"].median(), color="k", ls="--", lw=0.9,
               label=f"median {d['duration_h'].median():.0f} h")
    ax.set_xlabel("labelled-run duration (h, clipped at 48)")
    ax.set_ylabel("runs")
    ax.set_title(f"LEAD 1.0 labelled anomalous runs (n={len(d)}); "
                 f"95% <= {d['duration_h'].quantile(0.95):.0f} h", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_lead_duration.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig_waste(); fig_severity(); fig_transfer()
    fig_lead()
    print("all figures regenerated from results/final_consolidated/*.csv")
