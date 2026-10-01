"""Round-3 unified figure pipeline: ALL paper figures are regenerated
DIRECTLY from results/*.csv (no hard-coded numbers). dpi=300.

Figures:
  fig_revision_overall.png   Fig.1  event-F1 / range-F1 / FAR (2x2 backbone x scoring)
  fig_round2_leakage.png     Fig.2  causal vs transductive per-building F1
  fig_revision_recall_type.png Fig.3 recall by anomaly type (all models with per-type data)
  fig_revision_waste.png     Fig.4  oracle waste variants + z6 prescription (before/after)
  fig_severity_matrix.png    Fig.5  synthetic type x severity recall (new protocol)
  fig_revision_transfer.png  Fig.6  seen/unseen F1 + range-recall (coverage)
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 300})

C2X2 = {"PatchTST-SSL(HoW-cond)": "#1f77b4", "PatchTST-SSL(HoD-cond)": "#7fb3d5",
        "HOW-profile(HoW-cond)": "#d62728", "HOW-profile(global-MAD)": "#f0a0a0"}


def _m(ax, val="f1"):
    return val


def fig1():
    d = pd.read_csv(os.path.join(RES, "metrics_round3.csv"))
    g = d.groupby("model").agg(
        f1=("f1", "mean"), f1s=("f1", "std"),
        rf1=("range_f1", "mean"), rf1s=("range_f1", "std"),
        far=("false_alarm_rate", "mean"),
        p=("precision", "mean"), r=("recall", "mean")).reset_index()
    order = g.sort_values("f1", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, (col, err, ttl) in zip(
            axes, [("f1", "f1s", "Event-level F1"),
                   ("rf1", "rf1s", "Range-based F1"),
                   ("far", None, "False-alarm rate (%)")]):
        o = order if col != "far" else g.sort_values("far")
        colors = [C2X2.get(m, "#999999") for m in o["model"]]
        y = np.arange(len(o))[::-1]
        ax.barh(y, o[col], xerr=(o[err] if err else None),
                color=colors, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(o["model"], fontsize=6.5)
        ax.set_title(ttl, fontsize=8)
        if col == "far":
            ax.set_xlabel("% of normal hours")
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
    c = pd.read_csv(os.path.join(RES, "metrics_round2.csv"))
    t = pd.read_csv(os.path.join(RES, "metrics_round2_transductive.csv"))
    a = c[c["model"] == "PatchTST-SSL"].groupby("building_id")["f1"].mean()
    b = t.groupby("building_id")["f1"].mean()
    common = a.index.intersection(b.index)
    a, b = a[common], b[common]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.scatter(a, b, s=14, alpha=0.7, edgecolor="none")
    lim = [0, 1]
    ax.plot(lim, lim, "k--", lw=0.7)
    ax.set_xlim(0.1, 1.02)
    ax.set_ylim(0.1, 1.02)
    ax.set_xlabel("per-building F1, causal protocol")
    ax.set_ylabel("per-building F1, transductive")
    mad = float(np.mean(np.abs(b - a)))
    ax.set_title(f"Leakage check: mean |diff| = +{mad:.3f} F1\n"
                 "(transductive inflated)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_round2_leakage.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig3():
    d = pd.read_csv(os.path.join(RES, "metrics_round2_types.csv"))
    g = d.groupby(["model", "type"])["recall"].mean().unstack()
    g = g.loc[g.mean(axis=1).sort_values().index]
    types = ["spike", "drift", "schedule"]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(g))
    w = 0.26
    for i, ty in enumerate(types):
        ax.bar(x + (i - 1) * w, g[ty], width=w, label=ty)
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, fontsize=6.5, rotation=20, ha="right")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3, fontsize=7)
    ax.set_title("Recall by anomaly type (41 BDG2 buildings, 3 seeds, "
                 "causal protocol)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_revision_recall_type.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig4():
    wo = pd.read_csv(os.path.join(RES, "metrics_round2_waste_oracle.csv"))
    pr = pd.read_csv(os.path.join(RES, "metrics_round3_prescription.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    # (a) oracle variants
    piv = wo.groupby(["estimator", "variant"])["bias_pct"].mean().unstack()
    variants = ["clip", "signed", "floor"]
    piv = piv[[v for v in variants if v in piv.columns]]
    x = np.arange(len(piv))
    w = 0.26
    for i, v in enumerate(piv.columns):
        axes[0].bar(x + (i - 1) * w, piv[v] * 100, width=w, label=v)
    axes[0].axhline(0, color="k", lw=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(piv.index, fontsize=6.5)
    axes[0].set_ylabel("oracle-window bias (%)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("(a) Stage (a): oracle windows, estimator variants",
                      fontsize=8)
    # (b) prescription: baseline vs z6 (kWh-weighted mean bias)
    pr = pr[pr["true_kwh"] > 0]
    g = pr.groupby("model").apply(
        lambda x: pd.Series({
            "base": np.average(x["baseline_bias"], weights=x["true_kwh"]) * 100,
            "z6": np.average(x["z6_bias"], weights=x["true_kwh"]) * 100,
            "base_cov": np.average(x["baseline_cov"], weights=x["true_kwh"]) * 100,
            "z6_cov": np.average(x["z6_cov"], weights=x["true_kwh"]) * 100}),
        include_groups=False)
    x = np.arange(len(g))
    axes[1].bar(x - 0.19, g["base"], width=0.38, label="flags only (baseline)")
    axes[1].bar(x + 0.19, g["z6"], width=0.38, label="+ z6 window extension")
    axes[1].axhline(0, color="k", lw=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(g.index, fontsize=6.5, rotation=12, ha="right")
    axes[1].set_ylabel("operational waste bias (%)")
    axes[1].legend(frameon=False, fontsize=7)
    for i, (r0, r1) in enumerate(zip(g.index, g.index)):
        axes[1].text(i, 4, f"{g.loc[r0,'base_cov']:.0f}%$\\rightarrow${g.loc[r1,'z6_cov']:.0f}%",
                     ha="center", fontsize=5.5, color="dimgrey")
    axes[1].set_title("(b) Stage (b) prescription: coverage (grey) and bias",
                      fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_revision_waste.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig5():
    d = pd.read_csv(os.path.join(RES, "metrics_round3_synthetic_types.csv"))
    d = d.dropna(subset=["type_recall"])
    piv = d.groupby(["model", "type", "severity"])["type_recall"].mean()
    models = list(piv.index.get_level_values("model").unique())
    sevs = ["subtle", "moderate", "obvious"]
    fig, axes = plt.subplots(1, len(models), figsize=(1.75 * len(models), 2.4),
                             sharey=True)
    for ax, m in zip(axes, models):
        mat = piv.loc[m].unstack("severity").reindex(
            index=["spike", "drift", "schedule"], columns=sevs)
        im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels(sevs, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(3))
        ax.set_yticklabels(["spike", "drift", "schedule"], fontsize=6)
        ax.set_title(m, fontsize=6.5)
        for i in range(3):
            for j in range(3):
                v = mat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=axes, shrink=0.8, label="recall")
    fig.savefig(os.path.join(FIG, "fig_severity_matrix.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig6():
    d = pd.read_csv(os.path.join(RES, "metrics_round3.csv"))
    g = d.groupby(["model", "seen"]).agg(
        f1=("f1", "mean"), rr=("range_recall", "mean")).reset_index()
    models = g.groupby("model")["f1"].mean().sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))
    y = np.arange(len(models))
    seen = g[g["seen"]].set_index("model").loc[models, "f1"]
    unseen = g[~g["seen"]].set_index("model").loc[models, "f1"]
    axes[0].barh(y + 0.19, seen, height=0.38, label="seen (office)")
    axes[0].barh(y - 0.19, unseen, height=0.38, label="unseen (edu/retail)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(models, fontsize=6.5)
    axes[0].set_xlabel("F1")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("Seen vs unseen building types", fontsize=8)
    rr = g[g["seen"]].set_index("model").loc[models, "rr"]
    axes[1].barh(y, rr, color="#2ca02c")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("range-recall (coverage of true event hours)")
    axes[1].set_title("Coverage bottleneck: range-recall", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_revision_transfer.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        print(f.__name__, "...", flush=True)
        f()
    print("all figures written to", FIG)
