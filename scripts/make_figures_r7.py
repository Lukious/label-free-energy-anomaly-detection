"""Round-7 figures, generated only from results/final_r7/*.csv (dpi 300).

Palette: reference categorical slots 1-3 (blue / orange / aqua), validated
all-pairs; generic detectors in neutral gray. Scoring level is encoded by
fill vs. hatch, so identity never depends on colour alone.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "final_r7")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})

PT, PTG, PTD = "PatchTST-SSL(HoW-cond)", "PatchTST-SSL(global-MAD)", "PatchTST-SSL(HoD-cond)"
HW, HWG, HWD = "HOW-profile(HoW-cond)", "HOW-profile(global-MAD)", "HOW-profile(HoD-cond)"
PM, TW, SL = "PatchTST-SSL(HoW-cond, patch-mask train)", "TOWT(HoW-cond)", "HOW-profile+z(slot)"
ORDER = [(PT, "PatchTST + HoW (ours)", BLUE, ""), (PTD, "PatchTST + HoD", BLUE, ".."),
         (PTG, "PatchTST + global", BLUE, "//"), (PM, "PatchTST, patch-mask train", BLUE, "xx"),
         (HW, "HOW-residual + HoW", ORANGE, ""), (HWD, "HOW-residual + HoD", ORANGE, ".."),
         (HWG, "HOW-residual + global", ORANGE, "//"), (SL, "slotwise HOW-z", ORANGE, "\\\\"),
         (TW, "TOWT-residual + HoW", AQUA, ""),
         ("IsolationForest(t)", "Isolation Forest", GRAY, ""), ("OC-SVM(t)", "OC-SVM", GRAY, ""),
         ("Autoencoder(t)", "Dense AE", GRAY, ""), ("LSTM-AE(t)", "LSTM-AE", GRAY, "")]


def load(n):
    p = os.path.join(MC, n)
    return pd.read_csv(p) if os.path.exists(p) else None


def hbar(ax, vals, errs, title, xlabel, fmt="{:.3f}"):
    y = np.arange(len(ORDER))[::-1]
    for yi, (m, lab, c, h), v, e in zip(y, ORDER, vals, errs):
        ax.barh(yi, v, height=0.62, color=c if not h else "white", edgecolor=c,
                hatch=h, linewidth=1.0)
        ax.errorbar(v, yi, xerr=e, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
        ax.text(v + (e if np.isfinite(e) else 0) + 0.004 * ax.get_xlim()[1], yi, fmt.format(v),
                va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def fig_overall(det):
    ps = det.groupby(["model", "seed"])[["f1", "range_f1", "false_alarm_rate"]].mean()
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 4.6), sharey=True)
    for ax, col, title, xl, k in [(axs[0], "f1", "Event F1", "F1", 1),
                                  (axs[1], "range_f1", "Range F1", "F1", 1),
                                  (axs[2], "false_alarm_rate", "Non-injected alarm rate", "% of non-injected hours", 100)]:
        v = [k * ps.loc[m][col].mean() for m, *_ in ORDER]
        e = [k * ps.loc[m][col].std(ddof=1) for m, *_ in ORDER]
        ax.set_xlim(0, max(v) * 1.22)
        hbar(ax, v, e, title, xl, "{:.1f}" if k == 100 else "{:.3f}")
    axs[0].set_yticklabels([lab for _, lab, *_ in ORDER])
    fig.text(0.01, -0.02, "Bars: mean over 3 seeds of the building macro-average; whiskers: seed SD. "
             "Blue = PatchTST backbone, orange = HOW profile, aqua = TOWT, gray = generic detectors; "
             "fill = HoW scoring, dots = HoD, hatching = global.", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_r7_overall.png"))
    plt.close(fig)


def fig_factorial(det):
    bl = det.groupby(["model", "building_id"])["f1"].mean().unstack(0)
    bl = bl[[PT, PTG, HW, HWG]].dropna()
    n = len(bl)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
    x = np.array([0, 1])
    for (g, h), c, lab, mk in [((HWG, HW), ORANGE, "HOW profile", "s"), ((PTG, PT), BLUE, "PatchTST", "o")]:
        m = [bl[g].mean(), bl[h].mean()]
        ci = [1.96 * bl[g].std(ddof=1) / np.sqrt(n), 1.96 * bl[h].std(ddof=1) / np.sqrt(n)]
        a.errorbar(x, m, yerr=ci, color=c, marker=mk, ms=7, lw=2, capsize=3,
                   markeredgecolor="white", markeredgewidth=1.5)
        a.text(1.06, m[1], lab, color=INK, va="center", fontsize=8)
    a.set_xticks(x); a.set_xticklabels(["global scoring", "HoW-conditional"])
    a.set_xlim(-0.25, 1.55)
    a.set_ylabel("event F1 (building mean, 95% CI)")
    a.set_title("(a) 2×2 cell means", loc="left", fontsize=9)
    a.grid(axis="x", visible=False)
    d = (bl[PT] - bl[HW]).rename("d").to_frame()
    d["site"] = d.index.str.split("_").str[0]
    sites = d.groupby("site")["d"].agg(["mean", "size"]).sort_values("size", ascending=False)
    for i, s in enumerate(sites.index):
        v = d[d["site"] == s]["d"].to_numpy()
        jit = np.linspace(-0.18, 0.18, len(v)) if len(v) > 1 else np.zeros(1)
        b.scatter(np.full(len(v), i) + jit, v, s=16, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.6, zorder=3)
        b.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=INK, lw=1.6, zorder=4)
    b.axhline(0, color=INK2, lw=0.8)
    b.axhline(d["d"].mean(), color=BLUE, lw=1, ls="--")
    b.text(len(sites) - 0.5, d["d"].mean(), f" all-building mean {d['d'].mean():+.3f}", color=INK2,
           fontsize=7, va="bottom", ha="right")
    b.set_xticks(range(len(sites)))
    b.set_xticklabels([f"{s}\n(n={int(sites.loc[s, 'size'])})" for s in sites.index], fontsize=7)
    b.set_ylabel("PatchTST − HOW, event F1\n(both with HoW scoring)")
    b.set_title("(b) backbone effect per building, grouped by site (bar = site mean)", loc="left", fontsize=9)
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_r7_factorial.png"))
    plt.close(fig)


def fig_extension(wci):
    modes = ["none", "z6", "z12", "cusum"]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    ext = load("extension.csv")
    for ax, (m, lab, c) in zip(axs, [(PT, "PatchTST", BLUE), (HW, "HOW profile", ORANGE), (TW, "TOWT", AQUA)]):
        ch = ext[(ext["model"] == m) & ext["chosen"]].groupby("seed")["mode"].first().value_counts()
        for i, mode in enumerate(modes):
            g = wci[(wci["model"] == m) & (wci["estimator"] == f"ext:{mode}")].set_index("stat")["value"]
            ax.bar(i, 100 * g["pb_gross_rel"], width=0.6, color="white", edgecolor=c, lw=1.2, hatch="//")
            ax.scatter(i, 100 * g["agg_bias"], s=40, color=c, edgecolor="white", linewidth=1.5, zorder=4)
            if mode in ch.index:
                ax.text(i, 100 * g["pb_gross_rel"] + 6, f"chosen {ch[mode]}/3", ha="center", fontsize=7, color=INK)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(4)); ax.set_xticklabels(modes)
        ax.set_title(lab, loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("% of true injected kWh (test)")
    fig.text(0.01, -0.05, "Hatched bar: per-building gross error (no cancellation between buildings or error terms). "
             "Dot: aggregate signed bias (cancellation allowed). 'chosen k/3': seeds in which validation selected the rule.",
             fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_r7_extension.png"))
    plt.close(fig)


def fig_sm(leak, det, inj, synevd):
    if leak is not None:
        c = det[det["model"] == PT].groupby("building_id")["f1"].mean()
        t = leak.groupby("building_id")["f1"].mean()
        k = c.index.intersection(t.index)
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        ax.scatter(c[k], t[k], s=18, color=BLUE, edgecolor="white", linewidth=0.6)
        lo, hi = min(c[k].min(), t[k].min()) - 0.02, max(c[k].max(), t[k].max()) + 0.02
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
        ax.set_xlabel("forward-only protocol, F1"); ax.set_ylabel("transductive variant, F1")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_r7_leakage.png")); plt.close(fig)
    if inj is not None:
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        pos, labs = [], []
        for i, t in enumerate(["spike", "drift", "schedule"]):
            for j, s in enumerate((1, 2, 3)):
                v = inj[(inj["type"] == t) & (inj["severity"] == s)]["mean_add_over_resid_sd"].to_numpy()
                p = i * 4 + j
                if len(v):
                    ax.boxplot([np.clip(v, 1e-2, None)], positions=[p], widths=0.6, showfliers=False,
                               patch_artist=True, boxprops=dict(facecolor="white", edgecolor=BLUE),
                               medianprops=dict(color=INK), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
                pos.append(p); labs.append(f"{t}\n{['subtle', 'moderate', 'obvious'][j]}\n(n={len(v)})")
        ax.set_yscale("log")
        ax.axhline(2.33, color=ORANGE, lw=1, ls="--")
        ax.text(pos[-1] + 0.5, 2.33, " flagger threshold (z = 2.33)", color=INK2, fontsize=7, va="bottom", ha="right")
        ax.set_xticks(pos); ax.set_xticklabels(labs, fontsize=6.5)
        ax.set_ylabel("mean injected increment /\nresidual robust SD")
        ax.grid(axis="x", visible=False)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_r7_injection.png")); plt.close(fig)
    if synevd is not None:
        models = [(PT, "PatchTST+HoW"), (PTD, "PatchTST+HoD"), (HW, "HOW+HoW"), (TW, "TOWT+HoW"),
                  ("IsolationForest(t)", "IF"), ("OC-SVM(t)", "OC-SVM"), ("Autoencoder(t)", "AE"), ("LSTM-AE(t)", "LSTM-AE")]
        cells = [(t, s) for t in ("spike", "drift", "schedule") for s in (1, 2, 3)]
        Mx = np.full((len(models), len(cells)), np.nan)
        for i, (m, _) in enumerate(models):
            g = synevd[synevd["model"] == m].groupby(["type", "severity"])["detected"].mean()
            for j, c in enumerate(cells):
                Mx[i, j] = g.get(c, np.nan)
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        im = ax.imshow(Mx, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                if np.isfinite(Mx[i, j]):
                    ax.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if Mx[i, j] > 0.6 else INK)
        ax.set_yticks(range(len(models))); ax.set_yticklabels([m[1] for m in models])
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([f"{t}\n{['subtle', 'mod.', 'obvious'][s - 1]}" for t, s in cells], fontsize=6.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="recall", shrink=0.8)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_r7_severity_synthetic.png")); plt.close(fig)


def main():
    det = load("detection_metrics.csv")
    fig_overall(det)
    fig_factorial(det)
    fig_extension(load("waste_ci.csv"))
    fig_sm(load("leakage.csv"), det, load("injection_realism.csv"), load("synthetic_event_detail.csv"))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
