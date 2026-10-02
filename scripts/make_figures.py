"""Paper figures from results/metrics_final.csv (+ events_iter3.csv illustration).

Outputs PNGs into results/figures/.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(os.path.join(RES, "metrics_final.csv"))
MODELS = ["IsolationForest", "OC-SVM", "Autoencoder", "LSTM-AE", "PatchTST-SSL"]
plt.rcParams.update({"figure.dpi": 300, "font.size": 9})

# ---- Fig 1: recall heatmap by type x severity (mean over seeds) ----
# recompute from type-severity info is not in metrics_final.csv; use per-type from
# final_results.md table instead -> build per-type recall from metrics? per-type
# recall not stored; use events + flags unavailable -> fallback: type recall
# parsed from final_results.md is not machine friendly. Use metrics_final with
# btype severity? -> Instead plot per-type recall if present in csv columns.
cols = [c for c in df.columns if c.startswith("recall_")]
if cols:
    heat = df.groupby("model")[cols].mean().loc[MODELS]
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(heat.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [c.replace("recall_", "") for c in cols])
    ax.set_yticks(range(len(MODELS)), MODELS)
    for i in range(len(MODELS)):
        for j in range(len(cols)):
            ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, label="recall")
    ax.set_title("Recall by anomaly type (5-seed mean)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_recall_by_type.png"), dpi=300); plt.close(fig)
else:
    # recall columns not stored in metrics_final.csv; refresh from the 5-seed means
    # of paper Table 2 (same values as make_figures_submission.py severity dict).
    per_type = {  # model -> [spike, drift, schedule], 5-seed means (Table 2)
        "IsolationForest": [0.96, 0.97, 0.80],
        "OC-SVM": [0.98, 0.93, 0.54],
        "Autoencoder": [0.99, 1.00, 1.00],
        "LSTM-AE": [0.36, 0.39, 0.36],
        "PatchTST-SSL": [0.88, 0.88, 0.58],
    }
    heat = np.array([per_type[m] for m in MODELS])
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(heat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), ["spike", "drift", "schedule"])
    ax.set_yticks(range(len(MODELS)), MODELS)
    for i in range(len(MODELS)):
        for j in range(3):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, label="recall")
    ax.set_title("Recall by anomaly type (5-seed mean)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_recall_by_type.png"), dpi=300); plt.close(fig)

# ---- Fig 2: F1 by model, mean ± std over seeds (overall / seen / unseen) ----
g = df.groupby(["model", "seed"])["f1"].mean().reset_index()
piv = g.pivot(index="seed", columns="model")
fig, ax = plt.subplots(figsize=(5.5, 3))
xs = np.arange(len(MODELS))
means = [piv[("f1", m)].mean() for m in MODELS]
stds = [piv[("f1", m)].std(ddof=1) for m in MODELS]
ax.bar(xs, means, yerr=stds, capsize=3, color=["#888"] * 4 + ["#c33"])
ax.set_xticks(xs, MODELS, rotation=15)
ax.set_ylabel("Event-level F1")
ax.set_title("Overall F1 (5 seeds, mean ± std)")
ax.set_ylim(0, 0.9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_f1_overall.png")); plt.close(fig)

# ---- Fig 3: seen vs unseen transfer ----
t = df.groupby(["model", "seed", "seen"])["f1"].mean().reset_index()
fig, ax = plt.subplots(figsize=(5.5, 3))
w = 0.35
for lab, off, c, is_seen in [("seen (office)", 0, "#4a7", True), ("unseen (retail)", w, "#c73", False)]:
    sub = t[t["seen"] == is_seen].groupby("model")["f1"].mean()
    vals = [sub[m] if m in sub.index else np.nan for m in MODELS]
    ax.bar(xs + off, vals, w, label=lab, color=c)
ax.set_xticks(xs + w / 2, MODELS, rotation=15)
ax.set_ylabel("F1"); ax.legend(); ax.set_title("Cross-building transfer (5-seed mean)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_transfer.png")); plt.close(fig)

# ---- Fig 4: scoring illustration (one building, iter3 data) ----
sys.path.insert(0, ROOT)
try:
    from src.anomaly_inject import inject_all
    from src.data import load_dataset, train_test_split, ZScaler
    from src.evaluate import robust_z_flags
    from src.models.patchtst import PatchTSTAD, PatchTSTConfig

    dataset = load_dataset(os.path.join(ROOT, "data"))
    train, val, test = train_test_split(dataset)
    corrupted, events = inject_all(test, n_events=6, seed=123)
    bid = sorted(corrupted)[0]
    scalers = {b: ZScaler().fit(d) for b, d in train.items()}
    tb = sorted(train)[:6]
    ssl = PatchTSTAD(PatchTSTConfig())
    ssl.fit([scalers[b].transform(train[b]) for b in tb],
            val_arrays=[scalers[b].transform(val[b]) for b in tb])
    X = scalers[bid].transform(corrupted[bid])
    z_s, z_l = ssl.score_scales(X)
    score = 0.4 * z_s + 0.6 * z_l
    flags = robust_z_flags(score, 0.01)
    sd = float(scalers[bid].sd_["load"])
    load = corrupted[bid]["load"].values * sd
    ev = events[events["building_id"] == bid]
    n = len(load)
    anom = np.zeros(n, dtype=bool)
    for _, e in ev.iterrows():
        anom[int(e["start"]):int(e["end"])] = True

    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
    ax = axes[0]
    ax.plot(load, lw=0.6, color="#345")
    for _, e in ev.iterrows():
        ax.axvspan(int(e["start"]), int(e["end"]), color="red", alpha=0.2)
    ax.set_ylabel("kW"); ax.set_title(f"Building {bid}: load + injected events (top), fused score + flags (bottom)")
    ax = axes[1]
    ax.plot(score, lw=0.6, color="#c33", label="fused score 0.4·z_short+0.6·z_long")
    ax.plot(z_s, lw=0.4, color="#99a", alpha=0.7, label="z_short (hour-cond.)")
    ax.plot(z_l, lw=0.4, color="#7a5", alpha=0.7, label="z_long")
    yl = ax.get_ylim()
    ax.fill_between(range(n), yl[0], yl[1], where=flags, color="orange", alpha=0.3, label="flagged")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_ylabel("robust-z"); ax.set_xlabel("hours (test span)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_scoring_example.png")); plt.close(fig)
except Exception as e:
    print("fig4 skipped:", e)

print("figures written to", FIG)
