"""Submission figures: severity heatmap, synthetic-vs-real bars, delay-FAR scatter.
Data: results/metrics_final.csv (synthetic 5-seed), results/metrics_realdata.csv (BDG2 3-seed),
and severity matrix from paper Table 2 (results/final_results.md).
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "results", "figures")
fin = pd.read_csv(os.path.join(BASE, "results", "metrics_final.csv"))
real = pd.read_csv(os.path.join(BASE, "results", "metrics_realdata.csv"))
models = ["IsolationForest", "OC-SVM", "Autoencoder", "LSTM-AE", "PatchTST"]
nice = {"IsolationForest": "IsolationForest", "OC-SVM": "OC-SVM",
        "Autoencoder": "Autoencoder", "LSTM-AE": "LSTM-AE",
        "PatchTST": "PatchTST-SSL (ours)"}

# --- Fig 5: severity-matrix heatmaps (values from Table 2, 5-seed means) ---
sev = {
    "IsolationForest": {"spike": [0.99, 0.92, 0.98], "drift": [0.90, 0.97, 0.98], "schedule": [1.00, 0.81, 0.76]},
    "OC-SVM":          {"spike": [0.97, 1.00, 1.00], "drift": [0.91, 0.95, 0.98], "schedule": [0.50, 0.37, 0.51]},
    "Autoencoder":     {"spike": [0.99, 1.00, 1.00], "drift": [1.00, 1.00, 1.00], "schedule": [1.00, 1.00, 1.00]},
    "LSTM-AE":         {"spike": [0.18, 0.39, 0.75], "drift": [0.41, 0.29, 0.48], "schedule": [0.67, 0.40, 0.31]},
    "PatchTST":        {"spike": [0.87, 0.76, 0.94], "drift": [0.59, 0.97, 0.98], "schedule": [0.00, 0.32, 0.76]},
}
fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
for ax, m in zip(axes, models):
    mat = np.array([sev[m][t] for t in ["spike", "drift", "schedule"]])
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["subtle", "moderate", "obvious"], fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels(["spike", "drift", "schedule"], fontsize=8)
    ax.set_title(nice[m], fontsize=9)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.suptitle("Recall by anomaly type x severity (synthetic, 5-seed mean)", fontsize=11)
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
fig.savefig(os.path.join(FIG, "fig_severity_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Fig 6: synthetic vs BDG2 grouped bars (F1 and FAR), computed from CSVs ---
def agg(df):
    g = df.groupby("model").agg(f1=("f1", "mean"), f1s=("f1", "std"),
                                far=("false_alarm_rate", "mean"), fars=("false_alarm_rate", "std"))
    return g.reindex(models)
a, b = agg(fin), agg(real)
x = np.arange(len(models)); w = 0.35
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.bar(x - w/2, a.f1, w, yerr=a.f1s, capsize=3, label="Synthetic (5 seeds)")
ax1.bar(x + w/2, b.f1, w, yerr=b.f1s, capsize=3, label="BDG2 real (3 seeds)")
ax1.set_xticks(x); ax1.set_xticklabels([nice[m].replace(" (ours)", "*") for m in models], rotation=20, fontsize=8)
ax1.set_ylabel("Event-level F1"); ax1.set_title("F1: synthetic vs real"); ax1.legend(fontsize=8); ax1.set_ylim(0, 0.9)
ax2.bar(x - w/2, a.far * 100, w, yerr=a.fars * 100, capsize=3, label="Synthetic (5 seeds)")
ax2.bar(x + w/2, b.far * 100, w, yerr=b.fars * 100, capsize=3, label="BDG2 real (3 seeds)")
ax2.set_xticks(x); ax2.set_xticklabels([nice[m].replace(" (ours)", "*") for m in models], rotation=20, fontsize=8)
ax2.set_ylabel("False-alarm rate (%)"); ax2.set_title("FAR: synthetic vs real"); ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_synthetic_vs_real.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Fig 7: delay-FAR scatter per model, both datasets ---
fig, ax = plt.subplots(figsize=(6.5, 4.5))
markers = {"Synthetic (5 seeds)": "o", "BDG2 real (3 seeds)": "s"}
for df, lab in [(fin, "Synthetic (5 seeds)"), (real, "BDG2 real (3 seeds)")]:
    g = df.groupby("model").agg(d=("detection_delay_h", "mean"), ds=("detection_delay_h", "std"),
                                f=("false_alarm_rate", "mean"), fs=("false_alarm_rate", "std")).reindex(models)
    for m in models:
        r = g.loc[m]
        ax.errorbar(r.f * 100, r.d, xerr=r.fs * 100, yerr=r.ds, fmt=markers[lab],
                    ms=8 if m == "PatchTST" else 6, capsize=2,
                    color="tab:red" if m == "PatchTST" else None,
                    markeredgecolor="k" if m == "PatchTST" else None, alpha=0.85)
        if lab.startswith("Synthetic"):
            ax.annotate(nice[m].replace(" (ours)", ""), (r.f * 100, r.d),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
# legend proxies
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", label="Synthetic (5 seeds)"),
                   Line2D([], [], marker="s", ls="", label="BDG2 real (3 seeds)")], fontsize=8)
ax.set_xlabel("False-alarm rate (%)"); ax.set_ylabel("Detection delay (h)")
ax.set_title("Detection delay vs false-alarm rate (mean $\\pm$ std)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_delay_far_scatter.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("done:", os.listdir(FIG))
