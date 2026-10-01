"""Revision figures (v4) from results/metrics_revision*.csv. dpi=300. Figure regeneration only."""
import pandas as pd, numpy as np, os, ast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "results", "figures")
df = pd.read_csv(os.path.join(BASE, "results", "metrics_revision.csv"))

models = ["IsolationForest(t)", "OC-SVM(t)", "Autoencoder(t)", "LSTM-AE(t)",
          "PatchTST-SSL(w-tuned)", "PatchTST-SSL(w=0.5 fixed)"]
nice = {"IsolationForest(t)": "IsolationForest(t)", "OC-SVM(t)": "OC-SVM(t)",
        "Autoencoder(t)": "Autoencoder(t)", "LSTM-AE(t)": "LSTM-AE(t)",
        "PatchTST-SSL(w-tuned)": "PatchTST-SSL (ours)", "PatchTST-SSL(w=0.5 fixed)": "PatchTST-SSL ($w{=}0.5$ fixed)"}
C = {"IsolationForest(t)": "#8c8c8c", "OC-SVM(t)": "#b07aa1", "Autoencoder(t)": "#c44e52",
     "LSTM-AE(t)": "#dd8452", "PatchTST-SSL(w-tuned)": "#4c72b0", "PatchTST-SSL(w=0.5 fixed)": "#9ecbdd"}

# --- Fig: overall F1 (real BDG2, 41 buildings) with precision/recall ---
g = df.groupby(["model", "seed"]).agg(p=("precision","mean"), r=("recall","mean"),
                                      f1=("f1","mean"), d=("detection_delay_h","mean"),
                                      far=("false_alarm_rate","mean")).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for m in models:
    s = g[g.model == m]
    x = nice[m]
    axes[0].bar(x, s.f1.mean(), yerr=s.f1.std(), color=C[m], capsize=3)
    axes[1].errorbar(s.far.mean()*100, s.d.mean(), xerr=s.far.std()*100, yerr=s.d.std(),
                     fmt="o", color=C[m], label=nice[m], capsize=3)
axes[0].set_ylabel("Event-level F1"); axes[0].tick_params(axis="x", rotation=35)
axes[0].set_title("Real BDG2 meters (41 buildings, 3 seeds)")
axes[1].set_xlabel("False-alarm rate (%)"); axes[1].set_ylabel("Detection delay (h)")
axes[1].legend(fontsize=7); axes[1].set_title("Delay vs false-alarm rate")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_revision_overall.png"), dpi=300); plt.close()

# --- Fig: recall by type ---
t = pd.read_csv(os.path.join(BASE, "results", "metrics_revision_types.csv"))
tt = t.groupby(["model","type","seed"]).recall.mean().reset_index()
tt = tt.groupby(["model","type"]).recall.agg(["mean","std"]).reset_index()
fig, ax = plt.subplots(figsize=(8.5, 4))
w_ = 0.12
for i, m in enumerate(models):
    sub = tt[tt.model == m].set_index("type").reindex(["spike","drift","schedule"])
    ax.bar(np.arange(3) + (i-2.5)*w_, sub["mean"], w_, yerr=sub["std"], label=nice[m],
           color=C[m], capsize=2)
ax.set_xticks(range(3)); ax.set_xticklabels(["spike","drift","schedule"])
ax.set_ylabel("Recall"); ax.legend(fontsize=7)
ax.set_title("Recall by anomaly type (real BDG2, 41 buildings)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_revision_recall_type.png"), dpi=300); plt.close()

# --- Fig: transfer seen vs unseen ---
fig, ax = plt.subplots(figsize=(7, 4))
for m in models:
    s = g.merge(df[["model","seed","seen"]].drop_duplicates(), on=["model","seed"])
    se = s[s.seen]; un = s[~s.seen]
    ax.errorbar([0], se.f1.mean(), yerr=se.f1.std(), fmt="o", color=C[m])
    ax.errorbar([1], un.f1.mean(), yerr=un.f1.std(), fmt="o", color=C[m], label=nice[m])
    ax.plot([0,1], [se.f1.mean(), un.f1.mean()], color=C[m], lw=1, alpha=.7)
ax.set_xticks([0,1]); ax.set_xticklabels(["seen (office, n=24)","unseen (education 12 + retail 5)"])
ax.set_ylabel("Event-level F1"); ax.legend(fontsize=7)
ax.set_title("Cross-building-type transfer (real BDG2)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_revision_transfer.png"), dpi=300); plt.close()

# --- Fig: waste comparison (SSL vs TOWT per-building bias) ---
wd = pd.read_csv(os.path.join(BASE, "results", "metrics_revision_waste.csv"))
sel = ["PatchTST-SSL", "TOWT", "Autoencoder(t)", "LSTM-AE(t)"]
fig, ax = plt.subplots(figsize=(7, 4))
data, labels = [], []
for m in sel:
    b = wd[wd.model == m].groupby("building_id").bias_pct.mean()*100
    data.append(b.values); labels.append(m)
ax.boxplot(data, tick_labels=labels, showfliers=False)
ax.axhline(0, color="k", lw=.8, ls="--")
ax.set_ylabel("Signed waste bias (%)")
ax.set_title("Waste-estimation bias per building (real BDG2)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_revision_waste.png"), dpi=300); plt.close()

print("done:", [f for f in os.listdir(FIG) if f.startswith("fig_revision")])
