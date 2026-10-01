"""Round-2 manuscript figures (v5).

Fig. A (leakage): per-building F1, causal vs transductive normalization under
  identical buildings/seeds/injections (metrics_round2.csv vs
  metrics_round2_transductive.csv).
Fig. B (scoring example): one real BDG2 building under the current protocol
  (seed 123, w=1.0 causal scoring, trailing flagger) — load with injected
  events, z_short, flags.

New script only; existing experiment code untouched.
"""
from __future__ import annotations

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.run_revision import FR, causal_flags, causal_scale_z, fuse  # noqa: E402
from scripts.run_round2 import how_of  # noqa: E402
from src.anomaly_inject import inject_all  # noqa: E402
from src.data import ZScaler, load_dataset  # noqa: E402
from src.models.patchtst import PatchTSTAD, PatchTSTConfig  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "figures")
DATA_DIR = os.path.join(ROOT, "data", "bdg2", "selected_v2")
SEED = 123


def fig_leakage():
    caus = pd.read_csv(os.path.join(ROOT, "results", "metrics_round2.csv"))
    trd = pd.read_csv(
        os.path.join(ROOT, "results", "metrics_round2_transductive.csv"))
    c = caus[caus.model == "PatchTST-SSL"].groupby(
        ["seed", "building_id"])["f1"].mean().rename("causal")
    t = trd.groupby(["seed", "building_id"])["f1"].mean().rename("transductive")
    cmp = pd.concat([c, t], axis=1).dropna()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.4, 4.0), dpi=300)
    j = rng.uniform(-0.12, 0.12, len(cmp))
    ax.scatter(cmp.causal + j, cmp.transductive, s=14, alpha=0.55,
               color="#1f77b4", edgecolor="none", label="building × seed")
    lims = [0, 1.02]
    ax.plot(lims, lims, "k--", lw=0.8, label="y = x")
    d = (cmp.transductive - cmp.causal).mean()
    ax.set_xlabel("F1, causal protocol (train-frozen statistics)")
    ax.set_ylabel("F1, transductive protocol\n(full-series statistics)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Per-building F1: leakage inflates by {d:+.3f} on average",
                 fontsize=9)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_round2_leakage.png"))
    plt.close(fig)
    print("wrote fig_round2_leakage.png", f"n={len(cmp)} Δ={d:+.4f}")


def fig_scoring_example():
    warnings.filterwarnings("ignore")
    dataset = load_dataset(DATA_DIR)
    train_bids = sorted(b for b in dataset
                        if dataset[b]["btype"].iloc[0] == "office")
    bid = "Hog_office_Denita"
    n = len(dataset[bid])
    a, b = int(n * FR[0]), int(n * (FR[0] + FR[1]))
    train = {bb: dataset[bb].iloc[:int(len(dataset[bb]) * FR[0])]
             .reset_index(drop=True) for bb in dataset}
    val = {bb: dataset[bb].iloc[
        int(len(dataset[bb]) * FR[0]):int(len(dataset[bb]) * (FR[0] + FR[1]))]
        .reset_index(drop=True) for bb in dataset}
    test = {bb: dataset[bb].iloc[
        int(len(dataset[bb]) * (FR[0] + FR[1])):].reset_index(drop=True)
        for bb in dataset}
    corrupted, events = inject_all(test, n_events=6, seed=SEED)

    scalers = {bb: ZScaler().fit(train[bb]) for bb in dataset}
    tr_arr = [scalers[bb].transform(train[bb]) for bb in train_bids]
    va_arr = [scalers[bb].transform(val[bb]) for bb in train_bids]
    ssl = PatchTSTAD(PatchTSTConfig())
    ssl.fit(tr_arr, val_arrays=va_arr)

    n_tv = int(n * (FR[0] + FR[1]))
    full = scalers[bid].transform(dataset[bid])
    r_pt = ssl.point_residuals(full)
    r_pl = ssl.residuals(full)
    dl = np.zeros(n)
    dl[n_tv:] = (corrupted[bid]["load"].to_numpy()
                 - dataset[bid]["load"].to_numpy()[n_tv:]) \
        / scalers[bid].sd_["load"]
    rp = r_pt + dl
    z_s, _ = causal_scale_z(rp, r_pl, full, n_tv)
    score = fuse(1.0, z_s, np.zeros_like(z_s))
    sc = score[n_tv:]
    flags = causal_flags(sc)
    ev = events[events["building_id"] == bid]

    t = corrupted[bid]["timestamp"].to_numpy()
    load = corrupted[bid]["load"].to_numpy()
    hours = np.arange(len(load))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, dpi=300)
    ax = axes[0]
    ax.plot(hours, load, lw=0.6, color="#333", label="metered load (injected)")
    ax.plot(hours, dataset[bid]["load"].to_numpy()[n_tv:], lw=0.6,
            color="#aaa", ls="--", label="clean load")
    for _, e in ev.iterrows():
        ax.axvspan(int(e["start"]), int(e["end"]), color="#d62728", alpha=0.18)
    ax.set_ylabel("load (kW)")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title(f"{bid} (real BDG2, causal protocol, w=1.0, seed {SEED})",
                 fontsize=9)
    ax = axes[1]
    ax.plot(hours, sc, lw=0.5, color="#1f77b4", label=r"$z_{\mathrm{short}}$")
    thr = ax.get_ylim()
    af = sc[flags.astype(bool)]
    ax.scatter(hours[flags.astype(bool)], af, s=3, color="#d62728",
               label="flagged hours")
    for _, e in ev.iterrows():
        ax.axvspan(int(e["start"]), int(e["end"]), color="#d62728", alpha=0.18)
    ax.set_ylabel("short-scale $z$")
    ax.set_xlabel("test-span hours")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for a_ in axes:
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_round2_scoring_example.png"))
    plt.close(fig)
    print("wrote fig_round2_scoring_example.png", bid,
          f"events={len(ev)} flagged_h={int(flags.sum())}")


if __name__ == "__main__":
    fig_leakage()
    fig_scoring_example()
