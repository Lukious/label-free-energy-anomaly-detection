"""Round-7 MC4: how large are injected events relative to (i) the building's
median load and (ii) the natural, uninjected residual scatter of the same test
span? Training-free (HOW-profile residual). Output:
results/final_r7/injection_realism.csv (one row per injected event) and
results/final_r7/natural_excursions.csv (per building: share of clean test
hours whose HOW-residual robust z exceeds 3 / 5, and longest such run)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from scripts.run_final_r7 import (DATA_DIR, FR, N_EVENTS, OUT, SEEDS,  # noqa: E402
                                  how_backbone_residuals, inject_all, load_dataset)
from scripts.run_revision import _runs  # noqa: E402


def main():
    ds = load_dataset(DATA_DIR)
    n_tv = {b: int(len(df) * (FR[0] + FR[1])) for b, df in ds.items()}
    off = {b: int(len(df) * FR[0]) for b, df in ds.items()}
    ev_rows, nat_rows = [], []
    im = pd.read_csv(os.path.join(OUT, "interp_mask.csv"))
    filled = im.groupby("building_id")["idx"].apply(lambda x: set(x.tolist())).to_dict()
    for b, df in ds.items():
        tr = df.iloc[:off[b]].reset_index(drop=True)
        r = how_backbone_residuals(tr, df, 1.0)  # kW
        ref = r[:n_tv[b]]
        med = np.median(ref)
        rsd = np.median(np.abs(ref - med)) * 1.4826
        zt = (r[n_tv[b]:] - med) / rsd
        runs3 = _runs(np.abs(zt) > 3)
        nat_rows.append({"building_id": b, "resid_robust_sd_kw": rsd,
                         "median_load_kw": float(df["load"].median()),
                         "share_abs_z_gt3": float((np.abs(zt) > 3).mean()),
                         "share_abs_z_gt5": float((np.abs(zt) > 5).mean()),
                         "longest_run_gt3_h": max([e - s for s, e in runs3], default=0)})
    rej_rows = []
    trs = {b: ds[b].iloc[:off[b]].reset_index(drop=True) for b in ds}
    for seed in SEEDS:
        te = {b: ds[b].iloc[n_tv[b]:].reset_index(drop=True) for b in ds}
        lg = []
        cor, ev = inject_all(te, n_events=N_EVENTS, seed=seed, ref=trs, log=lg)
        rej_rows += [dict(r, seed=seed) for r in lg]
        nat = {r["building_id"]: r for r in nat_rows}
        for _, e in ev.iterrows():
            b, s, t = e["building_id"], int(e["start"]), int(e["end"])
            add = cor[b]["load"].to_numpy()[s:t] - te[b]["load"].to_numpy()[s:t]
            on = add > 0
            ev_rows.append({
                "seed": seed, "building_id": b, "type": e["type"],
                "severity": int(e["severity"]), "duration_h": t - s,
                "mean_add_kw_active": float(add[on].mean()) if on.any() else 0.0,
                "peak_add_kw": float(add.max()),
                "mean_add_over_resid_sd": float(add[on].mean() / nat[b]["resid_robust_sd_kw"]) if on.any() else 0.0,
                "peak_add_over_resid_sd": float(add.max() / nat[b]["resid_robust_sd_kw"]),
                "mean_add_over_median_load": float(add[on].mean() / nat[b]["median_load_kw"]) if on.any() else 0.0,
                "excess_kwh": float(e["injected_excess_kwh"]),
                "excess_over_test_energy": float(e["injected_excess_kwh"] / te[b]["load"].sum()),
                # 3.7: injection is applied AFTER interpolation; hours of the
                # event window that were filled in preparation
                "interp_hours": int(sum((n_tv[b] + i) in filled.get(b, set()) for i in range(s, t))),
            })
    pd.DataFrame(ev_rows).to_csv(os.path.join(OUT, "injection_realism.csv"), index=False)
    pd.DataFrame(nat_rows).to_csv(os.path.join(OUT, "natural_excursions.csv"), index=False)
    # C8: every draw (accepted or rejected) for the test injections
    pd.DataFrame(rej_rows).to_csv(os.path.join(OUT, "injection_draws.csv"), index=False)
    e = pd.DataFrame(ev_rows)
    print(e.groupby(["type", "severity"])[["duration_h", "mean_add_over_resid_sd",
                                          "peak_add_over_resid_sd", "mean_add_over_median_load",
                                          "excess_over_test_energy"]].median().round(3))
    print(e.groupby(["type", "severity"]).size().unstack())
    print(pd.DataFrame(nat_rows)[["share_abs_z_gt3", "share_abs_z_gt5", "longest_run_gt3_h"]].describe().round(3))


if __name__ == "__main__":
    main()
