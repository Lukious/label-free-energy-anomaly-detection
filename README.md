# Label-Free Energy Anomaly Detection on 41 BDG2 Buildings

Code, building selection and result tables for the manuscript

> **Calendar-Conditional Scoring Drives Label-Free Energy Anomaly Detection: A Leakage-Free Factorial Evaluation of Learned and Profile Backbones on 41 Measured Buildings**
> Suwhan Baek (Department of Architecture & Urban Engineering, Hanyang Cyber University) and Dongyeop Lee (Department of Economics, Korea University).
> Submitted to the *Journal of Building Engineering* (Elsevier).

## Summary

The study asks which components of a label-free anomaly-detection pipeline for building electricity meters determine its performance. A masked-reconstruction PatchTST and a learning-free hour-of-week (HOW) profile are each combined with the same calendar-conditional or global robust-$z$ scoring layer, the same forward-only flagger and the same waste estimator. The evaluation uses 41 measured BDG2 buildings with controlled fault injections.

Model parameters, residual-normalization statistics and hyperparameters are frozen before the test span. The only test-time adaptation is a flagger that uses observations up to the preceding hour.

Main results (3 seeds, building-level inference):

| Detector | Precision | Recall | Event F1 | Range F1 | Non-injected alarm rate |
|---|---|---|---|---|---|
| PatchTST + hour-of-week scoring | 0.58 | 0.83 | 0.646 | 0.399 | 2.05% |
| PatchTST + global scoring | 0.58 | 0.72 | 0.595 | 0.347 | 2.41% |
| HOW profile + hour-of-week scoring | 0.57 | 0.83 | 0.641 | 0.418 | 2.77% |
| HOW profile + global scoring | 0.50 | 0.79 | 0.569 | 0.378 | 3.85% |
| PatchTST + hour-of-day scoring | 0.64 | 0.78 | 0.671 | 0.389 | 1.41% |
| TOWT + hour-of-week scoring | 0.62 | 0.80 | 0.659 | 0.414 | 2.05% |

**Scoring.** Calendar-conditional scoring increases event F1 by 0.062 on average (Holm *p* < 0.001), with no interaction with the backbone. The effect is confirmed by permutation tests, site-cluster bootstrap and mixed-model analyses.

**Backbone.** Under identical scoring, the backbone difference is +0.006. A building-level TOST indicates practical equivalence within ±0.03 F1 (*p* = 0.012). Across six training seeds the margin ranges from +0.002 to +0.033.

**Waste.** On true event windows, the HOW profile and a TOWT regression halve the per-event waste error of the learned counterfactual. All detectors underestimate operational waste by 36–50%.

The manuscript reports all results, including the sensitivity analyses and an out-of-domain check on LEAD 1.0.

## Repository structure

```
src/                 pipeline modules (data, injection, features, evaluation, models)
scripts/             experiment entry points; the *_r7 scripts form the canonical pipeline
results/final_r7/    master CSVs of the canonical run: every number, table and figure
                     of the manuscript is generated from these files
results/figures/     figures generated from the master CSVs (fig_r7_*)
data/selection/      list of the 41 selected BDG2 buildings (selected_v3.txt)
docs/                data provenance notes
results/ (other)     earlier development runs, kept for the record
```

## Data

The experiments use the **Building Data Genome 2 (BDG2)** open dataset: C. Miller et al., *Scientific Data* 7 (2020) 368, doi:10.1038/s41597-020-00712-x. The data are archived at doi:10.5281/zenodo.3887306.

They use the hourly electricity meters (`electricity_cleaned.csv`), the metadata and the site weather files, placed in `data/bdg2/`. The raw data are not redistributed here.

`scripts/prepare_r7.py` applies the deterministic selection rule and writes the 41-building panel to `data/bdg2/selected_v3/`. The rule is: at least 95% completeness over 2016, a non-degenerate 2016 meter, the 24 most complete offices, the 12 most complete education buildings and all 5 qualifying retail buildings. The resulting list is in `data/selection/selected_v3.txt`.

The optional LEAD 1.0 check uses `lead1.0-small.csv` from <https://github.com/samy101/lead-dataset>, placed in `data/lead/`.

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_r7.py          # build the 41-building panel from BDG2
scripts/run_all_r7.sh                 # canonical run + all diagnostics, then tables and figures
```

`run_all_r7.sh` is resumable. It runs the following steps:

1. `run_final_r7.py`, one process per seed, for the real data and for the synthetic data.
2. Interpolation and injection-magnitude analyses.
3. Window, training-seed, leakage, schedule-injector and LEAD diagnostics.
4. Statistics (`stats_r7.py`).
5. `make_tables_r7.py` and `make_figures_r7.py`.

Seeds 123–125 control injections and model initialization; the validation injections use seed + 333. Training is deterministic for a given seed, and `run_leak_r7.py` asserts that retraining reproduces the canonical F1 exactly. The reference environment is Python 3.13 with PyTorch 2.14 (Apple MPS); `requirements.txt` pins the versions.

## License

MIT (see `LICENSE`). BDG2 and LEAD 1.0 are subject to their own licenses.

## Citation

See `CITATION.cff`. The archived release of this repository has a Zenodo DOI, given in the badge and in the manuscript.
