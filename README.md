# Label-Free Energy Anomaly Detection and Waste Quantification

Code and results for **"Calendar-Conditional Scoring for Label-Free Energy
Anomaly Detection and Waste Quantification: A Causally Calibrated
Evaluation on 41 Measured Buildings"** (Su-Hwan Baek, Core Process
Engineering Research Institute, POSCO HOLDINGS).

This repository accompanies the manuscript submitted to the *Journal of
Building Engineering* (Elsevier).

## What this is

A fully unsupervised pipeline that detects energy anomalies in commercial
building electricity meters and quantifies the wasted energy (kWh) of each
anomaly, without requiring any fault labels. A pre-registered complete-factorial
experiment (backbone × scoring, Holm-corrected building-level paired
tests) separates the two components: the learned backbone adds **+0.040
F1 over an hour-of-week profile under identical scoring** (0.674 vs
0.634, Holm-adjusted p = 0.003; TOST rules out equivalence at ±0.03),
while **calendar-conditional scoring adds +0.06 to +0.09 F1 within each
backbone** (profile 0.55 → 0.63, PatchTST 0.60 → 0.66) and hour-of-week
conditioning lifts schedule-fault recall from 0.45 to 0.60. Reported
honestly: the ranking **inverts under overlap-based range-F1** (profile
0.411 vs Transformer 0.402), the profile family pays a 2.8–4.0% false
alarm rate against 1.4–1.8%, and a six-hour extend-until-recovery window
rule moves the learned detector's operational waste bias from **−51% to
+1%** (near-unbiased) where the dense profile baseline should be left
unextended. The paper is framed as a causally calibrated evaluation
methodology with calendar-conditional scoring as its methodological
core.

Key components:

- **PatchTST-SSL backbone** — masked-reconstruction Transformer producing
  point-scale residuals (channel-mixing variant)
- **Calendar-conditional scoring** — hour-of-day / hour-of-week conditional robust-z normalization
  that removes the heteroscedastic schedule-transition noise that floods
  global-MAD baselines at night and in the evening
- **Causal calibration** — normalization statistics frozen on the
  train/validation span; trailing (causal) 2-week rolling robust-z
  thresholds at a 1% FPR target; no transductive access to the test series
- **Waste quantification** — integration of positive residuals over
  flagged windows, evaluated against injected ground truth and compared
  with a TOWT (time-of-week-and-temperature) M&V regression baseline
- **Fair evaluation protocol** — identical validation-event tuning for
  every model, building-level paired statistics (n = 41 buildings,
  3 seeds), and a type × severity anomaly-injection grid
  (spike / drift / schedule breakdown × 3 severities)

## Repository structure

```
src/                  # pipeline modules
  data.py             # synthetic physics-based load generator (8 buildings)
  anomaly_inject.py   # type × severity injection with event labels
  features.py         # calendar / weather features
  evaluate.py         # event-level P/R/F1, delay, FAR, waste metrics
  models/
    baselines.py      # IsolationForest, OC-SVM, AE, LSTM-AE
    patchtst.py       # PatchTST-SSL + multi-scale scoring
scripts/              # one-step entry points (see below)
results/              # metrics CSVs (synthetic 5-seed, BDG2 3-seed,
                      # revision experiments) and publication figures
data/                 # created by the fetch/prepare scripts
docs/                 # data provenance notes
```

## Data

The measured-data experiments use **Building Data Genome 2 (BDG2)**
(Miller, Kathirgamanathan et al., 2020, *Scientific Data* 7:368):
hourly electricity meters with site-matched measured weather for 41
buildings selected by a deterministic rule (≥95% completeness, balanced
usage types: 24 office seen, 12 education + 5 retail held out).

```bash
# Option A — direct download (no account needed)
python scripts/fetch_bdg2.py            # downloads via GitHub LFS media URLs

# Option B — Kaggle (requires kaggle.json)
kaggle datasets download -d claytonmiller/buildingdatagenomeproject2

# Then select the 41-building panel
python scripts/prepare_bdg2.py
python scripts/prepare_revision.py      # joins measured weather, splits
```

## Reproducing the experiments

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/run_smoke.py        # quick synthetic demo (~3 min)
python scripts/generate_data.py    # full synthetic dataset
python scripts/run_final.py        # synthetic 5-seed main results
python scripts/run_realdata.py     # BDG2 3-seed measured results
python scripts/run_revision.py     # causal/fair-tuning/ablation suite
python scripts/run_round2.py       # two-stage waste, TOST, HoW variant, leakage
python scripts/run_round3.py       # pre-registered 2x2 factorial + Holm/TOST,
                                   # range-F1, prescription, TOWT diagnosis
python scripts/run_round3_lead.py  # LEAD 1.0-small out-of-domain check
python scripts/run_round3_synthetic_types.py  # synthetic type x severity, new protocol (incl. LSTM-AE, TOWT)
python scripts/run_round2_synthetic.py  # synthetic re-run under new protocol
python scripts/make_all_figures.py       # regenerate ALL figures from results CSVs
```

All scripts are seeded; results in `results/` were produced with the
seeds recorded in each script.

## Requirements

Python 3.12. Key dependencies (see `requirements.txt`): numpy, pandas,
scikit-learn, scipy, matplotlib, torch (CPU or MPS).

## Results snapshot (BDG2, 41 measured buildings, 3 seeds)

Pre-registered 2x2 (backbone x scoring), identical causal flagger:

| Configuration | Precision | Recall | Event F1 | Range F1 | FAR |
|---|---|---|---|---|---|
| PatchTST + HoW-conditional scoring | 0.61 | 0.83 | **0.674 ± 0.02** | 0.402 | 1.8% |
| PatchTST + HoD-conditional scoring | 0.64 | 0.74 | 0.659 ± 0.01 | 0.389 | 1.4% |
| HOW profile + HoW-conditional scoring | 0.56 | 0.82 | 0.634 ± 0.02 | 0.411 | 3.0% |
| HOW profile + global MAD | 0.47 | 0.78 | 0.548 ± 0.03 | 0.359 | 4.0% |
| TOWT + HoW-conditional scoring | 0.59 | 0.79 | 0.644 ± 0.01 | 0.401 | 2.1% |
| LSTM-AE (tuned) | 0.71 | 0.43 | 0.511 ± 0.03 | 0.251 | 9.0% |

Building-level paired (Holm-corrected): backbone effect under identical
scoring +0.040 (p = 0.0034, TOST rejects equivalence); scoring effect
within the profile +0.085 (p < 1e-4); HoD->HoW within PatchTST +0.015
(not significant). Note the honest range-F1 inversion: profile family
0.411-0.413 > PatchTST(HoW) 0.402.

Waste (operational, own flags): total bias -40% (profile) / -59% (PatchTST
HoW) / -62% (PatchTST HoD). Level-bias hypothesis rejected (in-event
residuals +1.8-2.0 sd for all backbones; target-only masking unchanged).
Prescription: 6-hour extend-until-recovery window reconstruction moves
PatchTST(HoD) from **-51.0% to +0.9%** (coverage 11% -> 25%); z12/CUSUM
over-correct (+18%/+97%); the profile baseline is already closest to
unbiased (-15%) and should not be extended.

LEAD 1.0-small (200 labeled buildings, out-of-domain check): 2.2% of
flagged hours coincide with labeled anomalies; 10.4% of labeled anomalous
hours recovered --- an honest negative result, reported as a robustness
check only.

All figures are regenerated from the CSVs by
`python scripts/make_all_figures.py` (dpi 300).

## License

MIT License — see [LICENSE](LICENSE).

## Citation

```bibtex
@article{baek2026calendarconditional,
  title   = {Calendar-Conditional Scoring for Label-Free Energy
             Anomaly Detection and Waste Quantification: A Causally
             Calibrated Evaluation on 41 Measured Buildings},
  author  = {Baek, Su-Hwan},
  journal = {Journal of Building Engineering (submitted)},
  year    = {2026}
}
```
