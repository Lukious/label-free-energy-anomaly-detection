# Label-Free Energy Anomaly Detection and Waste Quantification

Code and results for **"Time-Conditional Multi-Scale Scoring for Label-Free
Energy Anomaly Detection and Waste Quantification: A Causally Calibrated
Evaluation on 41 Measured Buildings"** (Su-Hwan Baek, Core Process
Engineering Research Institute, POSCO HOLDINGS).

This repository accompanies the manuscript submitted to the *Journal of
Building Engineering* (Elsevier).

## What this is

A fully unsupervised pipeline that detects energy anomalies in commercial
building electricity meters and quantifies the wasted energy (kWh) of each
anomaly, without requiring any fault labels. The central experimental
finding is that the detection performance earned by this pipeline comes
from **time-conditional multi-scale scoring with causal threshold
calibration**, not from the Transformer backbone: a simple
hour-of-week profile baseline with the same scoring reaches within 0.01 F1
of the learned model, whereas classical unsupervised baselines
(Isolation Forest, OC-SVM, autoencoders) lose false-alarm control on
measured data.

Key components:

- **PatchTST-SSL backbone** — masked-reconstruction Transformer producing
  point-scale residuals (channel-mixing variant)
- **Time-conditional scoring** — hour-conditional robust-z normalization
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
python scripts/make_figures_revision.py   # regenerate publication figures
```

All scripts are seeded; results in `results/` were produced with the
seeds recorded in each script.

## Requirements

Python 3.12. Key dependencies (see `requirements.txt`): numpy, pandas,
scikit-learn, scipy, matplotlib, torch (CPU or MPS).

## Results snapshot (BDG2, 41 measured buildings, 3 seeds)

| Model | F1 | FAR |
|---|---|---|
| PatchTST-SSL + time-conditional scoring | 0.66 ± 0.01 | 1.4% |
| Hour-of-week profile + robust-z | 0.65 | — |
| LSTM-AE (tuned) | 0.58 ± 0.01 | 9.0% |
| OC-SVM (tuned) | 0.52 ± 0.06 | 6.1% |

Building-level paired test vs the strongest baseline: +0.081 F1
(p = 2.9e-04, bootstrap CI [+0.04, +0.12]).

## License

MIT License — see [LICENSE](LICENSE).

## Citation

```bibtex
@article{baek2026timeconditional,
  title   = {Time-Conditional Multi-Scale Scoring for Label-Free Energy
             Anomaly Detection and Waste Quantification: A Causally
             Calibrated Evaluation on 41 Measured Buildings},
  author  = {Baek, Su-Hwan},
  journal = {Journal of Building Engineering (submitted)},
  year    = {2026}
}
```
