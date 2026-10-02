# Final Consolidated (Canonical) Run — 2026-10-02

**One run, one set of master CSVs.** Every final number in the paper derives
from `results/final_consolidated/*.csv` via `scripts/make_tables.py`
(LaTeX table fragments + `paper/latex/numbers.tex` macros) and
`scripts/make_all_figures.py` (all figures incl. Fig. 5b). No legacy
per-round result file is read anywhere. Seeds {123,124,125}, w=1.0 in every
seed. n = 41 buildings in detection CSVs; paired contrasts use n = 40
buildings with defined F1 on all four 2×2 arms.

Command: `python scripts/run_final_consolidated.py all`
(log: `results/final_consolidated_run.log`)

---

## P-A. The two "HOW-profile" implementations — what they are, and which is
the official control (review B1d)

**They are NOT the same detector.** Code trace:

| | HOW-profile (HoW-cond.) — OFFICIAL CONTROL | HOW-profile + z (slot-level) — VARIANT |
|---|---|---|
| Residual | load − train HoW-slot **median profile**, standardized by train-global sd | load − train HoW-slot median (same profile idea) |
| Slot definition | dayofweek × hour = 168 slots | dayofweek × hour = 168 slots (identical) |
| Normalization target | the **profile residual**, train-frozen slot median/MAD via `cond_z(..., how_of)` with hierarchical fallback | the **raw load**, train-frozen slot MAD directly |
| Scoring | fused short+long causal score (`fuse(w, z_short, z_long)`, w tuned on validation injection events) + causal robust-z flagger | raw slot z only, own flagger, no fusion, no long scale |
| F1 (canonical) | **0.634 ± 0.019** | 0.645 ± 0.026 |

**Official control = HOW-profile (HoW-cond.)**, because its conditioning is
*identical* to the 2×2 design's scoring factor: train-frozen HoW-slot robust
z on the backbone residual, same causal flagger, same fusion weight, only
the backbone differs (profile vs PatchTST). The slot-level variant changes
three things at once (normalization target, no fusion, no long scale) and
therefore cannot serve as the backbone-only control.

**Both are reported** because the comparator choice is not innocuous:

- vs **HOW-profile (HoW-cond.)**: Δ = PatchTST − HOW = **+0.0398**
  (90% CI [+0.0207, +0.0589]), paired Holm p = 0.0034 (family of 6),
  TOST(±0.03) p = 0.804 → equivalence NOT established; CI **contains**
  values inside the margin (lower bound 0.021 < 0.03), so "difference
  exceeds the practical margin" is **undecided**, not proven (B2 fix).
- vs **HOW-profile+z (slot)**: Δ = **+0.0285** (90% CI [+0.0078, +0.0492]),
  TOST p = 0.452 — the headline contrast falls **inside** the ±0.03 margin
  under the variant comparator. Stated honestly in the paper.

## Canonical detection numbers (building macro-avg F1, 3 seeds)

| Model | F1 |
|---|---|
| PatchTST-SSL (HoW-cond., ours) | **0.674 ± 0.021** |
| PatchTST-SSL (HoD-cond.) | 0.659 ± 0.008 |
| HOW-profile+z (slot variant) | 0.645 ± 0.026 |
| TOWT + HoW-cond. scoring | 0.644 ± 0.012 |
| HOW-profile (HoW-cond.) | 0.634 ± 0.019 |
| HOW-profile (global-MAD) | 0.548 ± 0.029 |
| OC-SVM (t) | 0.525 ± 0.058 |
| LSTM-AE (own tuned flagger) | **0.511 ± 0.025** ← final value (old 0.58 was the fusion-scored variant) |
| Isolation Forest (t) | 0.503 ± 0.048 |
| Autoencoder (t) | 0.459 ± 0.007 |

2×2 Holm (6 contrasts): PatchTST−HOW(HoW) +0.0398 (p=0.0034);
HOW HoW−global +0.0854 (p<1e-4); PatchTST HoW−HoD +0.0147 (n.s.);
PatchTST(HoD)−HOW(HoW) +0.0251 (n.s.).

Leakage under identical conditions (same buildings/injections,
normalization only): transductive − causal ΔF1 = **+0.009** (n=41).

## Waste (canonical; fixes Table 11/12/Fig 5b mismatch — B1a/b)

**Operational, own flags, kWh totals:** PatchTST(HoW) **−59.1%**,
HOW(HoW) **−40.1%**, TOWT(HoW) **−55.7%**. (Single source for Table 12 and
Fig. 5b.)

**Oracle (true windows), 3 estimator variants:**

| Model | signed | clip | floor |
|---|---|---|---|
| PatchTST(HoW) | −86.8% | −0.3% | −41.7% |
| HOW(HoW) | −8.0% | +16.8% | −3.2% |
| TOWT(HoW) | −4.8% | +21.1% | −3.9% |

**Prescription (B3 fix: rule chosen on VALIDATION injection events, then
applied once to test):** validation chose z12 for PatchTST in all 3 seeds
(val biases −0.15/−0.21/−0.09) and z6 for HOW (+0.22/−0.01/+0.30).

| Model [chosen rule] | test bias | matched err (kWh) | false add (kWh) | missed (kWh) |
|---|---|---|---|---|
| PatchTST(HoW) [z12] | **−0.1%** (from −59.1% baseline) | −538,758 | +746,776 | −209,397 |
| TOWT(HoW) [z6] | −4.4% (from −55.7%) | −451,147 | +635,953 | −292,310 |
| HOW(HoW) [z6] | +34.7% (from −40.1%) | −246,236 | +1,282,475 | −188,540 |

Post-extension Eq.(6) decomposition is reported for every rule (master CSV
`prescription.csv`); missed kWh does **not** decrease with extension
(missed events carry no flags — B3a logic fixed). Per-building |err|
quartiles (chosen rule, PatchTST): Q1 0.17, median 0.36, Q3 0.82.

**Level bias (B3f, resolves +1.77σ vs −112% "contradiction"):**
full-window-mask PatchTST residual mean **+1.77σ** (event-simple) vs
**+1.41σ** kWh-weighted; target-only mask +1.82σ; HOW +2.03σ. Per type:
spike +3.28σ, drift +0.61σ, schedule −0.21σ. The event-simple mean is
dominated by short high-amplitude spikes, while kWh-weighted (and total
signed oracle bias, −86.8%) is driven by long drift/schedule under-capture
— two different weightings of the same signed errors, not a contradiction.

## Synthetic (new causal protocol, same-condition comparisons — B4)

See `synthetic_metrics.csv` / `table_synthetic.tex` (generated by the same
canonical run; HoW-cond arms paired backbone-vs-backbone with identical
scoring, unlike earlier mixed-condition comparisons).

## LEAD 1.0 label duration distribution (appendix evidence)

5,503 labeled anomalous runs: median **2 h**, IQR [1, 7] h, Q90 22 h, mean
6.8 h; 71% ≤ 6 h, 95% ≤ 24 h.

## Automation (review recommendation)

- `scripts/make_tables.py` → `paper/latex/tables/*.tex` +
  `paper/latex/numbers.tex` (40 macros; abstract/body use `\PatchTSTHowF`
  etc.). Reads ONLY `results/final_consolidated/*.csv`.
- `scripts/make_all_figures.py` → all figures incl. Fig. 5b from the same
  master CSVs (figure/table mismatch structurally impossible).
- `main.tex` body untouched in this step (numbers.tex/tables to be
  `\input` in the next editing pass).
