#!/bin/zsh
# Round-7 canonical pipeline: one process per seed (resumable: a finished
# seed directory is skipped), then merge + post-processing.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=results/final_r7
for s in 123 124 125; do
  [[ -f $OUT/real_seed$s/selection_log.csv ]] && { echo "skip real $s"; continue; }
  $PY -u scripts/run_final_r7.py real-seed $s
done
$PY scripts/run_final_r7.py merge-real
for s in 123 124 125; do
  [[ -f $OUT/synth_seed$s/synthetic_metrics.csv ]] && { echo "skip synth $s"; continue; }
  $PY -u scripts/run_final_r7.py synth-seed $s
done
$PY scripts/run_final_r7.py merge-synth
$PY scripts/interp_mask_r7.py
$PY scripts/injection_realism_r7.py
[[ -f $OUT/window_sensitivity.csv ]] || $PY -u scripts/run_window_r7.py
[[ -f $OUT/trainseed_sensitivity.csv ]] || $PY -u scripts/run_trainseed_r7.py   # post-hoc diagnostic
$PY -u scripts/run_leak_r7.py                                                       # leakage contrast (NaN-safe)
if [[ -f data/lead/lead1.0-small.csv ]]; then $PY -u scripts/run_lead_r7.py; fi   # LEAD 1.0 (public: github.com/samy101/lead-dataset)
[[ -f $OUT/schedule_sens/factorial_stats.csv ]] || $PY -u scripts/run_schedule_sens_r7.py   # H4: additive schedule injector
$PY scripts/stats_r7.py
$PY scripts/make_tables_r7.py
$PY scripts/make_figures_r7.py
echo ALL R7 DONE
