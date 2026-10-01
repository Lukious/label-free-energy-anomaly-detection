# Data provenance

## Building Data Genome 2 (BDG2)

- Original dataset: Miller, C., Kathirgamanathan, A., et al. (2020).
  "The Building Data Genome Project 2, energy meter data from the ASHRAE
  Great Energy Predictor III competition." *Scientific Data* 7:368.
  https://doi.org/10.1038/s41597-020-00712-x
- License: CC BY 4.0
- Obtained via the official GitHub repository
  (buds-lab/building-data-genome-project-2), Git LFS media endpoint:
  - `data/meters/cleaned/electricity_cleaned.csv` (175 MB)
  - `data/metadata/metadata.csv`
  - `data/weather/weather.csv` (measured site weather, 19.5 MB)

Raw files are not redistributed in this repository; run
`scripts/fetch_bdg2.py` to download them from the source above.

## Derived panel

`scripts/prepare_bdg2.py` + `scripts/prepare_revision.py` select a
41-building panel by a deterministic completeness rule (>= 95% non-missing
hours in 2016) and balanced usage types (24 office as seen buildings,
12 education + 5 retail as held-out), join site-matched measured
airTemperature, and produce chronological 60/15/25 train/validation/test
splits in local building time.

## Synthetic benchmark

`src/data.py` generates an 8-building synthetic benchmark
(physically motivated load model: base load + occupancy schedule +
degree-day weather response + AR(1) noise) used only for controlled
type x severity sensitivity analysis. Full generator equations and
parameters are documented in the manuscript appendix.
