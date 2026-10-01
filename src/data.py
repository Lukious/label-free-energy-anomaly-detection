"""Synthetic building energy dataset generation + loading + normalization.

Physical load model (per building, hourly, 1 year = 8760 h):
  load = base + occupancy_schedule + weather_sensitive(CDD/HDD) + AR(1) noise

Buildings: 6 office + 2 retail. 6 train, 2 held out for transfer eval.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

SEED = 42
N_HOURS = 8760  # 1 year, hourly
N_BUILDINGS = 8  # 6 office + 2 retail
TRAIN_BUILDINGS = 6
LATITUDE_TEMP = {"annual_mean": 12.0, "annual_amp": 12.0}

# Fraction of the year used as test span (anomalies injected only here)
TEST_FRACTION = 0.25
VAL_FRACTION = 0.15  # of the pre-test span, used for threshold calibration

FEATURE_COLS = ["load", "temp", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TARGET_COL = "load"


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _outdoor_temp(rng: np.random.Generator) -> np.ndarray:
    """Annual sine + diurnal sine + noise."""
    t = np.arange(N_HOURS)
    doy = t / 24.0
    hod = (t % 24) / 24.0
    annual = LATITUDE_TEMP["annual_mean"] + LATITUDE_TEMP["annual_amp"] * np.sin(
        2 * np.pi * (doy - 100) / 365.0
    )
    diurnal = 5.0 * np.sin(2 * np.pi * (hod - 0.35))
    noise = rng.normal(0, 1.5, N_HOURS)
    return annual + diurnal + noise


def _occupancy_schedule(rng: np.random.Generator, btype: str) -> np.ndarray:
    """Weekday/weekend schedule with building-specific start/end hours."""
    t = np.arange(N_HOURS)
    hod = t % 24
    dow = (t // 24) % 7  # 0=Mon
    if btype == "office":
        start = int(rng.integers(7, 10))
        end = int(rng.integers(16, 20))
        peak = rng.uniform(60, 120)  # kW occupancy-driven load
        weekend_frac = rng.uniform(0.05, 0.15)
    else:  # retail
        start = int(rng.integers(9, 11))
        end = int(rng.integers(20, 22))
        peak = rng.uniform(80, 160)
        weekend_frac = rng.uniform(0.5, 0.8)
    occ = np.where((hod >= start) & (hod <= end), 1.0, 0.0).astype(float)
    # smooth ramp (1h edges) for realism
    occ = np.convolve(np.pad(occ, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    is_weekend = dow >= 5
    sched = np.where(is_weekend, occ * weekend_frac, occ) * peak
    return sched


def _weather_load(temp: np.ndarray, bp_cool: float, bp_heat: float,
                  cdd_k: float, hdd_k: float) -> np.ndarray:
    cdd = np.clip(temp - bp_cool, 0, None)
    hdd = np.clip(bp_heat - temp, 0, None)
    return cdd_k * cdd + hdd_k * hdd


def generate_synthetic_dataset(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Return {building_id: DataFrame(timestamp, load, temp, btype)}."""
    rng = _rng(seed)
    temp = _outdoor_temp(rng)
    out: dict[str, pd.DataFrame]
    out = {}
    ts = pd.date_range("2016-01-01", periods=N_HOURS, freq="h")
    t = np.arange(N_HOURS)
    hour = t % 24
    dow = (t // 24) % 7
    cal = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
        }
    )
    for i in range(N_BUILDINGS):
        bid = f"B{i + 1:02d}"
        btype = "office" if i < TRAIN_BUILDINGS else "retail"
        base = rng.uniform(15, 35)  # kW base load
        sched = _occupancy_schedule(rng, btype)
        wl = _weather_load(
            temp,
            bp_cool=rng.uniform(16, 22),
            bp_heat=rng.uniform(14, 19),
            cdd_k=rng.uniform(1.5, 4.0),
            hdd_k=rng.uniform(0.8, 2.5),
        )
        # AR(1) noise
        eps = rng.normal(0, 1.0, N_HOURS)
        ar = np.zeros(N_HOURS)
        for k in range(1, N_HOURS):
            ar[k] = 0.6 * ar[k - 1] + eps[k]
        noise = ar * rng.uniform(1.5, 3.0)
        load = np.clip(base + sched + wl + noise, 1.0, None)
        df = pd.DataFrame({"timestamp": ts, "load": load, "temp": temp,
                           "btype": btype, "building_id": bid})
        df = pd.concat([df.reset_index(drop=True), cal], axis=1)
        out[bid] = df
    return out


def save_dataset(dataset: dict[str, pd.DataFrame], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for bid, df in dataset.items():
        df.to_csv(os.path.join(out_dir, f"{bid}.csv"), index=False)


def load_dataset(data_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    for fn in sorted(os.listdir(data_dir)):
        if fn.endswith(".csv"):
            out[fn[:-4]] = pd.read_csv(os.path.join(data_dir, fn),
                                       parse_dates=["timestamp"])
    return out


def train_test_split(dataset: dict[str, pd.DataFrame]):
    """Split by time: train = first (1 - TEST_FRACTION - VAL_FRACTION),
    val = next VAL_FRACTION, test = last TEST_FRACTION."""
    n = N_HOURS
    n_test = int(n * TEST_FRACTION)
    n_val = int(n * VAL_FRACTION)
    n_train = n - n_test - n_val
    train, val, test = {}, {}, {}
    for bid, df in dataset.items():
        train[bid] = df.iloc[:n_train].reset_index(drop=True)
        val[bid] = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test[bid] = df.iloc[n_train + n_val:].reset_index(drop=True)
    return train, val, test


class ZScaler:
    """Per-building z-scaler fitted on training data."""

    def __init__(self, cols=FEATURE_COLS):
        self.cols = cols
        self.mu_ = None
        self.sd_ = None

    def fit(self, df: pd.DataFrame):
        self.mu_ = df[self.cols].mean()
        self.sd_ = df[self.cols].std().replace(0, 1e-8)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return ((df[self.cols] - self.mu_) / self.sd_).to_numpy(dtype=np.float32)
