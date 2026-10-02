"""Anomaly injection: spike / drift / schedule-breakdown x 3 severity levels.

Injects only into the test span. Returns the corrupted series + event labels
(start_idx, end_idx, type, severity, injected_excess_kwh).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 123

# (magnitude_lo, magnitude_hi) per severity level (subtle / moderate / obvious)
SPIKE_SIGMA = {1: (3.0, 4.5), 2: (5.0, 7.0), 3: (8.0, 10.0)}  # x sigma of series
SPIKE_DURATION_H = (1, 6)
DRIFT_PCT_PER_DAY = {1: (0.02, 0.04), 2: (0.06, 0.09), 3: (0.12, 0.15)}
DRIFT_DURATION_DAYS = (5, 14)  # plan allows 1-4 weeks; capped to limit contamination
SCHEDULE_OFFHOUR_FRAC = {1: (0.30, 0.50), 2: (0.55, 0.75), 3: (0.85, 1.00)}

SEVERITY_NAMES = {1: "subtle", 2: "moderate", 3: "obvious"}


def inject_anomalies(df: pd.DataFrame, rng: np.random.Generator,
                     n_events: int = 10, ref: pd.DataFrame | None = None,
                     log: list | None = None, schedule_mode: str = "lift",
                     min_peak_frac: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """df: span DataFrame with 'load' column. Returns corrupted df + events.

    ref (Round-7 v11): if given, the magnitude scales (sigma for spikes, median
    load for drift / schedule targets) are computed on this UNINJECTED
    reference span (the training span), so no statistic of the injected span
    is used. ref=None keeps the original behaviour (statistics of `df`).
    log: optional list receiving one record per draw (accepted or rejected).
    schedule_mode (sensitivity, H4): "lift" (default) raises off-hours TO a
    fraction of median load; "additive" ADDS a fraction of the reference
    span's on-hour minus off-hour mean load to every off-hour, so the
    increment does not vanish where off-hour load is already high.
    min_peak_frac: if given, a draw is rejected when its peak increment is
    below this fraction of the reference median load (relative threshold),
    instead of the absolute 0.5 kW.
    """
    df = df.copy()
    n = len(df)
    load = df["load"].to_numpy(copy=True)
    ref_load = load if ref is None else ref["load"].to_numpy()
    sigma = float(np.std(ref_load))
    ref_med = None if ref is None else float(np.median(ref_load))

    if ref is not None:
        rh = ref["timestamp"].dt.hour.to_numpy()
        ron = (rh >= 8) & (rh < 18)
        sched_delta = max(float(ref_load[ron].mean() - ref_load[~ron].mean()), 0.0)
    else:
        sched_delta = None
    min_peak = 0.5 if min_peak_frac is None else min_peak_frac * float(np.median(ref_load))

    def med_now():
        # legacy (ref=None): median of the CURRENT, partly injected series
        return float(np.median(load)) if ref_med is None else ref_med
    # off-hours mask: bottom-30% hourly quantile defines "non-operating" baseline
    hourly_med = df.groupby(df["timestamp"].dt.hour)["load"].median()
    base_med = hourly_med.quantile(0.3)
    events: list[dict] = []

    guard = 0
    while len(events) < n_events and guard < 1000:
        guard += 1
        etype = rng.choice(["spike", "drift", "schedule"])
        sev = int(rng.integers(1, 4))
        if etype == "spike":
            dur = int(rng.integers(*SPIKE_DURATION_H))
            start = int(rng.integers(0, n - dur - 1))
            end = start + dur
            mag = rng.uniform(*SPIKE_SIGMA[sev]) * sigma
            shape = np.hanning(dur * 2)[:dur] * 2  # smooth ramp up/down
            if shape.max() < 0.5:
                shape = np.ones(dur)
            add = mag * shape / shape.max()
        elif etype == "drift":
            dur = int(rng.integers(*DRIFT_DURATION_DAYS)) * 24
            start = int(rng.integers(0, n - dur - 1))
            end = start + dur
            slope = rng.uniform(*DRIFT_PCT_PER_DAY[sev]) / 24.0  # fraction/hour
            add = np.linspace(0, slope * dur, dur) * med_now()
        else:  # schedule breakdown
            dur = int(rng.integers(3, 10)) * 24  # several days
            start = int(rng.integers(0, n - dur - 1))
            end = start + dur
            frac = rng.uniform(*SCHEDULE_OFFHOUR_FRAC[sev])
            sub = df.iloc[start:end]
            hrs = sub["timestamp"].dt.hour.to_numpy()
            is_on = np.array(
                [h in range(8, 18) for h in hrs]
            )  # off-hour = outside 8-18 approximation
            if schedule_mode == "additive":
                assert sched_delta is not None, "additive schedule needs ref"
                add = np.where(~is_on, frac * sched_delta, 0.0)
            else:
                # add to off-hour samples to lift them to target fraction of median
                target = med_now() * frac
                add = np.where(~is_on, np.clip(target - load[start:end], 0, None), 0.0)
            add = pd.Series(add).rolling(3, center=True, min_periods=1).mean().to_numpy()
        # overlap check
        ok = all(not (start < e["end"] and e["start"] < end) for e in events)
        if log is not None:
            log.append({"type": etype, "severity": sev,
                        "accepted": bool(ok and add.max() >= min_peak),
                        "reason": "overlap" if not ok else
                        ("peak<0.5kW" if add.max() < min_peak else "")})
        if not ok or add.max() < min_peak:
            continue
        excess_kwh = float(np.sum(np.clip(add, 0, None)))
        load[start:end] += add
        events.append({
            "start": start, "end": end, "type": etype,
            "severity": sev, "severity_name": SEVERITY_NAMES[sev],
            "injected_excess_kwh": excess_kwh,
        })

    df["load"] = load
    return df, pd.DataFrame(events)


def inject_all(test: dict[str, pd.DataFrame],
               n_events: int = 10, seed: int = SEED,
               ref: dict[str, pd.DataFrame] | None = None, log: list | None = None,
               schedule_mode: str = "lift", min_peak_frac: float | None = None):
    rng = np.random.default_rng(seed)
    corrupted, all_events = {}, {}
    for bid, df in test.items():
        lg = [] if log is not None else None
        c, ev = inject_anomalies(df, rng, n_events=n_events,
                                 ref=None if ref is None else ref[bid], log=lg,
                                 schedule_mode=schedule_mode, min_peak_frac=min_peak_frac)
        if log is not None:
            log += [dict(r, building_id=bid) for r in lg]
        corrupted[bid] = c
        ev = ev.copy()
        ev["building_id"] = bid
        all_events[bid] = ev
    return corrupted, pd.concat(all_events.values(), ignore_index=True)
