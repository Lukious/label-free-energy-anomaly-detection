"""Evaluation: event-level P/R/F1, detection delay, false alarm rate, waste error.

Threshold rule (plan 5.3): calibrate on validation-period scores so that
FPR ~= 1% (99th percentile of validation scores).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def threshold_from_val(val_scores: np.ndarray, fpr_target: float = 0.01) -> float:
    return float(np.quantile(val_scores, 1.0 - fpr_target))


def _mad_scale(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) * 1.4826)


def adaptive_threshold(val_scores: np.ndarray, test_scores: np.ndarray,
                       fpr_target: float = 0.01) -> float:
    """Validation-quantile threshold adapted for seasonal scale shift.

    thr = quantile(val, 1-FPR) * (MAD(test) / MAD(val))

    Rationale (plan 5.3): threshold targets a validation FPR (~1%), but the
    test span sits in a different season whose reconstruction-error scale can
    differ multiplicatively. The MAD ratio corrects that drift robustly
    (anomalies are a minority of points, so MAD is barely affected).
    """
    base = threshold_from_val(val_scores, fpr_target)
    ratio = _mad_scale(test_scores) / (_mad_scale(val_scores) + 1e-9)
    return float(base * ratio)


def robust_z_flags(scores: np.ndarray, fpr_target: float = 0.01,
                   window: int = 336) -> np.ndarray:
    """Rolling robust (MAD) z-score flags — plan 5.2: s_t = |r_t| / (k*sigma_hat).

    A rolling median/MAD (2-week window) adapts to seasonal scale drift between
    train and test spans while remaining robust to anomaly contamination
    (anomalies are a minority within any 2-week window except long drifts).
    """
    from scipy.stats import norm

    s = pd.Series(scores)
    med = s.rolling(window, center=True, min_periods=window // 4).median()
    mad = (s - med).abs().rolling(window, center=True,
                                  min_periods=window // 4).median() * 1.4826
    med = med.bfill().ffill()
    mad = mad.bfill().ffill()
    z = (s - med) / (mad + 1e-9)
    return (z > norm.ppf(1.0 - fpr_target)).to_numpy()


def _flag_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs, s = [], None
    for i, f in enumerate(flags):
        if f and s is None:
            s = i
        elif not f and s is not None:
            runs.append((s, i))
            s = None
    if s is not None:
        runs.append((s, len(flags)))
    return runs


def evaluate_building(scores: np.ndarray, flags: np.ndarray,
                      events: pd.DataFrame, n_points: int) -> dict:
    """Event-level metrics for one building."""
    if len(events) == 0:
        return {}
    tp_events, delays = 0, []
    for _, ev in events.iterrows():
        s, e = int(ev["start"]), int(ev["end"])
        hit = np.where(flags[s:e])[0]
        if len(hit) > 0:
            tp_events += 1
            delays.append(int(hit[0]))  # hours after event start
    recall = tp_events / len(events)
    # precision: flagged runs overlapping any true event
    runs = _flag_runs(flags)
    anom_mask = np.zeros(n_points, dtype=bool)
    for _, ev in events.iterrows():
        anom_mask[int(ev["start"]):int(ev["end"])] = True
    tp_runs = sum(1 for s, e in runs if anom_mask[s:e].any())
    precision = tp_runs / len(runs) if runs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    far = float(flags[~anom_mask].mean()) if (~anom_mask).any() else 0.0
    return {
        "n_events": len(events), "tp_events": tp_events,
        "precision": precision, "recall": recall, "f1": f1,
        "detection_delay_h": float(np.mean(delays)) if delays else np.nan,
        "detection_rate": tp_events / len(events),
        "false_alarm_rate": far,
    }


def per_type_recall(flags: np.ndarray, events: pd.DataFrame) -> dict:
    """Recall and mean delay per anomaly type."""
    out = {}
    for etype, ev in events.groupby("type"):
        tp, delays = 0, []
        for _, row in ev.iterrows():
            s, e = int(row["start"]), int(row["end"])
            hit = np.where(flags[s:e])[0]
            if len(hit):
                tp += 1
                delays.append(int(hit[0]))
        out[etype] = {"recall": tp / len(ev), "n": len(ev),
                      "delay_h": float(np.mean(delays)) if delays else np.nan}
    return out


def merge_flag_runs(flags: np.ndarray, gap: int = 6) -> np.ndarray:
    """iter2: alarm-aggregation hysteresis — merge flagged runs separated by
    <= `gap` hours into one alarm. Fragmented flag runs (a single anomaly
    crossing the threshold intermittently) inflate the run count and destroy
    event-level precision; merging them is the standard BMS alarm-aggregation
    postprocess and does not remove any flagged hour, so recall can only
    improve (gap hours become flagged) while the run count shrinks."""
    out = flags.copy()
    idx = np.where(out)[0]
    if len(idx) < 2:
        return out
    i = 0
    while i < len(idx) - 1:
        if idx[i + 1] - idx[i] <= gap:
            out[idx[i]:idx[i + 1] + 1] = True
        i += 1
    return out


def per_type_severity_recall(flags: np.ndarray, events: pd.DataFrame) -> dict:
    """iter2: recall broken down by anomaly type x severity (3 levels)."""
    out = {}
    for (etype, sev), ev in events.groupby(["type", "severity"]):
        tp = sum(1 for _, row in ev.iterrows()
                 if flags[int(row["start"]):int(row["end"])].any())
        out[(etype, int(sev))] = {"recall": tp / len(ev), "n": len(ev)}
    return out


def waste_metrics(residuals: np.ndarray, flags: np.ndarray,
                  events: pd.DataFrame) -> dict:
    """Residual-integral waste vs injected ground-truth excess."""
    est_list, true_list = [], []
    for _, ev in events.iterrows():
        s, e = int(ev["start"]), int(ev["end"])
        est = float(np.sum(np.clip(residuals[s:e], 0, None)))
        est_list.append(est)
        true_list.append(float(ev["injected_excess_kwh"]))
    est, true = np.array(est_list), np.array(true_list)
    err = np.abs(est - true) / np.maximum(true, 1e-6)
    return {
        "waste_est_kwh": float(est.sum()),
        "waste_true_kwh": float(true.sum()),
        "waste_error_mean": float(np.mean(err)),
        "waste_error_total": float(abs(est.sum() - true.sum()) / max(true.sum(), 1e-6)),
    }


def aggregate(per_building: dict[str, dict]) -> dict:
    keys = [k for d in per_building.values() for k in d]
    out = {}
    for k in set(keys):
        vals = [d[k] for d in per_building.values()
                if k in d and d[k] == d[k]
                and isinstance(d[k], (int, float, bool, np.floating, np.integer))]
        if vals:
            out[k] = float(np.mean(vals))
    return out
