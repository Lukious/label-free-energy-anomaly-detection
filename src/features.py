"""Feature construction: calendar + weather + load for baselines and SSL model."""
from __future__ import annotations

import numpy as np

from .data import FEATURE_COLS


def point_features(df) -> np.ndarray:
    """Per-timestamp feature matrix used by IF / OC-SVM / dense AE."""
    return df[FEATURE_COLS].to_numpy(dtype=np.float32)


def make_windows(arr: np.ndarray, seq_len: int, stride: int = 1):
    """Sliding windows: (N, seq_len, C)."""
    n = (len(arr) - seq_len) // stride + 1
    idx = np.arange(seq_len)[None, :] + stride * np.arange(n)[:, None]
    return arr[idx]


def recon_point_error(win_err: np.ndarray, seq_len: int, n_points: int) -> np.ndarray:
    """Collapse per-window errors to per-timestep error (mean over covering windows)."""
    acc = np.zeros(n_points)
    cnt = np.zeros(n_points)
    for i in range(win_err.shape[0]):
        acc[i:i + seq_len] += win_err[i]
        cnt[i:i + seq_len] += 1
    cnt[cnt == 0] = 1
    return acc / cnt
