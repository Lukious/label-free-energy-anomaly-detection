"""PatchTST-style masked-reconstruction SSL model (channel-independent).

Compact implementation using torch nn.TransformerEncoder — no external libs.
Channels: [load, temp, hour_sin, hour_cos, dow_sin, dow_cos].
Only the load channel is masked; all channels are reconstructed (the model
must infer expected load from calendar/weather when load is masked).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..features import make_windows

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Full-series robust z-score (median/MAD) — anomaly-contamination robust."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / (mad + 1e-9)


def _hour_cond_z(r: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Robust z with per-hour-of-day median/MAD (see score_scales docstring).

    hour decoded from the hour_sin/hour_cos channels (indices 2, 3)."""
    hour = (np.degrees(np.arctan2(X[:, 2], X[:, 3])) % 360.0 / 15.0).round().astype(int) % 24
    z = np.empty_like(r)
    for h in range(24):
        m = hour == h
        if m.sum() < 24:  # too few samples -> fall back to global
            z[m] = _robust_z(r[m])
            continue
        med = np.median(r[m])
        mad = np.median(np.abs(r[m] - med)) * 1.4826
        z[m] = (r[m] - med) / (mad + 1e-9)
    return z


class PatchTSTConfig:
    seq_len = 168
    stride = 24
    patch_len = 16
    patch_stride = 8
    d_model = 128       # iter2: 64 -> 128 (capacity up, early stopping guards fit)
    nhead = 4
    num_layers = 3      # iter2: 2 -> 3
    dim_ff = 256        # iter2: 128 -> 256
    mask_ratio = 0.4
    epochs = 80         # iter2: 10 -> 80 with early stopping (patience 8)
    batch = 64
    lr = 1e-3
    patience = 8
    infer_batch = 512   # stride-1 point-residual inference batch
    target_idx = 0  # load channel
    seed = 7
    mix_channels = True  # False -> channel-independent ablation M1(e)


class PatchEmbed(nn.Module):
    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.lin = nn.Linear(patch_len, d_model)

    def forward(self, x):  # x: (B, C, P, patch_len)
        return self.lin(x)  # (B, C, P, d)


class PatchTST(nn.Module):
    """Patch tokens from ALL channels attend to each other (with learned
    channel embeddings), so the masked load channel is inferred from
    weather/calendar patches — required for expected-consumption SSL."""

    def __init__(self, cfg: PatchTSTConfig, n_channels: int):
        super().__init__()
        self.cfg = cfg
        self.n_patches = (cfg.seq_len - cfg.patch_len) // cfg.patch_stride + 1
        self.embed = PatchEmbed(cfg.patch_len, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, cfg.d_model))
        self.mix_channels = getattr(cfg, "mix_channels", True)
        if self.mix_channels:
            self.chan = nn.Embedding(n_channels, cfg.d_model)
        else:  # channel-independent (PatchTST original design) — ablation M1(e)
            self.chan = None
        enc_layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.nhead, cfg.dim_ff, batch_first=True,
            dropout=0.1, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.num_layers)
        self.head = nn.Linear(cfg.d_model, cfg.patch_len)
        # linear skip: predict each load patch directly from the same-time
        # patches of ALL channels (weather/calendar) — gives the model an
        # instant near-regression solution the attention refines
        self.skip = nn.Linear(n_channels * cfg.patch_len, cfg.patch_len)
        self.skip_ci = nn.Linear(cfg.patch_len, cfg.patch_len)  # channel-independent skip
        self.channels = n_channels

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        P = self.n_patches
        ps = self.cfg.patch_stride
        pl = self.cfg.patch_len
        idx = torch.arange(P, device=x.device) * ps
        patches = torch.stack([x[:, :, i:i + pl] for i in idx], dim=2)  # (B,C,P,pl)
        h = self.embed(patches) + self.pos.unsqueeze(0)
        if self.mix_channels:
            h = h + self.chan(
                torch.arange(C, device=x.device)).view(1, C, 1, -1)
            h = h.reshape(B, C * P, -1)  # channel tokens mixed via attention
            h = self.encoder(h)
            out = self.head(h).view(B, C, P, pl)
            # skip path on the load channel: (B,P,C*pl) -> (B,P,pl)
            skip_in = patches.permute(0, 2, 1, 3).reshape(B, P, C * pl)
            out[:, self.cfg.target_idx, :, :] = (out[:, self.cfg.target_idx, :, :]
                                                  + self.skip(skip_in))
        else:
            # channel-independent: each channel encoded separately (B*C, P, d)
            h = h.reshape(B * C, P, -1)
            h = self.encoder(h)
            out = self.head(h).view(B, C, P, pl)
            ci = self.cfg.target_idx
            skip_in = patches[:, ci]  # (B, P, pl)
            out[:, ci, :, :] = out[:, ci, :, :] + self.skip_ci(skip_in)
        rec = torch.zeros(B, C, T, device=x.device)
        cnt = torch.zeros(B, C, T, device=x.device)
        for j, i in enumerate(idx):
            rec[:, :, i:i + pl] += out[:, :, j]
            cnt[:, :, i:i + pl] += 1
        return rec / cnt


class PatchTSTAD:
    """Channel-independent masked-reconstruction anomaly detector.

    fit: multiple buildings' normalized feature arrays (list) -> shared encoder.
    score: single building array -> per-timestep load-channel reconstruction error.
    """

    name = "PatchTST-SSL"

    def __init__(self, cfg: PatchTSTConfig | None = None):
        self.cfg = cfg or PatchTSTConfig()
        self.model = None
        torch.manual_seed(self.cfg.seed)

    def _windows(self, X):
        return make_windows(X, self.cfg.seq_len, self.cfg.stride)

    def fit(self, arrays: list[np.ndarray], val_arrays: list[np.ndarray] | None = None):
        """iter2: validation-monitored training with early stopping.

        val_arrays (clean validation spans of the same buildings) are used to
        monitor masked-reconstruction loss; training stops when the validation
        loss stops improving for `patience` epochs (overfitting guard while
        epochs/capacity were increased).
        """
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        C = arrays[0].shape[1]
        self.model = PatchTST(cfg, C).to(DEVICE)
        W = np.concatenate([self._windows(a) for a in arrays], axis=0)
        Wt = torch.tensor(W, device=DEVICE)  # (N, T, C)
        Vt = None
        if val_arrays:
            V = np.concatenate([self._windows(a) for a in val_arrays], axis=0)
            Vt = torch.tensor(V, device=DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        lossf = nn.MSELoss()

        def masked_loss(xb):
            x_in = xb.clone()
            # SSL task: mask the ENTIRE load channel; predict it from
            # weather/calendar channels (aligns train and inference)
            x_in[:, cfg.target_idx, :] = 0.0
            rec = self.model(x_in)
            return lossf(rec[:, cfg.target_idx, :], xb[:, cfg.target_idx, :])

        n = len(Wt)
        best_val, best_state, bad = float("inf"), None, 0
        self.history: list[dict] = []
        for ep in range(cfg.epochs):
            self.model.train()
            perm = torch.randperm(n, device=DEVICE)
            tot, nb = 0.0, 0
            for i in range(0, n, cfg.batch):
                xb = Wt[perm[i:i + cfg.batch]].permute(0, 2, 1)  # (B,C,T)
                opt.zero_grad()
                loss = masked_loss(xb)
                loss.backward()
                opt.step()
                tot += loss.item(); nb += 1
            tr_loss = tot / max(nb, 1)
            rec = {"epoch": ep + 1, "train_loss": tr_loss}
            if Vt is not None:
                self.model.eval()
                with torch.no_grad():
                    vl = 0.0
                    for i in range(0, len(Vt), cfg.batch * 4):
                        xb = Vt[i:i + cfg.batch * 4].permute(0, 2, 1)
                        vl += masked_loss(xb).item()
                vl /= max((len(Vt) - 1) // (cfg.batch * 4) + 1, 1)
                rec["val_loss"] = vl
                if vl < best_val - 1e-5:
                    best_val, bad = vl, 0
                    best_state = {k: v.detach().clone() for k, v in
                                  self.model.state_dict().items()}
                else:
                    bad += 1
                self.history.append(rec)
                if bad >= cfg.patience:
                    self.history.append({"epoch": ep + 1, "stopped": True})
                    break
            else:
                self.history.append(rec)
        if best_state is not None:
            self.model.load_state_dict(best_state)  # restore best-val weights
        self.model.eval()
        return self

    @torch.no_grad()
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Full-sequence (no masking) reconstruction of the load channel."""
        cfg = self.cfg
        W = self._windows(X)
        Wt = torch.tensor(W, device=DEVICE).permute(0, 2, 1)  # (B,C,T)
        rec = self.model(Wt)[:, cfg.target_idx, :]  # (B,T)
        return rec.cpu().numpy()

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        """iter2: MULTI-SCALE anomaly score (design decision, see iter2_results).

        Root cause of the iter1 spike weakness (recall 0.07): the only score was
        the squared residual *pooled over all overlapping 168h windows*, so a
        1-6h spike residual was averaged with ~160 normal hours and diluted far
        below the threshold.

        Fix — two complementary scales:
          short (point) scale: stride-1 windows; take the residual at the LAST
            timestep of each window. Full 168h receptive field (history), no
            averaging across the event, no future leakage. Sensitive to spikes
            and schedule anomalies (off-hour excess localized to single hours).
          long (pooled) scale: the original stride-24 window-mean residual
            (iter1 score). Slow, low-variance signal — sensitive to drift,
            robust to patch reconstruction noise.
        Combination: each scale is standardized by its OWN full-series
        median/MAD (robust to anomaly contamination), then combined by
        element-wise MAX: an hour is anomalous if it is extreme on ANY scale.
        Max (rather than a weighted sum) keeps each scale's noise floor from
        inflating the other's calibrated threshold; the downstream rolling
        robust-z flagger re-standardizes the combined score.
        """
        z_short, z_long = self.score_scales(X)
        return np.maximum(z_short, z_long)

    def score_scales(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """iter3: return the two standardized scales separately so the fusion
        weights can be calibrated on validation data (see run_iter3.py).

        Short scale uses an HOUR-CONDITIONAL robust z (median/MAD of the point
        residual computed per hour-of-day, full series): the iter3 false-alarm
        analysis showed the FP noise floor is hour-dependent (daytime ramp
        hours and night hours each carry ~1/3 of false runs) because schedule
        transitions make the reconstruction error heteroscedastic across the
        day — a single full-series MAD mixes those regimes and over-flags the
        high-noise hours. Conditioning the scale on hour-of-day equalizes the
        noise floor while preserving spike/drift magnitudes."""
        r_point = self.point_residuals(X)
        r_pooled = self.residuals(X)
        z_short = _hour_cond_z(r_point, X)
        z_long = _robust_z(r_pooled * r_pooled)  # squared: iter1 long-scale score
        return z_short, z_long

    @torch.no_grad()
    def point_residuals(self, X: np.ndarray) -> np.ndarray:
        """Signed residual of the LAST hour of each stride-1 window.

        r_t = y_t - yhat_t where yhat_t conditions on the previous 168h of
        weather/calendar (load masked). No pooling: a spike hour keeps its full
        magnitude. Hours before seq_len reuse the first computable value.
        """
        cfg = self.cfg
        W = make_windows(X, cfg.seq_len, stride=1)  # (N, T, C)
        Wt = torch.tensor(W, device=DEVICE).permute(0, 2, 1)  # (B,C,T)
        res = []
        for i in range(0, len(Wt), cfg.infer_batch):
            xb = Wt[i:i + cfg.infer_batch]
            x_in = xb.clone()
            x_in[:, cfg.target_idx, :] = 0.0
            rec = self.model(x_in)[:, cfg.target_idx, -1]
            res.append((xb[:, cfg.target_idx, -1] - rec).cpu().numpy())
        res = np.concatenate(res)  # residual of hour t = res[t - seq_len + 1]
        out = np.empty(len(X))
        out[cfg.seq_len - 1:] = res
        out[:cfg.seq_len - 1] = res[0]
        return out

    @torch.no_grad()
    def residuals(self, X: np.ndarray) -> np.ndarray:
        """Long scale: signed per-timestep residual pooled over all covering
        168h windows (stride 24). Same as iter1."""
        cfg = self.cfg
        W = self._windows(X)
        Wt = torch.tensor(W, device=DEVICE).permute(0, 2, 1)
        Win = Wt.clone()
        Win[:, cfg.target_idx, :] = 0.0
        rec = self.model(Win)[:, cfg.target_idx, :]
        res = (Wt[:, cfg.target_idx, :] - rec).mean(dim=1).cpu().numpy()
        n = len(X)
        out = np.zeros(n)
        cnt = np.zeros(n)
        for i, r in enumerate(res):
            s = i * cfg.stride
            out[s:s + cfg.seq_len] += r
            cnt[s:s + cfg.seq_len] += 1
        cnt[cnt == 0] = 1
        return out / cnt

    def waste_residuals(self, X: np.ndarray) -> np.ndarray:
        """iter2: point-scale residual for waste estimation.

        The pooled residual spreads a spike's excess over its whole 168h window
        (under-counting event-window waste); the point residual keeps the excess
        where it was injected."""
        return self.point_residuals(X)
