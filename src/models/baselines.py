"""Unsupervised baselines: IsolationForest, One-Class SVM, dense AE, LSTM-AE.

Common interface:
  model.fit(X)            # X: (n, C) normalized point features
  model.score(X) -> (n,)  # per-point anomaly scores (higher = more anomalous)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SVM_MAX_TRAIN = 5000  # OC-SVM subsample cap (kernel SVM is O(n^2)+)


class IsolationForestAD:
    name = "IsolationForest"

    def __init__(self, seed=0, **kw):
        self.clf = IsolationForest(n_estimators=200, random_state=seed,
                                   contamination="auto", **kw)

    def fit(self, X):
        self.clf.fit(X)
        return self

    def score(self, X):
        return -self.clf.score_samples(X)


class OCSVMAD:
    name = "OC-SVM"

    def __init__(self, seed=0):
        self.clf = OneClassSVM(kernel="rbf", nu=0.02, gamma="scale")
        self.rng = np.random.default_rng(seed)

    def fit(self, X):
        if len(X) > SVM_MAX_TRAIN:
            idx = self.rng.choice(len(X), SVM_MAX_TRAIN, replace=False)
            X = X[idx]
        self.clf.fit(X)
        return self

    def score(self, X):
        return -self.clf.score_samples(X)


class DenseAE(nn.Module):
    """MLP autoencoder on point features."""

    def __init__(self, d_in: int, hidden=(64, 32, 64)):
        super().__init__()
        enc, dec = [], []
        prev = d_in
        for h in hidden[: len(hidden) // 2]:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        for h in hidden[len(hidden) // 2:]:
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec += [nn.Linear(prev, d_in)]
        self.net = nn.Sequential(*enc, *dec)

    def forward(self, x):
        return self.net(x)


class DenseAEAD:
    name = "Autoencoder"
    epochs = 40
    batch = 256
    lr = 1e-3

    def __init__(self, seed=0):
        torch.manual_seed(seed)
        self.model = None
        self.target_idx = 0  # load is first column

    def fit(self, X):
        d_in = X.shape[1]
        self.model = DenseAE(d_in).to(DEVICE)
        Xs = torch.tensor(X, device=DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lossf = nn.MSELoss(reduction="none")
        n = len(Xs)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, self.batch):
                xb = Xs[perm[i:i + self.batch]]
                opt.zero_grad()
                loss = lossf(self.model(xb), xb)[:, self.target_idx].mean()
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    def score(self, X):
        with torch.no_grad():
            Xs = torch.tensor(X, device=DEVICE)
            err = (self.model(Xs) - Xs).pow(2)[:, self.target_idx]
        return err.cpu().numpy()


class LSTMAE(nn.Module):
    def __init__(self, d_in: int, hidden=64):
        super().__init__()
        self.enc = nn.LSTM(d_in, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, d_in, batch_first=True)

    def forward(self, x):  # x: (B, T, C)
        z, _ = self.enc(x)
        # bottleneck: keep last hidden repeated (Seq2Seq AE with bottleneck vector)
        z = z[:, -1:, :].expand(-1, x.shape[1], -1)
        y, _ = self.dec(z)
        return y


class LSTMAEAD:
    name = "LSTM-AE"
    seq_len = 48
    stride = 24
    epochs = 30
    batch = 128
    lr = 1e-3

    def __init__(self, seed=0):
        torch.manual_seed(seed)
        self.model = None

    def _windows(self, X):
        from ..features import make_windows
        return make_windows(X, self.seq_len, self.stride)

    def fit(self, X):
        d_in = X.shape[1]
        W = self._windows(X)
        self.model = LSTMAE(d_in).to(DEVICE)
        Wt = torch.tensor(W, device=DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        n = len(Wt)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, self.batch):
                xb = Wt[perm[i:i + self.batch]]
                opt.zero_grad()
                loss = (self.model(xb) - xb).pow(2)[..., 0].mean()
                loss.backward()
                opt.step()
        self.model.eval()
        return self

    def score(self, X):
        W = self._windows(X)
        with torch.no_grad():
            Wt = torch.tensor(W, device=DEVICE)
            err = (self.model(Wt) - Wt).pow(2)[..., 0].mean(dim=1).cpu().numpy()
        # max-pool window errors onto timestamps: sparse spikes are not diluted
        out = np.full(len(X), -np.inf)
        for i, e in enumerate(err):
            s = i * self.stride
            out[s:s + self.seq_len] = np.maximum(out[s:s + self.seq_len], e)
        out[out == -np.inf] = 0.0
        return out
