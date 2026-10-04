"""8. LSTM Autoencoder for time-series anomaly detection (server metrics / IoT sensors)."""
import os

import numpy as np
import torch
import torch.nn as nn

from .. import config
from ..core import BaseModel, torch_param_stats

CKPT = config.CKPT_DIR / "lstm_ae.pt"
WIN = 48


def synth_signal(n=2000, seed=0, anomalies=0, kinds=("spike", "shift", "noise", "dropout")):
    """Server-load-like signal: daily + weekly seasonality, trend-free, Gaussian noise. Returns (x, labels)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = (50 + 18 * np.sin(2 * np.pi * t / 96) + 6 * np.sin(2 * np.pi * t / 672 + 1.3)
         + 3 * np.sin(2 * np.pi * t / 24 + rng.uniform(0, 6)) + rng.normal(0, 1.6, n))
    y = np.zeros(n, dtype=int)
    for k in range(anomalies):
        kind = kinds[k % len(kinds)]
        s = int(rng.integers(WIN, n - WIN - 30))
        if kind == "spike":
            L = int(rng.integers(2, 5)); x[s:s + L] += rng.choice([-1, 1]) * rng.uniform(25, 40)
        elif kind == "shift":
            L = int(rng.integers(15, 30)); x[s:s + L] += rng.choice([-1, 1]) * rng.uniform(15, 22)
        elif kind == "noise":
            L = int(rng.integers(15, 30)); x[s:s + L] += rng.normal(0, 9, L)
        else:  # dropout / flatline
            L = int(rng.integers(10, 20)); x[s:s + L] = x[s] * 0 + rng.uniform(0, 5)
        y[s:s + L] = 1
    return x.astype(np.float32), y


class LSTMAE(nn.Module):
    def __init__(self, hidden=64, latent=16):
        super().__init__()
        self.enc = nn.LSTM(1, hidden, batch_first=True, num_layers=2)
        self.to_lat = nn.Linear(hidden, latent)
        self.from_lat = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True, num_layers=2)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):  # x: B,T,1
        _, (h, _) = self.enc(x)
        z = self.to_lat(h[-1])
        rep = self.from_lat(z).unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.dec(rep)
        return self.out(y)


def windows(x, stride=1):
    idx = np.arange(0, len(x) - WIN + 1, stride)
    return np.stack([x[i:i + WIN] for i in idx]), idx


def train_and_save(steps=None, log=print):
    steps = steps or int(os.environ.get("LSTM_AE_STEPS", "1200"))
    x, _ = synth_signal(6000, seed=1)
    mu, sd = float(x.mean()), float(x.std())
    xn = (x - mu) / sd
    W, _ = windows(xn, 2)
    W = torch.tensor(W).unsqueeze(-1)
    dev = "cpu"  # LSTMs are fastest on CPU at this size
    model = LSTMAE().to(dev)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    curve = []
    for step in range(1, steps + 1):
        b = W[torch.randint(len(W), (64,))].to(dev)
        loss = ((model(b) - b) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 10) == 0:
            curve.append((step, round(loss.item(), 5)))
            log(f"[lstm-ae] step {step}/{steps} loss {loss.item():.5f}")
    model.eval()
    with torch.no_grad():
        err = point_errors(model, xn)
    thr = float(np.percentile(err, 99.5)) * 1.1
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "thr": thr, "curve": curve}, CKPT)


@torch.no_grad()
def point_errors(model, xn, return_recon=False):
    W, idx = windows(xn, 1)
    out = []
    for i in range(0, len(W), 512):
        b = torch.tensor(W[i:i + 512]).unsqueeze(-1)
        out.append(model(b).squeeze(-1).numpy())
    R = np.concatenate(out)
    err_sum = np.zeros(len(xn)); rec_sum = np.zeros(len(xn)); cnt = np.zeros(len(xn))
    for r, s, w in zip(R, idx, W):
        err_sum[s:s + WIN] += (r - w) ** 2
        rec_sum[s:s + WIN] += r
        cnt[s:s + WIN] += 1
    cnt[cnt == 0] = 1
    err = err_sum / cnt
    return (err, rec_sum / cnt) if return_recon else err


class LSTMAnomaly(BaseModel):
    key = "lstm_ae"
    name = "LSTM Autoencoder"
    family = "Recurrent seq2seq autoencoder"
    task = "Time-series anomaly detection"
    modality = "time-series"
    description = "A 2-layer LSTM encoder squeezes 48-step windows of server metrics into a 16-d latent and a decoder rebuilds them. Points with high reconstruction error are flagged as anomalies."
    compute = "CPU · ~90K params · trains in ~30 s"

    def _load(self):
        if not CKPT.exists():
            self.notes = ["Trained on synthetic normal telemetry on first load."]
            train_and_save()
        ck = torch.load(CKPT, map_location="cpu")
        self.m_model = LSTMAE()
        self.m_model.load_state_dict(ck["state"])
        self.m_model.eval()
        self.mu, self.sd, self.thr, self.curve = ck["mu"], ck["sd"], ck["thr"], ck["curve"]
        self.variant = "LSTM-AE h64 z16 (cpu)"
        s = torch_param_stats(self.m_model)
        self.params, self.param_mb = s["params"], s["param_mb"]

    def detect(self, series=None, n=600, anomalies=4, seed=None, sensitivity=1.0):
        labels = None
        if not series:
            seed = int(seed if seed is not None else np.random.randint(0, 10_000))
            x, labels = synth_signal(int(n), seed=seed, anomalies=int(anomalies))
            xn = (x - self.mu) / self.sd
            scale, shift = self.sd, self.mu
        else:
            x = np.asarray(series, dtype=np.float32)
            if len(x) < WIN:
                raise ValueError(f"Need at least {WIN} points")
            # Robust per-series standardisation for user data
            med, iqr = np.median(x), np.subtract(*np.percentile(x, [75, 25])) or 1.0
            xn = (x - med) / (iqr / 1.35)
            scale, shift = iqr / 1.35, med
        err, rec = point_errors(self.m_model, xn, True)
        thr = self.thr / max(0.1, float(sensitivity))
        flags = (err > thr).astype(int)
        recon = rec * scale + shift
        # group into events
        events, start = [], None
        for i, f in enumerate(list(flags) + [0]):
            if f and start is None:
                start = i
            if not f and start is not None:
                events.append({"start": start, "end": i - 1, "peak_error": round(float(err[start:i].max()), 3)})
                start = None
        res = {"values": [round(float(v), 3) for v in x], "reconstruction": [round(float(v), 3) for v in recon],
               "error": [round(float(v), 4) for v in err], "threshold": round(thr, 4), "flags": flags.tolist(),
               "events": events, "n_anomalous_points": int(flags.sum()), "window": WIN}
        if labels is not None:
            res["labels"] = labels.tolist()
            res["eval"] = self._score(flags, labels, err)
            res["seed"] = seed
        return res

    @staticmethod
    def _score(flags, labels, err):
        from sklearn.metrics import roc_auc_score

        # tolerance: predictions within +/-3 points of a true anomaly count as hits
        lab_d = np.convolve(labels, np.ones(7), "same") > 0
        tp = int(((flags == 1) & lab_d).sum())
        fp = int(((flags == 1) & ~lab_d).sum())
        fl_d = np.convolve(flags, np.ones(7), "same") > 0
        rec = float(((labels == 1) & fl_d).sum() / max(1, labels.sum()))
        prec = tp / max(1, tp + fp)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        auc = float(roc_auc_score(labels, err)) if 0 < labels.sum() < len(labels) else None
        return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
                "roc_auc": round(auc, 3) if auc else None}

    def benchmark(self):
        scores = [self.detect(n=1000, anomalies=6, seed=s)["eval"] for s in range(100, 105)]
        avg = lambda k: round(float(np.mean([s[k] for s in scores if s[k] is not None])), 3)  # noqa: E731
        return {"primary": {"name": "F1 (point, ±3 tol)", "value": avg("f1"), "better": "higher"},
                "metrics": {"precision": avg("precision"), "recall": avg("recall"), "roc_auc": avg("roc_auc"),
                            "series": len(scores), "threshold": round(self.thr, 4)}}
