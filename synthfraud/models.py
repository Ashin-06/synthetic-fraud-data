"""Synthesizers.

GaussianCopula  fast baseline: marginals kept exactly (empirical quantiles), dependence
                captured by a correlation matrix of normal scores. Conditional sampling by
                rejection on the condition column.
TVAE            conditional variational autoencoder over the mode-normalised encoding,
                conditioned on fraud type so rare fraud can be generated on demand. Optional
                DP-SGD (per-example clipping + Gaussian noise) with an RDP accountant gives a
                formal (epsilon, delta) guarantee for every person in the training table.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.special import gammaln, logsumexp
from torch import nn
from torch.func import functional_call, grad, vmap

from .source import SCHEMA, Schema
from .transform import DataTransformer


# --------------------------------------------------------------------------- copula

class GaussianCopula:
    name = "gaussian_copula"

    def __init__(self, schema: Schema = SCHEMA, seed: int = 0):
        self.schema, self.seed = schema, seed

    def fit(self, df: pd.DataFrame) -> "GaussianCopula":
        self.cols = self.schema.columns
        self.sorted, self.cats = {}, {}
        z = []
        for c in self.cols:
            if c in self.schema.continuous:
                v = df[c].to_numpy(float)
                self.sorted[c] = np.sort(v)
                u = (stats.rankdata(v) - 0.5) / len(v)
            else:
                freq = df[c].astype(str).value_counts(normalize=True).sort_index()
                self.cats[c] = (freq.index.to_numpy(), np.r_[0, freq.cumsum().to_numpy()])
                lo = self.cats[c][1][:-1][pd.Categorical(df[c].astype(str), categories=freq.index).codes]
                hi = self.cats[c][1][1:][pd.Categorical(df[c].astype(str), categories=freq.index).codes]
                u = np.random.default_rng(self.seed).uniform(lo, hi)  # spread ties inside the category's interval
            z.append(stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6)))
        self.corr = np.corrcoef(np.vstack(z))
        self.integer = {c: bool(np.allclose(df[c] % 1, 0)) for c in self.schema.continuous}
        return self

    def sample(self, n: int, condition: dict | None = None, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed if seed is not None else self.seed + 1)
        out, have = [], 0
        while have < n:
            m = max(2 * (n - have), 1000) * (20 if condition else 1)
            z = rng.multivariate_normal(np.zeros(len(self.cols)), self.corr, m, method="cholesky")
            u = stats.norm.cdf(z)
            df = pd.DataFrame({c: self._inv(c, u[:, i]) for i, c in enumerate(self.cols)})
            if condition:
                for k, v in condition.items():
                    df = df[df[k] == v]
            out.append(df)
            have += len(df)
        return pd.concat(out).head(n).reset_index(drop=True)

    def _inv(self, c, u):
        if c in self.sorted:
            x = np.quantile(self.sorted[c], u)
            return np.round(x) if self.integer[c] else np.round(x, 2)
        labels, edges = self.cats[c]
        return labels[np.clip(np.searchsorted(edges, u, side="right") - 1, 0, len(labels) - 1)]


# --------------------------------------------------------------------------- privacy accounting

def rdp_epsilon(q: float, sigma: float, steps: int, delta: float = 1e-5) -> float:
    """(epsilon, delta) of the sampled Gaussian mechanism over `steps` (Mironov et al. 2019)."""
    if sigma <= 0:
        return math.inf
    best = math.inf
    for a in range(2, 256):
        k = np.arange(a + 1)
        logb = gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1)
        terms = logb + (a - k) * np.log1p(-q) + k * np.log(q) + (k * k - k) / (2 * sigma**2)
        rdp = steps * logsumexp(terms) / (a - 1)
        best = min(best, rdp + math.log(1 / delta) / (a - 1))
    return float(best)


def noise_for_epsilon(target: float, q: float, steps: int, delta: float = 1e-5) -> float:
    lo, hi = 0.3, 50.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if rdp_epsilon(q, mid, steps, delta) > target:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------- TVAE

class _Net(nn.Module):
    def __init__(self, d: int, c: int, n_cont: int, latent: int = 32, hidden: int = 192):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.full((max(n_cont, 1),), -2.0))  # per-column output noise (TVAE)
        self.enc = nn.Sequential(nn.Linear(d + c, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2 * latent))
        self.dec = nn.Sequential(nn.Linear(latent + c, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.latent = latent


class TVAE:
    name = "tvae"

    def __init__(self, schema: Schema = SCHEMA, epochs: int = 150, batch: int = 512, lr: float = 1e-3,
                 dp_epsilon: float | None = None, delta: float = 1e-5, clip: float = 1.0, seed: int = 0, device: str | None = None):
        self.schema, self.epochs, self.batch, self.lr = schema, epochs, batch, lr
        self.dp_epsilon, self.delta, self.clip, self.seed = dp_epsilon, delta, clip, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.spent_epsilon = math.inf

    def _cond(self, df: pd.DataFrame) -> np.ndarray:
        idx = pd.Categorical(df[self.schema.condition].astype(str), categories=self.conditions).codes
        return np.eye(len(self.conditions), dtype=np.float32)[idx]

    def _loss(self, net, x, c):
        h = net.enc(torch.cat([x, c], -1))
        mu, logvar = h.chunk(2, -1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        out = net.dec(torch.cat([z, c], -1))
        rec, j = 0.0, 0
        for col, s, e in self.tf.spans():
            if col.kind == "continuous":
                sig = net.log_sigma[j].exp()
                rec = rec + (torch.tanh(out[..., s]) - x[..., s]) ** 2 / (2 * sig**2) + net.log_sigma[j]
                rec = rec + F.cross_entropy(out[..., s + 1 : e], x[..., s + 1 : e].argmax(-1), reduction="none")
                j += 1
            else:
                rec = rec + F.cross_entropy(out[..., s:e], x[..., s:e].argmax(-1), reduction="none")
        kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), -1)
        return rec + kl

    def fit(self, df: pd.DataFrame, log=None) -> "TVAE":
        torch.manual_seed(self.seed)
        self.conditions = sorted(df[self.schema.condition].astype(str).unique())
        self.tf = DataTransformer(seed=self.seed).fit(df, self.schema.continuous, self.schema.categorical)
        x = torch.as_tensor(self.tf.transform(df), device=self.device)
        c = torch.as_tensor(self._cond(df), device=self.device)
        # rare conditions are sampled more often so the decoder learns them (not under DP: it changes sampling rates)
        freq = c.mean(0)
        w = (1 / freq[c.argmax(1)]) ** 0.5 if self.dp_epsilon is None else torch.ones(len(x), device=self.device)
        self.cond_freq = freq.cpu().numpy()
        net = _Net(x.shape[1], c.shape[1], len(self.schema.continuous)).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-5)
        n = len(x)
        steps_per_epoch = max(1, n // self.batch)
        q = self.batch / n
        sigma = noise_for_epsilon(self.dp_epsilon, q, steps_per_epoch * self.epochs, self.delta) if self.dp_epsilon else 0.0
        self.sigma = sigma
        per_example = vmap(grad(lambda p, xi, ci: self._loss_fn(net, p, xi, ci)), in_dims=(None, 0, 0), randomness="different")
        for epoch in range(self.epochs):
            net.train()
            for _ in range(steps_per_epoch):
                if sigma == 0:
                    idx = torch.multinomial(w, self.batch, replacement=True)
                    opt.zero_grad()
                    self._loss(net, x[idx], c[idx]).mean().backward()
                    opt.step()
                else:
                    mask = torch.rand(n, device=self.device) < q  # Poisson sampling for the accountant
                    if not mask.any():
                        continue
                    params = {k: v.detach() for k, v in net.named_parameters()}
                    g = per_example(params, x[mask], c[mask])
                    norms = torch.sqrt(sum(v.reshape(v.shape[0], -1).pow(2).sum(1) for v in g.values()))
                    f = (self.clip / (norms + 1e-6)).clamp(max=1.0)
                    opt.zero_grad()
                    for k, p in net.named_parameters():
                        gs = (g[k] * f.view(-1, *[1] * (g[k].dim() - 1))).sum(0)
                        p.grad = (gs + torch.randn_like(p) * sigma * self.clip) / (q * n)
                    opt.step()
            if log and (epoch + 1) % max(1, self.epochs // 6) == 0:
                with torch.no_grad():
                    log(f"  epoch {epoch + 1}/{self.epochs}  loss {self._loss(net, x[:4096], c[:4096]).mean().item():.3f}")
        self.net = net.eval()
        self.spent_epsilon = rdp_epsilon(q, sigma, steps_per_epoch * self.epochs, self.delta) if sigma else math.inf
        return self

    def _loss_fn(self, net, p, xi, ci):
        h = functional_call(net.enc, {k[4:]: v for k, v in p.items() if k.startswith("enc.")}, (torch.cat([xi, ci], -1).unsqueeze(0),))
        mu, logvar = h.chunk(2, -1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        out = functional_call(net.dec, {k[4:]: v for k, v in p.items() if k.startswith("dec.")}, (torch.cat([z, ci.unsqueeze(0)], -1),))
        loss, j = 0.0, 0
        x = xi.unsqueeze(0)
        for col, s, e in self.tf.spans():
            if col.kind == "continuous":
                ls = p["log_sigma"][j]
                loss = loss + ((torch.tanh(out[:, s]) - x[:, s]) ** 2 / (2 * ls.exp() ** 2) + ls).sum()
                loss = loss + F.cross_entropy(out[:, s + 1 : e], x[:, s + 1 : e].argmax(-1))
                j += 1
            else:
                loss = loss + F.cross_entropy(out[:, s:e], x[:, s:e].argmax(-1))
        return loss + (-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()))

    @torch.no_grad()
    def sample(self, n: int, condition: dict | None = None, seed: int | None = None) -> pd.DataFrame:
        g = torch.Generator(device=self.device).manual_seed(seed if seed is not None else self.seed + 1)
        if condition and self.schema.condition in condition:
            ci = np.full(n, self.conditions.index(str(condition[self.schema.condition])))
        else:
            ci = np.random.default_rng(seed).choice(len(self.conditions), n, p=self.cond_freq / self.cond_freq.sum())
        c = torch.as_tensor(np.eye(len(self.conditions), dtype=np.float32)[ci], device=self.device)
        z = torch.randn(n, self.net.latent, generator=g, device=self.device)
        out = self.net.dec(torch.cat([z, c], -1))
        parts, j = [], 0
        for col, s, e in self.tf.spans():
            if col.kind == "continuous":
                alpha = torch.tanh(out[:, s : s + 1]) + torch.randn(n, 1, generator=g, device=self.device) * self.net.log_sigma[j].exp()
                j += 1
                parts += [alpha.clamp(-1, 1), F.one_hot(torch.multinomial(F.softmax(out[:, s + 1 : e], -1), 1, generator=g).squeeze(1), e - s - 1).float()]
            else:
                parts.append(F.one_hot(torch.multinomial(F.softmax(out[:, s:e], -1), 1, generator=g).squeeze(1), e - s).float())
        df = self.tf.inverse(torch.cat(parts, 1).cpu().numpy())
        # condition is authoritative for the label columns
        df[self.schema.condition] = np.array(self.conditions)[ci]
        df[self.schema.target] = np.where(df[self.schema.condition] == "none", "0", "1")
        return df[self.schema.columns]


# --------------------------------------------------------------------------- conditional copula (optionally DP)

def _zcdp_to_eps(rho: float, delta: float) -> float:
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def _rho_for_eps(eps: float, delta: float) -> float:
    lo, hi = 0.0, eps
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _zcdp_to_eps(mid, delta) < eps else (lo, mid)
    return lo


class ConditionalCopula:
    """One Gaussian copula per fraud type, so each kind of fraud keeps its own joint
    pattern. With dp_epsilon set it becomes differentially private:

    * every person belongs to exactly one fraud type, so classes are disjoint and the
      budget applies to each in parallel
    * marginals are histograms over public bins (`source.BOUNDS`), released with the
      Gaussian mechanism
    * the copula correlation comes from a noisy sum of clipped normal scores
    * class sizes are released with noise too
    Accounting is in zero-concentrated DP and converted to (epsilon, delta).
    """

    def __init__(self, schema: Schema = SCHEMA, dp_epsilon: float | None = None, delta: float = 1e-5, bins: int = 48,
                 min_rows: int = 25, seed: int = 0):
        self.schema, self.dp_epsilon, self.delta, self.bins, self.min_rows, self.seed = schema, dp_epsilon, delta, bins, min_rows, seed
        self.name = "conditional_copula"
        self.spent_epsilon = math.inf

    # ---- binning on public bounds
    def _edges(self, c):
        from .source import BOUNDS

        lo, hi, scale = BOUNDS[c]
        if scale == "log":
            return np.geomspace(lo, hi, self.bins + 1)
        if scale == "log1p":
            return np.expm1(np.linspace(np.log1p(lo), np.log1p(hi), self.bins + 1))
        return np.linspace(lo, hi + 1e-9, self.bins + 1)

    def fit(self, df: pd.DataFrame) -> "ConditionalCopula":
        from .source import CATEGORIES

        rng = np.random.default_rng(self.seed)
        cols = self.schema.columns
        k = len(cols)
        dp = self.dp_epsilon is not None
        if dp:
            rho = _rho_for_eps(self.dp_epsilon, self.delta)
            # rho split: 5% class sizes, 55% marginals (all columns jointly, L2 sensitivity sqrt(k)), 40% correlation
            sig_n = math.sqrt(1 / (2 * 0.05 * rho))
            sig_m = math.sqrt(k / (2 * 0.55 * rho))
            clipz = 2.5
            sig_c = math.sqrt((k * clipz**2) ** 2 / (2 * 0.40 * rho))
            self.spent_epsilon = _zcdp_to_eps(rho, self.delta)
        self.classes = {}
        cond = df[self.schema.condition].astype(str)
        for cls in (CATEGORIES[self.schema.condition] if dp else sorted(cond.unique())):
            part = df[cond == cls]
            n_noisy = len(part) + (rng.normal(0, sig_n) if dp else 0.0)
            if n_noisy < (self.min_rows if not dp else 5):
                continue
            margins = {}
            for c in cols:
                if c in self.schema.continuous:
                    edges = self._edges(c)
                    h = np.histogram(np.clip(part[c].to_numpy(float), edges[0], edges[-1]), edges)[0].astype(float)
                else:
                    cats = CATEGORIES[c] if dp else sorted(df[c].astype(str).unique())
                    h = part[c].astype(str).value_counts().reindex(cats).fillna(0).to_numpy(float)
                    edges = np.array(cats)
                if dp:
                    h = np.clip(h + rng.normal(0, sig_m, len(h)), 0, None)
                if h.sum() <= 0:
                    h = np.ones_like(h)
                margins[c] = (edges, h / h.sum())
            # normal scores of each row through the (released) marginal CDFs
            z = np.column_stack([self._to_normal(c, part[c], margins[c], rng) for c in cols])
            if dp:
                z = np.clip(z, -clipz, clipz)
                s = z.T @ z + rng.normal(0, sig_c, (k, k))
                s = (s + s.T) / 2
                cov = s / max(n_noisy, 1.0)
            else:
                cov = np.cov(z, rowvar=False) if len(part) > 2 else np.eye(k)
            self.classes[cls] = (max(n_noisy, 0.0), margins, self._nearest_corr(cov))
        tot = sum(v[0] for v in self.classes.values())
        self.weights = {c: v[0] / tot for c, v in self.classes.items()}
        return self

    def _to_normal(self, c, values, margin, rng):
        edges, p = margin
        if c in self.schema.continuous:
            v = np.clip(values.to_numpy(float), edges[0], edges[-1])
            b = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(p) - 1)
            within = (v - edges[b]) / np.maximum(edges[b + 1] - edges[b], 1e-12)
        else:
            b = pd.Categorical(values.astype(str), categories=list(edges)).codes
            b = np.clip(b, 0, len(p) - 1)
            within = rng.random(len(b))
        cdf = np.r_[0, np.cumsum(p)]
        u = cdf[b] + p[b] * within
        return stats.norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))

    @staticmethod
    def _nearest_corr(cov):
        cov = np.nan_to_num(cov)
        w, v = np.linalg.eigh((cov + cov.T) / 2)
        cov = v @ np.diag(np.clip(w, 1e-3, None)) @ v.T
        d = np.sqrt(np.diag(cov))
        return cov / np.outer(d, d)

    def sample(self, n: int, condition: dict | None = None, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed if seed is not None else self.seed + 1)
        if condition and self.schema.condition in condition:
            counts = {str(condition[self.schema.condition]): n}
        else:
            names = list(self.weights)
            draw = rng.choice(len(names), n, p=np.array([self.weights[c] for c in names]))
            counts = {names[i]: int((draw == i).sum()) for i in range(len(names))}
        frames = []
        cols = self.schema.columns
        for cls, m in counts.items():
            if m == 0 or cls not in self.classes:
                continue
            _, margins, corr = self.classes[cls]
            u = stats.norm.cdf(rng.multivariate_normal(np.zeros(len(cols)), corr, m, method="eigh"))
            out = {}
            for i, c in enumerate(cols):
                edges, p = margins[c]
                cdf = np.r_[0, np.cumsum(p)]
                b = np.clip(np.searchsorted(cdf, u[:, i], side="right") - 1, 0, len(p) - 1)
                if c in self.schema.continuous:
                    frac = (u[:, i] - cdf[b]) / np.maximum(p[b], 1e-12)
                    x = edges[b] + np.clip(frac, 0, 1) * (edges[b + 1] - edges[b])
                    out[c] = np.round(x) if c in ("customer_age", "tenure_months", "hour", "txn_count_24h", "device_age_days") else np.round(x, 2)
                else:
                    out[c] = np.asarray(edges)[b]
            df = pd.DataFrame(out)
            df[self.schema.condition] = cls
            df[self.schema.target] = "0" if cls == "none" else "1"
            frames.append(df)
        return pd.concat(frames).sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)[cols]
