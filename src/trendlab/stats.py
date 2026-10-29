"""Statistical inference for backtests: is the Sharpe ratio real, or luck and selection?

Everything works on *per-period* simple returns. Sharpe ratios are per period
unless a name says ``annual``; ``annualize_sharpe`` converts.

* **Probabilistic Sharpe ratio** (Bailey & Lopez de Prado 2012, "The Sharpe
  Ratio Efficient Frontier", J. of Risk 15(2)): the probability that the true
  Sharpe exceeds a benchmark ``SR*`` given the sample length, skewness ``g3``
  and (non-excess) kurtosis ``g4``::

      PSR = Phi[(SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2)]

* **Deflated Sharpe ratio** (Bailey & Lopez de Prado 2014, "The Deflated Sharpe
  Ratio", J. of Portfolio Management 40(5)): PSR against the Sharpe the best
  of ``N`` *skill-less* trials would reach by chance, ``SR0``.
* **Lo (2002)** "The Statistics of Sharpe Ratios", FAJ 58(4): annualising with
  ``sqrt(q)`` is wrong when returns are autocorrelated.
* **PBO via CSCV** (Bailey, Borwein, Lopez de Prado & Zhu 2017, "The
  Probability of Backtest Overfitting", J. of Computational Finance 20(4)):
  how often the in-sample winner of a parameter search is below the median
  out of sample.
* **Stationary bootstrap** (Politis & Romano 1994, JASA 89(428)) for confidence
  intervals that respect volatility clustering; the Sharpe-difference test
  follows the paired, time-series bootstrap idea of Ledoit & Wolf (2008),
  "Robust performance hypothesis testing with the Sharpe ratio", J. of
  Empirical Finance 15(5).
* **Random-entry test**: a Monte Carlo null in which the strategy keeps its
  trades (count, lengths, exposure path) but they are placed at random times.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import metrics

EULER_GAMMA = 0.5772156649015329
_CHUNK_CELLS = 2_000_000  # elements per vectorised Monte Carlo chunk


# --------------------------------------------------------------------------- helpers


def _excess(returns, rf=None) -> np.ndarray:
    """Excess returns as a float array; a Series ``rf`` is aligned on the index."""
    if isinstance(returns, pd.Series):
        r = returns.astype(float)
        if rf is None:
            return r.to_numpy()
        if np.isscalar(rf):
            return (r - float(rf)).to_numpy()
        if isinstance(rf, pd.Series):
            return (r - rf.reindex(r.index).fillna(0.0)).to_numpy()
        return r.to_numpy() - np.asarray(rf, dtype=float)
    r = np.asarray(returns, dtype=float)
    return r if rf is None else r - np.asarray(rf, dtype=float)


def _sr(x: np.ndarray) -> float:
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return float(x.mean() / sd) if sd and np.isfinite(sd) else float("nan")


def _moments(x: np.ndarray) -> tuple[float, float, float]:
    """Per-period Sharpe, skewness and non-excess kurtosis (pandas' bias-corrected moments)."""
    s = pd.Series(x)
    return _sr(x), float(s.skew()), float(s.kurt() + 3.0)


def annualize_sharpe(sr: float | np.ndarray, periods: int = 252) -> float | np.ndarray:
    """Per-period Sharpe -> annual, assuming iid returns (``sr * sqrt(periods)``)."""
    return sr * math.sqrt(periods)


def deannualize_sharpe(sr_annual: float | np.ndarray, periods: int = 252) -> float | np.ndarray:
    """Annual Sharpe -> per period (``sr / sqrt(periods)``)."""
    return sr_annual / math.sqrt(periods)


def sharpe_per_period(returns, rf=None) -> float:
    """Per-period Sharpe ratio of (excess) returns, sample std with ddof=1; NaN bars dropped."""
    x = _excess(returns, rf)
    return _sr(x[np.isfinite(x)])


def trial_sharpes(returns_matrix: pd.DataFrame, rf=None) -> pd.Series:
    """Per-period Sharpe of every column (one column per configuration tried)."""
    return pd.Series(
        {c: sharpe_per_period(returns_matrix[c], rf) for c in returns_matrix.columns},
        name="sharpe",
    )


# --------------------------------------------------------------------------- PSR / DSR


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0, rf=None) -> float:
    """P(true per-period Sharpe > ``sr_benchmark``) under Bailey & Lopez de Prado (2012).

    ``sr_benchmark`` is per period (divide an annual target by ``sqrt(periods)``).
    """
    x = _excess(returns, rf)
    x = x[np.isfinite(x)]
    t = len(x)
    if t < 3:
        return float("nan")
    sr, g3, g4 = _moments(x)
    if not np.isfinite(sr):
        return float("nan")
    denom = max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2, 1e-12)
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(denom)))


def min_track_record_length(
    returns, sr_benchmark: float = 0.0, alpha: float = 0.05, rf=None
) -> float:
    """Bars needed for PSR(``sr_benchmark``) to reach ``1 - alpha`` (inf if SR <= benchmark)."""
    x = _excess(returns, rf)
    x = x[np.isfinite(x)]
    sr, g3, g4 = _moments(x)
    if not np.isfinite(sr) or sr <= sr_benchmark:
        return float("inf")
    denom = max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2, 1e-12)
    return float(1.0 + denom * (norm.ppf(1.0 - alpha) / (sr - sr_benchmark)) ** 2)


def sharpe_difference_test(returns_a, returns_b, rf=None, alpha: float = 0.05) -> dict:
    """Paired PSR and MinTRL for ``SR_a - SR_b`` of two return series on the same dates.

    Unlike :func:`probabilistic_sharpe_ratio` with ``sr_benchmark = SR_b``, this
    does not treat the benchmark's *estimated* Sharpe as a known constant: the
    per-observation variance of the difference is

        V = V_a + V_b - 2 * (rho + SR_a * SR_b * rho**2 / 2)

    with ``V_i = 1 - g3_i SR_i + (g4_i - 1)/4 SR_i**2`` (Mertens 2002, as in the
    PSR) and the Jobson-Korkie / Memmel (2003) covariance term. Highly
    correlated series (a strategy and its own underlying) have a much smaller
    ``V`` than ``V_a`` alone. Returns per-period ``sr_a``, ``sr_b``, ``rho``,
    ``se`` (per period), ``psr`` = P(SR_a > SR_b) and ``mintrl`` (bars needed
    for ``psr`` to reach ``1 - alpha``; inf if ``SR_a <= SR_b``).
    """
    xa, xb = _excess(returns_a, rf), _excess(returns_b, rf)
    ok = np.isfinite(xa) & np.isfinite(xb)
    xa, xb = xa[ok], xb[ok]
    t = len(xa)
    nan = float("nan")
    if t < 3:
        return {"sr_a": nan, "sr_b": nan, "rho": nan, "se": nan, "psr": nan, "mintrl": nan}
    sa, g3a, g4a = _moments(xa)
    sb, g3b, g4b = _moments(xb)
    rho = float(np.corrcoef(xa, xb)[0, 1])
    va = 1.0 - g3a * sa + (g4a - 1.0) / 4.0 * sa**2
    vb = 1.0 - g3b * sb + (g4b - 1.0) / 4.0 * sb**2
    v = max(va + vb - 2.0 * (rho + 0.5 * sa * sb * rho**2), 1e-12)
    diff = sa - sb
    psr = float(norm.cdf(diff * math.sqrt(t - 1) / math.sqrt(v)))
    mintrl = float(1.0 + v * (norm.ppf(1.0 - alpha) / diff) ** 2) if diff > 0 else float("inf")
    return {
        "sr_a": sa,
        "sr_b": sb,
        "rho": rho,
        "se": math.sqrt(v / (t - 1)),
        "psr": psr,
        "mintrl": mintrl,
    }


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum of ``n_trials`` zero-mean Sharpe estimates with variance ``sr_variance``.

    ``SR0 = sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))``,
    ``g`` = Euler-Mascheroni (Bailey & Lopez de Prado 2014, eq. 6). Units follow
    ``sr_variance`` (per-period variance -> per-period SR0).
    """
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    returns,
    trial_sharpes: np.ndarray | pd.Series | list[float],
    rf=None,
    n_trials: int | None = None,
    periods: int = 252,
) -> dict[str, float]:
    """Deflated Sharpe ratio: PSR of ``returns`` against the selection benchmark ``SR0``.

    ``trial_sharpes`` are the per-period Sharpe ratios of *every* configuration
    tried (including the chosen one); their cross-sectional variance sets
    ``SR0``. ``n_trials`` overrides ``len(trial_sharpes)`` with an effective
    number of independent trials (e.g. after clustering correlated configs).

    Returns ``dsr`` (probability), ``sr``/``sr0`` (per period), their annualised
    versions, ``n_trials``, ``sr_variance``, ``skew``, ``kurtosis`` and ``t``.
    """
    ts = np.asarray(trial_sharpes, dtype=float)
    ts = ts[np.isfinite(ts)]
    n = int(n_trials if n_trials is not None else len(ts))
    var = float(ts.var(ddof=1)) if len(ts) > 1 else 0.0
    sr0 = expected_max_sharpe(var, n)
    x = _excess(returns, rf)
    x = x[np.isfinite(x)]
    sr, g3, g4 = _moments(x)
    return {
        "dsr": probabilistic_sharpe_ratio(x, sr0),
        "sr": sr,
        "sr0": sr0,
        "sr_annual": float(annualize_sharpe(sr, periods)),
        "sr0_annual": float(annualize_sharpe(sr0, periods)),
        "n_trials": n,
        "sr_variance": var,
        "skew": g3,
        "kurtosis": g4,
        "t": len(x),
    }


def lo_adjusted_sharpe(returns, rf=None, periods: int = 252, lags: int = 10) -> float:
    """Annual Sharpe corrected for serial correlation (Lo 2002, eq. 21).

    ``SR_annual = SR * q / sqrt(q + 2 sum_{k=1}^{q-1} (q - k) rho_k)`` with
    ``q = periods``; autocorrelations beyond ``lags`` are taken as zero because
    sample estimates at long lags are mostly noise. Positive autocorrelation
    (typical of smoothed or trend-following P&L) shrinks the annual Sharpe.
    Returns NaN if the implied variance ratio is non-positive.
    """
    x = _excess(returns, rf)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    sr = _sr(x)
    q = int(periods)
    xc = x - x.mean()
    denom = float(xc @ xc)
    kmax = min(lags, q - 1, len(x) - 2)
    rho = np.array([xc[k:] @ xc[:-k] / denom for k in range(1, kmax + 1)])
    k = np.arange(1, kmax + 1)
    vr = q + 2.0 * float(((q - k) * rho).sum())
    if vr <= 0 or not np.isfinite(sr):
        return float("nan")
    return float(sr * q / math.sqrt(vr))


# --------------------------------------------------------------------------- PBO / CSCV


def _block_bounds(t: int, n_blocks: int) -> np.ndarray:
    return np.linspace(0, t, n_blocks + 1).round().astype(int)


def _split_metric(n, s1, s2, dn, mu, metric: str) -> np.ndarray:
    """Metric from aggregated sums; ``s1``/``s2`` are of data centred by ``mu``."""
    n = n[:, None]
    mean_c = s1 / n
    if metric == "mean":
        return mean_c + mu
    if metric == "sharpe":
        var = (s2 - n * mean_c**2) / (n - 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (mean_c + mu) / np.sqrt(var)
    if metric == "sortino":
        with np.errstate(divide="ignore", invalid="ignore"):
            return (mean_c + mu) / np.sqrt(dn / n)
    raise ValueError(f"unknown metric {metric!r}")


def pbo_cscv(
    returns_matrix: pd.DataFrame | np.ndarray,
    n_blocks: int = 16,
    metric: str | Callable[[np.ndarray], np.ndarray] = "sharpe",
    rf=None,
) -> dict:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation.

    The ``T x N`` matrix (rows = bars, columns = configurations) is cut into
    ``n_blocks`` contiguous blocks. For each of the ``C(S, S/2)`` ways to pick
    half the blocks as in-sample (IS), the IS-best configuration ``n*`` is
    ranked out of sample (OOS, the complementary blocks). With relative rank
    ``w = rank / (N + 1)`` and logit ``l = ln(w / (1 - w))``, ``PBO = P(l <= 0)``.

    ``metric`` is ``'sharpe'``, ``'sortino'``, ``'mean'`` (exact and vectorised
    via per-block sums) or a callable mapping a ``t x N`` array to ``N`` scores
    (looped, slow). ``rf`` (scalar or a Series aligned on the rows) is
    subtracted from every column first.

    Returns ``pbo``, ``logits`` (per combination), ``is_best_oos_rank`` (``w``),
    ``selected`` (column index of ``n*``), ``is_metric``/``oos_metric`` of
    ``n*``, ``performance_degradation`` (OLS slope of OOS on IS metric of
    ``n*``), ``degradation_intercept``, ``prob_oos_loss`` (P(OOS metric < 0)),
    ``n_combinations``, ``n_blocks``, ``n_configs``.
    """
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even integer >= 2")
    if isinstance(returns_matrix, pd.DataFrame):
        if rf is not None and not np.isscalar(rf):
            rf = pd.Series(rf).reindex(returns_matrix.index).fillna(0.0).to_numpy()[:, None]
        x = returns_matrix.to_numpy(dtype=float)
    else:
        x = np.asarray(returns_matrix, dtype=float)
        if rf is not None and not np.isscalar(rf):
            rf = np.asarray(rf, dtype=float)[:, None]
    if rf is not None:
        x = x - rf
    if not np.isfinite(x).all():
        raise ValueError("returns_matrix contains NaN/inf; align or trim configs first")
    t, n_cfg = x.shape
    if n_cfg < 2 or t < 2 * n_blocks:
        raise ValueError("need >= 2 configurations and >= 2 rows per block")

    combos = np.array(list(itertools.combinations(range(n_blocks), n_blocks // 2)))
    is_mask = np.zeros((len(combos), n_blocks), dtype=bool)
    is_mask[np.arange(len(combos))[:, None], combos] = True
    oos_mask = ~is_mask

    bounds = _block_bounds(t, n_blocks)
    if callable(metric):
        blocks = [x[bounds[i] : bounds[i + 1]] for i in range(n_blocks)]
        is_m = np.empty((len(combos), n_cfg))
        oos_m = np.empty((len(combos), n_cfg))
        for c in range(len(combos)):
            is_m[c] = metric(np.concatenate([blocks[i] for i in np.flatnonzero(is_mask[c])]))
            oos_m[c] = metric(np.concatenate([blocks[i] for i in np.flatnonzero(oos_mask[c])]))
    else:
        mu = x.mean(axis=0)
        xc = x - mu  # centring keeps the sum-of-squares variance numerically stable
        cuts = bounds[:-1]
        cnt = np.diff(bounds).astype(float)
        s1 = np.add.reduceat(xc, cuts, axis=0)
        s2 = np.add.reduceat(xc**2, cuts, axis=0)
        dn = np.add.reduceat(np.minimum(x, 0.0) ** 2, cuts, axis=0)
        im, om = is_mask.astype(float), oos_mask.astype(float)
        is_m = _split_metric(im @ cnt, im @ s1, im @ s2, im @ dn, mu, metric)
        oos_m = _split_metric(om @ cnt, om @ s1, om @ s2, om @ dn, mu, metric)

    is_m = np.where(np.isfinite(is_m), is_m, -np.inf)
    oos_m = np.where(np.isfinite(oos_m), oos_m, -np.inf)
    rows = np.arange(len(combos))
    # Configs that coincide over the IS blocks tie exactly; pick the lowest index
    # among near-ties so the choice does not depend on summation order.
    top = is_m.max(axis=1, keepdims=True)
    tied = is_m >= top - 1e-10 * (1.0 + np.abs(top))
    sel = np.argmax(tied, axis=1)
    v = oos_m[rows, sel]
    less = (oos_m < v[:, None]).sum(axis=1)
    ties = (oos_m == v[:, None]).sum(axis=1)
    rank = less + (ties + 1) / 2.0  # 1 = worst, N = best; ties get the average rank
    w = rank / (n_cfg + 1.0)
    logits = np.log(w / (1.0 - w))

    is_sel = is_m[rows, sel]
    ok = np.isfinite(is_sel) & np.isfinite(v)
    if ok.sum() > 2 and np.ptp(is_sel[ok]) > 0:
        slope, intercept = np.polyfit(is_sel[ok], v[ok], 1)
    else:
        slope = intercept = float("nan")
    return {
        "pbo": float((logits <= 0).mean()),
        "logits": logits,
        "is_best_oos_rank": w,
        "selected": sel,
        "is_metric": is_sel,
        "oos_metric": v,
        "performance_degradation": float(slope),
        "degradation_intercept": float(intercept),
        "prob_oos_loss": float((v < 0).mean()),
        "n_combinations": int(len(combos)),
        "n_blocks": int(n_blocks),
        "n_configs": int(n_cfg),
    }


# --------------------------------------------------------------------------- bootstrap


def _sb_chunk(rng: np.random.Generator, n: int, p: float, m: int) -> np.ndarray:
    """``m`` stationary-bootstrap index paths of length ``n`` (circular wrap)."""
    starts = rng.integers(0, n, size=(m, n))
    new = rng.random((m, n)) < p
    new[:, 0] = True
    t = np.arange(n)
    last = np.maximum.accumulate(np.where(new, t, 0), axis=1)
    return (np.take_along_axis(starts, last, axis=1) + (t - last)) % n


def _iter_sb(n: int, mean_block: float, n_boot: int, seed: int | None) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    p = 1.0 / max(float(mean_block), 1.0)
    chunk = max(1, _CHUNK_CELLS // max(n, 1))
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        yield _sb_chunk(rng, n, p, m)
        done += m


def stationary_bootstrap_indices(
    n: int, mean_block: float = 20.0, n_boot: int = 1000, seed: int | None = 0
) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap: ``(n_boot, n)`` integer index array.

    Each path starts at a uniform random bar; at every step it starts a new
    block (uniform random start) with probability ``1 / mean_block``, otherwise
    continues to the next bar, wrapping around the end. Block lengths are thus
    geometric with mean ``mean_block`` and the resampled series is stationary.
    """
    return np.concatenate(list(_iter_sb(n, mean_block, n_boot, seed)), axis=0)


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return x.mean(axis=1) / sd


def _align_pair(r_a, r_b, rf) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(r_a, pd.Series) and isinstance(r_b, pd.Series):
        idx = r_a.index.intersection(r_b.index)
        r_a, r_b = r_a.loc[idx], r_b.loc[idx]
    a, b = _excess(r_a, rf), _excess(r_b, rf)
    if len(a) != len(b):
        raise ValueError("r_a and r_b must have the same length")
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def bootstrap_sharpe_difference(
    r_a,
    r_b,
    rf=None,
    n_boot: int = 5000,
    mean_block: float = 20.0,
    seed: int | None = 0,
    periods: int = 252,
    alpha: float = 0.05,
) -> dict:
    """Paired stationary-bootstrap test of ``SR(a) - SR(b)`` (annualised).

    Both series are resampled with the *same* indices, preserving their
    correlation and volatility clustering. The CI is the percentile interval.
    The one-sided p-value for H0: ``diff <= 0`` centres the bootstrap
    distribution on zero (``samples - diff``) and counts how often the centred
    statistic is at least the observed difference.

    Returns ``diff``, ``sharpe_a``, ``sharpe_b``, ``ci_low``, ``ci_high``,
    ``p_value``, ``samples`` (array of ``n_boot`` bootstrap differences).
    """
    a, b = _align_pair(r_a, r_b, rf)
    scale = math.sqrt(periods)
    sa, sb = _sr(a) * scale, _sr(b) * scale
    diff = sa - sb
    out = []
    for idx in _iter_sb(len(a), mean_block, n_boot, seed):
        out.append((_sharpe_rows(a[idx]) - _sharpe_rows(b[idx])) * scale)
    samples = np.concatenate(out)
    fin = samples[np.isfinite(samples)]
    if len(fin) == 0 or not np.isfinite(diff):
        lo = hi = p = float("nan")
    else:
        lo, hi = np.quantile(fin, [alpha / 2, 1 - alpha / 2])
        p = (1.0 + np.sum(fin - diff >= diff)) / (len(fin) + 1.0)
    return {
        "diff": float(diff),
        "sharpe_a": float(sa),
        "sharpe_b": float(sb),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p),
        "samples": samples,
    }


_SERIES_METRICS: dict[str, Callable[..., float]] = {
    "cagr": metrics.cagr,
    "max_drawdown": lambda r, periods: metrics.max_drawdown(r),
    "sortino": metrics.sortino,
    "vol": metrics.ann_vol,
}


def bootstrap_metric_ci(
    returns,
    metric: str | Callable[[pd.Series], float] = "sharpe",
    rf=None,
    n_boot: int = 2000,
    mean_block: float = 20.0,
    seed: int | None = 0,
    alpha: float = 0.05,
    periods: int = 252,
) -> dict:
    """Stationary-bootstrap percentile CI for any return metric.

    Built-ins: ``'sharpe'`` and ``'mean'`` (annualised, vectorised; excess of
    ``rf``), ``'sortino'`` (excess of ``rf``), ``'cagr'``, ``'vol'``,
    ``'max_drawdown'`` (raw returns, via :mod:`trendlab.metrics`). A callable
    receives each resampled raw-return Series and returns a float.

    Returns ``estimate``, ``ci_low``, ``ci_high``, ``se`` and ``samples``.
    """
    raw = _excess(returns, None)
    ex = _excess(returns, rf)
    ok = np.isfinite(raw) & np.isfinite(ex)
    raw, ex = raw[ok], ex[ok]
    scale = math.sqrt(periods)
    samples: list[np.ndarray] = []
    if metric in ("sharpe", "mean"):
        if metric == "sharpe":
            est = _sr(ex) * scale
            fn = lambda m: _sharpe_rows(m) * scale  # noqa: E731
        else:
            est = ex.mean() * periods
            fn = lambda m: m.mean(axis=1) * periods  # noqa: E731
        for idx in _iter_sb(len(ex), mean_block, n_boot, seed):
            samples.append(fn(ex[idx]))
    else:
        if callable(metric):
            data, f = raw, metric
        elif metric in _SERIES_METRICS:
            g = _SERIES_METRICS[metric]
            data = ex if metric == "sortino" else raw
            f = lambda s: g(s, periods=periods)  # noqa: E731
        else:
            raise ValueError(f"unknown metric {metric!r}")
        est = f(pd.Series(data))
        for idx in _iter_sb(len(data), mean_block, n_boot, seed):
            samples.append(np.array([f(pd.Series(data[row])) for row in idx]))
    s = np.concatenate(samples)
    lo, hi = np.nanquantile(s, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(est),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "se": float(np.nanstd(s, ddof=1)),
        "samples": s,
    }


# --------------------------------------------------------------------------- random entry


def _segments(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Starts and lengths of maximal runs of non-zero exposure."""
    nz = np.round(w, 12) != 0
    edges = np.diff(np.concatenate([[0], nz.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return starts, ends - starts


def random_entry_test(
    asset_returns: pd.Series | np.ndarray,
    exposure: pd.Series | np.ndarray,
    strategy_returns: pd.Series | np.ndarray | None = None,
    rf: pd.Series | np.ndarray | float | None = None,
    n_sims: int = 5000,
    seed: int | None = 0,
    periods: int = 252,
) -> dict:
    """Is the strategy's *timing* better than random timing with the same trades?

    Null model (segment permutation): the in-market segments (maximal runs of
    non-zero exposure, carrying their exact exposure path) and the flat gaps
    between them (including the leading and trailing gaps) are each shuffled
    independently and re-interleaved. Every simulated strategy therefore has
    the same number of trades, the same holding periods and the same average
    exposure as the real one; only *when* it is in the market is random. (Two
    segments can abut when a zero-length leading/trailing gap lands inside.)

    Simulated excess returns are ``w_sim * (g - rf)`` minus the actual
    strategy's mean per-bar drag ``mean(w*(g - rf) + rf - r)`` (costs and
    execution effects), so they pay the same total frictions. ``exposure`` must
    be the *held* exposure aligned with ``asset_returns`` (``BacktestResult.exposure``).
    If ``strategy_returns`` is None the frictionless ``rf + w*(g - rf)`` is used.

    Returns ``actual_sharpe`` (annualised, excess of ``rf``), ``p_value``
    (``(1 + #null >= actual) / (n_sims + 1)``), ``null_sharpes`` (array),
    ``null_mean``, ``n_segments`` and ``exposure_fraction``.
    """
    g = np.asarray(asset_returns, dtype=float)
    w = np.asarray(exposure, dtype=float)
    n = len(g)
    if len(w) != n:
        raise ValueError("asset_returns and exposure must have the same length")
    if rf is None:
        rfv = np.zeros(n)
    elif isinstance(rf, pd.Series) and isinstance(asset_returns, pd.Series):
        rfv = rf.reindex(asset_returns.index).fillna(0.0).to_numpy(dtype=float)
    else:
        rfv = np.broadcast_to(np.asarray(rf, dtype=float), (n,)).copy()
    x = np.nan_to_num(g - rfv)
    w = np.nan_to_num(w)
    frictionless = w * x
    if strategy_returns is None:
        actual_ex = frictionless
    else:
        r = np.asarray(strategy_returns, dtype=float)
        if len(r) != n:
            raise ValueError("strategy_returns must have the same length as asset_returns")
        # a missing bar (e.g. a leading pct_change NaN) earns the risk-free rate, like x above
        actual_ex = np.where(np.isfinite(r), r - rfv, 0.0)
    drag = float(np.mean(frictionless - actual_ex))
    scale = math.sqrt(periods)
    actual = _sr(actual_ex) * scale

    starts, lens = _segments(w)
    k = len(starts)
    out = {
        "actual_sharpe": float(actual),
        "n_segments": int(k),
        "exposure_fraction": float(np.mean(np.round(w, 12) != 0)),
    }
    if k == 0:
        return out | {"p_value": float("nan"), "null_sharpes": np.array([]), "null_mean": np.nan}

    ends = starts + lens
    gaps = np.concatenate([[starts[0]], starts[1:] - ends[:-1], [n - ends[-1]]])
    src = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends, strict=True)])
    seg_of = np.repeat(np.arange(k), lens)
    offset = src - starts[seg_of]
    wv = w[src]

    rng = np.random.default_rng(seed)
    chunk = max(1, _CHUNK_CELLS // max(len(src), k + 1))
    res = []
    done = 0
    while done < n_sims:
        m = min(chunk, n_sims - done)
        perm = np.argsort(rng.random((m, k)), axis=1)
        gperm = np.argsort(rng.random((m, k + 1)), axis=1)
        len_r = lens[perm]
        start_r = np.cumsum(gaps[gperm][:, :k], axis=1) + np.cumsum(len_r, axis=1) - len_r
        seg_pos = np.take_along_axis(start_r, np.argsort(perm, axis=1), axis=1)
        vals = wv * x[seg_pos[:, seg_of] + offset]
        s1, s2 = vals.sum(axis=1), (vals**2).sum(axis=1)
        sd = np.sqrt(np.maximum(s2 - s1**2 / n, 0.0) / (n - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            res.append((s1 / n - drag) / sd * scale)
        done += m
    null = np.concatenate(res)
    if not np.isfinite(actual):
        p = float("nan")
    else:
        # exact ties (e.g. a permutation reproducing the actual path) must count as >=, but
        # the null uses a one-pass variance, so allow for floating-point noise
        tol = 1e-9 * max(abs(actual), 1.0)
        p = (1.0 + np.sum(null >= actual - tol)) / (n_sims + 1.0)
    fin = null[np.isfinite(null)]
    null_mean = float(fin.mean()) if len(fin) else float("nan")
    return out | {"p_value": float(p), "null_sharpes": null, "null_mean": null_mean}


def random_entry_test_result(result, n_sims: int = 5000, seed: int | None = 0) -> dict:
    """:func:`random_entry_test` on a :class:`~trendlab.engine.BacktestResult`."""
    return random_entry_test(
        result.asset_returns,
        result.exposure,
        result.returns,
        result.rf,
        n_sims=n_sims,
        seed=seed,
        periods=result.periods_per_year,
    )
