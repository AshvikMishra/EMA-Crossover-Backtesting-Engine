"""Trading-cost estimation and cost sensitivity.

**Corwin & Schultz (2012)**, "A Simple Way to Estimate Bid-Ask Spreads from
Daily High and Low Prices", J. of Finance 67(2). Daily highs are almost always
buyer-initiated (at the ask) and lows seller-initiated (at the bid), so the
high/low range contains volatility *plus* the spread. Volatility scales with
the horizon but the spread does not, so comparing one-day and two-day ranges
separates them::

    beta  = ln(H_t/L_t)^2 + ln(H_{t+1}/L_{t+1})^2
    gamma = ln(max(H_t, H_{t+1}) / min(L_t, L_{t+1}))^2
    alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2) - sqrt(gamma / (3 - 2 sqrt 2))
    S     = 2 (e^alpha - 1) / (1 + e^alpha)

``S`` is the *full* proportional spread; a market order pays roughly half of
it, i.e. ``CostModel.half_spread_bps ~ S / 2``.

**Break-even cost**: costs in the engine are ``|dw| * bps``, so they reduce
the arithmetic annual return by exactly ``turnover_per_year * bps / 1e4`` and
the edge vanishes at ``bps* = edge / turnover * 1e4``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from . import metrics
from .engine import BacktestConfig, BacktestResult, CostModel, run_backtest
from .strategies import Strategy

_K = 3.0 - 2.0 * math.sqrt(2.0)


def corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
    window: int | None = None,
    close: pd.Series | None = None,
    clip_daily: bool = True,
) -> pd.Series:
    """Corwin-Schultz proportional bid-ask spread estimate per bar.

    The two-day estimate for days ``t-1, t`` is stored at ``t`` (first bar NaN)
    and negative estimates are set to 0, as in the paper. With ``close`` the
    paper's overnight adjustment is applied: day ``t``'s high and low are
    shifted so the range includes the previous close when the market gapped.
    With ``window`` the estimates are averaged over a trailing window (the
    paper averages over a month, ``window=21``). ``clip_daily=False`` keeps
    negative two-day estimates, so only the window average (if any) is clipped
    (the paper's robustness variant): clipping each two-day estimate biases the
    spread *upwards* when volatility is large relative to the spread, which is
    the usual case for liquid ETFs.
    """
    h = high.astype(float)
    lo = low.astype(float)
    h_prev, l_prev = h.shift(1), lo.shift(1)
    if close is not None:
        pc = close.astype(float).shift(1)
        up = (lo - pc).clip(lower=0.0).fillna(0.0)  # gap up: whole day above prior close
        down = (pc - h).clip(lower=0.0).fillna(0.0)  # gap down
        h, lo = h - up + down, lo - up + down  # only the second day of each pair moves
    beta = np.log(h / lo) ** 2 + np.log(h_prev / l_prev) ** 2
    h2 = np.maximum(h, h_prev)
    l2 = np.minimum(lo, l_prev)
    gamma = np.log(h2 / l2) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = s.where(beta.notna() & gamma.notna())
    if clip_daily:
        s = s.clip(lower=0.0)
    if window:
        s = s.rolling(window, min_periods=max(1, window // 2)).mean().clip(lower=0.0)
    return s.rename("cs_spread")


def median_spread_bps(
    prices: pd.DataFrame, window: int = 21, overnight: bool = True, clip_daily: bool = False
) -> float:
    """Median over time of the ``window``-bar average Corwin-Schultz spread, in bps (full spread).

    Defaults to clipping the window averages rather than each two-day
    estimate, which is far less biased for low-spread, high-volatility assets.
    """
    s = corwin_schultz_spread(
        prices["High"],
        prices["Low"],
        window=window,
        close=prices["Close"] if overnight else None,
        clip_daily=clip_daily,
    )
    return float(s.median() * 1e4)


def breakeven_cost_bps(gross_excess_return_annual: float, turnover_per_year: float) -> float:
    """Per-side cost (bps of traded notional) that wipes out an annual arithmetic edge.

    ``turnover_per_year`` is ``sum |dw|`` per year (a full in-and-out round trip
    of a 0/1 rule counts 2). A negative result means there is no gross edge to
    lose; zero turnover gives inf (positive edge) or NaN.
    """
    if turnover_per_year <= 0:
        return float("inf") if gross_excess_return_annual > 0 else float("nan")
    return float(gross_excess_return_annual / turnover_per_year * 1e4)


def breakeven_from_result(result: BacktestResult, benchmark: pd.Series | None = None) -> dict:
    """Break-even cost for a backtest, relative to cash (``rf``) or a ``benchmark``.

    Uses the gross (pre-cost) returns ``result.returns + result.costs``. The
    edge is the arithmetic annual mean of gross minus hurdle returns; at
    ``breakeven_bps`` the mean net excess return over the same window is zero.
    Returns ``edge_annual``, ``turnover_per_year`` and ``breakeven_bps``.
    """
    gross = result.returns + result.costs
    hurdle = result.rf if benchmark is None else benchmark.reindex(gross.index).fillna(0.0)
    p = result.periods_per_year
    edge_total = float((gross - hurdle).sum())
    turn_total = float(result.turnover.sum())
    years = len(gross) / p  # same convention as the arithmetic mean, so the ratio is exact
    return {
        "edge_annual": edge_total / years,
        "turnover_per_year": turn_total / years,
        "breakeven_bps": breakeven_cost_bps(edge_total, turn_total),
    }


def cost_sensitivity(
    prices: pd.DataFrame,
    strategy: Strategy,
    bps_grid: Iterable[float] = (0, 2, 5, 10, 20, 50),
    rf_yield: pd.Series | float | None = None,
    config_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Backtest ``strategy`` at each flat per-side cost in ``bps_grid``.

    ``config_kwargs`` are passed to :class:`BacktestConfig` (``execution``,
    ``start``, ``end``, ...; not ``costs``). The target is computed once.
    Returns a DataFrame indexed by ``bps`` with columns ``cagr``, ``sharpe``,
    ``vol``, ``max_drawdown``, ``total_return``, ``turnover_per_year`` and
    ``cost_drag_annual`` (annual arithmetic return lost to costs, which equals
    ``turnover_per_year * bps / 1e4``).
    """
    kw = dict(config_kwargs or {})
    if "costs" in kw:
        raise ValueError("pass costs through bps_grid, not config_kwargs")
    target = strategy.target_position(prices)
    rows = {}
    for bps in bps_grid:
        cfg = BacktestConfig(costs=CostModel.flat(float(bps)), **kw)
        res = run_backtest(prices, strategy, cfg, rf_yield, target=target)
        p = res.periods_per_year
        years = len(res.returns) / p  # matches costs.mean() * p, so drag = turnover * bps
        rows[float(bps)] = {
            "cagr": metrics.cagr(res.returns, p),
            "sharpe": metrics.sharpe(res.returns, res.rf, p),
            "vol": metrics.ann_vol(res.returns, p),
            "max_drawdown": metrics.max_drawdown(res.returns),
            "total_return": metrics.total_return(res.returns),
            "turnover_per_year": float(res.turnover.sum() / years),
            "cost_drag_annual": float(res.costs.mean() * p),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "bps"
    return out
