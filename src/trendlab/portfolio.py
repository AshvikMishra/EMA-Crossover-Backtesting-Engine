"""Multi-asset trend-following portfolios and static benchmarks.

A single-asset trend rule is a poor test of trend following: its result is
dominated by one market's path. The literature evaluates trend as a *portfolio*
of sleeves, one per asset, each switched between the asset and cash by its own
signal (Faber 2007, "A Quantitative Approach to Tactical Asset Allocation";
Moskowitz, Ooi & Pedersen 2012, "Time Series Momentum"; Hurst, Ooi & Pedersen
2017, "A Century of Evidence on Trend-Following Investing").

Mechanics of :func:`portfolio_backtest` (bar ``t`` runs from close ``t-1`` to
close ``t``):

* **Strategic budget** ``s_i``: equal or inverse-volatility weights (risk
  parity-lite, as in MOP 2012) among the assets that are *available* -- listed
  and warmed up -- at close ``t``, scaled to ``gross_target``. It is reset on
  each ``rebalance`` period end and whenever the available set changes, so a
  late-listing asset joins as soon as its signal exists and its budget is
  otherwise redistributed.
* **Signal** ``x_i`` from the asset's :class:`~trendlab.strategies.Strategy`.
* **Optional portfolio vol target**: decided exposure ``s_i x_i`` is scaled by
  ``min(max_leverage, vol_target / sigma_t)``, with ``sigma_t`` the trailing
  realised vol of the unscaled book or its ex-ante vol ``sqrt(w' C w)`` from a
  trailing covariance matrix (Harvey et al. 2018, "The Impact of Volatility
  Targeting"; Moreira & Muir 2017 scale by inverse *variance* instead).
* The decision at close ``t`` is held from close ``t+1`` (``next_close``) or
  close ``t`` (``same_close``), exactly like :func:`trendlab.engine.run_backtest`,
  and the per-bar return is ``sum_i w_i g_i + (1 - sum_i w_i) rf - sum_i |dw_i| c_i``.

Like the single-asset engine, sleeves are rebalanced to target every bar
(intra-month drift is ignored and costs only on target changes). Static
benchmarks (:func:`static_mix_backtest`, :func:`blend`) instead let weights
drift between rebalances and pay for the trades that restore them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import per_bar_rate, periods_per_year
from .engine import CostModel
from .indicators import realized_vol
from .strategies import Strategy

WEIGHTINGS = ("equal", "inverse_vol")
VOL_ESTIMATORS = ("realized", "ex_ante")
_PERIOD_ALIASES = {
    "ME": "M",
    "BME": "M",
    "M": "M",
    "QE": "Q",
    "BQE": "Q",
    "Q": "Q",
    "YE": "Y",
    "BYE": "Y",
    "Y": "Y",
    "A": "Y",
}

PriceInput = pd.DataFrame | pd.Series
StrategySpec = Strategy | Mapping[str, Strategy] | Callable[[str], Strategy]
CostSpec = CostModel | Mapping[str, CostModel] | None


@dataclass
class PortfolioResult:
    """Output of :func:`portfolio_backtest` / :func:`static_mix_backtest`.

    Frames are indexed by the evaluation window and have one column per ticker.
    ``weights`` is the exposure actually held over each bar; ``contributions``
    has the per-asset ``w_i * g_i`` plus ``cash`` and ``costs`` columns and its
    rows sum to ``returns``.
    """

    name: str
    returns: pd.Series
    equity: pd.Series
    weights: pd.DataFrame
    strategic: pd.DataFrame
    signals: pd.DataFrame | None
    scale: pd.Series
    turnover: pd.Series
    costs: pd.Series
    contributions: pd.DataFrame
    asset_returns: pd.DataFrame
    rf: pd.Series
    periods_per_year: int

    @property
    def start(self) -> pd.Timestamp:
        return self.returns.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.returns.index[-1]

    @property
    def cash(self) -> pd.Series:
        return (1.0 - self.weights.sum(axis=1)).rename("cash")

    @property
    def gross_exposure(self) -> pd.Series:
        return self.weights.abs().sum(axis=1).rename("gross")

    def summary(self, benchmark: PortfolioResult | pd.Series | object | None = None) -> dict:
        """:func:`trendlab.metrics.summary` with gross exposure and total turnover."""
        from .metrics import summary

        bench = getattr(benchmark, "returns", benchmark)
        return summary(
            self.returns,
            rf=self.rf,
            periods=self.periods_per_year,
            exposure=self.gross_exposure,
            turnover=self.turnover,
            benchmark=bench,
        )


def rebalance_mask(index: pd.DatetimeIndex, rebalance: str | None) -> np.ndarray:
    """True on the last bar of each ``rebalance`` period ('ME', 'QE', 'YE', 'W', ...).

    ``'D'``/``'B'`` rebalance every bar; ``None`` never (only on forced events).
    The final bar of the sample is always flagged, which is harmless because a
    decision taken there is never filled.
    """
    n = len(index)
    if rebalance is None:
        return np.zeros(n, dtype=bool)
    if rebalance in ("D", "B"):
        return np.ones(n, dtype=bool)
    if n == 0:
        return np.zeros(0, dtype=bool)
    periods = pd.DatetimeIndex(index).to_period(_PERIOD_ALIASES.get(rebalance, rebalance))
    return np.append(np.asarray(periods[1:] != periods[:-1], dtype=bool), True)


def common_calendar(
    prices_dict: Mapping[str, PriceInput], calendar: str | pd.DatetimeIndex | None = None
) -> pd.DatetimeIndex:
    """Trading calendar for a universe.

    ``calendar`` may be a ticker in ``prices_dict`` (its dates), ``'union'``
    (every weekday on which any asset traded; weekends are kept only when every
    asset trades 24/7, so a crypto-only universe keeps its 365-day calendar) or
    an explicit index. ``None`` means ``'SPY'`` when present, otherwise ``'union'``.
    """
    if isinstance(calendar, pd.DatetimeIndex):
        cal = calendar
    else:
        if calendar is None:
            calendar = "SPY" if "SPY" in prices_dict else "union"
        if calendar == "union":
            cal = pd.DatetimeIndex([])
            all_week = bool(prices_dict)
            for px in prices_dict.values():
                dates = pd.DatetimeIndex(px.index)
                cal = cal.union(dates)
                all_week &= bool((dates.day_of_week >= 5).any())
            if not all_week:
                cal = cal[cal.day_of_week < 5]
        elif calendar in prices_dict:
            cal = pd.DatetimeIndex(prices_dict[calendar].index)
        else:
            raise ValueError(
                f"calendar must be a ticker in the universe or 'union', got {calendar!r}"
            )
    return pd.DatetimeIndex(cal.unique().sort_values(), name="Date")


def _as_ohlcv(px: PriceInput) -> pd.DataFrame:
    if isinstance(px, pd.Series):
        c = px.astype(float)
        return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 0.0})
    return px


def align_universe(
    prices_dict: Mapping[str, PriceInput],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    calendar: str | pd.DatetimeIndex | None = None,
) -> dict[str, pd.DataFrame]:
    """Reindex every asset's OHLCV onto one business-day calendar.

    On calendar days an asset did not trade (a local holiday) the close is
    carried forward and Open/High/Low are set to it with zero volume, so the
    bar's return is exactly 0. Each frame runs from the asset's first calendar
    date on or after its inception to its last observation (no stale prices
    after a delisting or a shorter download); assets with no data on the
    calendar are dropped. Close-to-close returns between calendar dates stay
    exact, but Open/High/Low on a calendar day only describe that session (e.g.
    Bitcoin's weekend range is not folded into Monday's bar).
    """
    cal = common_calendar(prices_dict, calendar)
    if start is not None:
        cal = cal[cal >= pd.Timestamp(start)]
    if end is not None:
        cal = cal[cal <= pd.Timestamp(end)]
    out: dict[str, pd.DataFrame] = {}
    for ticker, px in prices_dict.items():
        df = _as_ohlcv(px)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if df.empty:
            continue
        close = df["Close"].reindex(cal, method="ffill")
        live = close.notna().to_numpy() & (cal <= df.index[-1])
        if not live.any():
            continue
        traded = cal.isin(df.index)
        frame = pd.DataFrame(index=cal)
        for col in ("Open", "High", "Low"):
            if col in df:
                frame[col] = df[col].reindex(cal).where(traded, close)
        frame["Close"] = close
        if "Volume" in df:
            frame["Volume"] = df["Volume"].reindex(cal).where(traded, 0.0)
        out[ticker] = frame.loc[live]
    return out


def _universe_index(aligned: Mapping[str, pd.DataFrame]) -> pd.DatetimeIndex:
    if not aligned:
        raise ValueError("no asset has data on the calendar")
    idx = pd.DatetimeIndex([])
    for df in aligned.values():
        idx = idx.union(df.index)
    return pd.DatetimeIndex(idx, name="Date")


def _asset_returns(aligned: Mapping[str, pd.DataFrame], idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Close-to-close returns on ``idx``; NaN where the asset is not listed (or on its first bar)."""
    rets = {t: df["Close"].pct_change().reindex(idx) for t, df in aligned.items()}
    return pd.DataFrame(rets, index=idx)


def _resolve_strategies(spec: StrategySpec, tickers: list[str]) -> dict[str, Strategy]:
    if isinstance(spec, Strategy):
        return {t: spec for t in tickers}
    if isinstance(spec, Mapping):
        missing = [t for t in tickers if t not in spec]
        if missing:
            raise KeyError(f"no strategy given for {missing}")
        return {t: spec[t] for t in tickers}
    if callable(spec):
        return {t: spec(t) for t in tickers}
    raise TypeError("strategy_factory must be a Strategy, a mapping or a callable")


def _bps(costs: CostSpec, tickers: list[str]) -> np.ndarray:
    if costs is None:
        costs = CostModel()
    if isinstance(costs, CostModel):
        return np.full(len(tickers), costs.bps)
    return np.array([costs.get(t, CostModel()).bps for t in tickers])


def _window(idx: pd.DatetimeIndex, first: int, start, end, what: str) -> tuple[int, int]:
    """Mirror run_backtest: open one bar after the first decision, honour start/end."""
    start_pos = first + 1
    if start is not None:
        start_pos = max(start_pos, int(idx.searchsorted(pd.Timestamp(start))))
    end_pos = len(idx) - 1
    if end is not None:
        end_pos = int(idx.searchsorted(pd.Timestamp(end), side="right")) - 1
    if start_pos >= end_pos:
        raise ValueError(f"{what}: empty evaluation window")
    return start_pos, end_pos


def _ex_ante_vol(weights: pd.DataFrame, rets: pd.DataFrame, window: int, periods: int) -> pd.Series:
    """Annualised ``sqrt(w_t' C_t w_t)``, ``C_t`` the covariance of the last ``window`` bars.

    ``rets`` is NaN where an asset is not listed. Each ``C_ij`` uses only the
    bars on which both assets traded (pairwise-complete), so a newly listed asset
    is not credited with zero volatility for its pre-inception bars; a pair with
    fewer than ``max(10, window // 2)`` common bars (as in ``realized_vol``)
    leaves ``sigma_t`` undefined when both carry weight.
    """
    live = rets.notna().to_numpy()
    g = np.where(live, rets.to_numpy(), 0.0)
    lv = live.astype(float)
    x = weights.to_numpy()
    n, k = g.shape
    out = np.full(n, np.nan)
    if n < window or window < 2:
        return pd.Series(out, index=weights.index)

    def rolling_sum(a: np.ndarray) -> np.ndarray:
        c = np.cumsum(np.concatenate([np.zeros((1, *a.shape[1:])), a]), axis=0)
        return c[window:] - c[:-window]

    cnt = rolling_sum(lv[:, :, None] * lv[:, None, :])
    s_ij = rolling_sum(g[:, :, None] * lv[:, None, :])  # sum of g_i on bars where j also traded
    p_ij = rolling_sum(g[:, :, None] * g[:, None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = (p_ij - s_ij * s_ij.transpose(0, 2, 1) / cnt) / (cnt - 1.0)
    cov[cnt < max(10, window // 2)] = np.nan
    xw = x[window - 1 :]
    ww = xw[:, :, None] * xw[:, None, :]
    undefined = (np.isnan(cov) & (ww != 0)).any(axis=(1, 2))
    var = (ww * np.nan_to_num(cov)).sum(axis=(1, 2))
    var[undefined] = np.nan
    out[window - 1 :] = np.sqrt(np.maximum(var, 0.0) * periods)
    return pd.Series(out, index=weights.index)


def _vol_scale(
    decided: pd.DataFrame,
    rets: pd.DataFrame,
    lag: int,
    vol_target: float,
    window: int,
    max_leverage: float,
    estimator: str,
    periods: int,
) -> pd.Series:
    if estimator == "realized":
        held = decided.shift(lag).fillna(0.0)
        book = (held * rets.fillna(0.0)).sum(axis=1)
        started = (held.abs().sum(axis=1) > 0).cummax()
        sigma = realized_vol(book.where(started), window, periods)
    else:
        sigma = _ex_ante_vol(decided, rets, window, periods)
    with np.errstate(divide="ignore"):
        scale = vol_target / sigma
    # undefined estimate: keep the last scale; before the first one, unlevered (capped)
    scale = scale.clip(upper=max_leverage).ffill()
    return scale.fillna(min(1.0, max_leverage)).rename("scale")


def portfolio_backtest(
    prices_dict: Mapping[str, PriceInput],
    strategy_factory: StrategySpec,
    *,
    weighting: str = "equal",
    vol_window: int = 63,
    rebalance: str | None = "ME",
    gross_target: float = 1.0,
    vol_target: float | None = None,
    max_leverage: float = 1.0,
    vol_estimator: str = "realized",
    costs: CostSpec = None,
    execution: str = "next_close",
    rf_yield: pd.Series | float | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    calendar: str | pd.DatetimeIndex | None = None,
    initial_capital: float = 100_000.0,
    name: str = "Trend portfolio",
) -> PortfolioResult:
    """Backtest one trend sleeve per asset inside a multi-asset portfolio.

    ``strategy_factory`` is a Strategy used for every asset, a ``{ticker:
    Strategy}`` mapping or a ``ticker -> Strategy`` callable. ``costs`` may be a
    single :class:`CostModel` or a per-ticker mapping. ``vol_window`` serves both
    the inverse-vol weights and the vol target. Signals are computed on the full
    aligned history so data before ``start`` is warm-up, as in the engine.
    """
    if execution == "next_open":
        raise NotImplementedError("portfolio_backtest supports 'next_close' and 'same_close'")
    if execution not in ("next_close", "same_close"):
        raise ValueError("execution must be 'next_close' or 'same_close'")
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    if vol_estimator not in VOL_ESTIMATORS:
        raise ValueError(f"vol_estimator must be one of {VOL_ESTIMATORS}")

    aligned = align_universe(prices_dict, calendar=calendar)
    idx = _universe_index(aligned)
    tickers = list(aligned)
    ppy = periods_per_year(idx)
    strategies = _resolve_strategies(strategy_factory, tickers)
    signals = pd.DataFrame(
        {t: strategies[t].target_position(aligned[t]).reindex(idx) for t in tickers}, index=idx
    )
    raw_rets = _asset_returns(aligned, idx)
    rets = raw_rets.fillna(0.0)

    avail = signals.notna()
    if weighting == "inverse_vol":
        vol = pd.DataFrame(
            {
                t: realized_vol(aligned[t]["Close"].pct_change(), vol_window, ppy).reindex(idx)
                for t in tickers
            },
            index=idx,
        )
        inv = (1.0 / vol).where(vol > 0)
        avail &= inv.notna()
        raw = inv.where(avail)
    else:
        raw = avail.astype(float).where(avail)
    raw_w = raw.div(raw.sum(axis=1), axis=0).fillna(0.0)

    changed = avail.ne(avail.shift(1, fill_value=False)).any(axis=1).to_numpy()
    events = rebalance_mask(idx, rebalance) | changed
    strategic = raw_w.loc[events].reindex(idx).ffill().fillna(0.0) * gross_target
    decided = strategic * signals.fillna(0.0)

    lag = 2 if execution == "next_close" else 1
    if vol_target is None:
        scale = pd.Series(1.0, index=idx, name="scale")
    else:
        scale = _vol_scale(
            decided, raw_rets, lag, vol_target, vol_window, max_leverage, vol_estimator, ppy
        )
        decided = decided.mul(scale, axis=0)

    any_avail = avail.any(axis=1).to_numpy()
    if not any_avail.any():
        raise ValueError(f"{name}: no asset has enough history to warm up")
    start_pos, end_pos = _window(idx, int(np.argmax(any_avail)), start, end, name)
    sl = slice(start_pos, end_pos + 1)
    win = idx[sl]

    w = decided.shift(lag).fillna(0.0).iloc[sl].to_numpy().copy()
    w[0] = 0.0
    g = rets.iloc[sl].to_numpy()
    rf = per_bar_rate(idx, rf_yield).iloc[sl].to_numpy()
    dw = np.abs(np.diff(w, axis=0, prepend=0.0))
    cost = (dw * _bps(costs, tickers) / 1e4).sum(axis=1)
    contrib = w * g
    cash = (1.0 - w.sum(axis=1)) * rf
    cash[0] = 0.0
    ret = contrib.sum(axis=1) + cash - cost
    ret[0] = 0.0
    return _result(
        name,
        win,
        ret,
        w,
        tickers,
        contrib,
        cash,
        cost,
        dw.sum(axis=1),
        g,
        rf,
        strategic=strategic.iloc[sl],
        signals=signals.iloc[sl],
        scale=scale.iloc[sl],
        periods=ppy,
        initial_capital=initial_capital,
    )


def _result(
    name,
    win,
    ret,
    w,
    tickers,
    contrib,
    cash,
    cost,
    turnover,
    g,
    rf,
    *,
    strategic,
    signals,
    scale,
    periods,
    initial_capital,
) -> PortfolioResult:
    returns = pd.Series(ret, index=win, name=name)
    contributions = pd.DataFrame(contrib, index=win, columns=tickers)
    contributions["cash"] = cash
    contributions["costs"] = -cost
    return PortfolioResult(
        name=name,
        returns=returns,
        equity=(initial_capital * (1.0 + returns).cumprod()).rename(name),
        weights=pd.DataFrame(w, index=win, columns=tickers),
        strategic=strategic,
        signals=signals,
        scale=scale,
        turnover=pd.Series(turnover, index=win, name="turnover"),
        costs=pd.Series(cost, index=win, name="costs"),
        contributions=contributions,
        asset_returns=pd.DataFrame(g, index=win, columns=tickers),
        rf=pd.Series(rf, index=win, name="rf"),
        periods_per_year=periods,
    )


def _simulate_drift(
    R: np.ndarray,
    target: np.ndarray,
    events: np.ndarray,
    rf: np.ndarray,
    bps: np.ndarray,
    h0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Buy-and-hold between ``events``; at an event's close trade back to ``target``.

    Returns held weights (start of each bar), per-bar returns and per-asset
    traded notional, with a trade at close ``t`` booked on bar ``t+1`` as in
    the engine.
    """
    n, k = R.shape
    held = np.zeros((n, k))
    ret = np.zeros(n)
    traded = np.zeros((n, k))
    h = h0.astype(float).copy()
    pending = np.zeros(k)
    for t in range(n):
        held[t] = h
        traded[t] = pending
        r = h @ R[t] + (1.0 - h.sum()) * rf[t] - pending @ bps / 1e4
        ret[t] = r
        drift = h * (1.0 + R[t]) / (1.0 + r) if r != -1.0 else h
        if events[t]:
            pending = np.abs(target[t] - drift)
            h = target[t].copy()
        else:
            pending = np.zeros(k)
            h = drift
    return held, ret, traded


def static_mix_backtest(
    prices_dict: Mapping[str, PriceInput],
    weights: Mapping[str, float],
    *,
    rebalance: str | None = "ME",
    costs: CostSpec = None,
    rf_yield: pd.Series | float | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    calendar: str | pd.DatetimeIndex | None = None,
    initial_capital: float = 100_000.0,
    name: str = "Static mix",
) -> PortfolioResult:
    """Constant-weight benchmark (60/40, equal-weight buy & hold, ...) with drift.

    Weights drift with prices and are reset to ``weights`` at each ``rebalance``
    period end (``None`` = never, i.e. true buy & hold). ``1 - sum(weights)``
    sits in cash at ``rf``. Before an asset's inception its weight is spread
    pro rata over the listed assets, and it joins at the close it first
    trades. Weights use no price information, so trading at the rebalance
    close carries no look-ahead.
    """
    cal = common_calendar(prices_dict, calendar)
    aligned = align_universe({t: prices_dict[t] for t in weights}, calendar=cal)
    idx = _universe_index(aligned)
    tickers = list(aligned)
    rets = _asset_returns(aligned, idx).fillna(0.0)
    avail = pd.DataFrame({t: idx.isin(aligned[t].index) for t in tickers}, index=idx)
    base = pd.Series({t: float(weights[t]) for t in tickers})
    raw = avail.astype(float) * base
    norm = raw.abs().sum(axis=1)
    target = raw.mul(base.abs().sum() / norm.where(norm > 0), axis=0).fillna(0.0)

    changed = avail.ne(avail.shift(1, fill_value=False)).any(axis=1).to_numpy()
    events = rebalance_mask(idx, rebalance) | changed
    start_pos, end_pos = _window(
        idx, int(np.argmax(avail.any(axis=1).to_numpy())), start, end, name
    )
    sl = slice(start_pos, end_pos + 1)
    win = idx[sl]
    ev = events[sl].copy()
    ev[0] = True
    g = rets.iloc[sl].to_numpy()
    rf = per_bar_rate(idx, rf_yield).iloc[sl].to_numpy()
    rf_sim = rf.copy()
    rf_sim[0] = 0.0
    bps = _bps(costs, tickers)
    tgt = target.iloc[sl].to_numpy()
    held, ret, traded = _simulate_drift(g, tgt, ev, rf_sim, bps, np.zeros(len(tickers)))
    cost = (traded * bps / 1e4).sum(axis=1)
    cash = (1.0 - held.sum(axis=1)) * rf_sim
    return _result(
        name,
        win,
        ret,
        held,
        tickers,
        held * g,
        cash,
        cost,
        traded.sum(axis=1),
        g,
        rf,
        strategic=target.iloc[sl],
        signals=None,
        scale=pd.Series(1.0, index=win, name="scale"),
        periods=periods_per_year(idx),
        initial_capital=initial_capital,
    )


def blend(
    returns_a: pd.Series,
    returns_b: pd.Series,
    w_a: float,
    rebalance: str | None = "ME",
    cost_bps: float = 0.0,
) -> pd.Series:
    """Returns of ``w_a`` in strategy A and ``1 - w_a`` in B, e.g. 80% 60/40 + 20% trend.

    Exact drift simulation: sleeves drift between ``rebalance`` period ends and
    are reset there (``'D'`` gives the daily constant mix ``w_a a + (1-w_a) b``),
    paying ``cost_bps`` on the notional moved between sleeves. Inputs are
    aligned on their overlapping span; a date missing in one stream counts as
    a zero return for it (a holiday), so mixed 252/365 calendars are handled.
    """
    lo = max(returns_a.index[0], returns_b.index[0])
    hi = min(returns_a.index[-1], returns_b.index[-1])
    df = pd.concat([returns_a, returns_b], axis=1).sort_index().loc[lo:hi].fillna(0.0)
    idx = pd.DatetimeIndex(df.index)
    target = np.tile([w_a, 1.0 - w_a], (len(idx), 1))
    events = rebalance_mask(idx, rebalance)
    _, ret, _ = _simulate_drift(
        df.to_numpy(), target, events, np.zeros(len(idx)), np.full(2, cost_bps), target[0]
    )
    name_a = returns_a.name if returns_a.name is not None else "A"
    name_b = returns_b.name if returns_b.name is not None else "B"
    return pd.Series(ret, index=idx, name=f"{w_a:.0%} {name_a} / {1 - w_a:.0%} {name_b}")


def correlation_matrix(
    returns: Mapping[str, pd.Series] | pd.DataFrame,
    freq: str | None = "ME",
    min_periods: int = 12,
) -> pd.DataFrame:
    """Pairwise Pearson correlation of returns compounded to ``freq`` (None = per bar).

    Monthly compounding (the default) sidesteps asynchronous closes and mixed
    trading calendars, which bias daily correlations towards zero.
    """
    df = returns if isinstance(returns, pd.DataFrame) else pd.concat(dict(returns), axis=1)
    df = df.sort_index()
    if freq is not None:
        df = (1.0 + df).resample(freq).prod(min_count=1) - 1.0
    return df.corr(min_periods=min_periods)
