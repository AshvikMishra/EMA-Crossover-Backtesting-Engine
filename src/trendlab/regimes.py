"""Where does a trend-following edge come from? Crisis, regime and convexity diagnostics.

Trend following is usually sold as "crisis alpha": it should step aside during
long bear markets and re-enter afterwards, giving a convex ("smile") payoff
against its own underlying (Fung & Hsieh 2001, *Rev. Financial Studies* 14(2);
Hurst, Ooi & Pedersen 2017, "A Century of Evidence on Trend-Following
Investing", AQR). The tools here test that claim instead of assuming it:

* :func:`crisis_table` -- returns, drawdowns, exit and re-entry timing in named
  stress windows;
* :func:`label_regimes` / :func:`performance_by_regime` -- performance split by
  trend, volatility and drawdown regimes, with labels known one bar *before*
  the return they classify (no look-ahead);
* :func:`capture_ratios` -- Morningstar-style up/down-market capture;
* :func:`convexity_smile` -- quadratic fit ``y = a + b x + c x^2`` of strategy
  on benchmark returns at a coarse frequency; ``c > 0`` is the trend smile;
* :func:`rolling_sharpe`, :func:`rolling_excess_vs_benchmark` -- stability;
* :func:`whipsaw_stats` -- the cost of short losing round trips, the price
  trend followers pay for their crisis protection.

All return inputs are simple per-bar returns as produced by
:func:`trendlab.engine.run_backtest`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import metrics
from .engine import BacktestResult
from .universe import CRISES

TREND_LABELS = ["bull", "bear"]
VOL_LABELS = ["low", "mid", "high"]
DD_LABELS = ["0 to -10%", "-10 to -20%", "< -20%"]
_TOL = 1e-12
_CRISIS_COLS = [
    "start",
    "end",
    "bars",
    "trough",
    "bench_return",
    "strat_return",
    "protection",
    "bench_max_dd",
    "strat_max_dd",
    "avg_exposure",
    "exit_date",
    "exit_vs_start_bars",
    "reentry_date",
    "reentry_lag_bars",
    "missed_rebound",
]


def _compound(r: pd.Series | np.ndarray) -> float:
    return float(np.prod(1.0 + np.asarray(r, dtype=float)) - 1.0)


def _returns_on(result: BacktestResult, index: pd.DatetimeIndex) -> pd.Series:
    """Per-bar returns of ``result`` between consecutive dates of ``index`` (a subset of its own).

    Built from equity ratios, so bars missing from ``index`` (e.g. crypto weekends when
    the other leg trades only on weekdays) are compounded in rather than dropped.
    """
    eq = result.equity.reindex(index).to_numpy(dtype=float)
    r = np.empty(len(eq))
    r[0] = float(result.returns.loc[index[0]])
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = eq[1:] / eq[:-1] - 1.0
    return pd.Series(r, index=index)


def _window_max_dd(r: pd.Series) -> float:
    """Max drawdown of a return window measured from the window's starting value."""
    return metrics.max_drawdown(pd.Series(np.concatenate([[0.0], r.to_numpy(dtype=float)])))


def crisis_table(
    strategy_result: BacktestResult,
    benchmark_result: BacktestResult,
    crises: dict[str, tuple[str, str]] = CRISES,
    *,
    lookback: int = 60,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Strategy vs benchmark inside each crisis window ``(start, end]``.

    Returns compound from the close on ``start`` to the close on ``end``. A
    crisis is skipped unless both backtests cover it entirely (or, with
    ``allow_partial``, overlap it; the window is then clipped). Both legs are
    evaluated on their common dates; returns on bars only one leg has (crypto
    weekends) are compounded into the next common bar, not dropped.

    Timing columns (a bar is "invested" when exposure > 0, so shorts count as out;
    dates are the first bar *held* with the new exposure, one bar after the fill):

    * ``trough`` -- bar of the benchmark's lowest equity inside the window;
    * ``exit_date`` -- the *last* switch to flat (first bar held out of the
      market) between ``lookback`` bars before the window and the trough; NaT
      if none. If the strategy re-entered before the trough this exit is no
      longer in force, which shows up as ``reentry_lag_bars <= 0``;
    * ``exit_vs_start_bars`` -- bars from the first window bar to the exit
      (negative = exited before the crisis began);
    * ``reentry_date`` -- first invested bar after ``exit_date`` (or after the
      trough if the strategy was already flat before the search range);
    * ``reentry_lag_bars`` -- bars from trough to re-entry (<= 0: back in
      before the trough; NaN: never left or never returned);
    * ``missed_rebound`` -- benchmark return between the trough and re-entry;
    * ``protection`` -- ``strat_return - bench_return``.

    Index: crisis name. Dates are Timestamps, returns and drawdowns fractions.
    """
    common = strategy_result.returns.index.intersection(benchmark_result.returns.index)
    if len(common) < 2:
        return pd.DataFrame(columns=_CRISIS_COLS).rename_axis("crisis")
    rs = _returns_on(strategy_result, common)
    rb = _returns_on(benchmark_result, common)
    invested = strategy_result.exposure.reindex(common).fillna(0.0).to_numpy() > _TOL
    exposure = strategy_result.exposure.reindex(common).fillna(0.0)
    lo, hi = common[0], common[-1]
    rows = {}
    for name, (s, e) in crises.items():
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        if e <= lo or s >= hi:
            continue
        if not allow_partial and (s < lo or e > hi):
            continue
        s, e = max(s, lo), min(e, hi)
        p0 = int(common.searchsorted(s, side="right"))  # first bar after start close
        p1 = int(common.searchsorted(e, side="right")) - 1
        if p1 < p0:
            continue
        wb, ws = rb.iloc[p0 : p1 + 1], rs.iloc[p0 : p1 + 1]
        trough_pos = p0 + int(np.argmin(np.cumprod(1.0 + wb.to_numpy())))

        search_from = max(1, p0 - lookback)
        exits = [
            i for i in range(search_from, trough_pos + 1) if invested[i - 1] and not invested[i]
        ]
        exit_pos = exits[-1] if exits else None
        if exit_pos is not None:
            after = exit_pos + 1
        elif not invested[trough_pos]:
            after = trough_pos + 1
        else:
            after = None
        reentry_pos = None
        if after is not None:
            later = np.flatnonzero(invested[after:])
            reentry_pos = after + int(later[0]) if len(later) else None

        lag = float(reentry_pos - trough_pos) if reentry_pos is not None else math.nan
        missed = (
            _compound(rb.iloc[trough_pos + 1 : reentry_pos])
            if reentry_pos is not None
            else math.nan
        )
        bench_ret, strat_ret = _compound(wb), _compound(ws)
        rows[name] = {
            "start": s,
            "end": e,
            "bars": p1 - p0 + 1,
            "trough": common[trough_pos],
            "bench_return": bench_ret,
            "strat_return": strat_ret,
            "protection": strat_ret - bench_ret,
            "bench_max_dd": _window_max_dd(wb),
            "strat_max_dd": _window_max_dd(ws),
            "avg_exposure": float(exposure.iloc[p0 : p1 + 1].mean()),
            "exit_date": common[exit_pos] if exit_pos is not None else pd.NaT,
            "exit_vs_start_bars": float(exit_pos - p0) if exit_pos is not None else math.nan,
            "reentry_date": common[reentry_pos] if reentry_pos is not None else pd.NaT,
            "reentry_lag_bars": lag,
            "missed_rebound": missed,
        }
    out = pd.DataFrame.from_dict(rows, orient="index", columns=_CRISIS_COLS)
    out.index.name = "crisis"
    return out


def _categorical(values: np.ndarray, mask: np.ndarray, categories: list[str], index) -> pd.Series:
    vals = np.where(mask, values, None)
    return pd.Series(pd.Categorical(vals, categories=categories, ordered=True), index=index)


def _expanding_terciles(x: pd.Series, min_periods: int) -> pd.Series:
    """Label each value against the terciles of all values up to and including it."""
    exp = x.expanding(min_periods=min_periods)
    q1, q2 = exp.quantile(1 / 3), exp.quantile(2 / 3)
    vals = np.select([x <= q1, x <= q2], ["low", "mid"], default="high")
    ok = (x.notna() & q1.notna()).to_numpy()
    return _categorical(vals, ok, VOL_LABELS, x.index)


def label_regimes(
    prices: pd.DataFrame,
    vix: pd.Series | pd.DataFrame | None = None,
    *,
    sma_window: int = 200,
    vol_window: int = 21,
    min_history: int = 252,
) -> pd.DataFrame:
    """Regime labels for each bar, using only information up to the *previous* close.

    Columns (ordered categoricals, NaN while warming up):

    * ``trend`` -- ``bull`` if Close > ``sma_window``-bar SMA, else ``bear``;
    * ``vol`` -- ``low``/``mid``/``high`` terciles of the VIX close (or, without
      ``vix``, of ``vol_window``-bar realised volatility) against an *expanding*
      history of at least ``min_history`` observations. VIX terciles use the
      VIX's own full calendar and history (also before ``prices`` starts) and the
      latest VIX label is then carried onto each price date;
    * ``drawdown`` -- the asset's drawdown from its running peak, bucketed as
      ``0 to -10%``, ``-10 to -20%``, ``< -20%``.

    Every label is computed at close t-1 and attached to bar t, so it can be
    used to classify the return of bar t without look-ahead.
    """
    close = prices["Close"].astype(float)
    idx = close.index

    ma = close.rolling(sma_window, min_periods=sma_window).mean()
    trend = _categorical(
        np.where(close > ma, "bull", "bear"), ma.notna().to_numpy(), TREND_LABELS, idx
    )

    if vix is not None:
        v = vix["Close"] if isinstance(vix, pd.DataFrame) else vix
        v = v.astype(float).dropna().sort_index()
        v = v[~v.index.duplicated(keep="last")]
        vol = _expanding_terciles(v, min_history).reindex(idx, method="ffill")
    else:
        level = close.pct_change().rolling(vol_window, min_periods=vol_window).std()
        vol = _expanding_terciles(level, min_history)

    dd = close / close.cummax() - 1.0
    dd_vals = np.select([dd > -0.10, dd > -0.20], DD_LABELS[:2], default=DD_LABELS[2])
    drawdown = _categorical(dd_vals, dd.notna().to_numpy(), DD_LABELS, idx)

    out = pd.DataFrame({"trend": trend, "vol": vol, "drawdown": drawdown}, index=idx)
    return out.shift(1)


def performance_by_regime(
    returns_dict: dict[str, pd.Series],
    labels: pd.Series,
    periods: int = 252,
    rf: pd.Series | float | None = None,
) -> pd.DataFrame:
    """Annualised performance of each return series within each regime.

    Series are aligned on the dates they share with each other and with a
    non-missing label. Index: regime value (in category order); columns: a
    MultiIndex ``(strategy, metric)`` with metrics ``ann_return`` (geometric,
    ``prod(1+r)^(periods/n) - 1`` over the regime's bars), ``ann_vol``,
    ``sharpe`` (excess of ``rf`` if given) and ``share`` (fraction of bars).
    """
    lab = labels.dropna()
    common = lab.index
    for r in returns_dict.values():
        common = common.intersection(r.dropna().index)
    lab = lab.reindex(common)
    if isinstance(lab.dtype, pd.CategoricalDtype):
        regimes = [c for c in lab.cat.categories if (lab == c).any()]
    else:
        regimes = sorted(lab.unique())
    rows = {}
    for reg in regimes:
        mask = (lab == reg).to_numpy()
        row = {}
        for name, r in returns_dict.items():
            sub = r.reindex(common)[mask]
            n = len(sub)
            growth = float((1.0 + sub).prod())
            row[(name, "ann_return")] = growth ** (periods / n) - 1.0 if growth > 0 else -1.0
            row[(name, "ann_vol")] = metrics.ann_vol(sub, periods) if n > 1 else math.nan
            row[(name, "sharpe")] = metrics.sharpe(sub, rf, periods) if n > 1 else math.nan
            row[(name, "share")] = n / len(common)
        rows[reg] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    if not out.empty:
        out.columns = pd.MultiIndex.from_tuples(out.columns, names=["strategy", "metric"])
    out.index.name = labels.name or "regime"
    return out


def _compound_by(r: pd.Series, freq: str) -> pd.Series:
    grouped = (1.0 + r).resample(freq)
    return (grouped.prod() - 1.0)[grouped.size() > 0]


def _period_returns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    freq: str,
    rf: pd.Series | float | None = None,
) -> pd.DataFrame:
    """Compounded ``freq`` returns of benchmark (``x``) and strategy (``y``).

    Both legs are clipped to their common date span and then compounded on their
    *own* bars, so a 7-day crypto leg keeps its weekend returns against a
    weekday benchmark. With ``rf`` (per-bar rate), each leg's compounded rf over
    the same bars is subtracted, giving period excess returns.
    """
    s, b = strategy_returns.dropna(), benchmark_returns.dropna()
    if s.empty or b.empty:
        return pd.DataFrame(columns=["x", "y"], dtype=float)
    lo, hi = max(s.index[0], b.index[0]), min(s.index[-1], b.index[-1])

    def agg(r: pd.Series) -> pd.Series:
        r = r.loc[lo:hi]
        out = _compound_by(r, freq)
        if rf is not None:
            rfr = pd.Series(float(rf), index=r.index) if np.isscalar(rf) else rf
            out = out - _compound_by(rfr.reindex(r.index).fillna(0.0), freq)
        return out

    return pd.concat({"x": agg(b), "y": agg(s)}, axis=1, join="inner").dropna()


def capture_ratios(
    strategy_returns: pd.Series, benchmark_returns: pd.Series, freq: str = "ME"
) -> dict[str, float]:
    """Up- and down-market capture on compounded ``freq`` returns (Morningstar definition).

    ``upside_capture`` = geometric mean strategy return over benchmark-up periods
    divided by the benchmark's geometric mean over the same periods; likewise
    for down periods (benchmark return < 0). A trend follower aims for upside
    near 1 and downside well below 1; ``capture_ratio`` = upside / downside.
    """
    df = _period_returns(strategy_returns, benchmark_returns, freq)

    def geo(r: pd.Series) -> float:
        return float(np.prod(1.0 + r) ** (1.0 / len(r)) - 1.0) if len(r) else math.nan

    up, down = df[df["x"] > 0], df[df["x"] < 0]
    up_cap = geo(up["y"]) / geo(up["x"]) if len(up) else math.nan
    down_cap = geo(down["y"]) / geo(down["x"]) if len(down) else math.nan
    return {
        "upside_capture": up_cap,
        "downside_capture": down_cap,
        "capture_ratio": up_cap / down_cap if down_cap else math.nan,
        "up_periods": int(len(up)),
        "down_periods": int(len(down)),
    }


def convexity_smile(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    freq: str = "QE",
    rf: pd.Series | float | None = None,
) -> dict:
    """Fit ``y = a + b x + c x^2`` to ``freq``-compounded strategy (y) and benchmark (x) returns.

    A significantly positive ``c`` is the "trend smile": the strategy does
    relatively best when its benchmark makes large moves in either direction
    (Fung & Hsieh 2001). Standard errors are plain OLS; with few quarterly
    observations treat ``t_c`` as indicative only.

    Returns ``{"data": DataFrame[x, y], "coef": Series[a, b, c], "tstat":
    Series[a, b, c], "t_c": float, "r2": float, "n": int}``. With ``rf`` (per-bar
    rate) both legs are period excess returns: compounded return minus
    compounded T-bill return over the same period.
    """
    df = _period_returns(strategy_returns, benchmark_returns, freq, rf)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    n = len(df)
    names = ["a", "b", "c"]
    if n < 4:
        nan = pd.Series(math.nan, index=names)
        return {
            "data": df,
            "coef": nan,
            "tstat": nan.copy(),
            "t_c": math.nan,
            "r2": math.nan,
            "n": n,
        }
    X = np.column_stack([np.ones(n), x, x**2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = float(resid @ resid) / (n - 3)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else math.nan
    return {
        "data": df,
        "coef": pd.Series(beta, index=names),
        "tstat": pd.Series(t, index=names),
        "t_c": float(t[2]),
        "r2": r2,
        "n": n,
    }


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252 * 3,
    rf: pd.Series | float | None = None,
    periods: int = 252,
) -> pd.Series:
    """Trailing ``window``-bar annualised Sharpe ratio (NaN until the window is full)."""
    ex = metrics._excess(returns, rf)
    roll = ex.rolling(window, min_periods=window)
    sr = roll.mean() / roll.std(ddof=1) * math.sqrt(periods)
    return sr.replace([np.inf, -np.inf], np.nan).rename("rolling_sharpe")


def rolling_excess_vs_benchmark(
    returns: pd.Series,
    benchmark: pd.Series,
    window: int = 252 * 3,
    periods: int = 252,
) -> pd.DataFrame:
    """Trailing ``window``-bar annualised returns of strategy and benchmark and their gap.

    Columns: ``strategy`` and ``benchmark`` (geometric annualised return over the
    window), ``excess`` (their difference) and ``information_ratio``
    (annualised mean / std of the per-bar active return).
    """
    df = pd.concat({"s": returns, "b": benchmark}, axis=1, join="inner").dropna()

    def ann(r: pd.Series) -> pd.Series:
        logs = np.log1p(r).rolling(window, min_periods=window).sum()
        return np.expm1(logs * periods / window)

    active = (df["s"] - df["b"]).rolling(window, min_periods=window)
    ir = (active.mean() / active.std(ddof=1) * math.sqrt(periods)).replace(
        [np.inf, -np.inf], np.nan
    )
    out = pd.DataFrame({"strategy": ann(df["s"]), "benchmark": ann(df["b"])})
    out["excess"] = out["strategy"] - out["benchmark"]
    out["information_ratio"] = ir
    return out


def whipsaw_stats(trades: pd.DataFrame, max_bars: int = 20) -> dict[str, float]:
    """Summarise whipsaws: closed round trips of fewer than ``max_bars`` bars that lost money.

    ``trades`` is :attr:`BacktestResult.trades`; trades still open at the end of
    the sample are excluded (reported as ``n_open``). Keys: ``n_trades``,
    ``n_losing``, ``losing_share``, ``n_whipsaws``, ``whipsaw_share`` (of closed
    trades), ``whipsaw_loss`` (compounded return of all whipsaws, i.e. their
    total drag on equity, costs included), ``whipsaw_avg_return``,
    ``whipsaw_avg_bars`` and ``n_open``.
    """
    if trades is None or trades.empty:
        closed = pd.DataFrame(columns=["bars", "return"])
        n_open = 0
    else:
        is_open = trades["open"].astype(bool)
        closed, n_open = trades[~is_open], int(is_open.sum())
    r = closed["return"].astype(float)
    whip = closed[(closed["bars"] < max_bars) & (r < 0)]
    n = len(closed)
    return {
        "n_trades": n,
        "n_losing": int((r < 0).sum()),
        "losing_share": float((r < 0).mean()) if n else math.nan,
        "n_whipsaws": int(len(whip)),
        "whipsaw_share": len(whip) / n if n else math.nan,
        "whipsaw_loss": _compound(whip["return"]) if len(whip) else 0.0,
        "whipsaw_avg_return": float(whip["return"].mean()) if len(whip) else math.nan,
        "whipsaw_avg_bars": float(whip["bars"].mean()) if len(whip) else math.nan,
        "n_open": n_open,
    }
