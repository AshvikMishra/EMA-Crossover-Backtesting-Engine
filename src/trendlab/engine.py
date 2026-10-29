"""Vectorised backtest engine.

Timing convention (bar ``i`` is the period from close ``i-1`` to close ``i``):

* the strategy observes close ``i`` and outputs ``target[i]``;
* ``execution`` decides when that decision is filled:

  ===============  =========================================================
  ``same_close``   filled at close ``i`` itself. Needs the close to compute
                   the signal *and* trade on it -- the optimistic convention
                   the original notebook used. Kept only to measure the bias.
  ``next_open``    filled at the open of bar ``i+1``.
  ``next_close``   filled at the close of bar ``i+1`` (default; most
                   conservative and needs no intraday data).
  ===============  =========================================================

Per-bar strategy return with exposure ``w`` (fraction of equity), asset return
``g`` and risk-free return ``rf``::

    r = w * g + (1 - w) * rf - |delta w| * cost_bps / 1e4

i.e. un-invested capital sits in T-bills, leverage (w > 1) or shorts (w < 0)
are financed at the same rate, and costs are proportional to traded notional.
Weights are rebalanced to target every bar (drift between trades is ignored,
which is exact for the 0/1 exposures most rules here produce).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import per_bar_rate, periods_per_year
from .strategies import Strategy

EXECUTIONS = ("same_close", "next_open", "next_close")


@dataclass(frozen=True)
class CostModel:
    """Proportional trading costs, in basis points of traded notional.

    Defaults (1 bp commission + 2 bp half-spread + 2 bp slippage = 5 bp per
    side) are typical for liquid US ETFs with a retail broker; crypto is
    usually 10-30 bp per side.
    """

    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 2.0

    @property
    def bps(self) -> float:
        return self.commission_bps + self.half_spread_bps + self.slippage_bps

    @classmethod
    def zero(cls) -> CostModel:
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def flat(cls, bps: float) -> CostModel:
        return cls(commission_bps=bps, half_spread_bps=0.0, slippage_bps=0.0)


@dataclass(frozen=True)
class BacktestConfig:
    execution: str = "next_close"
    costs: CostModel = field(default_factory=CostModel)
    earn_cash_rate: bool = True
    initial_capital: float = 100_000.0
    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None

    def __post_init__(self):
        if self.execution not in EXECUTIONS:
            raise ValueError(f"execution must be one of {EXECUTIONS}")


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    equity: pd.Series
    exposure: pd.Series
    target: pd.Series
    turnover: pd.Series
    costs: pd.Series
    asset_returns: pd.Series
    rf: pd.Series
    config: BacktestConfig
    periods_per_year: int

    @property
    def start(self) -> pd.Timestamp:
        return self.returns.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.returns.index[-1]

    @property
    def trades(self) -> pd.DataFrame:
        return extract_trades(self.exposure, self.returns, self.asset_returns, self.costs)

    def summary(self, benchmark: BacktestResult | None = None) -> dict:
        from .metrics import summary

        return summary(
            self.returns,
            rf=self.rf,
            periods=self.periods_per_year,
            exposure=self.exposure,
            turnover=self.turnover,
            benchmark=None if benchmark is None else benchmark.returns,
            trades=self.trades,
        )


def first_valid_decision(prices: pd.DataFrame, strategy: Strategy) -> pd.Timestamp:
    """Date of the first close at which ``strategy`` has a defined target."""
    idx = strategy.target_position(prices).first_valid_index()
    if idx is None:
        raise ValueError(f"{strategy.name}: not enough history to warm up")
    return idx


def evaluation_start(
    prices: pd.DataFrame, strategies: list[Strategy], start: str | pd.Timestamp | None = None
) -> pd.Timestamp:
    """Common window start at which every strategy in ``strategies`` is warmed up.

    Matches :func:`run_backtest`: the window opens one bar after the first
    warmed-up decision.
    """
    dates = [
        prices.index[prices.index.get_loc(first_valid_decision(prices, s)) + 1] for s in strategies
    ]
    if start is not None:
        dates.append(prices.index[prices.index.searchsorted(pd.Timestamp(start))])
    return max(dates)


def _resolve_rf(index: pd.DatetimeIndex, rf_yield, earn: bool) -> pd.Series:
    if not earn or rf_yield is None:
        return pd.Series(0.0, index=index)
    return per_bar_rate(index, rf_yield)


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig | None = None,
    rf_yield: pd.Series | float | None = None,
    *,
    target: pd.Series | None = None,
) -> BacktestResult:
    """Backtest ``strategy`` on ``prices`` (adjusted OHLCV) and return the full result.

    Indicators are computed on the *entire* ``prices`` history so that data
    before ``config.start`` serves as warm-up. The evaluation window opens at
    the close of the later of ``config.start`` and the first bar after the
    strategy is warmed up. At that close the account holds ``initial_capital``
    in cash and takes the strategy's current position at the first fill the
    execution rule allows, paying the opening cost -- so a rule that is
    already "in" a trend starts invested, exactly like buy-and-hold.

    ``target`` lets callers pass precomputed target exposures (e.g. a
    portfolio overlay); ``strategy`` is then only used for its name.
    """
    config = config or BacktestConfig()
    if target is None:
        target = strategy.target_position(prices)
    target = target.reindex(prices.index)

    first = target.first_valid_index()
    if first is None:
        raise ValueError(f"{strategy.name}: not enough history to warm up")
    start_pos = prices.index.get_loc(first) + 1
    if config.start is not None:
        start_pos = max(start_pos, int(prices.index.searchsorted(pd.Timestamp(config.start))))
    end_pos = len(prices) - 1
    if config.end is not None:
        end_pos = int(prices.index.searchsorted(pd.Timestamp(config.end), side="right")) - 1
    if start_pos >= end_pos:
        raise ValueError(f"{strategy.name}: empty evaluation window")

    decided = target.fillna(0.0)
    close = prices["Close"]
    gross = close.pct_change().fillna(0.0)
    rf_all = _resolve_rf(prices.index, rf_yield, config.earn_cash_rate)

    lag = 2 if config.execution == "next_close" else 1
    w_all = decided.shift(lag).fillna(0.0)
    sl = slice(start_pos, end_pos + 1)
    idx = prices.index[sl]
    w = w_all.iloc[sl].to_numpy().copy()
    g = gross.iloc[sl].to_numpy()
    rf = rf_all.iloc[sl].to_numpy()
    w[0] = 0.0  # in cash at the opening close

    if config.execution == "next_open":
        # bar i: weight decided at close i-2 held overnight, new weight from the open
        w_old = np.concatenate([[0.0], w[:-1]])
        o = prices["Open"].iloc[sl].to_numpy()
        c = close.iloc[sl].to_numpy()
        c_prev = close.shift(1).iloc[sl].to_numpy()
        risky = (1.0 + w_old * (o / c_prev - 1.0)) * (1.0 + w * (c / o - 1.0)) - 1.0
    else:
        risky = w * g

    turnover = np.abs(np.diff(w, prepend=0.0))
    cost = turnover * config.costs.bps / 1e4
    ret = risky + (1.0 - w) * rf - cost
    ret[0] = 0.0
    ret = pd.Series(ret, index=idx)
    equity = config.initial_capital * (1.0 + ret).cumprod()

    return BacktestResult(
        name=strategy.name,
        returns=ret.rename(strategy.name),
        equity=equity.rename(strategy.name),
        exposure=pd.Series(w, index=idx, name="exposure"),
        target=target.iloc[sl].rename("target"),
        turnover=pd.Series(turnover, index=idx, name="turnover"),
        costs=pd.Series(cost, index=idx, name="costs"),
        asset_returns=gross.iloc[sl].rename("asset"),
        rf=pd.Series(rf, index=idx, name="rf"),
        config=config,
        periods_per_year=periods_per_year(prices.index),
    )


def extract_trades(
    exposure: pd.Series,
    returns: pd.Series,
    asset_returns: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> pd.DataFrame:
    """Round trips = maximal runs of non-zero exposure with the same sign.

    ``return`` is the compounded *strategy* return over the run, including
    financing, the entry cost and (if ``costs`` is given) the exit cost booked
    on the first bar after the run. ``asset_return`` is the underlying's move.
    """
    w = exposure.to_numpy()
    sign = np.sign(np.round(w, 12))
    r = returns.to_numpy()
    c = np.zeros(len(w)) if costs is None else costs.to_numpy()
    rows = []
    i, n = 0, len(w)
    while i < n:
        if sign[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and sign[j + 1] == sign[i]:
            j += 1
        growth = np.prod(1.0 + r[i : j + 1])
        if j + 1 < n:
            growth *= 1.0 - c[j + 1]
        row = {
            "entry": exposure.index[i],
            "exit": exposure.index[j],
            "side": "long" if sign[i] > 0 else "short",
            "bars": j - i + 1,
            "return": float(growth - 1.0),
            "open": j + 1 >= n,
        }
        if asset_returns is not None:
            row["asset_return"] = float(np.prod(1.0 + asset_returns.to_numpy()[i : j + 1]) - 1.0)
        rows.append(row)
        i = j + 1
    cols = ["entry", "exit", "side", "bars", "return", "open"]
    if asset_returns is not None:
        cols.append("asset_return")
    return pd.DataFrame(rows, columns=cols)
