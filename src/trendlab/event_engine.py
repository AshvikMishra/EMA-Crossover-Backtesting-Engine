"""Event-driven (bar-by-bar) backtester and an exact replica of the original notebook.

The vectorised engine in :mod:`trendlab.engine` is fast but abstract: it works
with weights and per-bar returns. This module walks the bars one at a time and
keeps an explicit book (cash, shares, fees owed), which is how a broker sees an
account. It exists for two reasons:

1. **Parity testing.** With fractional shares, no fixed commission and 0/1
   exposures it reproduces :func:`trendlab.engine.run_backtest` to floating
   point precision for ``same_close`` and ``next_close`` (and for any exposure,
   shorts and leverage included, with ``rebalance='every_bar'`` and zero
   costs). Any disagreement is a bug in one of the two implementations.
2. **Quantifying the notebook's biases.** :func:`legacy_notebook_backtest`
   re-runs ``notebooks/00_original_exploration.ipynb`` line for line (EMAs with
   no warm-up, fills at the signal close, integer shares, $5 tickets, idle cash
   earning nothing) so every later correction can be measured against it.

Book-keeping conventions (chosen so that, for 0/1 exposures, they are exactly
the ones implied by the vectorised formula ``r = w*g + (1-w)*rf - |dw|*bps``):

* cash earns the per-bar T-bill return from :func:`trendlab.data.per_bar_rate`;
  a negative balance (leverage) is financed at the same rate;
* fees (``cost_bps`` of traded notional plus ``commission_per_trade`` dollars
  per fill) are settled at the first close *after* the fill -- the same close
  for an open fill -- like a T+1 debit. With fractional shares they are paid
  pro rata from the asset and cash legs so paying them does not move the
  exposure; with integer shares they come out of cash;
* by default the book only trades when the filled target changes
  (``rebalance='on_change'``), so any weight other than 0 or +1 (fractional,
  short, levered) drifts between signals. ``rebalance='every_bar'`` resizes to
  target at every fill opportunity, which is what the vectorised engine assumes;
* with ``integer_shares`` a whole-share order too small to fill (e.g. the
  capital cannot buy one share) stays pending and is retried at every later
  fill opportunity, and ``exposure`` is 0 whenever the book holds no shares.

For ``next_open`` the vectorised engine books interest and proportional costs
on the previous close's equity rather than the equity at the open, so the two
engines agree exactly only without costs and cash yield; otherwise they differ
by second-order terms (overnight gap x rate). It also applies the previous
target to the overnight gap as if the book had been rebalanced at the close, so
for weights other than 0/1 the two differ by the intraday drift as well.

References: M. López de Prado, *Advances in Financial Machine Learning*
(Wiley, 2018), ch. 11-12 on backtest pitfalls; E. Chan, *Quantitative Trading*
(Wiley, 2009), ch. 3 on look-ahead bias and transaction costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import per_bar_rate, periods_per_year
from .engine import EXECUTIONS, extract_trades
from .strategies import Strategy

REBALANCE = ("on_change", "every_bar")
FILL_COLUMNS = [
    "date",
    "when",
    "price",
    "shares",
    "notional",
    "fee",
    "target",
    "shares_after",
    "cash_after",
]


@dataclass
class EventBacktestResult:
    """Per-bar book of an event-driven backtest (all Series share the window index).

    ``exposure`` is the target exposure in force during each bar (comparable to
    :attr:`BacktestResult.exposure`); ``weight`` is the realised asset weight at
    the close after any fills; ``costs`` and ``turnover`` are fees and traded
    notional settled at that close as fractions of the previous close's equity.
    """

    name: str
    returns: pd.Series
    equity: pd.Series
    exposure: pd.Series
    weight: pd.Series
    cash: pd.Series
    shares: pd.Series
    turnover: pd.Series
    costs: pd.Series
    asset_returns: pd.Series
    rf: pd.Series
    fills: pd.DataFrame
    execution: str
    periods_per_year: int
    config: dict = field(default_factory=dict)

    @property
    def start(self) -> pd.Timestamp:
        return self.returns.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.returns.index[-1]

    @property
    def trades(self) -> pd.DataFrame:
        """Round trips in the same format as :func:`trendlab.engine.extract_trades`."""
        return extract_trades(self.exposure, self.returns, self.asset_returns, self.costs)

    def summary(self, benchmark=None) -> dict:
        from .metrics import summary

        bench = getattr(benchmark, "returns", benchmark)
        return summary(
            self.returns,
            rf=self.rf,
            periods=self.periods_per_year,
            exposure=self.exposure,
            turnover=self.turnover,
            benchmark=bench,
            trades=self.trades,
        )


def _window(index: pd.DatetimeIndex, target: pd.Series, start, end, name: str) -> tuple[int, int]:
    first = target.first_valid_index()
    if first is None:
        raise ValueError(f"{name}: not enough history to warm up")
    start_pos = index.get_loc(first) + 1
    if start is not None:
        start_pos = max(start_pos, int(index.searchsorted(pd.Timestamp(start))))
    end_pos = len(index) - 1
    if end is not None:
        end_pos = int(index.searchsorted(pd.Timestamp(end), side="right")) - 1
    if start_pos >= end_pos:
        raise ValueError(f"{name}: empty evaluation window")
    return start_pos, end_pos


def run_event_backtest(
    prices: pd.DataFrame,
    strategy_or_target: Strategy | pd.Series,
    *,
    execution: str = "next_close",
    initial_capital: float = 100_000.0,
    commission_per_trade: float = 0.0,
    cost_bps: float = 0.0,
    integer_shares: bool = False,
    rf_yield: pd.Series | float | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    rebalance: str = "on_change",
) -> EventBacktestResult:
    """Simulate a single-asset account bar by bar.

    ``strategy_or_target`` is a :class:`Strategy` or a precomputed Series of
    target exposures decided at each close (NaN = warming up). The window,
    opening rule and timing follow :func:`trendlab.engine.run_backtest`: the
    account opens with ``initial_capital`` in cash at the close of the first
    bar after warm-up (or ``start``) and takes the current target at the first
    fill ``execution`` allows. With ``integer_shares`` buys are sized with
    ``floor`` after reserving the fee, and targets must be non-negative.
    """
    if execution not in EXECUTIONS:
        raise ValueError(f"execution must be one of {EXECUTIONS}")
    if rebalance not in REBALANCE:
        raise ValueError(f"rebalance must be one of {REBALANCE}")
    if isinstance(strategy_or_target, Strategy):
        name = strategy_or_target.name
        target = strategy_or_target.target_position(prices)
    else:
        name = str(strategy_or_target.name or "target")
        target = strategy_or_target.astype(float)
    target = target.reindex(prices.index)
    start_pos, end_pos = _window(prices.index, target, start, end, name)
    decided = target.fillna(0.0).to_numpy()
    if integer_shares and (decided[start_pos - 1 : end_pos + 1] < 0).any():
        raise ValueError("integer_shares supports long-only (non-negative) targets")

    close = prices["Close"].to_numpy(dtype=float)
    open_ = prices["Open"].to_numpy(dtype=float) if execution == "next_open" else close
    rf_all = per_bar_rate(prices.index, rf_yield).to_numpy()
    k = cost_bps / 1e4
    idx = prices.index[start_pos : end_pos + 1]
    n = len(idx)

    equity = np.zeros(n)
    cash_s = np.zeros(n)
    shares_s = np.zeros(n)
    exposure = np.zeros(n)
    weight = np.zeros(n)
    turnover = np.zeros(n)
    costs = np.zeros(n)
    rf = rf_all[start_pos : end_pos + 1]
    fills: list[dict] = []

    cash, shares, held = float(initial_capital), 0.0, 0.0
    fee_due = notional_due = 0.0

    def fill(tgt: float, price: float, date: pd.Timestamp, when: str) -> None:
        nonlocal cash, shares, held, fee_due, notional_due
        if rebalance == "on_change" and tgt == held:
            return
        value = cash + shares * price
        if integer_shares:
            if tgt * value > shares * price:
                budget = tgt * value - shares * price - commission_per_trade
                delta = float(max(0, math.floor(budget / (price * (1.0 + k)))))
            else:
                delta = float(min(0, math.floor(tgt * value / price + 1e-9) - shares))
        else:
            delta = tgt * value / price - shares
            if abs(delta * price) <= 1e-9 * max(abs(value), 1.0):
                delta = 0.0
        if delta == 0.0:
            if not integer_shares:
                held = tgt  # already at target
            return  # an unaffordable whole-share order stays pending
        held = tgt
        notional = abs(delta) * price
        fee = notional * k + commission_per_trade
        cash -= delta * price
        shares = tgt * value / price if not integer_shares else shares + delta
        fee_due += fee
        notional_due += notional
        fills.append(
            {
                "date": date,
                "when": when,
                "price": price,
                "shares": delta,
                "notional": notional,
                "fee": fee,
                "target": tgt,
                "shares_after": shares,
                "cash_after": cash,
            }
        )

    pending = 0.0
    for j in range(n):
        pos = start_pos + j
        date, c = idx[j], close[pos]
        if j > 0:
            if execution == "next_open":
                fill(pending, open_[pos], date, "open")
            cash += cash * rf[j]
            if fee_due or notional_due:
                if integer_shares:
                    cash -= fee_due
                else:
                    book = cash + shares * c
                    a = shares * c / book if book else 0.0
                    shares -= fee_due * a / c
                    cash -= fee_due * (1.0 - a)
                costs[j] = fee_due / equity[j - 1]
                turnover[j] = notional_due / equity[j - 1]
                fee_due = notional_due = 0.0
            exposure[j] = held if shares else 0.0
        equity[j] = cash + shares * c
        if execution == "same_close":
            fill(decided[pos], c, date, "close")
        elif execution == "next_close":
            fill(decided[pos - 1], c, date, "close")
        else:
            pending = decided[pos]
        cash_s[j], shares_s[j] = cash, shares
        weight[j] = shares * c / equity[j] if equity[j] else 0.0

    eq = pd.Series(equity, index=idx, name=name)
    ret = eq.pct_change().fillna(0.0).rename(name)
    gross = prices["Close"].pct_change().fillna(0.0).iloc[start_pos : end_pos + 1]
    return EventBacktestResult(
        name=name,
        returns=ret,
        equity=eq,
        exposure=pd.Series(exposure, index=idx, name="exposure"),
        weight=pd.Series(weight, index=idx, name="weight"),
        cash=pd.Series(cash_s, index=idx, name="cash"),
        shares=pd.Series(shares_s, index=idx, name="shares"),
        turnover=pd.Series(turnover, index=idx, name="turnover"),
        costs=pd.Series(costs, index=idx, name="costs"),
        asset_returns=gross.rename("asset"),
        rf=pd.Series(rf, index=idx, name="rf"),
        fills=pd.DataFrame(fills, columns=FILL_COLUMNS),
        execution=execution,
        periods_per_year=periods_per_year(prices.index),
        config={
            "initial_capital": initial_capital,
            "commission_per_trade": commission_per_trade,
            "cost_bps": cost_bps,
            "integer_shares": integer_shares,
            "rebalance": rebalance,
        },
    )


def legacy_notebook_backtest(
    prices_window: pd.DataFrame,
    *,
    fast: int = 50,
    slow: int = 200,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 5.0,
) -> dict:
    """Replicate the original notebook's EMA50/200 backtest exactly, biases included.

    * EMAs use ``ewm(span, adjust=False, min_periods=0)`` on the window only, so
      EMA200 is dominated by its seed (the first close) for most of the sample;
    * a buy (sell) fires only on the bar where EMA50 crosses above (below)
      EMA200 and is filled at that *same* close (look-ahead);
    * buys take ``capital // close`` whole shares; every signal costs $5, even
      a sell while flat; idle cash earns nothing;
    * buy-and-hold buys ``(capital - 5) / first close`` fractional shares.

    Returns a dict with ``algo_final``, ``bnh_final``, ``algo_total_return``,
    ``bnh_total_return``, ``algo_max_drawdown``, ``bnh_max_drawdown`` (floats),
    ``algo_equity``/``bnh_equity`` (Series), ``signals`` (DataFrame with
    ``Close, ema_fast, ema_slow, buy, sell``) and ``trades`` (DataFrame with
    ``date, side, price, shares, fee, cash_after``).
    """
    px = prices_window["Close"].astype(float)
    ema_f = px.ewm(span=fast, min_periods=0, adjust=False).mean().to_numpy()
    ema_s = px.ewm(span=slow, min_periods=0, adjust=False).mean().to_numpy()
    c = px.to_numpy()
    n = len(c)
    buy = np.zeros(n, dtype=int)
    sell = np.zeros(n, dtype=int)
    for i in range(1, n):
        if ema_f[i] > ema_s[i] and ema_f[i - 1] <= ema_s[i - 1]:
            buy[i] = 1
        elif ema_f[i] < ema_s[i] and ema_f[i - 1] >= ema_s[i - 1]:
            sell[i] = -1

    capital, position = float(initial_capital), 0.0
    bnh_shares = (capital - transaction_cost) / c[0]
    algo_eq, bnh_eq, trades = np.zeros(n), np.zeros(n), []
    for i in range(n):
        if buy[i] == 1:
            qty = capital // c[i]
            position += qty
            capital -= qty * c[i] + transaction_cost
            trades.append((px.index[i], "buy", c[i], qty, transaction_cost, capital))
        elif sell[i] == -1:
            qty = position
            capital += position * c[i] - transaction_cost
            position = 0.0
            trades.append((px.index[i], "sell", c[i], qty, transaction_cost, capital))
        algo_eq[i] = capital + position * c[i]
        bnh_eq[i] = bnh_shares * c[i]

    algo = pd.Series(algo_eq, index=px.index, name="Algo_Equity")
    bnh = pd.Series(bnh_eq, index=px.index, name="BNH_Equity")
    return {
        "algo_final": float(algo.iloc[-1]),
        "bnh_final": float(bnh.iloc[-1]),
        "algo_total_return": float((algo.iloc[-1] - initial_capital) / initial_capital),
        "bnh_total_return": float((bnh.iloc[-1] - initial_capital) / initial_capital),
        "algo_max_drawdown": float(((algo - algo.cummax()) / algo.cummax()).min()),
        "bnh_max_drawdown": float(((bnh - bnh.cummax()) / bnh.cummax()).min()),
        "algo_equity": algo,
        "bnh_equity": bnh,
        "signals": pd.DataFrame(
            {"Close": px, "ema_fast": ema_f, "ema_slow": ema_s, "buy": buy, "sell": sell},
            index=px.index,
        ),
        "trades": pd.DataFrame(
            trades, columns=["date", "side", "price", "shares", "fee", "cash_after"]
        ),
    }
