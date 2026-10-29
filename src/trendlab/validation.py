"""Parameter search and walk-forward analysis.

Choosing the best-looking (fast, slow) pair on the full sample and reporting its
backtest is in-sample optimisation: with enough candidates the winner's Sharpe
is inflated by selection alone (Bailey, Borwein, Lopez de Prado & Zhu, 2014,
"Pseudo-Mathematics and Financial Charlatanism", Notices of the AMS 61(5)).

Walk-forward analysis (Pardo, 2008, *The Evaluation and Optimization of Trading
Strategies*, ch. 11) removes the look-ahead in the *choice*: on each fold the
candidate with the best objective on a training window is frozen and traded on
the following, untouched test window. Stitching the test windows gives an
out-of-sample (OOS) equity curve, and the walk-forward efficiency

    WFE = annualised OOS mean excess return / annualised IS mean excess return

(both of the chosen configurations, averaged across folds) measures how much of
the in-sample edge survives. Values well below 1 are the norm; values near 0 say
the optimisation was fitting noise. Returns are taken in excess of the T-bill
rate: un-invested capital earns ``rf`` in the engine, so on raw returns a rule
that sits mostly in cash would score a WFE near 1 with no edge at all.

Implementation notes
--------------------
* Every candidate is backtested **once** over the full history (indicators warm
  up on all available data, positions evolve continuously); folds are slices of
  these per-bar returns. Because the rules are causal, a candidate's return on
  bar t depends only on prices up to t, so slicing introduces no leakage.
* The choice made at the close ending a training window can only be filled per
  ``config.execution``. With ``next_close`` a decision at close t is held from
  bar t+2, so an ``embargo`` of one bar separates train end and test start
  (cf. the embargo of Lopez de Prado, 2018, *Advances in Financial Machine
  Learning*, sec. 7.4). Test windows stay contiguous.
* Switching configuration at a fold boundary is charged the turnover
  ``|w_new - w_old| * bps``, replacing the new configuration's own
  ``|w_new - w_new_prev|`` on that bar. With ``next_open`` the overnight leg of
  the boundary bar is also re-priced with the old configuration's exposure, so
  the stitched series is exactly what the engine would produce for the switched
  position path. The first fold is treated as a continuation of the chosen
  configuration (no switch), the same way the buy-and-hold benchmark is a slice
  of a continuous run.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import periods_per_year
from .engine import BacktestConfig, BacktestResult, run_backtest
from .metrics import cagr, calmar, max_drawdown, sharpe, sortino, summary
from .strategies import BuyAndHold, MACrossover, Strategy

Objective = Callable[[pd.Series, pd.Series, int], float]

OBJECTIVES: dict[str, Objective] = {
    "sharpe": lambda r, rf, p: sharpe(r, rf, p),
    "sortino": lambda r, rf, p: sortino(r, rf, p),
    "cagr": lambda r, rf, p: cagr(r, p),
    "calmar": lambda r, rf, p: calmar(r, p),
    "mean": lambda r, rf, p: float(r.mean() * p),
}

_EXECUTION_LAG = {"same_close": 1, "next_open": 1, "next_close": 2}


def parameter_grid(
    fast_values: Iterable[int], slow_values: Iterable[int], kind: str = "ema", **kwargs
) -> list[MACrossover]:
    """All ``MACrossover(fast, slow, kind)`` with ``fast < slow``, fast-major order."""
    slows = list(slow_values)
    return [
        MACrossover(fast=f, slow=s, kind=kind, **kwargs)
        for f in fast_values
        for s in slows
        if f < s
    ]


def _check_names(strategies: Sequence[Strategy]) -> None:
    names = [s.name for s in strategies]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate strategy names: {dupes}")
    if not names:
        raise ValueError("no strategies given")


def grid_backtests(
    prices: pd.DataFrame,
    strategies: Sequence[Strategy],
    config: BacktestConfig | None = None,
    rf_yield: pd.Series | float | None = None,
    start: str | pd.Timestamp | None = None,
) -> dict[str, BacktestResult]:
    """Backtest every strategy over one common window; ``{name: BacktestResult}``.

    The window opens where :func:`engine.evaluation_start` would put it (one bar
    after the slowest rule's first warmed-up decision, and not before ``start``
    / ``config.start``) and ends at ``config.end``, so all results share one
    index. Targets are computed once and handed to :func:`run_backtest`.
    """
    _check_names(strategies)
    config = config or BacktestConfig()
    floor = start if start is not None else config.start
    if start is not None and config.start is not None:
        floor = max(pd.Timestamp(start), pd.Timestamp(config.start))
    targets = {s.name: s.target_position(prices) for s in strategies}
    pos = 0
    for name, target in targets.items():
        first = target.first_valid_index()
        if first is None:
            raise ValueError(f"{name}: not enough history to warm up")
        pos = max(pos, prices.index.get_loc(first) + 1)
    if floor is not None:
        pos = max(pos, int(prices.index.searchsorted(pd.Timestamp(floor))))
    if pos >= len(prices.index):
        raise ValueError("no bars left after warm-up and start")
    cfg = dataclasses.replace(config, start=prices.index[pos])
    return {
        s.name: run_backtest(prices, s, cfg, rf_yield, target=targets[s.name]) for s in strategies
    }


def grid_returns(
    prices: pd.DataFrame,
    strategies: Sequence[Strategy],
    config: BacktestConfig | None = None,
    rf_yield: pd.Series | float | None = None,
    start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-bar returns (T x N, columns = ``strategy.name``) over the common window."""
    res = grid_backtests(prices, strategies, config, rf_yield, start)
    return pd.DataFrame({k: r.returns for k, r in res.items()})


def _metrics_row(strategy: Strategy, res: BacktestResult) -> dict:
    p = res.periods_per_year
    return {
        "fast": getattr(strategy, "fast", np.nan),
        "slow": getattr(strategy, "slow", np.nan),
        "sharpe": sharpe(res.returns, res.rf, p),
        "cagr": cagr(res.returns, p),
        "max_drawdown": max_drawdown(res.returns),
        "exposure": float(res.exposure.abs().mean()),
        "n_trades": int(len(res.trades)),
    }


def grid_metrics(
    prices: pd.DataFrame,
    strategies: Sequence[Strategy],
    config: BacktestConfig | None = None,
    rf_yield: pd.Series | float | None = None,
    start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per strategy (index = name): fast, slow, sharpe, cagr, max_drawdown,
    exposure, n_trades. ``fast``/``slow`` are NaN for non-crossover rules."""
    res = grid_backtests(prices, strategies, config, rf_yield, start)
    rows = {s.name: _metrics_row(s, res[s.name]) for s in strategies}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "strategy"
    return out


def grid_pivot(metrics: pd.DataFrame, value: str = "sharpe") -> pd.DataFrame:
    """Heatmap layout of :func:`grid_metrics`: index = slow, columns = fast.

    Rows without fast/slow are dropped; NaN cells (including fast >= slow) are
    kept. Raises ValueError if a (fast, slow) pair occurs twice (e.g. EMA and
    SMA grids mixed), instead of silently keeping one of them.
    """
    m = metrics.dropna(subset=["fast", "slow"])
    dupes = m.duplicated(subset=["fast", "slow"])
    if dupes.any():
        raise ValueError(f"duplicate (fast, slow) pairs: {list(m.index[dupes])}; filter first")
    return m.pivot(index="slow", columns="fast", values=value).sort_index().sort_index(axis=1)


def _resolve_objective(objective: str | Objective) -> Objective:
    if callable(objective):
        return objective
    try:
        return OBJECTIVES[objective]
    except KeyError:
        msg = f"unknown objective {objective!r}; choose from {list(OBJECTIVES)}"
        raise ValueError(msg) from None


def _fold_bounds(
    n: int, train_bars: int, test_bars: int, embargo: int, anchored: bool, min_test_bars: int
) -> list[tuple[int, int, int, int]]:
    """Positional (train_start, train_end, test_start, test_end), inclusive."""
    folds = []
    test_start = train_bars + embargo
    while test_start < n:
        test_end = min(test_start + test_bars, n) - 1
        if test_end - test_start + 1 < min_test_bars:
            break
        train_end = test_start - 1 - embargo
        train_start = 0 if anchored else train_end - train_bars + 1
        folds.append((train_start, train_end, test_start, test_end))
        test_start = test_end + 1
    return folds


@dataclass
class WalkForwardResult:
    """Output of :func:`walk_forward`.

    ``folds`` has one row per fold with columns train_start, train_end,
    test_start, test_end, chosen, fast, slow, is_objective, oos_objective,
    bench_objective, is_ann_return, oos_ann_return, is_ann_excess,
    oos_ann_excess (arithmetic mean x periods, raw / in excess of rf) and
    switch_cost. ``walk_forward_efficiency`` = mean oos_ann_excess / mean
    is_ann_excess (NaN if the latter is <= 0).
    ``oos_returns``/``oos_exposure``/``benchmark_oos``/``rf_oos`` are indexed by
    the union of test windows; ``is_returns_best[k]`` is the chosen config's
    train-window return series of fold k.
    """

    folds: pd.DataFrame
    oos_returns: pd.Series
    oos_exposure: pd.Series
    benchmark_oos: pd.Series
    rf_oos: pd.Series
    is_returns_best: list[pd.Series]
    walk_forward_efficiency: float
    candidate_returns: pd.DataFrame
    objective: str
    periods_per_year: int

    @property
    def chosen(self) -> pd.Series:
        """Name of the configuration in force on every OOS bar."""
        out = pd.Series(index=self.oos_returns.index, dtype=object, name="chosen")
        for row in self.folds.itertuples():
            out.loc[row.test_start : row.test_end] = row.chosen
        return out

    def summary(self) -> pd.DataFrame:
        """:func:`metrics.summary` of the stitched OOS strategy and buy-and-hold."""
        p = self.periods_per_year
        rows = {
            "Walk-forward OOS": summary(
                self.oos_returns, rf=self.rf_oos, periods=p, exposure=self.oos_exposure
            ),
            "Buy & Hold": summary(self.benchmark_oos, rf=self.rf_oos, periods=p),
        }
        return pd.DataFrame(rows).T


def walk_forward(
    prices: pd.DataFrame,
    candidates: Sequence[Strategy],
    config: BacktestConfig | None = None,
    rf_yield: pd.Series | float | None = None,
    train_years: float = 5,
    test_years: float = 1,
    objective: str | Objective = "sharpe",
    anchored: bool = False,
    start: str | pd.Timestamp | None = None,
    *,
    embargo: int | None = None,
    min_test_fraction: float = 0.25,
) -> WalkForwardResult:
    """Rolling (or anchored) walk-forward selection among ``candidates``.

    On each fold the candidate maximising ``objective`` on the training window
    (ties and NaNs: first candidate wins / NaN ranks last) is traded unchanged
    over the next ``test_years``. ``objective`` is a key of :data:`OBJECTIVES`
    or ``f(returns, rf, periods) -> float``. ``embargo`` (bars between train end
    and test start) defaults to the fill delay of ``config.execution`` minus one.
    A shorter embargo would trade the choice before it could be filled, i.e.
    look-ahead, and is rejected. A trailing partial test window is kept if it
    has at least ``min_test_fraction`` of a full one.
    """
    config = config or BacktestConfig()
    obj_fn = _resolve_objective(objective)
    candidates = list(candidates)
    _check_names(candidates)
    results = grid_backtests(prices, candidates, config, rf_yield, start)
    names = [c.name for c in candidates]
    rets = pd.DataFrame({k: results[k].returns for k in names})
    expo = pd.DataFrame({k: results[k].exposure for k in names})
    rf = results[names[0]].rf
    idx = rets.index

    ppy = periods_per_year(prices.index)
    min_embargo = _EXECUTION_LAG[config.execution] - 1
    if embargo is None:
        embargo = min_embargo
    if embargo < min_embargo:
        raise ValueError(f"embargo must be >= {min_embargo} bar(s) for {config.execution!r}")
    train_bars = int(round(train_years * ppy))
    test_bars = int(round(test_years * ppy))
    if train_bars < 2 or test_bars < 1:
        raise ValueError("train/test windows too short")
    min_test = max(1, int(np.ceil(min_test_fraction * test_bars)))
    bounds = _fold_bounds(len(idx), train_bars, test_bars, embargo, anchored, min_test)
    if not bounds:
        raise ValueError(
            f"not enough history: {len(idx)} bars after warm-up, need "
            f"{train_bars + embargo + min_test} for one fold"
        )

    bench_cfg = dataclasses.replace(config, start=idx[0])
    bench = run_backtest(prices, BuyAndHold(), bench_cfg, rf_yield).returns.reindex(idx)

    by_name = dict(zip(names, candidates, strict=True))
    bps = config.costs.bps / 1e4
    r_mat, w_mat = rets.to_numpy(), expo.to_numpy()
    rf_arr = rf.to_numpy()
    asset = results[names[0]].asset_returns.to_numpy()
    if config.execution == "next_open":
        gap = (prices["Open"] / prices["Close"].shift(1) - 1.0).reindex(idx).to_numpy()
        intraday = (prices["Close"] / prices["Open"] - 1.0).reindex(idx).to_numpy()

    def switch_return(t: int, w_old: float, w_new: float) -> float:
        """Engine return of bar ``t`` for a position moved from ``w_old`` to ``w_new``."""
        if config.execution == "next_open":
            risky = (1.0 + w_old * gap[t]) * (1.0 + w_new * intraday[t]) - 1.0
        else:
            risky = w_new * asset[t]
        return risky + (1.0 - w_new) * rf_arr[t] - abs(w_new - w_old) * bps

    oos_r, oos_w = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
    rows, is_best = [], []
    prev_j: int | None = None
    for a, b, c, d in bounds:
        train = slice(a, b + 1)
        scores = np.array(
            [obj_fn(rets.iloc[train, j], rf.iloc[train], ppy) for j in range(len(names))],
            dtype=float,
        )
        j = int(np.argmax(np.where(np.isnan(scores), -np.inf, scores)))
        seg_r = r_mat[c : d + 1, j].copy()
        seg_w = w_mat[c : d + 1, j]
        switch = 0.0
        if prev_j is not None and prev_j != j:
            w_new, w_old = w_mat[c, j], w_mat[c - 1, prev_j]
            switch = abs(w_new - w_old) * bps
            seg_r[0] = switch_return(c, w_old, w_new)
        oos_r[c : d + 1], oos_w[c : d + 1] = seg_r, seg_w
        prev_j = j

        test = slice(c, d + 1)
        is_r = rets.iloc[train, j]
        seg = pd.Series(seg_r, index=idx[test])
        strat = by_name[names[j]]
        rows.append(
            {
                "train_start": idx[a],
                "train_end": idx[b],
                "test_start": idx[c],
                "test_end": idx[d],
                "chosen": names[j],
                "fast": getattr(strat, "fast", np.nan),
                "slow": getattr(strat, "slow", np.nan),
                "is_objective": scores[j],
                "oos_objective": obj_fn(seg, rf.iloc[test], ppy),
                "bench_objective": obj_fn(bench.iloc[test], rf.iloc[test], ppy),
                "is_ann_return": float(is_r.mean() * ppy),
                "oos_ann_return": float(seg.mean() * ppy),
                "is_ann_excess": float((is_r - rf.iloc[train]).mean() * ppy),
                "oos_ann_excess": float((seg - rf.iloc[test]).mean() * ppy),
                "switch_cost": switch,
            }
        )
        is_best.append(is_r.rename(names[j]))

    folds = pd.DataFrame(rows)
    folds.index.name = "fold"
    mask = ~np.isnan(oos_r)
    oos_idx = idx[mask]
    is_mean = float(folds["is_ann_excess"].mean())
    wfe = float(folds["oos_ann_excess"].mean() / is_mean) if is_mean > 0 else float("nan")
    return WalkForwardResult(
        folds=folds,
        oos_returns=pd.Series(oos_r[mask], index=oos_idx, name="walk_forward"),
        oos_exposure=pd.Series(oos_w[mask], index=oos_idx, name="exposure"),
        benchmark_oos=bench[mask].rename("Buy & Hold"),
        rf_oos=rf[mask].rename("rf"),
        is_returns_best=is_best,
        walk_forward_efficiency=wfe,
        candidate_returns=rets,
        objective=objective
        if isinstance(objective, str)
        else getattr(objective, "__name__", "custom"),
        periods_per_year=ppy,
    )
