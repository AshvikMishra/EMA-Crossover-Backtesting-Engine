from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices

from trendlab.data import per_bar_rate
from trendlab.engine import BacktestConfig, CostModel, evaluation_start
from trendlab.metrics import sharpe
from trendlab.strategies import BuyAndHold, EMACrossover, scaled
from trendlab.validation import (
    grid_backtests,
    grid_metrics,
    grid_pivot,
    grid_returns,
    parameter_grid,
    walk_forward,
)

RF = 0.02


def _frame(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = pd.Series(close, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * 1.002,
            "Low": np.minimum(open_, close) * 0.998,
            "Close": close,
            "Volume": 1e6,
        }
    )


def trending_prices(years: int = 8, seed: int = 0) -> pd.DataFrame:
    """Alternating year-long up/down regimes with strong drift plus daily noise."""
    idx = pd.bdate_range("2000-01-03", periods=years * 252)
    rng = np.random.default_rng(seed)
    regime = np.where((np.arange(len(idx)) // 180) % 2 == 0, 0.002, -0.0015)
    r = regime + rng.normal(0, 0.01, len(idx))
    return _frame(100 * np.cumprod(1 + r), idx)


def noise_prices(years: int = 8, seed: int = 1) -> pd.DataFrame:
    idx = pd.bdate_range("2000-01-03", periods=years * 252)
    r = np.random.default_rng(seed).normal(0, 0.012, len(idx))
    return _frame(100 * np.cumprod(1 + r), idx)


GRID = parameter_grid([5, 10, 20], [20, 40, 60])
CFG = BacktestConfig()


@pytest.fixture(scope="module")
def trend():
    return trending_prices()


@pytest.fixture(scope="module")
def wf(trend):
    return walk_forward(trend, GRID, CFG, RF, train_years=2, test_years=1)


def test_parameter_grid_only_fast_below_slow():
    grid = parameter_grid([5, 20, 50], [20, 50], kind="sma")
    assert [(s.fast, s.slow) for s in grid] == [(5, 20), (5, 50), (20, 50)]
    assert all(s.kind == "sma" for s in grid)
    assert len({s.name for s in GRID}) == len(GRID) == 8


def test_grid_returns_common_window(trend):
    rets = grid_returns(trend, GRID, CFG, RF)
    assert list(rets.columns) == [s.name for s in GRID]
    assert rets.index[0] == evaluation_start(trend, GRID)
    assert rets.index[-1] == trend.index[-1]
    assert not rets.isna().any().any()
    later = grid_returns(trend, GRID, CFG, RF, start="2003-01-01")
    assert later.index[0] >= pd.Timestamp("2003-01-01")


def test_grid_metrics_and_pivot(trend):
    m = grid_metrics(trend, GRID, CFG, RF)
    assert list(m.columns) == [
        "fast",
        "slow",
        "sharpe",
        "cagr",
        "max_drawdown",
        "exposure",
        "n_trades",
    ]
    assert len(m) == len(GRID)
    assert (m["max_drawdown"] <= 0).all() and (m["n_trades"] > 0).all()
    piv = grid_pivot(m, "sharpe")
    assert list(piv.index) == [20, 40, 60] and list(piv.columns) == [5, 10, 20]
    assert np.isnan(piv.loc[20, 20])


def test_duplicate_names_rejected(trend):
    with pytest.raises(ValueError):
        grid_returns(trend, GRID + GRID[:1], CFG, RF)


def test_folds_are_contiguous_and_non_overlapping(wf, trend):
    f = wf.folds
    idx = wf.candidate_returns.index
    assert len(f) >= 4
    pos = {c: idx.get_indexer(f[c]) for c in ["train_start", "train_end", "test_start", "test_end"]}
    # next_close: decision at train-end close is held from two bars later -> 1-bar embargo
    assert (pos["test_start"] - pos["train_end"] == 2).all()
    assert (pos["test_start"][1:] == pos["test_end"][:-1] + 1).all()
    assert (pos["test_end"] >= pos["test_start"]).all()
    assert (pos["train_end"] - pos["train_start"] + 1 == 2 * 252).all()
    assert pos["train_start"][0] == 0


def test_oos_index_is_union_of_test_windows(wf):
    idx = wf.candidate_returns.index
    parts = [idx[(idx >= r.test_start) & (idx <= r.test_end)] for r in wf.folds.itertuples()]
    union = parts[0].append(parts[1:])
    assert wf.oos_returns.index.equals(union)
    assert wf.benchmark_oos.index.equals(union)
    assert wf.oos_exposure.index.equals(union)
    assert wf.chosen.notna().all()


def test_choice_is_argmax_of_train_objective(wf, trend):
    rets = grid_returns(trend, GRID, CFG, RF)
    rf = per_bar_rate(trend.index, RF).reindex(rets.index)
    for row in wf.folds.itertuples():
        win = rets.loc[row.train_start : row.train_end]
        rf_win = rf.loc[win.index]
        scores = win.apply(lambda col, rf_win=rf_win: sharpe(col, rf_win, 252))
        assert row.chosen == scores.idxmax()
        assert row.is_objective == pytest.approx(scores.max())
        assert (row.fast, row.slow) == (
            next((s.fast, s.slow) for s in GRID if s.name == row.chosen)
        )


def test_no_leakage_from_future_prices(trend, wf):
    cut = wf.folds["test_start"].iloc[2]
    shocked = trend.copy()
    after = shocked.index >= cut
    rng = np.random.default_rng(42)
    factor = np.cumprod(1 + rng.normal(0, 0.03, after.sum()))
    for col in ["Open", "High", "Low", "Close"]:
        shocked.loc[after, col] = shocked.loc[after, col].to_numpy() * factor
    wf2 = walk_forward(shocked, GRID, CFG, RF, train_years=2, test_years=1)
    early = wf.folds["test_start"] < cut
    early2 = wf2.folds["test_start"] < cut
    cols = ["train_start", "train_end", "test_start", "test_end", "chosen", "is_objective"]
    pd.testing.assert_frame_equal(wf.folds.loc[early, cols], wf2.folds.loc[early2, cols])
    # the fold whose test starts at the shock was chosen on unshocked data too
    k = int(early.sum())
    assert wf.folds["chosen"].iloc[k] == wf2.folds["chosen"].iloc[k]
    before = wf.oos_returns.index < cut
    pd.testing.assert_series_equal(
        wf.oos_returns[before], wf2.oos_returns[wf2.oos_returns.index < cut]
    )


def test_zero_cost_oos_is_slice_of_chosen_candidates(trend):
    cfg = BacktestConfig(costs=CostModel.zero())
    res = walk_forward(trend, GRID, cfg, RF, train_years=2, test_years=1)
    expected = pd.concat(
        [
            res.candidate_returns.loc[r.test_start : r.test_end, r.chosen]
            for r in res.folds.itertuples()
        ]
    )
    np.testing.assert_allclose(res.oos_returns.to_numpy(), expected.to_numpy())
    assert (res.folds["switch_cost"] == 0).all()


def test_switching_cost_replaces_own_turnover(trend, wf):
    bps = CFG.costs.bps / 1e4
    rets = wf.candidate_returns
    folds = wf.folds
    switched = 0
    for k in range(1, len(folds)):
        prev, cur = folds.iloc[k - 1], folds.iloc[k]
        t0 = cur["test_start"]
        diff = wf.oos_returns.loc[t0] - rets.loc[t0, cur["chosen"]]
        if prev["chosen"] == cur["chosen"]:
            assert diff == 0
            continue
        switched += 1
        assert cur["switch_cost"] == pytest.approx(
            abs(wf.oos_exposure.loc[t0] - wf.oos_exposure.loc[:t0].iloc[-2]) * bps
        )
        # cost difference is bounded by one full turnover of each leg
        assert abs(diff) <= 2 * bps + 1e-12
    assert switched > 0


def test_benchmark_is_buy_and_hold(wf, trend):
    g = trend["Close"].pct_change().reindex(wf.benchmark_oos.index)
    np.testing.assert_allclose(wf.benchmark_oos.to_numpy(), g.to_numpy())
    assert BuyAndHold().name == wf.benchmark_oos.name


def test_anchored_train_windows_start_at_beginning(trend):
    res = walk_forward(trend, GRID, CFG, RF, train_years=2, test_years=1, anchored=True)
    assert (res.folds["train_start"] == res.candidate_returns.index[0]).all()
    assert res.folds["train_end"].is_monotonic_increasing


def test_wfe_and_trend_vs_noise(wf):
    noise = walk_forward(noise_prices(), GRID, CFG, RF, train_years=2, test_years=1)
    f = wf.folds
    expected = f["oos_ann_excess"].mean() / f["is_ann_excess"].mean()
    assert wf.walk_forward_efficiency == pytest.approx(expected)
    rf = wf.rf_oos
    assert f["oos_ann_excess"].iloc[0] == pytest.approx(
        (wf.oos_returns - rf).loc[f["test_start"].iloc[0] : f["test_end"].iloc[0]].mean() * 252
    )
    # a strongly trending series rewards trend following out of sample
    assert wf.oos_returns.mean() > 0
    assert sharpe(wf.oos_returns, wf.rf_oos) > sharpe(noise.oos_returns, noise.rf_oos)
    s = wf.summary()
    assert list(s.index) == ["Walk-forward OOS", "Buy & Hold"]
    assert "sharpe" in s.columns


def test_custom_objective_and_errors(trend):
    res = walk_forward(
        trend, GRID, CFG, RF, train_years=2, test_years=1, objective=lambda r, rf, p: -r.std()
    )
    assert res.folds["chosen"].notna().all()
    with pytest.raises(ValueError):
        walk_forward(trend, GRID, CFG, RF, objective="bogus")
    with pytest.raises(ValueError):
        walk_forward(trend, GRID, CFG, RF, train_years=20)


def test_wfe_is_nan_for_rules_that_never_invest(trend):
    """On raw returns a rule sitting in T-bills would show WFE ~ 1 without any edge."""
    cash = [scaled(BuyAndHold(), 0.0), scaled(EMACrossover(5, 20), 0.0)]
    res = walk_forward(trend, cash, CFG, RF, train_years=2, test_years=1)
    f = res.folds
    assert f["oos_ann_return"].mean() / f["is_ann_return"].mean() == pytest.approx(1.0, abs=0.01)
    assert np.isnan(res.walk_forward_efficiency)


@pytest.mark.parametrize("execution", ["next_close", "next_open", "same_close"])
def test_stitched_returns_follow_engine_formula(execution):
    """r = w*g + (1-w)*rf - |dw|*bps on the stitched position path, incl. switch bars.

    Uses prices with overnight gaps so that the next_open overnight leg matters.
    """
    px = make_prices(n=2500, seed=4)
    grid = parameter_grid([2, 3, 5], [8, 13, 21])
    bps = 50.0
    cfg = BacktestConfig(execution=execution, costs=CostModel.flat(bps))
    res = walk_forward(px, grid, cfg, RF, train_years=0.5, test_years=0.1)
    f = res.folds
    assert (f["chosen"] != f["chosen"].shift()).iloc[1:].sum() >= 5
    idx = res.oos_returns.index
    first = grid_backtests(px, grid, cfg, RF)[f["chosen"].iloc[0]].exposure
    w = res.oos_exposure.to_numpy()
    w_prev = np.concatenate([[first.iloc[first.index.get_loc(idx[0]) - 1]], w[:-1]])
    c = px["Close"].reindex(idx).to_numpy()
    c_prev = px["Close"].shift(1).reindex(idx).to_numpy()
    if execution == "next_open":
        o = px["Open"].reindex(idx).to_numpy()
        risky = (1 + w_prev * (o / c_prev - 1)) * (1 + w * (c / o - 1)) - 1
    else:
        risky = w * (c / c_prev - 1)
    rf = per_bar_rate(px.index, RF).reindex(idx).to_numpy()
    expected = risky + (1 - w) * rf - np.abs(w - w_prev) * bps / 1e4
    np.testing.assert_allclose(res.oos_returns.to_numpy(), expected, rtol=0, atol=1e-12)
    t0 = idx.get_indexer(f["test_start"])
    changed = (f["chosen"] != f["chosen"].shift()).to_numpy().copy()
    changed[0] = False
    want = np.where(changed, np.abs(w[t0] - w_prev[t0]) * bps / 1e4, 0.0)
    np.testing.assert_allclose(f["switch_cost"].to_numpy(), want, rtol=0, atol=1e-15)


def test_no_leakage_from_bar_after_train_end(trend, wf):
    """The embargo bar (filled at its close, held from the next) must not inform the choice."""
    k = 2
    row = wf.folds.iloc[k]
    pos = trend.index.get_loc(row["train_end"])
    cols = ["chosen", "is_objective", "train_end", "test_start"]

    def shocked(from_pos: int) -> pd.DataFrame:
        out = trend.copy()
        after = np.arange(len(out)) >= from_pos
        factor = np.cumprod(1 + np.random.default_rng(7).normal(0.01, 0.05, after.sum()))
        for col in ["Open", "High", "Low", "Close"]:
            out.loc[after, col] = out.loc[after, col].to_numpy() * factor
        return out

    late = walk_forward(shocked(pos + 1), GRID, CFG, RF, train_years=2, test_years=1)
    pd.testing.assert_series_equal(late.folds.iloc[k][cols], row[cols])
    # sanity: the test is not vacuous -- touching the last train bar does move the IS objective
    early = walk_forward(shocked(pos), GRID, CFG, RF, train_years=2, test_years=1)
    assert early.folds["is_objective"].iloc[k] != row["is_objective"]


def test_embargo_shorter_than_fill_delay_rejected(trend):
    with pytest.raises(ValueError, match="embargo"):
        walk_forward(trend, GRID, CFG, RF, train_years=2, test_years=1, embargo=0)
    res = walk_forward(trend, GRID, CFG, RF, train_years=2, test_years=1, embargo=5)
    idx = res.candidate_returns.index
    gap = idx.get_indexer(res.folds["test_start"]) - idx.get_indexer(res.folds["train_end"])
    assert (gap == 6).all()
    same = walk_forward(
        trend, GRID, BacktestConfig(execution="same_close"), RF, train_years=2, test_years=1
    )
    gap = idx.get_indexer(same.folds["test_start"]) - idx.get_indexer(same.folds["train_end"])
    assert (gap == 1).all()


def test_grid_pivot_keeps_nan_columns_and_rejects_duplicates():
    m = pd.DataFrame(
        {
            "fast": [5, 5, 10, np.nan],
            "slow": [20, 40, 40, np.nan],
            "calmar": [np.nan, np.nan, 1.0, 2],
        },
        index=["a", "b", "c", "Buy & Hold"],
    )
    piv = grid_pivot(m, "calmar")
    assert list(piv.columns) == [5, 10] and list(piv.index) == [20, 40]
    assert piv.loc[40, 10] == 1.0 and piv[5].isna().all()
    with pytest.raises(ValueError, match="duplicate"):
        grid_pivot(pd.concat([m, m.iloc[[2]].rename(index={"c": "d"})]), "calmar")


def test_grid_backtests_shares_engine_window_and_returns(trend):
    from trendlab.engine import run_backtest

    res = grid_backtests(trend, GRID, CFG, RF)
    common = evaluation_start(trend, GRID)
    s = GRID[-1]
    ref = run_backtest(trend, s, BacktestConfig(start=common), RF)
    pd.testing.assert_series_equal(res[s.name].returns, ref.returns)
    with pytest.raises(ValueError):
        grid_backtests(trend, GRID, CFG, RF, start="2100-01-01")
