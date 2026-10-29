"""Tests for multi-asset portfolios: engine equivalence, budgets, look-ahead, benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_prices

from trendlab.engine import BacktestConfig, CostModel, run_backtest
from trendlab.portfolio import (
    _ex_ante_vol,
    align_universe,
    blend,
    common_calendar,
    correlation_matrix,
    portfolio_backtest,
    rebalance_mask,
    static_mix_backtest,
)
from trendlab.strategies import BuyAndHold, EMACrossover

RF = 0.03


def _rf_series(index: pd.DatetimeIndex) -> pd.Series:
    days = pd.date_range(index[0] - pd.Timedelta(days=10), index[-1], freq="D")
    return pd.Series(np.linspace(0.01, 0.05, len(days)), index=days)


def _universe(n: int = 900) -> dict[str, pd.DataFrame]:
    return {
        "A": make_prices(n, seed=1, vol=0.10),
        "B": make_prices(n, seed=2, vol=0.30),
        "C": make_prices(n, seed=3, vol=0.20, drift=-0.05),
    }


@pytest.mark.parametrize("execution", ["next_close", "same_close"])
@pytest.mark.parametrize("strategy", [BuyAndHold(), EMACrossover(10, 40)])
def test_single_asset_matches_engine(execution, strategy):
    px = make_prices(700, seed=5)
    rf = _rf_series(px.index)
    costs = CostModel()
    ref = run_backtest(px, strategy, BacktestConfig(execution=execution, costs=costs), rf_yield=rf)
    res = portfolio_backtest({"X": px}, strategy, execution=execution, costs=costs, rf_yield=rf)
    assert res.returns.index.equals(ref.returns.index)
    np.testing.assert_allclose(res.returns.to_numpy(), ref.returns.to_numpy(), atol=1e-10, rtol=0)
    np.testing.assert_allclose(res.weights["X"].to_numpy(), ref.exposure.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(res.turnover.to_numpy(), ref.turnover.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(res.equity.to_numpy(), ref.equity.to_numpy(), rtol=1e-10)


def test_single_asset_matches_engine_with_start():
    px = make_prices(700, seed=6)
    cfg = BacktestConfig(costs=CostModel.flat(10), start="2001-03-01")
    ref = run_backtest(px, BuyAndHold(), cfg, rf_yield=RF)
    res = portfolio_backtest(
        {"X": px}, BuyAndHold(), costs=CostModel.flat(10), rf_yield=RF, start="2001-03-01"
    )
    np.testing.assert_allclose(res.returns.to_numpy(), ref.returns.to_numpy(), atol=1e-10)


def test_next_open_not_implemented():
    with pytest.raises(NotImplementedError):
        portfolio_backtest(_universe(300), BuyAndHold(), execution="next_open")


def test_contributions_sum_to_returns():
    res = portfolio_backtest(
        _universe(), EMACrossover(10, 40), weighting="inverse_vol", rf_yield=RF
    )
    np.testing.assert_allclose(res.contributions.sum(axis=1), res.returns, atol=1e-14)
    assert list(res.contributions.columns) == ["A", "B", "C", "cash", "costs"]
    np.testing.assert_allclose(res.cash + res.weights.sum(axis=1), 1.0)


@pytest.mark.parametrize("weighting", ["equal", "inverse_vol"])
def test_weights_bounded_by_gross_target(weighting):
    res = portfolio_backtest(
        _universe(), EMACrossover(10, 40), weighting=weighting, gross_target=0.8
    )
    assert (res.weights >= -1e-15).all().all()
    assert res.gross_exposure.max() <= 0.8 + 1e-12
    assert res.gross_exposure.max() > 0.5


def test_inverse_vol_favours_low_vol_and_sums_to_one():
    uni = {"LO": make_prices(600, seed=1, vol=0.10), "HI": make_prices(600, seed=2, vol=0.30)}
    res = portfolio_backtest(uni, BuyAndHold(), weighting="inverse_vol", vol_window=63)
    s = res.strategic.iloc[100:]
    np.testing.assert_allclose(s.sum(axis=1), 1.0, atol=1e-12)
    assert (s["LO"] > s["HI"]).all()
    assert 2.0 < (s["LO"] / s["HI"]).median() < 4.5  # ~ vol ratio of 3


def test_late_inception_asset_has_zero_weight_and_budget_is_redistributed():
    early = make_prices(800, seed=1)
    late = make_prices(500, seed=2).set_axis(early.index[300:], axis=0)
    res = portfolio_backtest({"E": early, "L": late}, BuyAndHold(), costs=CostModel.zero())
    before = res.weights.loc[: early.index[300]]
    assert (before["L"] == 0).all()
    np.testing.assert_allclose(before["E"].iloc[2:], 1.0)
    after = res.weights.loc[early.index[305] :]
    np.testing.assert_allclose(after.to_numpy(), 0.5)
    first_l = res.weights.index[res.weights["L"] > 0][0]
    assert first_l == early.index[302]  # decided at its first close, filled at the next


def test_late_inception_inverse_vol_waits_for_vol_estimate():
    early = make_prices(800, seed=1)
    late = make_prices(500, seed=2).set_axis(early.index[300:], axis=0)
    res = portfolio_backtest(
        {"E": early, "L": late}, BuyAndHold(), weighting="inverse_vol", vol_window=40
    )
    assert (res.weights.loc[: early.index[300 + 20], "L"] == 0).all()
    assert res.weights["L"].iloc[-1] > 0


@pytest.mark.parametrize("execution", ["next_close", "same_close"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"weighting": "inverse_vol"},
        {"weighting": "equal", "vol_target": 0.10, "max_leverage": 2.0},
        {"weighting": "inverse_vol", "vol_target": 0.10, "vol_estimator": "ex_ante"},
    ],
)
def test_no_lookahead_under_future_perturbation(kwargs, execution):
    uni = _universe(700)
    run = {"rf_yield": RF, "execution": execution, **kwargs}
    base = portfolio_backtest(uni, EMACrossover(10, 40), **run)
    cut = uni["A"].index[500]
    shocked = {}
    rng = np.random.default_rng(9)
    for t, df in uni.items():
        df = df.copy()
        after = df.index >= cut
        df.loc[after, ["Open", "High", "Low", "Close"]] *= rng.uniform(0.5, 1.5, (after.sum(), 1))
        shocked[t] = df
    alt = portfolio_backtest(shocked, EMACrossover(10, 40), **run)
    k = base.returns.index.get_loc(cut)
    # weights held over bar j are decided at close j-lag, so they cannot see close k before j=k+lag
    held_ok = k + (1 if execution == "next_close" else 0)
    pd.testing.assert_frame_equal(base.weights.iloc[: held_ok + 1], alt.weights.iloc[: held_ok + 1])
    pd.testing.assert_series_equal(base.returns.iloc[:k], alt.returns.iloc[:k])
    assert not np.allclose(base.returns.iloc[k:], alt.returns.iloc[k:])
    assert not np.allclose(base.weights.iloc[held_ok + 1 :], alt.weights.iloc[held_ok + 1 :])


def test_ex_ante_vol_matches_pairwise_rolling_cov():
    uni = _universe(300)
    late = uni["C"].iloc[200:]
    aligned = align_universe({"A": uni["A"], "B": uni["B"], "C": late})
    rets = pd.DataFrame({t: df["Close"].pct_change() for t, df in aligned.items()})
    rng = np.random.default_rng(3)
    w = pd.DataFrame(rng.uniform(0, 1, rets.shape), index=rets.index, columns=rets.columns)
    window = 40
    got = _ex_ante_vol(w, rets, window, 252)
    for t in (60, 150, 225, 260, 299):
        cov = rets.iloc[t - window + 1 : t + 1].cov(min_periods=20).to_numpy()
        x = w.iloc[t].to_numpy()
        if np.isnan(cov).any():
            assert np.isnan(got.iloc[t])
        else:
            np.testing.assert_allclose(got.iloc[t], np.sqrt(x @ cov @ x * 252), rtol=1e-8)
    assert np.isnan(got.iloc[205]) and np.isfinite(got.iloc[225])


def test_ex_ante_does_not_treat_new_listing_as_riskless():
    early = make_prices(800, seed=1, vol=0.20)
    late = make_prices(500, seed=2, vol=0.60).set_axis(early.index[300:], axis=0)
    res = portfolio_backtest(
        {"E": early, "L": late},
        BuyAndHold(),
        vol_target=0.10,
        max_leverage=3.0,
        vol_estimator="ex_ante",
    )
    before = res.scale.loc[early.index[299]]
    joined = res.scale.loc[early.index[300] : early.index[360]]
    assert (joined <= before + 0.05).all()  # adding a 60%-vol asset must not raise leverage
    # 50/50 of 20% and 60% vol is ~32% book vol, so the scale settles near 0.3
    assert 0.2 < res.scale.loc[early.index[400] :].median() < 0.45


def test_delisted_asset_budget_is_redistributed():
    alive = make_prices(600, seed=1)
    dead = make_prices(300, seed=2)
    out = align_universe({"A": alive, "D": dead})
    assert out["D"].index[-1] == dead.index[-1]  # no stale prices after the last observation
    res = portfolio_backtest({"A": alive, "D": dead}, BuyAndHold(), costs=CostModel.zero())
    tail = res.weights.loc[alive.index[303] :]
    np.testing.assert_allclose(tail["A"], 1.0)
    np.testing.assert_allclose(tail["D"], 0.0)
    mix = static_mix_backtest({"A": alive, "D": dead}, {"A": 0.5, "D": 0.5})
    np.testing.assert_allclose(mix.weights.loc[alive.index[302] :, "D"], 0.0)


def test_vol_target_scales_book_towards_target():
    uni = {t: make_prices(1500, seed=s, vol=0.30) for t, s in (("A", 1), ("B", 2))}
    res = portfolio_backtest(uni, BuyAndHold(), vol_target=0.10, max_leverage=3.0)
    realised = res.returns.iloc[200:].std() * np.sqrt(252)
    assert 0.07 < realised < 0.13
    assert res.scale.max() <= 3.0 + 1e-12
    capped = portfolio_backtest(uni, BuyAndHold(), vol_target=1.0, max_leverage=1.0)
    assert capped.gross_exposure.max() <= 1.0 + 1e-12


def test_strategy_mapping_and_callable_agree():
    uni = _universe(500)
    spec = {t: EMACrossover(10, 40) for t in uni}
    a = portfolio_backtest(uni, spec)
    b = portfolio_backtest(uni, lambda t: EMACrossover(10, 40))
    pd.testing.assert_series_equal(a.returns, b.returns)
    with pytest.raises(KeyError):
        portfolio_backtest(uni, {"A": BuyAndHold()})


def test_per_asset_costs():
    uni = _universe(500)
    cheap = portfolio_backtest(uni, EMACrossover(10, 40), costs=CostModel.zero())
    mixed = portfolio_backtest(
        uni, EMACrossover(10, 40), costs={"A": CostModel.flat(100), "B": CostModel.zero()}
    )
    assert mixed.costs.sum() > 0
    assert (mixed.returns <= cheap.returns + 1e-15).all()


def test_align_universe_holidays_and_inception():
    idx = pd.bdate_range("2020-01-01", periods=30, name="Date")
    a = make_prices(30, seed=1).set_axis(idx, axis=0)
    b = make_prices(25, seed=2).set_axis(idx[5:], axis=0).drop(idx[10])  # holiday on idx[10]
    out = align_universe({"A": a, "B": b}, calendar="A")
    assert out["A"].index.equals(idx)
    assert out["B"].index[0] == idx[5]
    hol = out["B"].loc[idx[10]]
    assert hol["Close"] == b["Close"].loc[idx[9]]
    assert hol["Open"] == hol["High"] == hol["Low"] == hol["Close"]
    assert hol["Volume"] == 0
    assert out["B"]["Close"].pct_change().loc[idx[10]] == 0.0
    # the next session's return spans the holiday exactly
    np.testing.assert_allclose(
        out["B"]["Close"].pct_change().loc[idx[11]],
        b["Close"].loc[idx[11]] / b["Close"].loc[idx[9]] - 1,
    )


def test_union_calendar_drops_weekends_and_compounds_crypto():
    btc = make_prices(60, seed=1, freq="D")
    etf = make_prices(40, seed=2)
    cal = common_calendar({"BTC": btc, "ETF": etf}, "union")
    assert (cal.day_of_week < 5).all()
    out = align_universe({"BTC": btc, "ETF": etf}, calendar="union")
    mondays = out["BTC"].index[out["BTC"].index.day_of_week == 0]
    m = mondays[1]
    prev_fri = m - pd.Timedelta(days=3)
    np.testing.assert_allclose(
        out["BTC"]["Close"].pct_change().loc[m],
        btc["Close"].loc[m] / btc["Close"].loc[prev_fri] - 1,
    )


def test_crypto_only_universe_keeps_weekends_and_matches_engine():
    btc = make_prices(900, seed=7, freq="D", vol=0.6)
    assert len(common_calendar({"BTC": btc})) == len(btc)
    ref = run_backtest(btc, EMACrossover(10, 40), BacktestConfig(costs=CostModel.flat(20)))
    res = portfolio_backtest({"BTC": btc}, EMACrossover(10, 40), costs=CostModel.flat(20))
    assert res.periods_per_year == 365
    np.testing.assert_allclose(res.returns.to_numpy(), ref.returns.to_numpy(), atol=1e-12)


def test_rebalance_mask_month_ends():
    idx = pd.bdate_range("2021-01-01", "2021-04-30")
    mask = rebalance_mask(idx, "ME")
    assert list(idx[mask].strftime("%Y-%m-%d")) == [
        "2021-01-29",
        "2021-02-26",
        "2021-03-31",
        "2021-04-30",
    ]
    assert rebalance_mask(idx, "D").all()
    assert not rebalance_mask(idx, None).any()


def test_static_mix_single_asset_matches_buy_and_hold():
    px = make_prices(600, seed=4)
    rf = _rf_series(px.index)
    ref = run_backtest(px, BuyAndHold(), BacktestConfig(costs=CostModel.zero()), rf_yield=rf)
    res = static_mix_backtest({"X": px}, {"X": 1.0}, costs=CostModel.zero(), rf_yield=rf)
    np.testing.assert_allclose(res.returns.to_numpy(), ref.returns.to_numpy(), atol=1e-12)


def test_static_mix_drifts_between_rebalances():
    uni = {"SPY": make_prices(600, seed=1), "IEF": make_prices(600, seed=2, vol=0.07)}
    res = static_mix_backtest(uni, {"SPY": 0.6, "IEF": 0.4}, costs=CostModel(), rf_yield=RF)
    np.testing.assert_allclose(res.contributions.sum(axis=1), res.returns, atol=1e-14)
    np.testing.assert_allclose(res.weights.sum(axis=1).iloc[1:], 1.0, atol=1e-3)
    me = rebalance_mask(res.weights.index, "ME")
    after_reb = np.flatnonzero(me[:-1]) + 1
    np.testing.assert_allclose(res.weights["SPY"].iloc[after_reb], 0.6, atol=1e-12)
    assert res.weights["SPY"].iloc[1:].std() > 1e-3  # drift inside the month
    assert (res.turnover > 0).sum() <= me.sum() + 1  # trades only after rebalance closes
    never = static_mix_backtest(uni, {"SPY": 0.6, "IEF": 0.4}, rebalance=None)
    assert (never.turnover > 0).sum() == 1


def test_static_mix_late_asset_redistributes():
    early = make_prices(400, seed=1)
    late = make_prices(200, seed=2).set_axis(early.index[200:], axis=0)
    res = static_mix_backtest({"E": early, "L": late}, {"E": 0.5, "L": 0.5})
    assert (res.weights.loc[: early.index[200], "L"] == 0).all()
    np.testing.assert_allclose(res.weights.loc[early.index[201], ["E", "L"]], 0.5, atol=1e-12)


def test_blend():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 0.01, 300), index=idx, name="A")
    b = pd.Series(rng.normal(0, 0.02, 300), index=idx, name="B")
    np.testing.assert_allclose(blend(a, b, 1.0), a, atol=1e-15)
    np.testing.assert_allclose(blend(a, b, 0.8, rebalance="D"), 0.8 * a + 0.2 * b, atol=1e-15)
    m = blend(a, b, 0.8)
    assert not np.allclose(m, 0.8 * a + 0.2 * b)
    np.testing.assert_allclose((1 + m).prod(), (1 + 0.8 * a + 0.2 * b).prod(), rtol=0.01)
    # buy-and-hold blend compounds each sleeve separately
    bh = blend(a, b, 0.5, rebalance=None)
    np.testing.assert_allclose(
        (1 + bh).prod(), 0.5 * (1 + a).prod() + 0.5 * (1 + b).prod(), rtol=1e-12
    )


def test_correlation_matrix_monthly():
    idx = pd.bdate_range("2015-01-01", periods=1500)
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, 1500)
    rets = {
        "x": pd.Series(x, index=idx),
        "y": pd.Series(-x, index=idx),
        "z": pd.Series(rng.normal(0, 0.01, 1500), index=idx),
    }
    c = correlation_matrix(rets)
    assert list(c.columns) == ["x", "y", "z"]
    np.testing.assert_allclose(np.diag(c), 1.0)
    np.testing.assert_allclose(c.to_numpy(), c.to_numpy().T)
    assert c.loc["x", "y"] < -0.95
    assert abs(c.loc["x", "z"]) < 0.4
    daily = correlation_matrix(rets, freq=None)
    np.testing.assert_allclose(daily.loc["x", "y"], -1.0)
