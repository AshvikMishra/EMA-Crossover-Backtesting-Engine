"""Backtest-engine tests: timing, look-ahead, costs, cash interest, warm-up and trades.

The checks follow the backtesting pitfalls catalogued by Bailey & Lopez de Prado
(2014) and Lopez de Prado, "Advances in Financial Machine Learning" (2018, ch. 11):
a return at bar T may only depend on data up to T (no look-ahead), fills happen after
the signal, costs scale with traded notional, and results must not depend on how
much pre-sample history happened to be loaded (warm-up bias).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import frame_from_close, make_prices

from trendlab.engine import (
    EXECUTIONS,
    BacktestConfig,
    CostModel,
    evaluation_start,
    extract_trades,
    first_valid_decision,
    run_backtest,
)
from trendlab.strategies import BuyAndHold, DonchianBreakout, EMACrossover, VolTarget


def _cfg(execution: str = "next_close", bps: float = 0.0, **kw) -> BacktestConfig:
    return BacktestConfig(execution=execution, costs=CostModel.flat(bps), **kw)


def _run(prices, target, execution="next_close", bps=0.0, rf=None, **kw):
    cfg = _cfg(execution, bps, earn_cash_rate=rf is not None, **kw)
    return run_backtest(prices, BuyAndHold(), cfg, rf_yield=rf, target=target)


def _step(index: pd.Index, on: int, off: int | None = None) -> pd.Series:
    """Target 0 before bar ``on``, 1 from ``on`` (decided at that close), 0 again from ``off``."""
    w = np.zeros(len(index))
    w[on:off] = 1.0
    return pd.Series(w, index=index)


def _perturb_after(prices: pd.DataFrame, t: int, seed: int = 123) -> pd.DataFrame:
    out = prices.copy()
    shock = np.exp(np.random.default_rng(seed).normal(0.0, 0.05, len(prices) - t - 1))
    for col in ("Open", "High", "Low", "Close"):
        vals = out[col].to_numpy().copy()
        vals[t + 1 :] *= shock
        out[col] = vals
    return out


# --------------------------------------------------------------------------- config


def test_cost_model():
    assert CostModel().bps == pytest.approx(5.0)
    assert CostModel.zero().bps == 0.0
    assert CostModel.flat(12.5).bps == 12.5


def test_invalid_execution_raises():
    with pytest.raises(ValueError):
        BacktestConfig(execution="same_open")


def test_not_enough_history_and_empty_window_raise(prices):
    with pytest.raises(ValueError):
        run_backtest(prices.iloc[:50], EMACrossover(50, 200))
    with pytest.raises(ValueError):
        run_backtest(prices, BuyAndHold(), _cfg(start=prices.index[-1]))
    with pytest.raises(ValueError):
        first_valid_decision(prices.iloc[:50], EMACrossover(50, 200))


# ------------------------------------------------------------------- look-ahead


@pytest.mark.parametrize("execution", EXECUTIONS)
@pytest.mark.parametrize(
    "strategy",
    [EMACrossover(10, 40, allow_short=True), VolTarget(EMACrossover(10, 40), window=40)],
    ids=lambda s: s.name,
)
def test_prefix_invariance(execution, strategy):
    """Returns at bars <= T are identical when prices strictly after T are changed."""
    prices = make_prices(800, seed=21, drift=0.0, vol=0.3)
    rf = pd.Series(np.linspace(0.01, 0.06, len(prices)), index=prices.index)
    t = 500
    cfg = _cfg(execution, bps=5.0)
    base = run_backtest(prices, strategy, cfg, rf_yield=rf)
    rf_moved = rf.copy()
    rf_moved.iloc[t + 1 :] = 0.5
    moved = run_backtest(_perturb_after(prices, t), strategy, cfg, rf_yield=rf_moved)

    cut = prices.index[t]
    pd.testing.assert_series_equal(base.returns.loc[:cut], moved.returns.loc[:cut])
    pd.testing.assert_series_equal(base.equity.loc[:cut], moved.equity.loc[:cut])
    # the exposure *held over* a bar is fixed before that bar starts
    ahead = prices.index[t + (2 if execution == "next_close" else 1)]
    pd.testing.assert_series_equal(base.exposure.loc[:ahead], moved.exposure.loc[:ahead])
    assert not np.allclose(base.returns.loc[cut:].iloc[1:], moved.returns.loc[cut:].iloc[1:])


@pytest.mark.parametrize("execution", EXECUTIONS)
def test_exposure_lags_decision(execution):
    """Prefix invariance alone cannot see a same-bar fill; the explicit lag can."""
    prices = make_prices(600, seed=20, drift=0.0, vol=0.3)
    s = EMACrossover(10, 40, allow_short=True)
    res = run_backtest(prices, s, _cfg(execution), rf_yield=None)
    lag = 2 if execution == "next_close" else 1
    expected = s.target_position(prices).fillna(0.0).shift(lag).loc[res.start :].iloc[1:]
    np.testing.assert_array_equal(res.exposure.iloc[1:].to_numpy(), expected.to_numpy())
    assert (res.exposure.diff().abs() > 0).sum() > 5


@pytest.mark.parametrize("execution", EXECUTIONS)
def test_truncation_invariance(execution):
    prices = make_prices(700, seed=22)
    s = DonchianBreakout(40, 20)
    full = run_backtest(prices, s, _cfg(execution, bps=5.0), rf_yield=0.03)
    cut = run_backtest(prices.iloc[:501], s, _cfg(execution, bps=5.0), rf_yield=0.03)
    pd.testing.assert_series_equal(full.returns.iloc[: len(cut.returns)], cut.returns)


# ----------------------------------------------------------------------- timing


@pytest.fixture
def step_case():
    prices = make_prices(30, seed=23)
    return prices, _step(prices.index, on=10)


def test_timing_same_close(step_case):
    prices, target = step_case
    res = _run(prices, target, "same_close")
    w = res.exposure.reindex(prices.index)
    assert w.iloc[10] == 0.0 and w.iloc[11] == 1.0
    c = prices["Close"]
    assert res.returns.loc[prices.index[10]] == 0.0
    assert res.returns.loc[prices.index[11]] == pytest.approx(c.iloc[11] / c.iloc[10] - 1)


def test_timing_next_close(step_case):
    prices, target = step_case
    res = _run(prices, target, "next_close")
    w = res.exposure.reindex(prices.index)
    assert w.iloc[11] == 0.0 and w.iloc[12] == 1.0
    c = prices["Close"]
    assert res.returns.loc[prices.index[11]] == 0.0
    assert res.returns.loc[prices.index[12]] == pytest.approx(c.iloc[12] / c.iloc[11] - 1)


def test_timing_next_open(step_case):
    prices, target = step_case
    res = _run(prices, target, "next_open")
    w = res.exposure.reindex(prices.index)
    assert w.iloc[10] == 0.0 and w.iloc[11] == 1.0
    o, c = prices["Open"], prices["Close"]
    # bar 11: in cash overnight, long from the open -> only the intraday move
    assert res.returns.loc[prices.index[11]] == pytest.approx(c.iloc[11] / o.iloc[11] - 1)
    assert res.returns.loc[prices.index[12]] == pytest.approx(c.iloc[12] / c.iloc[11] - 1)


def test_next_open_exit_keeps_overnight_leg():
    prices = make_prices(30, seed=24)
    res = _run(prices, _step(prices.index, on=5, off=15), "next_open")
    o, c = prices["Open"], prices["Close"]
    # decided flat at close 15: still long overnight into bar 16, flat from its open
    assert res.returns.loc[prices.index[16]] == pytest.approx(o.iloc[16] / c.iloc[15] - 1)
    assert res.exposure.loc[prices.index[16]] == 0.0


def test_next_open_equals_same_close_without_gaps():
    close = make_prices(400, seed=25)["Close"].to_numpy()
    prices = frame_from_close(close)  # Open == previous Close
    s = EMACrossover(10, 40)
    a = run_backtest(prices, s, _cfg("same_close", bps=5.0), rf_yield=0.02)
    b = run_backtest(prices, s, _cfg("next_open", bps=5.0), rf_yield=0.02)
    pd.testing.assert_series_equal(a.returns, b.returns, rtol=1e-12)


def test_same_close_captures_gap_that_next_open_misses():
    close = np.r_[np.full(10, 100.0), np.full(10, 120.0)]
    open_ = np.r_[np.full(10, 100.0), np.full(10, 120.0)]
    prices = frame_from_close(close, open_=open_)
    target = _step(prices.index, on=9)  # decided at close 9, before the gap at bar 10
    sc = _run(prices, target, "same_close")
    no = _run(prices, target, "next_open")
    nc = _run(prices, target, "next_close")
    assert sc.equity.iloc[-1] / sc.equity.iloc[0] == pytest.approx(1.2)
    assert no.equity.iloc[-1] / no.equity.iloc[0] == pytest.approx(1.0)
    assert nc.equity.iloc[-1] / nc.equity.iloc[0] == pytest.approx(1.0)


# ------------------------------------------------------------------------ costs


@pytest.mark.parametrize("execution", EXECUTIONS)
def test_total_costs_equal_turnover_times_bps(execution):
    prices = make_prices(800, seed=26, drift=0.0, vol=0.3)
    s = VolTarget(EMACrossover(10, 40, allow_short=True), window=40, max_leverage=2.0)
    res = run_backtest(prices, s, _cfg(execution, bps=7.0), rf_yield=0.02)
    dw = np.abs(np.diff(res.exposure.to_numpy(), prepend=0.0))
    np.testing.assert_allclose(res.turnover.to_numpy(), dw)
    assert res.costs.sum() == pytest.approx(dw.sum() * 7.0 / 1e4)
    free = run_backtest(prices, s, _cfg(execution, bps=0.0), rf_yield=0.02)
    np.testing.assert_allclose((free.returns - res.returns).to_numpy(), res.costs.to_numpy())


@pytest.mark.parametrize("execution", ["same_close", "next_close"])
def test_buy_and_hold_equals_price_ratio(prices, execution):
    res = run_backtest(prices, BuyAndHold(), _cfg(execution), rf_yield=None)
    close = prices["Close"].loc[res.start :]
    np.testing.assert_allclose(
        (res.equity / res.config.initial_capital).to_numpy(),
        (close / close.iloc[0]).to_numpy(),
        rtol=1e-12,
    )
    assert res.exposure.iloc[0] == 0.0 and (res.exposure.iloc[1:] == 1.0).all()


def test_buy_and_hold_next_open_buys_at_first_open(prices):
    res = run_backtest(prices, BuyAndHold(), _cfg("next_open"), rf_yield=None)
    first_open = prices["Open"].loc[res.returns.index[1]]
    growth = res.equity.iloc[-1] / res.config.initial_capital
    assert growth == pytest.approx(prices["Close"].iloc[-1] / first_open, rel=1e-12)


@pytest.mark.parametrize("execution", EXECUTIONS)
def test_round_trip_cost_on_flat_prices_is_exact(execution):
    prices = frame_from_close(np.full(40, 100.0))
    bps = 25.0
    res = _run(prices, _step(prices.index, on=10, off=20), execution, bps=bps)
    assert res.turnover.sum() == pytest.approx(2.0)
    assert res.equity.iloc[-1] / res.equity.iloc[0] == pytest.approx((1 - bps / 1e4) ** 2)


def test_round_trip_cost_on_random_prices_is_approximately_two_sides():
    prices = make_prices(60, seed=27)
    target = _step(prices.index, on=10, off=40)
    free = _run(prices, target, bps=0.0)
    paid = _run(prices, target, bps=20.0)
    ratio = paid.equity.iloc[-1] / free.equity.iloc[-1]
    assert ratio == pytest.approx((1 - 20.0 / 1e4) ** 2, rel=1e-4)


# ---------------------------------------------------------------- cash interest


@pytest.mark.parametrize("rf", [0.05, "series"])
def test_flat_strategy_earns_cash_rate(prices, rf):
    if rf == "series":
        rf = pd.Series(0.05, index=pd.date_range("1999-01-01", "2010-01-01", freq="D"))
    res = _run(prices, pd.Series(0.0, index=prices.index), rf=rf)
    days = (res.end - res.start).days
    assert res.equity.iloc[-1] / res.equity.iloc[0] == pytest.approx(1.05 ** (days / 365))
    assert res.costs.sum() == 0.0


def test_no_interest_when_earn_cash_rate_off(prices):
    cfg = BacktestConfig(earn_cash_rate=False)
    res = run_backtest(prices, BuyAndHold(0.0), cfg, rf_yield=0.05)
    assert (res.returns == 0.0).all() and (res.rf == 0.0).all()


def test_cash_accrues_at_yield_known_at_previous_close():
    """A yield change observed at close d is first earned over bar d+1 (no rf look-ahead)."""
    prices = make_prices(30, seed=30)
    idx = prices.index
    y = pd.Series(np.where(np.arange(30) < 15, 0.01, 0.10), index=idx)
    res = _run(prices, pd.Series(0.0, index=idx), rf=y)
    days = idx.to_series().diff().dt.days
    assert res.rf.loc[idx[15]] == pytest.approx(1.01 ** (days.iloc[15] / 365) - 1)
    assert res.rf.loc[idx[16]] == pytest.approx(1.10 ** (days.iloc[16] / 365) - 1)
    np.testing.assert_allclose(res.returns.iloc[1:], res.rf.iloc[1:])


def test_leverage_and_short_are_financed_at_cash_rate():
    prices = frame_from_close(np.full(300, 100.0))
    lev = _run(prices, pd.Series(2.0, index=prices.index), rf=0.05)
    short = _run(prices, pd.Series(-1.0, index=prices.index), rf=0.05)
    # w=2 on a flat asset pays rf on the borrowed 100%; w=-1 earns rf on 200% of equity
    assert (lev.rf.iloc[1:] > 0).all()
    np.testing.assert_allclose(lev.returns.iloc[1:], -lev.rf.iloc[1:])
    np.testing.assert_allclose(short.returns.iloc[1:], 2.0 * short.rf.iloc[1:])


# --------------------------------------------------------------------- warm-up


def _window_cfg(prices: pd.DataFrame) -> BacktestConfig:
    return _cfg("next_close", start=prices.index[2000], end=prices.index[2500])


def test_warmup_invariance_with_enough_history():
    prices = make_prices(3000, seed=0)
    s = EMACrossover(10, 40)
    ref = run_backtest(prices, s, _window_cfg(prices))
    for history in (1500, 400):
        alt = run_backtest(prices.iloc[2000 - history :], s, _window_cfg(prices))
        assert alt.start == ref.start and alt.end == ref.end
        np.testing.assert_allclose(alt.returns.to_numpy(), ref.returns.to_numpy(), rtol=1e-6)
        rel = abs(alt.equity.iloc[-1] / ref.equity.iloc[-1] - 1)
        assert rel < 1e-6


def test_warmup_bias_with_too_little_history():
    """Seeding the EMA just before the window (the original notebook's bug) changes results."""
    prices = make_prices(3000, seed=0)
    s = EMACrossover(10, 40)
    ref = run_backtest(prices, s, _window_cfg(prices))
    window = slice(prices.index[2000], prices.index[2500])
    slow_ref = s.indicators(prices)["slow"].loc[window]

    gaps = {}
    for history in (400, s.warmup, 10):
        short = prices.iloc[2000 - history :]
        gaps[history] = float((s.indicators(short)["slow"].loc[window] / slow_ref - 1).abs().max())
    assert gaps[400] < 1e-6 < gaps[s.warmup] < 0.01 < gaps[10]

    short = prices.iloc[2000 - 10 :]
    raw = s._raw_target(short)  # bypass the warm-up mask, as the notebook did
    biased = run_backtest(short, s, _window_cfg(prices), target=raw)
    assert biased.start == ref.start
    assert (biased.exposure != ref.exposure).any()
    assert abs(biased.equity.iloc[-1] / ref.equity.iloc[-1] - 1) > 1e-3


def test_engine_refuses_to_start_before_warmup():
    prices = make_prices(3000, seed=0)
    s = EMACrossover(10, 40)
    short = prices.iloc[2000 - 10 :]
    res = run_backtest(short, s, _window_cfg(prices))
    assert res.start == short.index[s.warmup + 1] > prices.index[2000]


def test_evaluation_start_matches_run_backtest(prices):
    strategies = [EMACrossover(10, 40), DonchianBreakout(60, 20), BuyAndHold()]
    common = evaluation_start(prices, strategies)
    assert common == prices.index[max(s.warmup for s in strategies) + 1]
    for s in strategies:
        assert run_backtest(prices, s, _cfg(start=common)).start == common
        assert run_backtest(prices, s).start == evaluation_start(prices, [s])
    late = prices.index[300] + pd.offsets.Hour(1)
    assert evaluation_start(prices, strategies, start=late) == prices.index[301]


# ---------------------------------------------------------------- window opening


@pytest.mark.parametrize("execution", EXECUTIONS)
def test_window_opens_invested_mid_trend_with_one_opening_cost(execution):
    prices = frame_from_close(np.linspace(100.0, 300.0, 500))
    s = EMACrossover(10, 40)
    start = prices.index[300]
    assert s.target_position(prices).loc[:start].iloc[-5:].eq(1.0).all()
    res = run_backtest(prices, s, _cfg(execution, bps=10.0, start=start), rf_yield=None)
    assert res.start == start
    assert res.exposure.iloc[0] == 0.0 and (res.exposure.iloc[1:] == 1.0).all()
    assert (res.costs > 0).sum() == 1 and res.costs.iloc[1] == pytest.approx(10.0 / 1e4)
    bh = run_backtest(prices, BuyAndHold(), _cfg(execution, bps=10.0, start=start))
    pd.testing.assert_series_equal(res.equity, bh.equity, check_names=False)


def test_window_end_is_inclusive(prices):
    end = prices.index[600]
    res = run_backtest(prices, BuyAndHold(), _cfg(end=end))
    assert res.end == end
    res2 = run_backtest(prices, BuyAndHold(), _cfg(end=end + pd.offsets.Hour(12)))
    assert res2.end == end


# ----------------------------------------------------------------------- trades


def _trade_inputs():
    idx = pd.date_range("2024-01-01", periods=9, freq="D")
    w = pd.Series([0.0, 1.0, 1.0, 0.0, -0.5, -0.5, 0.0, 0.0, 2.0], index=idx)
    # exit bars earn cash (r != -c) and the asset keeps moving, so off-by-one runs are visible
    r = pd.Series([0.0, 0.01, 0.02, -0.0009, 0.03, -0.01, -0.0004, 0.0001, 0.05], index=idx)
    g = pd.Series([0.0, 0.012, 0.021, 0.004, -0.06, 0.02, -0.003, 0.007, 0.025], index=idx)
    c = pd.Series([0.0, 0.001, 0.0, 0.001, 0.0005, 0.0, 0.0005, 0.0, 0.002], index=idx)
    return w, r, g, c


def test_extract_trades_known_answer():
    w, r, g, c = _trade_inputs()
    t = extract_trades(w, r, g, c)
    assert list(t.columns) == ["entry", "exit", "side", "bars", "return", "open", "asset_return"]
    assert list(t["side"]) == ["long", "short", "long"]
    assert list(t["bars"]) == [2, 2, 1]
    assert list(t["open"]) == [False, False, True]
    assert list(t["entry"]) == [w.index[1], w.index[4], w.index[8]]
    assert list(t["exit"]) == [w.index[2], w.index[5], w.index[8]]
    np.testing.assert_allclose(
        t["return"],
        [1.01 * 1.02 * (1 - 0.001) - 1, 1.03 * 0.99 * (1 - 0.0005) - 1, 0.05],
    )
    np.testing.assert_allclose(t["asset_return"], [1.012 * 1.021 - 1, 0.94 * 1.02 - 1, 0.025])


def test_extract_trades_without_optional_inputs():
    w, r, _, _ = _trade_inputs()
    t = extract_trades(w, r)
    assert "asset_return" not in t.columns
    assert t["return"].iloc[0] == pytest.approx(1.01 * 1.02 - 1)
    empty = extract_trades(pd.Series(0.0, index=w.index), r)
    assert empty.empty and "return" in empty.columns


def test_trades_compound_to_strategy_growth_long_flat():
    prices = make_prices(800, seed=28, drift=0.0, vol=0.3)
    res = run_backtest(prices, EMACrossover(10, 40), _cfg(bps=10.0), rf_yield=None)
    trades = res.trades
    assert len(trades) > 3
    # with rf=0 the strategy only moves while invested or when paying an exit cost
    growth = res.equity.iloc[-1] / res.equity.iloc[0]
    assert np.prod(1.0 + trades["return"]) == pytest.approx(growth, rel=1e-12)


def test_extract_trades_splits_direct_flip_into_two_trades():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    w = pd.Series([0.0, 1.0, 1.0, -1.0, -1.0, -1.0], index=idx)
    t = extract_trades(w, pd.Series(0.0, index=idx))
    assert list(t["side"]) == ["long", "short"]
    assert list(t["entry"]) == [idx[1], idx[3]] and list(t["exit"]) == [idx[2], idx[5]]
    assert list(t["open"]) == [False, True]


def test_trades_compound_to_strategy_growth_with_flips():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    w = pd.Series([0.0, 1.0, 1.0, -1.0, -1.0, -1.0], index=idx)
    c = pd.Series(np.abs(np.diff(w.to_numpy(), prepend=0.0)) * 10 / 1e4, index=idx)
    r = -c  # flat asset, no financing: only costs move equity
    t = extract_trades(w, r, None, c)
    assert np.prod(1.0 + t["return"]) == pytest.approx(np.prod(1.0 + r.to_numpy()))


# ---------------------------------------------------------------------- result


def test_result_fields_and_summary(prices):
    s = EMACrossover(10, 40)
    res = run_backtest(prices, s, rf_yield=0.02)
    bh = run_backtest(prices, BuyAndHold(), _cfg(start=res.start), rf_yield=0.02)
    for series in (res.returns, res.equity, res.exposure, res.turnover, res.costs, res.rf):
        assert series.index.equals(res.returns.index)
    assert res.returns.iloc[0] == 0.0
    assert res.equity.iloc[0] == res.config.initial_capital
    assert res.periods_per_year == 252
    assert res.target.index.equals(res.returns.index)
    out = res.summary(bh)
    assert {"cagr", "sharpe", "beta", "n_trades", "exposure"} <= set(out)
    assert out["exposure"] == pytest.approx(res.exposure.abs().mean())


def test_pre_window_asset_move_is_never_earned(prices):
    res = run_backtest(prices, BuyAndHold(), _cfg("same_close"), rf_yield=None)
    window = prices["Close"].loc[res.start :]
    np.testing.assert_allclose(res.asset_returns.iloc[1:], window.pct_change().iloc[1:])
    # the move *into* the opening close is not part of the window
    assert res.asset_returns.iloc[0] == 0.0 and res.exposure.iloc[0] == 0.0
    assert res.returns.iloc[0] == 0.0


def test_crypto_calendar_uses_365(price_factory):
    btc = price_factory(800, seed=29, freq="D")
    assert run_backtest(btc, BuyAndHold()).periods_per_year == 365
