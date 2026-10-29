from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trendlab.engine import BacktestConfig, CostModel, run_backtest
from trendlab.event_engine import legacy_notebook_backtest, run_event_backtest
from trendlab.strategies import EMACrossover

NVDA = Path(__file__).resolve().parents[1] / "data" / "raw" / "NVDA.parquet"
STRAT = EMACrossover(10, 40)


def gbm_prices(n: int = 900, seed: int = 0, mu: float = 0.0003, sigma: float = 0.02):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(mu + sigma * rng.standard_normal(n)))
    gap = np.exp(0.005 * rng.standard_normal(n))
    open_ = np.r_[close[0], close[:-1]] * gap
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * 1.01,
            "Low": np.minimum(open_, close) * 0.99,
            "Close": close,
            "Volume": 1e6,
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def prices():
    return gbm_prices()


def _vec(prices, execution, bps=0.0, rf=None, **kw):
    cfg = BacktestConfig(execution=execution, costs=CostModel.flat(bps), **kw)
    return run_backtest(prices, STRAT, cfg, rf_yield=rf)


@pytest.mark.parametrize("execution", ["same_close", "next_close", "next_open"])
def test_parity_without_costs(prices, execution):
    v = _vec(prices, execution)
    e = run_event_backtest(prices, STRAT, execution=execution)
    assert v.returns.index.equals(e.returns.index)
    np.testing.assert_allclose(e.returns, v.returns, atol=1e-10)
    np.testing.assert_allclose(e.equity, v.equity, rtol=1e-10)
    np.testing.assert_allclose(e.exposure, v.exposure, atol=0)
    # next_open trades notional at the open, the vectorised engine books |dw| of the prior close
    tol = 0.05 if execution == "next_open" else 1e-10
    np.testing.assert_allclose(e.turnover, v.turnover, atol=tol)
    assert len(e.trades) == len(v.trades) > 2


@pytest.mark.parametrize("execution", ["same_close", "next_close"])
def test_parity_with_costs_and_cash_yield(prices, execution):
    kw = {"start": "2015-06-01", "end": "2018-01-31"}
    v = _vec(prices, execution, bps=7.0, rf=0.04, **kw)
    e = run_event_backtest(prices, STRAT, execution=execution, cost_bps=7.0, rf_yield=0.04, **kw)
    np.testing.assert_allclose(e.returns, v.returns, atol=1e-10)
    np.testing.assert_allclose(e.costs, v.costs, atol=1e-12)
    np.testing.assert_allclose(e.rf, v.rf, atol=0)
    assert e.start == v.start and e.end == v.end


def test_next_open_costs_differ_only_second_order(prices):
    v = _vec(prices, "next_open", bps=10.0, rf=0.05)
    e = run_event_backtest(prices, STRAT, execution="next_open", cost_bps=10.0, rf_yield=0.05)
    np.testing.assert_allclose(e.returns, v.returns, atol=5e-5)


def test_precomputed_target_and_fractional_exposure(prices):
    tgt = STRAT.target_position(prices) * 0.5
    cfg = BacktestConfig(execution="next_close", costs=CostModel.zero())
    v = run_backtest(prices, STRAT, cfg, target=tgt)
    e = run_event_backtest(prices, tgt, rebalance="every_bar")
    np.testing.assert_allclose(e.returns, v.returns, atol=1e-10)
    drift = run_event_backtest(prices, tgt)
    assert not np.allclose(drift.returns, v.returns, atol=1e-10)
    assert drift.weight.max() <= 1.0


def test_integer_shares_and_commission_reduce_equity(prices):
    base = run_event_backtest(prices, STRAT, initial_capital=10_000)
    ints = run_event_backtest(prices, STRAT, initial_capital=10_000, integer_shares=True)
    comm = run_event_backtest(prices, STRAT, initial_capital=10_000, commission_per_trade=5.0)
    both = run_event_backtest(
        prices, STRAT, initial_capital=10_000, integer_shares=True, commission_per_trade=5.0
    )
    assert ints.shares.eq(ints.shares.round()).all()
    assert (ints.cash >= -1e-9).all()
    assert comm.fills["fee"].eq(5.0).all()
    final = base.equity.iloc[-1]
    assert ints.equity.iloc[-1] != final and comm.equity.iloc[-1] < final
    assert both.equity.iloc[-1] < ints.equity.iloc[-1]
    # with a rising market, idle cash from rounding is a drag relative to fractional shares
    up = gbm_prices(mu=0.002, sigma=0.01, seed=3)
    f = run_event_backtest(up, STRAT, initial_capital=5_000)
    i = run_event_backtest(up, STRAT, initial_capital=5_000, integer_shares=True)
    assert i.equity.iloc[-1] < f.equity.iloc[-1]


@pytest.mark.parametrize("execution", ["same_close", "next_close", "next_open"])
def test_no_lookahead(prices, execution):
    t = 500
    shocked = prices.copy()
    shocked.iloc[t:, :4] = (
        shocked.iloc[t:, :4].to_numpy() * np.linspace(0.5, 2.0, len(prices) - t)[:, None]
    )
    a = run_event_backtest(prices, STRAT, execution=execution, cost_bps=5.0, rf_yield=0.03)
    b = run_event_backtest(shocked, STRAT, execution=execution, cost_bps=5.0, rf_yield=0.03)
    cut = prices.index[t - 1]
    pd.testing.assert_series_equal(a.equity.loc[:cut], b.equity.loc[:cut])
    assert not a.equity.equals(b.equity)


def test_fill_log_and_summary(prices):
    e = run_event_backtest(prices, STRAT, cost_bps=5.0)
    assert list(e.fills.columns[:4]) == ["date", "when", "price", "shares"]
    assert (e.fills["when"] == "close").all()
    np.testing.assert_allclose(e.fills["fee"], e.fills["notional"] * 5e-4)
    s = e.summary(benchmark=_vec(prices, "next_close"))
    assert s["n_trades"] == len(e.trades) and "beta" in s


def test_bad_arguments(prices):
    with pytest.raises(ValueError):
        run_event_backtest(prices, STRAT, execution="tomorrow")
    with pytest.raises(ValueError):
        run_event_backtest(prices, STRAT, rebalance="weekly")
    with pytest.raises(ValueError):
        run_event_backtest(prices, -STRAT.target_position(prices), integer_shares=True)


def test_legacy_notebook_synthetic():
    # flat, then a strong rally, then a crash: one golden cross and one death cross
    idx = pd.bdate_range("2020-01-01", periods=600)
    close = np.r_[np.full(50, 100.0), np.linspace(100, 300, 250), np.linspace(300, 60, 300)]
    px = pd.DataFrame({"Close": close}, index=idx)
    out = legacy_notebook_backtest(px)
    sides = out["trades"]["side"].tolist()
    assert "buy" in sides and sides[-1] == "sell"
    buy = out["trades"].query("side == 'buy'").iloc[0]
    assert buy["shares"] == 100_000 // buy["price"]
    assert out["bnh_final"] == pytest.approx((100_000 - 5) / 100 * 60)
    assert out["algo_equity"].index.equals(idx)
    assert out["algo_max_drawdown"] <= 0 and out["bnh_max_drawdown"] < out["algo_max_drawdown"]


@pytest.mark.skipif(not NVDA.exists(), reason="cached NVDA data not available")
def test_legacy_notebook_known_answer():
    from trendlab.data import load_prices

    out = legacy_notebook_backtest(load_prices("NVDA", offline=True).loc["2022-01-01":"2025-01-01"])
    assert out["algo_total_return"] == pytest.approx(5.0514, abs=0.002)
    assert out["bnh_total_return"] == pytest.approx(3.4655, abs=0.002)
    assert out["algo_max_drawdown"] == pytest.approx(-0.2705, abs=0.002)
    assert out["bnh_max_drawdown"] == pytest.approx(-0.627, abs=0.002)
    # The cache's dividend adjustment scales NVDA by ~0.2% vs the notebook's download, which
    # changes ``capital // close``. Rescaled to the notebook's first close (30.062769) the
    # replica reproduces its printed dollars ($605,091.68 / $446,554.24) to within a dollar.
    px = load_prices("NVDA", offline=True).loc["2022-01-01":"2025-01-01"]
    scaled = px.assign(Close=px["Close"] * 30.062769 / px["Close"].iloc[0])
    nb = legacy_notebook_backtest(scaled)
    assert nb["algo_final"] == pytest.approx(605_091.68, abs=1.0)
    assert nb["bnh_final"] == pytest.approx(446_554.24, abs=1.0)


def test_integer_shares_unaffordable_order_retries_and_reports_no_phantom_exposure():
    idx = pd.bdate_range("2020-01-01", periods=40)
    close = np.r_[np.full(10, 200.0), np.linspace(200.0, 100.0, 30)]
    px = pd.DataFrame({"Open": close, "Close": close}, index=idx)
    tgt = pd.Series(1.0, index=idx)
    e = run_event_backtest(px, tgt, initial_capital=150.0, integer_shares=True)
    first = e.fills["date"].iloc[0]
    assert len(e.fills) == 1 and px.loc[first, "Close"] <= 150.0
    assert (e.exposure.loc[:first] == 0).all() and e.exposure.iloc[-1] == 1.0
    assert len(e.trades) == 1 and e.trades["entry"].iloc[0] > first
    broke = run_event_backtest(px, tgt, initial_capital=50.0, integer_shares=True)
    assert broke.fills.empty and (broke.exposure == 0).all() and broke.trades.empty


@pytest.mark.parametrize(
    ("execution", "lag"), [("same_close", 1), ("next_open", 1), ("next_close", 2)]
)
def test_fill_lag_is_exact(prices, execution, lag):
    # flipping decisions from close T on must first move equity at close T + lag
    base = STRAT.target_position(prices)
    flipped = base.copy()
    t = 600
    flipped.iloc[t:] = 1.0 - base.iloc[t:]
    a = run_event_backtest(prices, base, execution=execution, cost_bps=5.0, rf_yield=0.03)
    b = run_event_backtest(prices, flipped, execution=execution, cost_bps=5.0, rf_yield=0.03)
    moved = a.equity.index[np.argmax(a.equity.to_numpy() != b.equity.to_numpy())]
    assert moved == prices.index[t + lag]


@pytest.mark.parametrize("execution", ["same_close", "next_close"])
def test_every_bar_parity_for_shorts_and_leverage(prices, execution):
    ls = EMACrossover(10, 40, allow_short=True)
    cfg = BacktestConfig(execution=execution, costs=CostModel.zero())
    v = run_backtest(prices, ls, cfg, rf_yield=0.03)
    e = run_event_backtest(prices, ls, execution=execution, rf_yield=0.03, rebalance="every_bar")
    np.testing.assert_allclose(e.returns, v.returns, atol=1e-10)
    lev = STRAT.target_position(prices) * 2.0
    v = run_backtest(prices, STRAT, cfg, rf_yield=0.03, target=lev)
    e = run_event_backtest(prices, lev, execution=execution, rf_yield=0.03, rebalance="every_bar")
    np.testing.assert_allclose(e.returns, v.returns, atol=1e-10)
    assert (e.cash < 0).any()  # leverage is financed


def test_trades_match_vectorised(prices):
    v = _vec(prices, "next_close", bps=5.0, rf=0.02)
    e = run_event_backtest(prices, STRAT, cost_bps=5.0, rf_yield=0.02)
    pd.testing.assert_frame_equal(e.trades, v.trades, atol=1e-10)


def test_negative_target_before_window_is_ignored_for_integer_shares(prices):
    tgt = STRAT.target_position(prices)
    tgt.iloc[300] = -1.0
    e = run_event_backtest(prices, tgt, integer_shares=True, start=prices.index[400])
    assert e.start == prices.index[400]
