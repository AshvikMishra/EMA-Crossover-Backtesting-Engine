"""Known-answer tests for the performance metrics.

Definitions follow Bacon, "Practical Portfolio Performance Measurement and
Attribution" (2008): Sharpe = mean excess / std * sqrt(P); Sortino uses the downside
deviation averaged over *all* periods (Sortino & Price 1994); Calmar = CAGR / |MDD|.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trendlab import metrics as m


def _s(values) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values)), dtype=float)


# A handcrafted equity path 1.2, 0.9, 1.0, 1.3, 1.17 (starting from 1.0).
EQ_PATH = [1.2, 0.9, 1.0, 1.3, 1.17]
PATH_RETURNS = _s(np.array(EQ_PATH) / np.array([1.0] + EQ_PATH[:-1]) - 1.0)


def test_equity_and_total_return():
    np.testing.assert_allclose(m.equity_curve(PATH_RETURNS).to_numpy(), EQ_PATH)
    np.testing.assert_allclose(m.equity_curve(PATH_RETURNS, 100.0).iloc[-1], 117.0)
    assert m.total_return(PATH_RETURNS) == pytest.approx(0.17)


def test_cagr_constant_return_engine_convention():
    # engine returns start with a 0 on the opening bar: n bars span n-1 periods
    r = 0.0004
    returns = _s([0.0] + [r] * 504)
    assert m.cagr(returns, 252) == pytest.approx((1 + r) ** 252 - 1, rel=1e-12)
    assert m.cagr(_s([0.0] + [r] * 730), 365) == pytest.approx((1 + r) ** 365 - 1, rel=1e-12)


def test_cagr_counts_len_minus_one_periods():
    r, n = 0.001, 253
    assert m.cagr(_s([r] * n), 252) == pytest.approx((1 + r) ** (n * 252 / (n - 1)) - 1)


def test_cagr_edge_cases():
    assert math.isnan(m.cagr(_s([0.01])))
    assert m.cagr(_s([0.0, -1.0, 0.1])) == -1.0


def test_ann_vol_and_sharpe_known_mean_std():
    mu, s = 0.001, 0.01
    returns = _s(np.tile([mu + s, mu - s], 126))
    sd = s * math.sqrt(252 / 251)  # ddof=1 on an even-length alternating series
    assert m.ann_vol(returns) == pytest.approx(sd * math.sqrt(252))
    assert m.sharpe(returns) == pytest.approx(mu / sd * math.sqrt(252))


def test_sharpe_subtracts_rf_scalar_and_series():
    mu, s, rf = 0.001, 0.01, 0.0002
    returns = _s(np.tile([mu + s, mu - s], 126))
    sd = s * math.sqrt(252 / 251)
    expected = (mu - rf) / sd * math.sqrt(252)
    assert m.sharpe(returns, rf=rf) == pytest.approx(expected)
    assert m.sharpe(returns, rf=pd.Series(rf, index=returns.index)) == pytest.approx(expected)


def test_sharpe_of_zero_excess_is_nan():
    assert math.isnan(m.sharpe(_s([0.0] * 10)))
    assert math.isnan(m.sharpe(_s([0.001] * 10), rf=0.001))


def test_sortino_uses_all_period_downside_deviation():
    returns = _s([0.02, -0.01, 0.03, -0.02])
    dd = math.sqrt((0.01**2 + 0.02**2) / 4)  # averaged over all 4 bars, not only the 2 losers
    assert m.sortino(returns, periods=252) == pytest.approx(returns.mean() / dd * math.sqrt(252))
    assert math.isnan(m.sortino(_s([0.01, 0.02])))


def test_sortino_measures_downside_relative_to_rf():
    returns, rf = _s([0.02, -0.01, 0.03, 0.001]), 0.002
    ex = returns - rf
    dd = math.sqrt(((ex.clip(upper=0.0)) ** 2).mean())  # 0.001 < rf counts as downside
    assert m.sortino(returns, rf=rf) == pytest.approx(ex.mean() / dd * math.sqrt(252))


def test_max_drawdown_and_duration_handcrafted():
    dd = m.drawdown_series(PATH_RETURNS)
    np.testing.assert_allclose(dd.to_numpy(), [0.0, -0.25, -1 / 6, 0.0, -0.1])
    assert m.max_drawdown(PATH_RETURNS) == pytest.approx(-0.25)
    assert m.max_drawdown_duration(PATH_RETURNS) == 2
    assert m.max_drawdown(_s([0.01, 0.02])) == 0.0
    assert m.max_drawdown_duration(_s([0.01, 0.02])) == 0


def test_max_drawdown_counts_first_bar_loss():
    assert m.max_drawdown(_s([-0.1, 0.05])) == pytest.approx(-0.1)


def test_calmar():
    returns = _s([0.0] + [0.001] * 252 + [-0.2] + [0.001] * 251)
    expected = m.cagr(returns) / abs(m.max_drawdown(returns))
    assert m.calmar(returns) == pytest.approx(expected)
    assert m.max_drawdown(returns) == pytest.approx(-0.2)
    assert math.isnan(m.calmar(_s([0.0, 0.01, 0.01])))


def test_ulcer_index():
    dd = np.array([0.0, -0.25, -1 / 6, 0.0, -0.1])
    assert m.ulcer_index(PATH_RETURNS) == pytest.approx(math.sqrt((dd**2).mean()))


def test_var_cvar_historical():
    returns = _s(np.arange(-0.10, 0.10, 0.001)[:200])
    var, cvar = m.var_cvar(returns, 0.05)
    assert var == pytest.approx(returns.quantile(0.05))
    assert cvar == pytest.approx(returns[returns <= var].mean())
    assert cvar <= var < 0


def test_alpha_beta_recovers_beta_two_alpha_zero():
    b = _s(np.random.default_rng(3).normal(0.0005, 0.01, 500))
    out = m.alpha_beta(2.0 * b, b)
    assert out["beta"] == pytest.approx(2.0)
    assert out["alpha"] == pytest.approx(0.0, abs=1e-12)
    assert out["correlation"] == pytest.approx(1.0)
    assert out["tracking_error"] == pytest.approx(b.std() * math.sqrt(252))


def test_alpha_beta_recovers_alpha_with_noise():
    rng = np.random.default_rng(4)
    b = _s(rng.normal(0.0004, 0.01, 5000))
    r = 0.0002 + 0.5 * b + _s(rng.normal(0.0, 0.002, 5000)).to_numpy()
    out = m.alpha_beta(r, b)
    assert out["beta"] == pytest.approx(0.5, abs=0.01)
    assert out["alpha"] == pytest.approx(0.0002 * 252, abs=0.025)  # ~3.5 standard errors
    assert out["alpha_t"] > 3


def test_alpha_beta_with_rf_shifts_alpha():
    b = _s(np.random.default_rng(5).normal(0.0005, 0.01, 500))
    rf = 0.0001
    # r - rf = 2 (b - rf)  =>  zero alpha in excess-return space
    out = m.alpha_beta(rf + 2.0 * (b - rf), b, rf=rf)
    assert out["beta"] == pytest.approx(2.0)
    assert out["alpha"] == pytest.approx(0.0, abs=1e-12)


def test_trade_stats_known_answer():
    trades = pd.DataFrame({"return": [0.1, -0.05, 0.2, 0.0], "bars": [10, 4, 6, 2]})
    out = m.trade_stats(trades)
    assert out["n_trades"] == 4
    assert out["hit_rate"] == pytest.approx(0.5)
    assert out["profit_factor"] == pytest.approx(0.3 / 0.05)
    assert out["avg_win"] == pytest.approx(0.15)
    assert out["avg_loss"] == pytest.approx(-0.025)
    assert out["avg_bars"] == pytest.approx(5.5)
    assert m.trade_stats(pd.DataFrame(columns=["return", "bars"]))["n_trades"] == 0
    assert m.trade_stats(None)["n_trades"] == 0


def test_summary_keys_and_consistency():
    rng = np.random.default_rng(6)
    r = _s(np.concatenate([[0.0], rng.normal(0.0004, 0.01, 504)]))
    b = _s(np.concatenate([[0.0], rng.normal(0.0003, 0.012, 504)]))
    trades = pd.DataFrame({"return": [0.1, -0.02], "bars": [100, 50]})
    out = m.summary(
        r,
        rf=0.0,
        exposure=pd.Series(0.5, index=r.index),
        turnover=pd.Series(0.01, index=r.index),
        benchmark=b,
        trades=trades,
    )
    assert out["years"] == pytest.approx(2.0)
    assert out["cagr"] == pytest.approx(m.cagr(r))
    assert out["sharpe"] == pytest.approx(m.sharpe(r))
    assert out["max_drawdown"] == pytest.approx(m.max_drawdown(r))
    assert out["exposure"] == pytest.approx(0.5)
    assert out["turnover_per_year"] == pytest.approx(0.01 * 505 / 2.0)
    assert out["trades_per_year"] == pytest.approx(1.0)
    for key in ("beta", "alpha", "information_ratio", "hit_rate", "kurtosis", "cvar_95"):
        assert key in out


def test_format_table():
    rows = {
        "A": {"cagr": 0.1234, "sharpe": 1.234, "n_trades": 7.0, "start": "2020-01-01"},
        "B": {"cagr": float("nan"), "sharpe": 0.5, "n_trades": 3, "start": "2021-01-01"},
    }
    t = m.format_table(rows, keys=["cagr", "sharpe", "n_trades", "missing"])
    assert list(t.columns) == ["cagr", "sharpe", "n_trades"]
    assert t.loc["A", "cagr"] == "12.3%"
    assert t.loc["A", "sharpe"] == "1.23"
    assert t.loc["A", "n_trades"] == "7"
    assert t.loc["B", "cagr"] == "–"
