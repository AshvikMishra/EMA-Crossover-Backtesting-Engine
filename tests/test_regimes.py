import numpy as np
import pandas as pd
import pytest

from trendlab import BacktestConfig, BuyAndHold, CostModel, SMACrossover, run_backtest
from trendlab.regimes import (
    capture_ratios,
    convexity_smile,
    crisis_table,
    label_regimes,
    performance_by_regime,
    rolling_excess_vs_benchmark,
    rolling_sharpe,
    whipsaw_stats,
)


def _frame(close: np.ndarray, start: str = "2010-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(close))
    c = pd.Series(close, index=idx)
    return pd.DataFrame({"Open": c, "High": c * 1.001, "Low": c * 0.999, "Close": c, "Volume": 1.0})


def _crash_prices() -> pd.DataFrame:
    up = np.linspace(0, 0.6, 300)
    down = 0.6 + np.linspace(0, -0.9, 120)[1:]
    rec = down[-1] + np.linspace(0, 0.8, 200)[1:]
    return _frame(100 * np.exp(np.concatenate([up, down, rec])))


def test_crisis_table_protection_on_crash():
    px = _crash_prices()
    cfg = BacktestConfig(costs=CostModel.zero(), earn_cash_rate=False)
    strat = run_backtest(px, SMACrossover(10, 30), cfg)
    bench = run_backtest(px, BuyAndHold(), cfg)
    peak, trough = px.index[299], px.index[299 + 119]
    crises = {
        "crash": (peak.date().isoformat(), trough.date().isoformat()),
        "outside": ("1990-01-01", "1990-06-01"),
    }
    tab = crisis_table(strat, bench, crises)
    assert list(tab.index) == ["crash"]
    row = tab.loc["crash"]
    assert row["bench_return"] == pytest.approx(np.exp(-0.9 * 119 / 119) - 1, rel=1e-6)
    assert row["protection"] > 0.3
    assert row["strat_max_dd"] > row["bench_max_dd"]
    assert row["trough"] == trough
    assert pd.notna(row["exit_date"]) and row["exit_date"] < trough
    assert row["reentry_date"] > trough
    assert row["reentry_lag_bars"] > 0
    assert row["missed_rebound"] > 0


def test_crisis_table_buy_and_hold_vs_itself():
    px = _crash_prices()
    bench = run_backtest(px, BuyAndHold(), BacktestConfig(costs=CostModel.zero()))
    tab = crisis_table(bench, bench, {"c": (str(px.index[100].date()), str(px.index[400].date()))})
    row = tab.iloc[0]
    assert row["protection"] == pytest.approx(0.0)
    assert pd.isna(row["exit_date"]) and pd.isna(row["reentry_date"])


def _random_prices(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _frame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))))


@pytest.mark.parametrize("use_vix", [False, True])
def test_labels_are_lagged(use_vix):
    px = _random_prices()
    rng = np.random.default_rng(1)
    vix = pd.Series(15 + rng.gamma(2.0, 3.0, len(px)), index=px.index) if use_vix else None
    base = label_regimes(px, vix, sma_window=50, min_history=60)
    k = 500
    px2 = px.copy()
    px2.iloc[k:, px2.columns.get_loc("Close")] *= 0.5  # crash from bar k onward
    vix2 = None
    if use_vix:
        vix2 = vix.copy()
        vix2.iloc[k:] = 80.0
    pert = label_regimes(px2, vix2, sma_window=50, min_history=60)
    pd.testing.assert_frame_equal(base.iloc[: k + 1], pert.iloc[: k + 1])
    assert not base.iloc[k + 1 :].equals(pert.iloc[k + 1 :])
    assert pert["drawdown"].iloc[k + 1] == "< -20%"
    assert pert["trend"].iloc[k + 1] == "bear"
    if use_vix:
        assert pert["vol"].iloc[k + 1] == "high"


def test_labels_values_and_warmup():
    px = _random_prices()
    lab = label_regimes(px, sma_window=50, min_history=60)
    assert list(lab.columns) == ["trend", "vol", "drawdown"]
    assert lab["trend"].iloc[:50].isna().all() and lab["trend"].iloc[50:].notna().all()
    assert set(lab["vol"].dropna().unique()) == {"low", "mid", "high"}
    shares = lab["vol"].value_counts(normalize=True)
    assert shares.min() > 0.15


def test_performance_by_regime():
    idx = pd.bdate_range("2020-01-01", periods=400)
    labels = pd.Series(np.where(np.arange(400) % 2 == 0, "bull", "bear"), index=idx, name="trend")
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.001, 400)
    a = pd.Series(np.where(labels == "bull", 0.001, -0.001) + noise, index=idx)
    b = pd.Series(0.0002 + noise, index=idx)
    out = performance_by_regime({"A": a, "B": b}, labels, periods=252)
    assert set(out.index) == {"bull", "bear"}
    assert out[("A", "ann_return")]["bull"] > 0 > out[("A", "ann_return")]["bear"]
    assert out[("A", "share")].sum() == pytest.approx(1.0)
    assert set(out.columns.get_level_values("metric")) == {
        "ann_return",
        "ann_vol",
        "sharpe",
        "share",
    }


def test_capture_ratios_identity_and_cash():
    idx = pd.bdate_range("2015-01-01", periods=1000)
    r = pd.Series(np.random.default_rng(3).normal(0.0004, 0.01, 1000), index=idx)
    cap = capture_ratios(r, r)
    assert cap["upside_capture"] == pytest.approx(1.0)
    assert cap["downside_capture"] == pytest.approx(1.0)
    half = capture_ratios(0.5 * r, r)
    assert 0.4 < half["downside_capture"] < 0.6


def test_convexity_smile_detects_quadratic():
    idx = pd.bdate_range("2000-01-01", periods=252 * 20)
    rng = np.random.default_rng(4)
    bench = pd.Series(rng.normal(0.0003, 0.012, len(idx)), index=idx)
    q = (1.0 + bench).resample("QE").prod() - 1.0
    # strategy pays x^2 at each quarter's last bar, so quarterly y ~= x^2
    strat = pd.Series(0.0, index=idx)
    last = bench.groupby(bench.index.to_period("Q")).tail(1).index
    strat.loc[last] = (q.to_numpy() ** 2)[: len(last)]
    res = convexity_smile(strat, bench)
    assert res["n"] == len(q)
    assert res["coef"]["c"] == pytest.approx(1.0, rel=1e-6)
    assert res["t_c"] > 10
    lin = convexity_smile(bench, bench)
    assert lin["coef"]["b"] == pytest.approx(1.0) and abs(lin["coef"]["c"]) < 1e-8


def test_rolling_metrics():
    idx = pd.bdate_range("2015-01-01", periods=600)
    r = pd.Series(np.random.default_rng(5).normal(0.0005, 0.01, 600), index=idx)
    sr = rolling_sharpe(r, window=252)
    assert sr.iloc[:251].isna().all()
    assert sr.iloc[-1] == pytest.approx(r.iloc[-252:].mean() / r.iloc[-252:].std() * np.sqrt(252))
    ex = rolling_excess_vs_benchmark(r + 0.0001, r, window=252)
    assert list(ex.columns) == ["strategy", "benchmark", "excess", "information_ratio"]
    assert (ex["excess"].dropna() > 0).all()
    total = (1 + r.iloc[-252:]).prod() - 1
    assert ex["benchmark"].iloc[-1] == pytest.approx(total)


def test_whipsaw_stats():
    trades = pd.DataFrame(
        {
            "bars": [3, 5, 40, 2, 10],
            "return": [-0.02, -0.03, 0.2, 0.01, -0.05],
            "open": [False, False, False, False, True],
        }
    )
    s = whipsaw_stats(trades, max_bars=10)
    assert s["n_trades"] == 4 and s["n_open"] == 1
    assert s["n_whipsaws"] == 2 and s["n_losing"] == 2
    assert s["losing_share"] == pytest.approx(0.5)
    assert s["whipsaw_loss"] == pytest.approx(0.98 * 0.97 - 1)
    empty = whipsaw_stats(pd.DataFrame(columns=["bars", "return", "open"]))
    assert empty["n_trades"] == 0 and empty["n_whipsaws"] == 0


def _stepped_target_backtests():
    """Flat 20 bars, -3%/bar for 10 bars (trough at bar 29), then +1 per bar; out on bars 22-34."""
    idx = pd.bdate_range("2020-01-01", periods=60)
    c = np.r_[np.full(20, 100.0), 100 - np.arange(1, 11) * 3.0, 70 + np.arange(1, 31) * 1.0]
    px = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1.0}, index=idx)
    target = pd.Series(1.0, index=idx)
    target.iloc[22:35] = 0.0
    cfg = BacktestConfig(costs=CostModel.zero(), earn_cash_rate=False)
    strat = run_backtest(px, BuyAndHold(), cfg, target=target)
    return idx, strat, run_backtest(px, BuyAndHold(), cfg)


def test_crisis_table_exact_timing():
    idx, strat, bench = _stepped_target_backtests()
    row = crisis_table(strat, bench, {"c": (str(idx[19].date()), str(idx[29].date()))}).iloc[0]
    # decided flat at close 22 -> filled close 23 (next_close) -> first flat bar 24
    assert row["exit_date"] == idx[24] and row["exit_vs_start_bars"] == 4
    assert row["trough"] == idx[29] and row["bars"] == 10
    assert row["reentry_date"] == idx[37] and row["reentry_lag_bars"] == 8
    assert row["missed_rebound"] == pytest.approx(77 / 70 - 1)
    assert row["bench_return"] == pytest.approx(-0.3)
    assert row["strat_return"] == pytest.approx(91 / 100 * 88 / 91 - 1)
    assert row["avg_exposure"] == pytest.approx(0.4)


def test_crisis_table_mixed_calendars_keep_weekend_returns():
    days = pd.date_range("2020-01-01", periods=400, freq="D")
    moves = np.where(days.day_of_week >= 5, -0.01, 0.0)  # crypto-like leg only moves on weekends
    crypto = _frame(100 * np.cumprod(1 + moves))
    crypto.index = days
    stock = _frame(np.full(len(pd.bdate_range(days[0], days[-1])), 100.0), str(days[0].date()))
    cfg = BacktestConfig(costs=CostModel.zero(), earn_cash_rate=False)
    c, s = run_backtest(crypto, BuyAndHold(), cfg), run_backtest(stock, BuyAndHold(), cfg)
    start, end = "2020-03-02", "2020-09-01"
    row = crisis_table(c, s, {"w": (start, end)}).iloc[0]
    expected = c.equity.loc[end] / c.equity.loc[start] - 1
    assert expected < -0.3
    assert row["strat_return"] == pytest.approx(expected)
    assert row["strat_max_dd"] == pytest.approx(expected)


def test_period_returns_mixed_calendars_and_rf():
    days = pd.date_range("2015-01-01", periods=900, freq="D")
    rng = np.random.default_rng(6)
    daily = pd.Series(rng.normal(0, 0.02, len(days)), index=days)
    px = (1 + daily).cumprod()
    wk = px[px.index.day_of_week < 5].pct_change().dropna()  # same asset sampled on weekdays
    res = convexity_smile(daily, wk)
    own = (1 + daily).resample("QE").prod() - 1
    inner = res["data"].index[1:-1]
    pd.testing.assert_series_equal(res["data"]["y"].loc[inner], own.loc[inner], check_names=False)
    # excess = compounded return minus compounded rf, not compounded per-bar excess
    rf = pd.Series(0.0004, index=days)
    ex = convexity_smile(daily, daily, rf=rf)["data"]
    raw = convexity_smile(daily, daily)["data"]
    rf_q = (1 + rf).resample("QE").prod() - 1
    pd.testing.assert_series_equal(ex["y"], raw["y"] - rf_q.loc[raw.index], check_names=False)
    assert convexity_smile(daily, daily, rf=0.0004)["data"]["y"].equals(ex["y"])


def test_vix_terciles_use_vix_history_before_prices():
    rng = np.random.default_rng(7)
    vix = pd.Series(
        15 + rng.gamma(2.0, 3.0, 1300), index=pd.bdate_range("2015-01-01", periods=1300)
    )
    px = _frame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))), "2019-06-03")
    lab = label_regimes(px, vix, sma_window=20, min_history=252)
    assert lab["vol"].iloc[1:].notna().all()
    v = vix.loc[: px.index[10]]
    q1, q2 = v.quantile([1 / 3, 2 / 3])
    expected = "low" if v.iloc[-1] <= q1 else ("mid" if v.iloc[-1] <= q2 else "high")
    assert lab["vol"].iloc[11] == expected


def test_performance_by_regime_values():
    idx = pd.bdate_range("2020-01-01", periods=6)
    labels = pd.Series(pd.Categorical(["bear", "bull"] * 3, categories=["bull", "bear"]), idx)
    r = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0, -0.02], index=idx)
    out = performance_by_regime({"s": r}, labels, periods=252)
    assert list(out.index) == ["bull", "bear"]
    bull = r.iloc[1::2]
    assert out.loc["bull", ("s", "ann_return")] == pytest.approx((1 + bull).prod() ** 84 - 1)
    assert out.loc["bull", ("s", "share")] == pytest.approx(0.5)
