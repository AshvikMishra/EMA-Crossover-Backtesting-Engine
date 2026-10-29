import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from trendlab import costs, stats
from trendlab.engine import BacktestConfig, CostModel, run_backtest
from trendlab.strategies import EMACrossover


def _iid(n, sr, seed=0, sd=0.01):
    return np.random.default_rng(seed).normal(sr * sd, sd, n)


# ---------------------------------------------------------------- PSR / DSR / Lo


def test_psr_matches_normal_closed_form():
    x = _iid(2000, 0.05, seed=1)
    sr = x.mean() / x.std(ddof=1)
    expected = norm.cdf(sr * math.sqrt(len(x) - 1) / math.sqrt(1 + sr**2 / 2))
    assert stats.probabilistic_sharpe_ratio(x) == pytest.approx(expected, abs=0.01)


def test_psr_calibrated_under_null():
    vals = [stats.probabilistic_sharpe_ratio(_iid(500, 0.03, seed=s), 0.03) for s in range(300)]
    assert 0.44 < np.mean(vals) < 0.56
    assert 0.03 < np.mean(np.array(vals) > 0.95) < 0.08


def test_psr_accepts_series_with_rf():
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.Series(_iid(300, 0.1, seed=2), index=idx)
    rf = pd.Series(0.0001, index=idx)
    assert stats.probabilistic_sharpe_ratio(r, rf=rf) == pytest.approx(
        stats.probabilistic_sharpe_ratio(r.to_numpy() - 0.0001)
    )


def test_min_track_record_length():
    x = _iid(3000, 0.06, seed=3)
    mtrl = stats.min_track_record_length(x)
    sr = x.mean() / x.std(ddof=1)
    assert mtrl == pytest.approx(1 + (1 + sr**2 / 2) * (norm.ppf(0.95) / sr) ** 2, rel=0.1)
    assert stats.min_track_record_length(_iid(3000, 0.12, seed=3)) < mtrl
    assert stats.min_track_record_length(x, sr_benchmark=1.0) == math.inf


def test_expected_max_sharpe_monotone():
    vals = [stats.expected_max_sharpe(0.01, n) for n in (1, 2, 10, 100, 1000)]
    assert vals[0] == 0.0
    assert all(b > a for a, b in zip(vals, vals[1:], strict=False))
    assert stats.expected_max_sharpe(0.04, 50) == pytest.approx(
        2 * stats.expected_max_sharpe(0.01, 50)
    )


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(0)
    sims = rng.normal(0, 0.1, (20000, 100)).max(axis=1).mean()
    assert stats.expected_max_sharpe(0.01, 100) == pytest.approx(sims, rel=0.03)


def test_dsr_decreases_with_trials():
    x = _iid(2000, 0.06, seed=4)
    trials = np.random.default_rng(5).normal(0, 0.03, 200)
    dsrs = [stats.deflated_sharpe_ratio(x, trials[:n])["dsr"] for n in (2, 10, 50, 200)]
    assert all(b < a for a, b in zip(dsrs, dsrs[1:], strict=False))
    out = stats.deflated_sharpe_ratio(x, trials, n_trials=1000)
    assert out["n_trials"] == 1000 and out["dsr"] < dsrs[-1]
    assert out["sr_annual"] == pytest.approx(out["sr"] * math.sqrt(252))
    assert {"dsr", "sr", "sr0", "n_trials", "sr_variance"} <= out.keys()


def test_annualize_roundtrip_and_lo():
    assert stats.deannualize_sharpe(stats.annualize_sharpe(0.05, 365), 365) == pytest.approx(0.05)
    x = _iid(5000, 0.05, seed=6)
    plain = x.mean() / x.std(ddof=1) * math.sqrt(252)
    assert stats.lo_adjusted_sharpe(x, lags=5) == pytest.approx(plain, rel=0.15)
    rng = np.random.default_rng(7)
    e = rng.normal(0, 0.01, 5000)
    ar = np.empty_like(e)
    ar[0] = e[0]
    for i in range(1, len(e)):
        ar[i] = 0.3 * ar[i - 1] + e[i]
    ar += 0.0005
    naive = ar.mean() / ar.std(ddof=1) * math.sqrt(252)
    assert stats.lo_adjusted_sharpe(ar, lags=10) < 0.85 * naive


# ---------------------------------------------------------------- PBO


def test_pbo_noise_near_half():
    pbos = [
        stats.pbo_cscv(np.random.default_rng(s).normal(0, 0.01, (800, 10)), n_blocks=10)["pbo"]
        for s in range(8)
    ]
    assert 0.3 < np.mean(pbos) < 0.8


def test_pbo_true_signal_near_zero():
    x = np.random.default_rng(3).normal(0, 0.01, (1600, 20))
    x[:, 5] += 0.003
    df = pd.DataFrame(x, index=pd.bdate_range("2000-01-01", periods=1600))
    out = stats.pbo_cscv(df)
    assert out["pbo"] < 0.05
    assert out["n_combinations"] == 12870
    assert len(out["logits"]) == 12870 and (out["selected"] == 5).mean() > 0.95
    assert np.all((out["is_best_oos_rank"] > 0) & (out["is_best_oos_rank"] < 1))


def test_pbo_vectorised_matches_callable():
    x = np.random.default_rng(9).normal(0.0002, 0.01, (300, 6))

    def sr(m):
        return m.mean(axis=0) / m.std(axis=0, ddof=1)

    a = stats.pbo_cscv(x, n_blocks=6)
    b = stats.pbo_cscv(x, n_blocks=6, metric=sr)
    np.testing.assert_allclose(a["is_metric"], b["is_metric"], rtol=1e-9)
    np.testing.assert_allclose(a["logits"], b["logits"])
    assert a["performance_degradation"] == pytest.approx(b["performance_degradation"])
    with pytest.raises(ValueError):
        stats.pbo_cscv(x, n_blocks=5)


# ---------------------------------------------------------------- bootstrap


def test_stationary_bootstrap_indices():
    idx = stats.stationary_bootstrap_indices(1000, mean_block=10, n_boot=50, seed=1)
    assert idx.shape == (50, 1000) and idx.min() >= 0 and idx.max() < 1000
    np.testing.assert_array_equal(idx, stats.stationary_bootstrap_indices(1000, 10, 50, seed=1))
    jumps = (np.diff(idx, axis=1) % 1000 != 1).mean()
    assert 1 / jumps == pytest.approx(10, rel=0.15)


def test_bootstrap_sharpe_difference_covers_truth():
    rng = np.random.default_rng(11)
    n, sd = 3000, 0.01
    common = rng.normal(0, sd, n)
    a = 0.6 * common + 0.8 * rng.normal(0, sd, n) + 0.08 * sd
    b = 0.6 * common + 0.8 * rng.normal(0, sd, n) + 0.02 * sd
    true = (0.08 - 0.02) * math.sqrt(252)
    out = stats.bootstrap_sharpe_difference(a, b, n_boot=1000, mean_block=10, seed=0)
    assert out["ci_low"] < true < out["ci_high"]
    assert out["samples"].shape == (1000,)
    assert 0 < out["p_value"] < 0.05
    same = stats.bootstrap_sharpe_difference(b, a, n_boot=500, seed=0)
    assert same["p_value"] > 0.9


def test_bootstrap_metric_ci():
    x = pd.Series(_iid(2000, 0.05, seed=12), index=pd.bdate_range("2000-01-01", periods=2000))
    out = stats.bootstrap_metric_ci(x, "sharpe", n_boot=500)
    assert out["ci_low"] < 0.05 * math.sqrt(252) < out["ci_high"]
    assert out["ci_low"] < out["estimate"] < out["ci_high"]
    mdd = stats.bootstrap_metric_ci(x, "max_drawdown", n_boot=40)
    assert mdd["ci_high"] <= 0 and len(mdd["samples"]) == 40
    custom = stats.bootstrap_metric_ci(x, lambda s: float(s.mean()), n_boot=40)
    assert custom["estimate"] == pytest.approx(x.mean())


# ---------------------------------------------------------------- random entry


def test_random_entry_noise_and_foresight():
    rng = np.random.default_rng(13)
    g = rng.normal(0.0003, 0.01, 4000)
    w = (pd.Series(rng.normal(size=4000)).rolling(20).mean() > 0).astype(float).to_numpy()
    out = stats.random_entry_test(g, w, n_sims=1000)
    assert 0 < out["p_value"] < 1 and out["null_sharpes"].shape == (1000,)
    assert out["exposure_fraction"] == pytest.approx(w.mean())
    perfect = stats.random_entry_test(g, (g > 0).astype(float), n_sims=1000)
    assert perfect["p_value"] < 0.01


def test_random_entry_always_invested_is_uninformative():
    g = _iid(500, 0.05, seed=14)
    out = stats.random_entry_test(g, np.ones(500), n_sims=200)
    assert out["n_segments"] == 1
    np.testing.assert_allclose(out["null_sharpes"], out["actual_sharpe"])
    assert out["p_value"] == 1.0


def test_random_entry_on_backtest_result(prices):
    res = run_backtest(prices, EMACrossover(5, 20), BacktestConfig(), rf_yield=0.02)
    out = stats.random_entry_test_result(res, n_sims=300)
    assert out["actual_sharpe"] == pytest.approx(res.summary()["sharpe"])
    assert 0 < out["p_value"] <= 1


# ---------------------------------------------------------------- costs


@pytest.fixture
def prices():
    rng = np.random.default_rng(21)
    n = 1500
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": 1e6}, index=idx
    )


def _intraday(spread, days=600, steps=390, sig=0.01, seed=0):
    rng = np.random.default_rng(seed)
    path = 100 * np.exp(np.cumsum(rng.normal(0, sig / math.sqrt(steps), days * steps)))
    p = path.reshape(days, steps)
    close = p[:, -1] * (1 + spread / 2 * rng.choice([-1, 1], days))
    return pd.DataFrame(
        {
            "Open": p[:, 0],
            "High": p.max(axis=1) * (1 + spread / 2),
            "Low": p.min(axis=1) * (1 - spread / 2),
            "Close": close,
        },
        index=pd.bdate_range("2020-01-01", periods=days),
    )


@pytest.mark.parametrize("spread", [0.005, 0.02])
def test_corwin_schultz_recovers_spread(spread):
    df = _intraday(spread)
    assert costs.median_spread_bps(df) == pytest.approx(spread * 1e4, rel=0.25)
    s = costs.corwin_schultz_spread(df["High"], df["Low"])
    assert np.isnan(s.iloc[0]) and (s.dropna() >= 0).all()
    assert s.mean() * 1e4 > 0.95 * costs.median_spread_bps(df)  # daily clipping biases up


def test_breakeven_cost():
    assert costs.breakeven_cost_bps(0.02, 4.0) == pytest.approx(50.0)
    assert costs.breakeven_cost_bps(0.02, 0.0) == math.inf


def test_breakeven_from_result_zeroes_edge(prices):
    strat = EMACrossover(5, 20)
    res = run_backtest(prices, strat, BacktestConfig(costs=CostModel.zero()), rf_yield=0.02)
    be = costs.breakeven_from_result(res)
    at_be = run_backtest(
        prices, strat, BacktestConfig(costs=CostModel.flat(be["breakeven_bps"])), rf_yield=0.02
    )
    assert (at_be.returns - at_be.rf).mean() == pytest.approx(0.0, abs=1e-12)


def test_cost_sensitivity(prices):
    grid = [0, 5, 20, 50]
    df = costs.cost_sensitivity(prices, EMACrossover(5, 20), grid, rf_yield=0.02)
    assert list(df.index) == [float(b) for b in grid]
    assert {"cagr", "sharpe", "turnover_per_year", "cost_drag_annual"} <= set(df.columns)
    assert df["sharpe"].is_monotonic_decreasing and df["cagr"].is_monotonic_decreasing
    assert df.loc[0.0, "cost_drag_annual"] == 0.0
    assert df["turnover_per_year"].nunique() == 1
    with pytest.raises(ValueError):
        costs.cost_sensitivity(prices, EMACrossover(5, 20), [1], config_kwargs={"costs": None})


# ---------------------------------------------------------------- regression (review)


def test_random_entry_nan_strategy_return_is_not_significant():
    rng = np.random.default_rng(30)
    g = rng.normal(0, 0.01, 1000)
    w = (pd.Series(rng.normal(size=1000)).rolling(20).mean() > 0).astype(float).to_numpy()
    r = w * g
    r[0] = np.nan  # e.g. a leading pct_change NaN
    out = stats.random_entry_test(g, w, r, n_sims=300)
    clean = stats.random_entry_test(g, w, np.nan_to_num(r), n_sims=300)
    assert np.isfinite(out["actual_sharpe"]) and np.isfinite(out["null_sharpes"]).all()
    assert out["p_value"] == pytest.approx(clean["p_value"])


def test_random_entry_exact_ties_count():
    for s in range(10):
        g = _iid(3000, 0.03, seed=s, sd=0.013)
        assert stats.random_entry_test(g, np.ones(3000), n_sims=50)["p_value"] == 1.0


def test_random_entry_calibrated_under_null():
    ps = []
    for s in range(60):
        rng = np.random.default_rng(1000 + s)
        g = rng.normal(0.0003, 0.01, 1000)
        w = (pd.Series(rng.normal(size=1000)).rolling(30).mean() > 0).astype(float).to_numpy()
        ps.append(stats.random_entry_test(g, w, n_sims=200, seed=s)["p_value"])
    assert 0.35 < np.mean(ps) < 0.65
    assert np.mean(np.array(ps) < 0.1) < 0.25


def test_psr_uses_skew_and_kurtosis():
    x = np.random.default_rng(31).lognormal(0, 0.5, 3000) - 1.05  # strongly right-skewed
    s = pd.Series(x)
    sr, g3, g4 = x.mean() / x.std(ddof=1), s.skew(), s.kurt() + 3
    z = sr * math.sqrt(len(x) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    assert stats.probabilistic_sharpe_ratio(x) == pytest.approx(norm.cdf(z))


def test_lo_short_sample_is_nan():
    assert math.isnan(stats.lo_adjusted_sharpe(np.array([0.01, -0.005])))


def test_bootstrap_difference_null_calibration():
    ps = []
    for s in range(40):
        rng = np.random.default_rng(200 + s)
        c = rng.normal(0, 0.01, 1000)
        a = 0.6 * c + 0.8 * rng.normal(0, 0.01, 1000) + 0.0003
        b = 0.6 * c + 0.8 * rng.normal(0, 0.01, 1000) + 0.0003
        ps.append(stats.bootstrap_sharpe_difference(a, b, n_boot=200, seed=s)["p_value"])
    assert 0.3 < np.mean(ps) < 0.7


def _cs(h0, l0, h1, l1):
    k = 3 - 2 * math.sqrt(2)
    b = math.log(h0 / l0) ** 2 + math.log(h1 / l1) ** 2
    g = math.log(max(h0, h1) / min(l0, l1)) ** 2
    a = (math.sqrt(2 * b) - math.sqrt(b)) / k - math.sqrt(g / k)
    return 2 * (math.exp(a) - 1) / (1 + math.exp(a))


def test_corwin_schultz_hand_computed_and_overnight():
    idx = pd.bdate_range("2020-01-01", periods=2)
    h = pd.Series([101.0, 101.2], idx)
    lo = pd.Series([99.0, 99.2], idx)
    s = costs.corwin_schultz_spread(h, lo)
    assert s.iloc[1] == pytest.approx(_cs(101, 99, 101.2, 99.2)) and s.iloc[1] > 0
    # gap up: day-1 low 102 is 2 above the prior close 100 -> shift day 1 down by 2
    h, lo, c = (pd.Series(v, idx) for v in ([101.0, 104.0], [99.0, 102.0], [100.0, 103.0]))
    raw = costs.corwin_schultz_spread(h, lo, close=c, clip_daily=False)
    assert raw.iloc[1] == pytest.approx(_cs(101, 99, 102, 100)) and raw.iloc[1] < 0
    assert costs.corwin_schultz_spread(h, lo, close=c).iloc[1] == 0.0
    # gap down: day-1 high 97 is 3 below the prior close 100 -> shift day 1 up by 3
    h, lo = pd.Series([101.0, 97.0], idx), pd.Series([99.0, 95.0], idx)
    raw = costs.corwin_schultz_spread(h, lo, close=c, clip_daily=False)
    assert raw.iloc[1] == pytest.approx(_cs(101, 99, 100, 98))


def test_cost_drag_equals_turnover_times_bps(prices):
    df = costs.cost_sensitivity(prices, EMACrossover(5, 20), [0, 10, 30], rf_yield=0.02)
    np.testing.assert_allclose(df["cost_drag_annual"], df["turnover_per_year"] * df.index / 1e4)


def test_trial_sharpes_ignore_warmup_nans():
    df = pd.DataFrame(np.random.default_rng(32).normal(0.0005, 0.01, (500, 3)))
    df.iloc[:50, 1] = np.nan
    ts = stats.trial_sharpes(df)
    assert np.isfinite(ts).all()
    assert ts.iloc[1] == pytest.approx(stats.sharpe_per_period(df.iloc[50:, 1]))
    assert stats.deflated_sharpe_ratio(df[0], ts)["n_trials"] == 3


def test_sharpe_difference_test_uses_paired_variance():
    from trendlab.stats import probabilistic_sharpe_ratio, sharpe_difference_test

    rng = np.random.default_rng(7)
    common = rng.normal(0.0004, 0.01, 5000)
    a = common + rng.normal(0.0002, 0.004, 5000)
    out = sharpe_difference_test(a, common)
    assert 0.8 < out["rho"] < 1.0
    unpaired = probabilistic_sharpe_ratio(a, sr_benchmark=out["sr_b"])
    # the paired test is more confident than treating SR_b as a known constant
    assert out["psr"] > unpaired
    assert np.isfinite(out["mintrl"]) and out["mintrl"] > 1
    assert sharpe_difference_test(common, a)["mintrl"] == float("inf")
