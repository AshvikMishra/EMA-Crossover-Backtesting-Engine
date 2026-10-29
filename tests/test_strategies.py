"""Trading-rule tests: signal logic, warm-up masking and absence of look-ahead.

A rule's target at close ``t`` must be a function of data up to ``t`` only; otherwise
the backtest suffers look-ahead bias (Bailey, Borwein, Lopez de Prado & Zhu 2014,
"Pseudo-mathematics and financial charlatanism"). Rules covered: moving-average
crossovers with a hysteresis band, Faber (2007) 10-month SMA, Moskowitz, Ooi &
Pedersen (2012) time-series momentum, Donchian/Turtle breakouts and volatility
targeting (Moreira & Muir 2017).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import frame_from_close, make_prices

from trendlab.strategies import (
    BuyAndHold,
    DonchianBreakout,
    EMACrossover,
    FaberSMA,
    MACrossover,
    SMACrossover,
    TimeSeriesMomentum,
    VolTarget,
    _hysteresis,
    month_end_mask,
    scaled,
    years_to_bars,
)


def _switches(target: pd.Series) -> int:
    t = target.dropna()
    return int((t.diff().abs() > 0).sum())


def _perturb_after(prices: pd.DataFrame, t: int, seed: int = 99) -> pd.DataFrame:
    """Scale every OHLC price strictly after bar ``t`` by a random positive factor."""
    out = prices.copy()
    n = len(prices) - t - 1
    shock = np.exp(np.random.default_rng(seed).normal(0.0, 0.05, n))
    for col in ("Open", "High", "Low", "Close"):
        vals = out[col].to_numpy().copy()
        vals[t + 1 :] *= shock
        out[col] = vals
    return out


STRATEGIES = [
    EMACrossover(10, 40),
    EMACrossover(10, 40, band=0.01, allow_short=True),
    SMACrossover(10, 40),
    FaberSMA(6),
    TimeSeriesMomentum(lookback=120, skip=21),
    TimeSeriesMomentum(lookback=60, monthly=False, allow_short=True),
    DonchianBreakout(40, 20),
    VolTarget(EMACrossover(10, 40), target_vol=0.10, window=40, max_leverage=2.0),
    scaled(SMACrossover(10, 40), 0.5),
    BuyAndHold(),
]


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
def test_target_unchanged_when_future_prices_perturbed(strategy):
    prices = make_prices(900, seed=7)
    t = 600
    base = strategy.target_position(prices)
    moved = strategy.target_position(_perturb_after(prices, t))
    pd.testing.assert_series_equal(base.iloc[: t + 1], moved.iloc[: t + 1])
    if not isinstance(strategy, BuyAndHold):
        assert not base.iloc[t + 1 :].equals(moved.iloc[t + 1 :])  # the test has teeth


NON_CALENDAR = [
    s for s in STRATEGIES if not isinstance(s, FaberSMA) and not getattr(s, "monthly", False)
]


@pytest.mark.parametrize("strategy", NON_CALENDAR, ids=lambda s: s.name)
def test_target_unchanged_when_history_truncated(strategy):
    prices = make_prices(700, seed=8)
    t = 500
    full = strategy.target_position(prices)
    cut = strategy.target_position(prices.iloc[: t + 1])
    pd.testing.assert_series_equal(full.iloc[: t + 1], cut)


@pytest.mark.parametrize(
    "strategy", [FaberSMA(6), TimeSeriesMomentum(lookback=120)], ids=lambda s: s.name
)
def test_monthly_rules_truncated_at_month_end(strategy):
    # month-end flags need the calendar, so truncate exactly on a month-end bar
    prices = make_prices(700, seed=8)
    t = int(np.flatnonzero(month_end_mask(prices.index))[-3])
    full = strategy.target_position(prices)
    cut = strategy.target_position(prices.iloc[: t + 1])
    pd.testing.assert_series_equal(full.iloc[: t + 1], cut)


@pytest.mark.parametrize(
    "strategy",
    [EMACrossover(10, 40), SMACrossover(20, 50), DonchianBreakout(30, 15), BuyAndHold()],
    ids=lambda s: s.name,
)
def test_warmup_nan_count_equals_warmup(strategy):
    target = strategy.target_position(make_prices(600, seed=1))
    assert int(target.isna().sum()) == strategy.warmup
    assert target.iloc[: strategy.warmup].isna().all()
    assert target.iloc[strategy.warmup :].notna().all()


@pytest.mark.parametrize(
    "strategy",
    [FaberSMA(6), TimeSeriesMomentum(lookback=120), VolTarget(BuyAndHold(), window=63)],
    ids=lambda s: s.name,
)
def test_warmup_masks_at_least_warmup_bars(strategy):
    target = strategy.target_position(make_prices(600, seed=1))
    assert target.iloc[: strategy.warmup].isna().all()
    assert pd.notna(target.iloc[-1])


def test_warmup_values():
    assert EMACrossover(50, 200).warmup == 461
    assert SMACrossover(50, 200).warmup == 200
    assert DonchianBreakout(100, 50).warmup == 101
    assert TimeSeriesMomentum(252).warmup == 253
    assert VolTarget(SMACrossover(10, 40), window=63).warmup == 63
    assert VolTarget(SMACrossover(10, 100), window=63).warmup == 100


def test_fast_not_shorter_than_slow_raises():
    with pytest.raises(ValueError):
        EMACrossover(200, 200)
    with pytest.raises(ValueError):
        MACrossover(fast=100, slow=50)


def test_hysteresis_state_machine():
    spread = np.array([np.nan, 0.02, 0.005, -0.005, -0.02, 0.0, 0.011])
    np.testing.assert_array_equal(_hysteresis(spread, 0.01, False), [np.nan, 1, 1, 1, 0, 0, 1])
    np.testing.assert_array_equal(_hysteresis(spread, 0.01, True), [np.nan, 1, 1, 1, -1, -1, 1])
    np.testing.assert_array_equal(_hysteresis(spread, 0.0, False), [np.nan, 1, 1, 0, 0, 0, 1])


def test_band_reduces_switches_on_noisy_series():
    prices = make_prices(3000, seed=11, drift=0.0, vol=0.15)
    plain = _switches(EMACrossover(5, 20).target_position(prices))
    banded = _switches(EMACrossover(5, 20, band=0.01).target_position(prices))
    assert plain > 20
    assert banded < 0.6 * plain


def test_ma_crossover_matches_definition():
    prices = make_prices(500, seed=2)
    s = SMACrossover(10, 40)
    px = prices["Close"]
    expected = (px.rolling(10).mean() > px.rolling(40).mean()).astype(float)
    target = s.target_position(prices)
    pd.testing.assert_series_equal(
        target.iloc[s.warmup :], expected.iloc[s.warmup :], check_names=False
    )
    ind = s.indicators(prices)
    assert list(ind.columns) == ["fast", "slow"]


def test_long_short_crossover_is_minus_one_in_downtrend():
    prices = frame_from_close(np.linspace(200, 50, 400))
    target = EMACrossover(10, 40, allow_short=True).target_position(prices).dropna()
    assert (target == -1.0).all()
    assert (EMACrossover(10, 40).target_position(prices).dropna() == 0.0).all()


def test_month_end_mask_known_answer():
    idx = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-28", "2024-03-01"])
    np.testing.assert_array_equal(month_end_mask(idx), [False, True, False, True, True])


def test_faber_only_changes_at_month_ends():
    prices = make_prices(1500, seed=3, drift=0.0, vol=0.25)
    target = FaberSMA(10).target_position(prices)
    me = month_end_mask(prices.index)
    changed = (target.diff().abs() > 0).to_numpy()
    assert changed.sum() > 3
    assert me[changed].all()


def test_faber_matches_monthly_definition():
    prices = make_prices(1500, seed=3, drift=0.0, vol=0.25)
    target = FaberSMA(10).target_position(prices)
    me = month_end_mask(prices.index)
    monthly = prices["Close"][me]
    expected = (monthly > monthly.rolling(10).mean()).astype(float)
    got = target[me]
    valid = got.notna()
    assert valid.sum() > 50
    pd.testing.assert_series_equal(got[valid], expected[valid], check_names=False)


def test_tsmom_sign_logic_daily():
    prices = make_prices(600, seed=4)
    px = prices["Close"]
    s = TimeSeriesMomentum(lookback=60, skip=10, monthly=False, allow_short=True)
    target = s.target_position(prices)
    past = px.shift(10) / px.shift(60) - 1
    expected = pd.Series(np.where(past > 0, 1.0, -1.0), index=px.index)
    pd.testing.assert_series_equal(
        target.iloc[s.warmup :], expected.iloc[s.warmup :], check_names=False
    )
    assert set(target.dropna().unique()) == {-1.0, 1.0}


def test_tsmom_trend_direction():
    up = frame_from_close(np.linspace(100, 200, 400))
    down = frame_from_close(np.linspace(200, 100, 400))
    assert (TimeSeriesMomentum(120).target_position(up).dropna() == 1.0).all()
    assert (TimeSeriesMomentum(120).target_position(down).dropna() == 0.0).all()
    short = TimeSeriesMomentum(120, allow_short=True).target_position(down).dropna()
    assert (short == -1.0).all()


def test_tsmom_monthly_only_changes_at_month_ends():
    prices = make_prices(1500, seed=5, drift=0.0, vol=0.25)
    target = TimeSeriesMomentum(lookback=63).target_position(prices)
    changed = (target.diff().abs() > 0).to_numpy()
    assert changed.sum() > 3
    assert month_end_mask(prices.index)[changed].all()


def test_donchian_enters_on_breakout_and_exits_on_breakdown():
    close = np.r_[np.full(30, 100.0), np.full(10, 110.0), np.full(10, 90.0)]
    prices = frame_from_close(close)
    target = DonchianBreakout(entry=10, exit=5).target_position(prices)
    assert target.iloc[:11].isna().all()
    assert (target.iloc[11:30] == 0.0).all()
    assert (target.iloc[30:40] == 1.0).all()  # close 110 > prior 10-bar high of 100
    assert (target.iloc[40:] == 0.0).all()  # close 90 < prior 5-bar low of 110


def test_donchian_ignores_touch_without_break():
    close = np.r_[np.full(30, 100.0), [100.0], np.full(5, 99.0)]
    target = DonchianBreakout(entry=10, exit=5).target_position(frame_from_close(close))
    assert (target.dropna() == 0.0).all()


def test_vol_target_caps_leverage():
    prices = make_prices(600, seed=6, vol=0.20)
    s = VolTarget(BuyAndHold(), target_vol=1.0, window=40, max_leverage=1.5)
    target = s.target_position(prices).dropna()
    np.testing.assert_allclose(target.to_numpy(), 1.5)


def test_vol_target_scales_inversely_to_vol():
    a = make_prices(600, seed=6, vol=0.10)
    ret = a["Close"].pct_change().fillna(0.0)
    b = frame_from_close(100.0 * np.cumprod(1.0 + 2.0 * ret.to_numpy()), index=a.index)
    s = VolTarget(BuyAndHold(), target_vol=0.10, window=40, max_leverage=100.0)
    ta, tb = s.target_position(a).dropna(), s.target_position(b).dropna()
    np.testing.assert_allclose((ta / tb).to_numpy(), 2.0, rtol=1e-9)
    rv = ret.rolling(40, min_periods=20).std() * np.sqrt(252)
    np.testing.assert_allclose(ta.to_numpy(), (0.10 / rv).loc[ta.index].to_numpy(), rtol=1e-9)


def test_vol_target_zero_where_base_flat():
    prices = make_prices(800, seed=9, drift=0.0)
    base = EMACrossover(10, 40)
    vt = VolTarget(base, target_vol=0.15, window=40).target_position(prices)
    bt = base.target_position(prices)
    both = vt.notna() & bt.notna()
    assert ((bt[both] == 0) == (vt[both] == 0)).all()
    assert (vt[both] <= 1.0 + 1e-12).all()


def test_scaled_and_names():
    prices = make_prices(300, seed=10)
    base = SMACrossover(10, 40)
    half = scaled(base, 0.5)
    pd.testing.assert_series_equal(
        half.target_position(prices), 0.5 * base.target_position(prices), check_names=False
    )
    assert half.warmup == base.warmup
    assert EMACrossover(50, 200).name == "EMA50/200"
    assert "band" in EMACrossover(50, 200, band=0.01).name
    assert "L/S" in EMACrossover(50, 200, allow_short=True).name
    assert BuyAndHold().target_position(prices).name == "Buy & Hold"


def test_years_to_bars():
    assert years_to_bars(1) == 252
    assert years_to_bars(1.5, 365) == 548
