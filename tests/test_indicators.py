"""Moving averages and warm-up arithmetic.

The recursive EMA ``e_t = a x_t + (1-a) e_{t-1}`` keeps weight ``(1-a)^t`` on its seed
(Hunter 1986, "The exponentially weighted moving average"), so its warm-up is
``log(tol) / log(1-a)`` bars, not ``span`` bars.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trendlab.indicators import (
    ema,
    ema_alpha,
    ema_warmup,
    ma_warmup,
    moving_average,
    realized_vol,
    sma,
)


def _series(values) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values)), dtype=float)


def test_ema_alpha():
    assert ema_alpha(9) == pytest.approx(0.2)
    assert ema_alpha(200) == pytest.approx(2.0 / 201.0)


def test_ema_matches_manual_recursion():
    x = _series(np.random.default_rng(1).normal(100, 5, 300))
    span = 20
    a = 2.0 / (span + 1.0)
    manual = np.empty(len(x))
    manual[0] = x.iloc[0]
    for i in range(1, len(x)):
        manual[i] = a * x.iloc[i] + (1 - a) * manual[i - 1]
    np.testing.assert_allclose(ema(x, span).to_numpy(), manual, rtol=1e-12)
    assert not ema(x, span).isna().any()


def test_ema_of_constant_is_constant():
    x = _series(np.full(50, 7.5))
    np.testing.assert_allclose(ema(x, 10).to_numpy(), 7.5)


def test_ema_warmup_200_is_about_461():
    assert ema_warmup(200) == 461
    a = ema_alpha(200)
    # the seed's weight has just crossed below 1%
    assert (1 - a) ** 461 < 0.01 <= (1 - a) ** 460


def test_ema_warmup_never_below_span_and_tighter_tol_needs_more():
    assert ema_warmup(2) == max(2, math.ceil(math.log(0.01) / math.log(1 / 3)))
    assert ema_warmup(50) >= 50
    assert ema_warmup(50, tol=1e-6) > ema_warmup(50, tol=0.01)
    # a loose tolerance would give ~0.35*(span+1) bars; the floor keeps at least ``span``
    assert math.ceil(math.log(0.5) / math.log(1 - ema_alpha(200))) < 200
    assert ema_warmup(200, tol=0.5) == 200


def test_sma_nan_count_and_values():
    x = _series(np.arange(1.0, 11.0))
    s = sma(x, 4)
    assert s.isna().sum() == 3
    assert s.iloc[3] == pytest.approx((1 + 2 + 3 + 4) / 4)
    assert s.iloc[-1] == pytest.approx((7 + 8 + 9 + 10) / 4)


def test_moving_average_dispatch():
    x = _series(np.linspace(1, 2, 30))
    pd.testing.assert_series_equal(moving_average(x, 5, "ema"), ema(x, 5))
    pd.testing.assert_series_equal(moving_average(x, 5, "sma"), sma(x, 5))
    with pytest.raises(ValueError):
        moving_average(x, 5, "wma")


def test_ma_warmup():
    assert ma_warmup(200, "sma") == 200
    assert ma_warmup(200, "ema") == ema_warmup(200)


def test_realized_vol_known_answer():
    r = _series(np.tile([0.01, -0.01], 50))
    vol = realized_vol(r, window=20, periods=252)
    # ddof=1 std of an even-length alternating +/-s window is s*sqrt(n/(n-1))
    expected = 0.01 * math.sqrt(20 / 19) * math.sqrt(252)
    assert vol.iloc[-1] == pytest.approx(expected)
    assert vol.isna().sum() == 9  # min_periods = max(10, window // 2)
