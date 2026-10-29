"""Data hygiene, the offline cache and risk-free accrual.

Cash earns the T-bill yield known at the start of each bar, compounded over the
calendar days the bar spans (actual/365, as for US T-bill money-market accrual; see
Hull, "Options, Futures and Other Derivatives", ch. 4). Using the bar's own closing
yield would be a (small) look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trendlab import data
from trendlab.data import (
    clean_prices,
    load_fred,
    load_prices,
    per_bar_rate,
    periods_per_year,
    risk_free_yield,
)


def test_per_bar_rate_none_and_scalar():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    assert (per_bar_rate(idx, None) == 0.0).all()
    r = per_bar_rate(idx, 0.05)
    assert r.iloc[0] == 0.0
    np.testing.assert_allclose(r.iloc[1:], 1.05 ** (1 / 365) - 1)


def test_per_bar_rate_accrues_calendar_days_over_weekend():
    idx = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])  # Thu..Tue
    r = per_bar_rate(idx, 0.05)
    one_day = 1.05 ** (1 / 365) - 1
    assert r.iloc[1] == pytest.approx(one_day)
    assert r.iloc[2] == pytest.approx(1.05 ** (3 / 365) - 1)  # Fri -> Mon
    assert r.iloc[3] == pytest.approx(one_day)


def test_per_bar_rate_uses_previous_bar_yield():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    y = pd.Series([0.01, 0.01, 0.01, 0.10, 0.10, 0.10], index=idx)
    r = per_bar_rate(idx, y)
    low, high = 1.01 ** (1 / 365) - 1, 1.10 ** (1 / 365) - 1
    # the jump is observed on day 3, so the bar ending on day 3 still earns the old yield
    np.testing.assert_allclose(r.iloc[1:4], low)
    np.testing.assert_allclose(r.iloc[4:], high)


def test_per_bar_rate_forward_fills_sparse_yield_onto_bars():
    y = pd.Series([0.02, 0.04], index=pd.DatetimeIndex(["2024-01-01", "2024-01-06"]))
    idx = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])  # Fri, Mon, Tue
    r = per_bar_rate(idx, y)
    assert r.iloc[1] == pytest.approx(1.02 ** (3 / 365) - 1)  # Fri yield = 2% (ffilled)
    assert r.iloc[2] == pytest.approx(1.04 ** (1 / 365) - 1)  # Mon yield = 4%


def test_per_bar_rate_is_prefix_invariant():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    y = pd.Series(np.linspace(0.01, 0.05, 30), index=idx)
    y2 = y.copy()
    y2.iloc[20:] = 0.5
    a, b = per_bar_rate(idx, y), per_bar_rate(idx, y2)
    pd.testing.assert_series_equal(a.iloc[:21], b.iloc[:21])
    assert not np.allclose(a.iloc[21:], b.iloc[21:])


def test_periods_per_year_detection():
    assert periods_per_year(pd.date_range("2020-01-01", periods=800, freq="D")) == 365
    assert periods_per_year(pd.date_range("2020-01-01", periods=800, freq="B")) == 252
    assert periods_per_year(pd.date_range("2020-01-01", periods=2, freq="D")) == 252


def _dirty_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        ["2024-01-03", "2024-01-02", "2024-01-04", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    return pd.DataFrame(
        {
            "Open": [10.0, 9.0, 11.0, 11.5, 0.0, np.nan],
            "High": [10.5, 9.5, 11.5, 12.0, 12.5, 13.0],
            "Low": [9.5, 8.5, 10.5, 11.0, 11.0, 12.0],
            "Close": [10.2, 9.2, 11.1, 11.8, 12.0, -1.0],
            "Volume": [1.0] * 6,
        },
        index=idx,
    )


def test_clean_prices_dedups_sorts_and_drops_nonpositive():
    raw = _dirty_frame()
    snapshot = raw.copy()
    out = clean_prices(raw)
    pd.testing.assert_frame_equal(raw, snapshot)  # input untouched
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert list(out.index.strftime("%m-%d")) == ["01-02", "01-03", "01-04", "01-05"]
    assert out.loc["2024-01-04", "Close"] == 11.8  # duplicate: keep last
    assert (out["Close"] > 0).all()


def test_clean_prices_repairs_bad_open_with_previous_close():
    out = clean_prices(_dirty_frame())
    assert out.loc["2024-01-05", "Open"] == pytest.approx(11.8)
    first = clean_prices(_dirty_frame().iloc[[1]].assign(Open=0.0))
    assert first["Open"].iloc[0] == first["Close"].iloc[0]  # no previous close: own close


def test_load_prices_offline_uncached_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prices("NOPE", offline=True, data_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_fred("NOPE", offline=True, data_dir=tmp_path)


def test_load_prices_offline_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("TRENDLAB_OFFLINE", "1")
    with pytest.raises(FileNotFoundError):
        load_prices("NOPE", data_dir=tmp_path)


def test_load_prices_reads_cache_and_slices_inclusive(tmp_path):
    idx = pd.date_range("2024-01-01", periods=10, freq="B", name="Date")
    df = pd.DataFrame({c: np.arange(1.0, 11.0) for c in data.OHLCV}, index=idx)
    df.to_parquet(tmp_path / "IDX_TEST.parquet")  # "^" is mapped to "IDX_" on disk
    out = load_prices("^TEST", "2024-01-03", "2024-01-05", offline=True, data_dir=tmp_path)
    assert list(out.index) == list(pd.date_range("2024-01-03", "2024-01-05", freq="B"))
    out.iloc[0, 0] = -999.0  # returned frame is a copy
    again = load_prices("^TEST", offline=True, data_dir=tmp_path)
    assert again["Open"].min() == 1.0


def test_risk_free_yield_from_cached_fred(tmp_path):
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-05"], name="Date")
    pd.DataFrame({"DTB3": [5.0, np.nan, 4.0]}, index=idx).to_parquet(tmp_path / "FRED_DTB3.parquet")
    y = risk_free_yield(offline=True, data_dir=tmp_path)
    assert y.index.freqstr == "D"
    assert y.loc["2024-01-02"] == pytest.approx(0.05)
    assert y.loc["2024-01-04"] == pytest.approx(0.05)  # NaN dropped, then forward-filled
    assert y.loc["2024-01-05"] == pytest.approx(0.04)
    assert y.index[-1] == pd.Timestamp.today().normalize()


def test_committed_cache_loads_offline():
    if not (data.DATA_DIR / "SPY.parquet").exists():
        pytest.skip("no committed cache")
    spy = load_prices("SPY", "2020-01-01", "2020-12-31", offline=True)
    assert list(spy.columns) == data.OHLCV
    assert spy.index.is_monotonic_increasing and (spy["Close"] > 0).all()
    assert periods_per_year(spy.index) == 252
