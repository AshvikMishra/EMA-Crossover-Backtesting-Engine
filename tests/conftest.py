"""Shared synthetic market data for the test-suite (no network, deterministic).

Prices follow a geometric random walk (Black-Scholes 1973 / Osborne 1959 log-normal
model): ``log C_t - log C_{t-1} ~ N(drift - vol^2/2, vol^2)``. The open gaps from the
previous close and high/low wrap the open-close range, so every bar is internally
consistent (``High >= max(Open, Close)``, ``Low <= min(Open, Close)``).

Other test modules can ``from conftest import make_prices, frame_from_close``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TRENDLAB_OFFLINE", "1")


def frame_from_close(
    close,
    start: str = "2000-01-03",
    freq: str = "B",
    open_=None,
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """OHLCV frame around a given close path; ``open_`` defaults to the previous close."""
    close = np.asarray(close, dtype=float)
    if index is None:
        index = pd.date_range(start, periods=len(close), freq=freq, name="Date")
    if open_ is None:
        open_ = np.concatenate([[close[0]], close[:-1]])
    open_ = np.asarray(open_, dtype=float)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close),
            "Low": np.minimum(open_, close),
            "Close": close,
            "Volume": np.full(len(close), 1_000_000.0),
        },
        index=index,
    )


def make_prices(
    n: int = 1000,
    seed: int = 0,
    drift: float = 0.08,
    vol: float = 0.20,
    start: str = "2000-01-03",
    freq: str = "B",
    s0: float = 100.0,
    periods: int | None = None,
) -> pd.DataFrame:
    """Synthetic daily OHLCV with annualised ``drift`` and ``vol``.

    ``periods`` (bars per year) defaults to 365 for calendar-daily ``freq='D'`` and 252
    otherwise. Columns: Open, High, Low, Close, Volume; index: DatetimeIndex "Date".
    """
    rng = np.random.default_rng(seed)
    periods = periods or (365 if freq == "D" else 252)
    mu, sig = drift / periods, vol / np.sqrt(periods)
    log_ret = rng.normal(mu - 0.5 * sig**2, sig, n)
    close = s0 * np.exp(np.cumsum(log_ret))
    gap = rng.normal(0.0, sig * 0.3, n)
    prev = np.concatenate([[s0], close[:-1]])
    open_ = prev * np.exp(gap)
    wick = np.abs(rng.normal(0.0, sig * 0.5, (2, n)))
    high = np.maximum(open_, close) * (1.0 + wick[0])
    low = np.minimum(open_, close) * (1.0 - wick[1])
    volume = rng.integers(500_000, 2_000_000, n).astype(float)
    index = pd.date_range(start, periods=n, freq=freq, name="Date")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


@pytest.fixture
def prices() -> pd.DataFrame:
    """1000 business days of synthetic OHLCV (seed 0, 8% drift, 20% vol)."""
    return make_prices()


@pytest.fixture
def price_factory():
    """The :func:`make_prices` function, for tests that need several paths."""
    return make_prices
