"""Moving-average indicators and warm-up bookkeeping."""

from __future__ import annotations

import math

import pandas as pd


def ema_alpha(span: int) -> float:
    return 2.0 / (span + 1.0)


def ema(series: pd.Series, span: int) -> pd.Series:
    """Recursive EMA, ``e_t = a*x_t + (1-a)*e_{t-1}`` seeded with the first value.

    This matches ``ewm(span=..., adjust=False)``. Values are only trustworthy
    after :func:`ema_warmup` bars, because the seed keeps weight ``(1-a)**t``.
    """
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema_warmup(span: int, tol: float = 0.01) -> int:
    """Bars until the seed's weight in an EMA falls below ``tol``.

    For EMA200 at 1% this is ~461 bars (about 1.8 trading years), which is why
    the original notebook's EMA200 was meaningless for most of 2022.
    """
    a = ema_alpha(span)
    return max(span, math.ceil(math.log(tol) / math.log(1.0 - a)))


def moving_average(series: pd.Series, window: int, kind: str = "ema") -> pd.Series:
    if kind == "ema":
        return ema(series, window)
    if kind == "sma":
        return sma(series, window)
    raise ValueError(f"unknown moving-average kind {kind!r}")


def ma_warmup(window: int, kind: str = "ema", tol: float = 0.01) -> int:
    return ema_warmup(window, tol) if kind == "ema" else window


def realized_vol(returns: pd.Series, window: int = 63, periods: int = 252) -> pd.Series:
    """Rolling annualised volatility of simple returns."""
    return returns.rolling(window, min_periods=max(10, window // 2)).std() * math.sqrt(periods)
