"""Trading rules expressed as *target exposures*.

Every strategy maps a price frame to a Series ``target`` where ``target[t]``
is the exposure the rule wants **after observing the close of bar t**. It may
only use information up to and including bar t; the backtest engine decides
when that decision is actually filled (next close, next open, ...). Values are
NaN while indicators are still warming up.

Exposures are fractions of equity: 1 = fully long, 0 = in cash (T-bills),
-1 = fully short.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import periods_per_year
from .indicators import ma_warmup, moving_average, realized_vol


class Strategy(ABC):
    """Base class: subclasses implement :meth:`target_position` and :attr:`warmup`."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Number of leading bars whose target must be treated as unknown."""

    @abstractmethod
    def _raw_target(self, prices: pd.DataFrame) -> pd.Series: ...

    def target_position(self, prices: pd.DataFrame) -> pd.Series:
        target = self._raw_target(prices).astype(float)
        target.iloc[: self.warmup] = np.nan
        return target.rename(self.name)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.name}>"


def _hysteresis(spread: np.ndarray, band: float, allow_short: bool) -> np.ndarray:
    """Turn a relative spread (fast/slow - 1) into a long/flat(/short) state.

    Enter long above ``+band``, exit below ``-band``; in between keep the prior
    state. ``band=0`` reduces to the plain crossover.
    """
    out = np.zeros(len(spread))
    state = 0.0
    low = -1.0 if allow_short else 0.0
    for i, s in enumerate(spread):
        if np.isnan(s):
            out[i] = np.nan
            continue
        if s > band:
            state = 1.0
        elif s < -band:
            state = low
        out[i] = state
    return out


@dataclass(frozen=True, repr=False)
class MACrossover(Strategy):
    """Long while the fast moving average is above the slow one.

    This is the *state* form of the golden/death cross: the position is a
    function of the current regime, not only of the crossing day, so a
    backtest that starts mid-trend is not artificially left in cash.
    """

    fast: int = 50
    slow: int = 200
    kind: str = "ema"
    band: float = 0.0
    allow_short: bool = False
    price_col: str = "Close"

    def __post_init__(self):
        if self.fast >= self.slow:
            raise ValueError("fast window must be shorter than slow window")

    @property
    def name(self) -> str:
        tag = f"{self.kind.upper()}{self.fast}/{self.slow}"
        if self.band:
            tag += f" band{self.band:.1%}"
        if self.allow_short:
            tag += " L/S"
        return tag

    @property
    def warmup(self) -> int:
        return ma_warmup(self.slow, self.kind)

    def indicators(self, prices: pd.DataFrame) -> pd.DataFrame:
        px = prices[self.price_col]
        return pd.DataFrame(
            {
                "fast": moving_average(px, self.fast, self.kind),
                "slow": moving_average(px, self.slow, self.kind),
            }
        )

    def _raw_target(self, prices: pd.DataFrame) -> pd.Series:
        ind = self.indicators(prices)
        spread = (ind["fast"] / ind["slow"] - 1.0).to_numpy()
        return pd.Series(_hysteresis(spread, self.band, self.allow_short), index=prices.index)


def EMACrossover(fast: int = 50, slow: int = 200, **kw) -> MACrossover:
    return MACrossover(fast=fast, slow=slow, kind="ema", **kw)


def SMACrossover(fast: int = 50, slow: int = 200, **kw) -> MACrossover:
    return MACrossover(fast=fast, slow=slow, kind="sma", **kw)


@dataclass(frozen=True, repr=False)
class BuyAndHold(Strategy):
    exposure: float = 1.0

    @property
    def name(self) -> str:
        return "Buy & Hold"

    @property
    def warmup(self) -> int:
        return 0

    def _raw_target(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(self.exposure, index=prices.index)


def month_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the last available bar of each calendar month."""
    s = index.to_series()
    return (s.groupby(index.to_period("M")).transform("max") == s).to_numpy()


@dataclass(frozen=True, repr=False)
class FaberSMA(Strategy):
    """Faber (2007): hold when the month-end close is above its 10-month SMA.

    Evaluated only at month ends; the decision is held for the following month.
    """

    months: int = 10

    @property
    def name(self) -> str:
        return f"Faber {self.months}M SMA"

    @property
    def warmup(self) -> int:
        return int(self.months * 21 + 21)

    def _raw_target(self, prices: pd.DataFrame) -> pd.Series:
        me = month_end_mask(prices.index)
        monthly = prices["Close"][me]
        sig = (monthly > monthly.rolling(self.months, min_periods=self.months).mean()).astype(float)
        sig[monthly.rolling(self.months, min_periods=self.months).mean().isna()] = np.nan
        return sig.reindex(prices.index).ffill()


@dataclass(frozen=True, repr=False)
class TimeSeriesMomentum(Strategy):
    """Moskowitz-Ooi-Pedersen style: long if the trailing ``lookback`` return is positive.

    ``skip`` drops the most recent bars (12-1 momentum uses skip=21). With
    ``monthly=True`` the signal is only re-evaluated at month ends.
    """

    lookback: int = 252
    skip: int = 0
    monthly: bool = True
    allow_short: bool = False

    @property
    def name(self) -> str:
        tag = f"TSMOM {round(self.lookback / 21)}"
        if self.skip:
            tag += f"-{round(self.skip / 21)}"
        return tag + ("M L/S" if self.allow_short else "M")

    @property
    def warmup(self) -> int:
        return self.lookback + 1

    def _raw_target(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["Close"]
        past = px.shift(self.skip) / px.shift(self.lookback) - 1.0
        low = -1.0 if self.allow_short else 0.0
        sig = pd.Series(np.where(past > 0, 1.0, low), index=px.index)
        sig[past.isna()] = np.nan
        if self.monthly:
            me = month_end_mask(px.index)
            sig = sig.where(me).ffill()
        return sig


@dataclass(frozen=True, repr=False)
class DonchianBreakout(Strategy):
    """Turtle-style channel breakout: enter on a new ``entry``-bar high, exit on an ``exit``-bar low."""

    entry: int = 100
    exit: int = 50

    @property
    def name(self) -> str:
        return f"Donchian {self.entry}/{self.exit}"

    @property
    def warmup(self) -> int:
        return max(self.entry, self.exit) + 1

    def _raw_target(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["Close"].to_numpy()
        hi = prices["High"].rolling(self.entry).max().shift(1).to_numpy()
        lo = prices["Low"].rolling(self.exit).min().shift(1).to_numpy()
        out = np.full(len(close), np.nan)
        state = 0.0
        for i in range(len(close)):
            if np.isnan(hi[i]) or np.isnan(lo[i]):
                continue
            if close[i] > hi[i]:
                state = 1.0
            elif close[i] < lo[i]:
                state = 0.0
            out[i] = state
        return pd.Series(out, index=prices.index)


@dataclass(frozen=True, repr=False)
class VolTarget(Strategy):
    """Scale another strategy's exposure to hit an annualised volatility target.

    ``w_t = base_t * min(max_leverage, target_vol / sigma_t)`` where ``sigma_t``
    is the trailing realised volatility of the *asset* up to bar t.
    """

    base: Strategy
    target_vol: float = 0.15
    window: int = 63
    max_leverage: float = 1.0

    @property
    def name(self) -> str:
        return f"{self.base.name} + VT{self.target_vol:.0%}"

    @property
    def warmup(self) -> int:
        return max(self.base.warmup, self.window)

    def _raw_target(self, prices: pd.DataFrame) -> pd.Series:
        base = self.base._raw_target(prices)
        rets = prices["Close"].pct_change()
        vol = realized_vol(rets, self.window, periods_per_year(prices.index))
        scale = (self.target_vol / vol).clip(upper=self.max_leverage)
        return base * scale


def scaled(strategy: Strategy, factor: float) -> Strategy:
    """Convenience: constant multiple of another strategy (e.g. 0.5x exposure)."""
    return _Scaled(strategy, factor)


@dataclass(frozen=True, repr=False)
class _Scaled(Strategy):
    base: Strategy
    factor: float

    @property
    def name(self) -> str:
        return f"{self.factor:g}x {self.base.name}"

    @property
    def warmup(self) -> int:
        return self.base.warmup

    def _raw_target(self, prices: pd.DataFrame) -> pd.Series:
        return self.base._raw_target(prices) * self.factor


def years_to_bars(years: float, periods: int = 252) -> int:
    return int(math.ceil(years * periods))
