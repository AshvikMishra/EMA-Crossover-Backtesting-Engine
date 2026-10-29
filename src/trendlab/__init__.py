"""trendlab: a bias-aware backtesting engine for trend-following rules."""

from .engine import BacktestConfig, BacktestResult, CostModel, run_backtest
from .strategies import (
    BuyAndHold,
    DonchianBreakout,
    EMACrossover,
    FaberSMA,
    MACrossover,
    SMACrossover,
    TimeSeriesMomentum,
    VolTarget,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BuyAndHold",
    "CostModel",
    "DonchianBreakout",
    "EMACrossover",
    "FaberSMA",
    "MACrossover",
    "SMACrossover",
    "TimeSeriesMomentum",
    "VolTarget",
    "run_backtest",
]
__version__ = "0.2.0"
