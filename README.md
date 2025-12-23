# EMA-Crossover-Backtesting-Engine

## Overview
This project implements an EMA (Exponential Moving Average) crossover trading strategy with backtesting capabilities to evaluate performance against a buy-and-hold strategy.

## Concepts

### Moving Averages
- **Moving Average**: Average of a series of data points to identify overall trends
- **SMA (Simple Moving Average)**: Mean with equal weights for all data points
- **EMA (Exponential Moving Average)**: Mean with variable weights (recent data weighted more)
- Used in combination to avoid false signals

### EMA Crossover Strategy
- Uses 2 EMAs (fast and slow) to signal trends
- **EMAs Used**: EMA12 (fast), EMA128 (medium), EMA200 (slow)
- EMA calculated using `ewm` function from pandas instead of manual formula
- **EMA12**: Shows short-term trends
- **EMA200**: Shows long-term trends

### Trading Signals
- **Golden Cross**: Bull market signal when short-term EMA crosses above long-term EMA → **Buy**
- **Death Cross**: Bear market signal when short-term EMA crosses below long-term EMA → **Sell**

### Backtesting
- Simulates trades on historical data
- Shows profit percentage difference between algorithm performance vs bought-and-held value
- **Note**: Not 100% reliable due to the random walk theory, which states that stock prices fluctuate randomly (not based on history) and cannot be predicted

## Additional Features

### Equity Curve
Tracks and visualizes portfolio value over time for both the algorithm and buy-and-hold strategies. Provides a clear comparison of how each strategy performs throughout the entire testing period, showing cumulative returns and relative performance at every point in time.

### Risk Metric - Max Drawdown
Maximum drawdown (MDD) measures the largest peak-to-trough decline in portfolio value, showing the worst-case scenario loss an investor would have experienced. This metric is crucial for understanding volatility and risk, helping assess whether higher returns justify the potential downside exposure.

## Sample Performance

### BTC-USD (Exploration & Results)

**Initial Backtest:**
- Strategy: EMA12 and EMA128 crossover
- Asset: BTC-USD
- Result: 72.9% profit

**Improved Strategy:**
- Created EMA50 and backtested EMA50 and EMA200 crossover
- Asset: BTC-USD
- Result: 80% profit (improvement over initial 72.9%)

**Extended Date Range:**
- Further increased date range to include 2025
- **Algorithm Performance**: 207.4% profit
- **Buy-and-Hold Strategy**: 98% profit

### NVDA (2022-01-01 to 2025-01-01)
Testing the same EMA50/EMA200 crossover strategy on NVIDIA stock:

**Returns:**
- **Algorithm Final Value**: $605,091.68 (505.1% return)
- **Buy-and-Hold Final Value**: $446,554.24 (346.6% return)
- **Outperformance**: Algorithm beat buy-and-hold by 158.5 percentage points

**Risk Metrics (Maximum Drawdown):**
- **Algorithm Maximum Drawdown**: -27.05%
- **Buy-and-Hold Maximum Drawdown**: -62.7%

The algorithm significantly outperformed buy-and-hold on NVDA, delivering higher returns with substantially lower risk (57% less drawdown). This demonstrates the strategy's effectiveness in volatile growth stocks by avoiding major downturns through timely sell signals.