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

## Exploration & Results

### Initial Backtest
- Strategy: EMA12 and EMA128 crossover
- Asset: BTC-USD
- Result: 72.9% profit

### Improved Strategy
- Created EMA50 and backtested EMA50 and EMA200 crossover
- Asset: BTC-USD
- Result: 80% profit (improvement over initial 72.9%)

### Extended Date Range
- Further increased date range to include 2025
- **Algorithm Performance**: 207.4% profit
- **Buy-and-Hold Strategy**: 98% profit