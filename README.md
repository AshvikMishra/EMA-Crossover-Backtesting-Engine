# EMA-Crossover-Backtesting-Engine
### Notes:
Moving average -> avg of series of data points to identifiy overall trends

SMA (Simple moving avg) -> mean + equal weights

EMA (Exponential moving avg) -> mean + variable weights (recent = more)

Used in combination to avoid false signals


EMA crossover -> 2 EMAs - fast and slow - to signal trends


EMA used -> EMA12, EMA128, and EMA200
            fast   medium      slow


EMA calculated using ewm fn from pandas instead of formula

EMA12 is used to show short term trends

EMA200 is used to show long term trends


Golden cross -> bull market [short term goes above long term] -> buy

Death cross -> bear market [long term goes aboce short term] -> sell


Backtesting -> trades on historic data and shows the difference between the profit percentage of the algorithm vs the bought and held value

this is not reliable due to the random walk theory that states that stock prices fluctuate randomly, not based on history, and hence, cannot be predicted

### Exploration:
Created a new EMA - EMA50 - and backtested the crossover of the EMA50 and EMA200 for BTC-USD improving the percentage profit from 72.9% to 80% in comparision to the EMA12 and EMA128 used in the initial backtest

Further increased the date range to include 2025 to get 207.4% profit from the algorithm over 98% from the bought and held strategy