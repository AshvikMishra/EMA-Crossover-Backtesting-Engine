"""The fixed research universe, chosen *before* looking at results.

One liquid, long-history ETF per major asset class (the set used in most
trend-following studies such as Faber 2007), plus Bitcoin as a separate case
and NVDA only because it is the asset the original notebook reported on.
"""

from __future__ import annotations

ASSETS: dict[str, dict[str, str]] = {
    "SPY": {"name": "S&P 500", "class": "US equity"},
    "QQQ": {"name": "Nasdaq-100", "class": "US equity"},
    "IWM": {"name": "Russell 2000", "class": "US equity"},
    "EFA": {"name": "Developed ex-US", "class": "Intl equity"},
    "EEM": {"name": "Emerging markets", "class": "Intl equity"},
    "IEF": {"name": "7-10y Treasuries", "class": "Bonds"},
    "TLT": {"name": "20y+ Treasuries", "class": "Bonds"},
    "GLD": {"name": "Gold", "class": "Commodities"},
    "DBC": {"name": "Commodity basket", "class": "Commodities"},
    "VNQ": {"name": "US REITs", "class": "Real estate"},
    "BTC-USD": {"name": "Bitcoin", "class": "Crypto"},
    "NVDA": {"name": "NVIDIA", "class": "Single stock"},
}

# Diversified multi-asset sleeve (excludes the hindsight-picked NVDA and BTC).
CORE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "VNQ"]
EXTRA = ["^VIX"]  # regime labels only; not tradable
ALL_TICKERS = list(ASSETS) + EXTRA

# Named stress periods (peak-to-trough windows of the S&P 500 / crypto).
CRISES: dict[str, tuple[str, str]] = {
    "Dot-com bust": ("2000-03-24", "2002-10-09"),
    "Global financial crisis": ("2007-10-09", "2009-03-09"),
    "2011 euro / US downgrade": ("2011-04-29", "2011-10-03"),
    "2015-16 China / oil": ("2015-05-21", "2016-02-11"),
    "Q4 2018 selloff": ("2018-09-20", "2018-12-24"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 inflation bear": ("2022-01-03", "2022-10-12"),
    "2025 tariff shock": ("2025-02-19", "2025-04-08"),
}
RESEARCH_END = "2026-10-28"  # last complete session when the data snapshot was taken
