"""Market data loading with a local parquet cache.

Prices come from Yahoo Finance (via ``yfinance``) and the risk-free rate from
FRED's 3-month T-bill series (DTB3). Every download is written to
``data/raw`` so that results are reproducible offline and CI never touches the
network.

Caveats worth knowing about the free data:

* Yahoo's adjusted prices are *not* point-in-time: they are rewritten after
  every later split/dividend. The cache freezes them; ``data/raw/manifest.json``
  records when each file was fetched.
* ``auto_adjust=True`` adjusts OHLC for splits and dividends, so close-to-close
  returns are total returns.
* DTB3 is quoted on a discount basis in percent; :func:`risk_free_yield`
  converts it to an effective annual rate before it is accrued.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def _default_data_dir() -> Path:
    env = os.environ.get("TRENDLAB_DATA_DIR")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[2]  # src/trendlab/data.py -> repo root
    if (root / "pyproject.toml").exists():
        return root / "data" / "raw"
    return Path.cwd() / "data" / "raw"  # non-editable install: use the working directory


DATA_DIR = _default_data_dir()


def _safe_name(ticker: str) -> str:
    return ticker.replace("^", "IDX_").replace("=", "_").replace("/", "_")


def _cache_path(ticker: str, data_dir: Path) -> Path:
    return data_dir / f"{_safe_name(ticker)}.parquet"


def _update_manifest(data_dir: Path, key: str, info: dict) -> None:
    path = data_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[key] = info
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _download_yahoo(ticker: str, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                period="max",
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
                threads=False,
            )
            if df is not None and not df.empty:
                df = df[OHLCV].copy()
                df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
                df.index.name = "Date"
                return df
            last_err = RuntimeError(f"empty frame for {ticker}")
        except Exception as err:  # network / rate-limit errors
            last_err = err
        time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {ticker}") from last_err


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Basic hygiene: sorted unique index, no non-positive or missing closes."""
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Close"].notna() & (df["Close"] > 0)]
    # Occasional Yahoo rows have a missing/zero open; fall back to previous close.
    for col in ("Open", "High", "Low"):
        bad = df[col].isna() | (df[col] <= 0)
        if bad.any():
            df.loc[bad, col] = df["Close"].shift(1)[bad].fillna(df["Close"][bad])
    return df


def load_prices(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    *,
    refresh: bool = False,
    offline: bool | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Return adjusted daily OHLCV for ``ticker`` between ``start`` and ``end`` (inclusive).

    The full history is cached on first use; ``offline=True`` (or the env var
    ``TRENDLAB_OFFLINE=1``) raises instead of hitting the network.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    if offline is None:
        offline = os.environ.get("TRENDLAB_OFFLINE", "0") == "1"
    path = _cache_path(ticker, data_dir)
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
    else:
        if offline:
            raise FileNotFoundError(f"{ticker} not cached at {path} and offline mode is on")
        df = clean_prices(_download_yahoo(ticker))
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        _update_manifest(
            data_dir,
            ticker,
            {
                "source": "yahoo",
                "fetched": date.today().isoformat(),
                "first": str(df.index[0].date()),
                "last": str(df.index[-1].date()),
                "rows": len(df),
            },
        )
    return df.loc[start:end].copy()


def load_fred(
    series: str,
    *,
    refresh: bool = False,
    offline: bool | None = None,
    data_dir: Path | None = None,
) -> pd.Series:
    """Download (and cache) a FRED series as a float Series indexed by date."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    if offline is None:
        offline = os.environ.get("TRENDLAB_OFFLINE", "0") == "1"
    path = data_dir / f"FRED_{series}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)[series]
    if offline:
        raise FileNotFoundError(f"FRED {series} not cached at {path} and offline mode is on")
    with urllib.request.urlopen(FRED_URL.format(series=series), timeout=30) as resp:
        raw = resp.read().decode()
    df = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True, na_values=".")
    df.index.name = "Date"
    df.columns = [series]
    df[series] = pd.to_numeric(df[series], errors="coerce")
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    _update_manifest(
        data_dir,
        f"FRED:{series}",
        {"source": "fred", "fetched": date.today().isoformat(), "rows": len(df)},
    )
    return df[series]


def discount_to_effective(d: pd.Series | float, days: int = 91) -> pd.Series | float:
    """Convert a T-bill discount-basis yield to an effective annual (compounded) rate.

    A bill bought at ``1 - d * days/360`` pays 1 after ``days``; compounding that
    holding-period return over a 365-day year gives the effective rate.
    """
    return (1.0 / (1.0 - d * days / 360.0)) ** (365.0 / days) - 1.0


def risk_free_yield(**kwargs) -> pd.Series:
    """Effective annual 3-month T-bill yield (0.05 == 5%), forward-filled daily.

    FRED's DTB3 is quoted on a discount basis; it is converted with
    :func:`discount_to_effective` (about +10 bp on average, +25 bp at 5% rates).
    """
    y = discount_to_effective(load_fred("DTB3", **kwargs).dropna() / 100.0)
    full = pd.date_range(y.index[0], pd.Timestamp.today().normalize(), freq="D")
    return y.reindex(full).ffill()


def per_bar_rate(index: pd.DatetimeIndex, annual_yield: pd.Series | float | None) -> pd.Series:
    """Risk-free return earned over each bar of ``index``.

    The bar ending at ``index[i]`` accrues interest for the calendar days since
    ``index[i-1]`` at the yield *known at the previous bar* (no look-ahead), so
    a Friday→Monday equity bar earns three days and a crypto bar earns one.
    """
    if annual_yield is None:
        return pd.Series(0.0, index=index)
    if np.isscalar(annual_yield):
        y_prev = pd.Series(float(annual_yield), index=index)
    else:
        y = annual_yield.sort_index()
        y_prev = y.reindex(y.index.union(index)).ffill().reindex(index).shift(1)
        y_prev = y_prev.bfill().fillna(0.0)
    days = index.to_series().diff().dt.days.fillna(0).to_numpy()
    rate = (1.0 + y_prev.to_numpy()) ** (days / 365.0) - 1.0
    return pd.Series(rate, index=index)


def periods_per_year(index: pd.DatetimeIndex) -> int:
    """252 for exchange-traded instruments, 365 for 24/7 markets such as crypto."""
    if len(index) < 3:
        return 252
    years = (index[-1] - index[0]).days / 365.25
    if years <= 0:
        return 252
    bars = (len(index) - 1) / years
    return 365 if bars > 300 else 252
