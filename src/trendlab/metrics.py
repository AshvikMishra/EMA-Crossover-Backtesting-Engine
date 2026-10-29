"""Performance and risk metrics.

All functions take a Series of *simple* per-bar returns. ``periods`` is the
number of bars per year (252 for exchange-traded assets, 365 for crypto).
Sharpe-type ratios use returns in excess of the risk-free series ``rf`` when
it is given.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _excess(returns: pd.Series, rf: pd.Series | float | None) -> pd.Series:
    if rf is None:
        return returns
    if np.isscalar(rf):
        return returns - rf
    return returns - rf.reindex(returns.index).fillna(0.0)


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1.0 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series, periods: int = 252) -> float:
    n = len(returns)
    if n < 2:
        return float("nan")
    growth = (1.0 + returns).prod()
    years = (n - 1) / periods if n > 1 else float("nan")
    if growth <= 0:
        return -1.0
    return float(growth ** (1.0 / years) - 1.0)


def ann_vol(returns: pd.Series, periods: int = 252) -> float:
    return float(returns.std(ddof=1) * math.sqrt(periods))


def sharpe(returns: pd.Series, rf=None, periods: int = 252) -> float:
    ex = _excess(returns, rf)
    sd = ex.std(ddof=1)
    if not sd or np.isnan(sd):
        return float("nan")
    return float(ex.mean() / sd * math.sqrt(periods))


def sortino(returns: pd.Series, rf=None, periods: int = 252) -> float:
    """Mean excess return over downside deviation (averaged over *all* bars)."""
    ex = _excess(returns, rf)
    dd = math.sqrt(float((np.minimum(ex, 0.0) ** 2).mean()))
    if dd == 0:
        return float("nan")
    return float(ex.mean() / dd * math.sqrt(periods))


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest stretch, in bars, spent below a previous equity high."""
    under = (drawdown_series(returns) < 0).to_numpy()
    best = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return int(best)


def calmar(returns: pd.Series, periods: int = 252) -> float:
    mdd = max_drawdown(returns)
    return float(cagr(returns, periods) / abs(mdd)) if mdd < 0 else float("nan")


def ulcer_index(returns: pd.Series) -> float:
    return float(math.sqrt((drawdown_series(returns) ** 2).mean()))


def var_cvar(returns: pd.Series, level: float = 0.05) -> tuple[float, float]:
    """Historical one-bar VaR and expected shortfall at ``level`` (as negative returns)."""
    q = float(returns.quantile(level))
    tail = returns[returns <= q]
    return q, float(tail.mean()) if len(tail) else q


def alpha_beta(
    returns: pd.Series, benchmark: pd.Series, rf=None, periods: int = 252
) -> dict[str, float]:
    """OLS of strategy excess returns on benchmark excess returns.

    Returns annualised alpha, its t-statistic (plain OLS standard errors),
    beta, correlation, tracking error and information ratio.
    """
    y = _excess(returns, rf)
    x = _excess(benchmark.reindex(returns.index).fillna(0.0), rf)
    n = len(y)
    xm, ym = x.mean(), y.mean()
    vx = ((x - xm) ** 2).sum()
    beta = float(((x - xm) * (y - ym)).sum() / vx) if vx > 0 else float("nan")
    a = ym - beta * xm
    resid = y - a - beta * x
    s2 = (resid**2).sum() / max(n - 2, 1)
    se_a = math.sqrt(s2 * (1.0 / n + xm**2 / vx)) if vx > 0 else float("nan")
    active = returns - benchmark.reindex(returns.index).fillna(0.0)
    te = active.std(ddof=1) * math.sqrt(periods)
    return {
        "alpha": float(a * periods),
        "alpha_t": float(a / se_a) if se_a else float("nan"),
        "beta": beta,
        "correlation": float(np.corrcoef(x, y)[0, 1]) if n > 2 else float("nan"),
        "tracking_error": float(te),
        "information_ratio": float(active.mean() * periods / te) if te else float("nan"),
    }


def trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades is None or trades.empty:
        return {
            "n_trades": 0,
            "hit_rate": float("nan"),
            "profit_factor": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "avg_bars": float("nan"),
        }
    r = trades["return"]
    wins, losses = r[r > 0], r[r <= 0]
    gross_loss = -losses.sum()
    return {
        "n_trades": int(len(r)),
        "hit_rate": float((r > 0).mean()),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "avg_bars": float(trades["bars"].mean()),
    }


def summary(
    returns: pd.Series,
    *,
    rf: pd.Series | float | None = None,
    periods: int = 252,
    exposure: pd.Series | None = None,
    turnover: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Standard tear-sheet metrics as a flat dict (fractions, not percent)."""
    var, cvar = var_cvar(returns)
    out = {
        "start": returns.index[0].date().isoformat(),
        "end": returns.index[-1].date().isoformat(),
        "years": (len(returns) - 1) / periods,
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods),
        "vol": ann_vol(returns, periods),
        "sharpe": sharpe(returns, rf, periods),
        "sortino": sortino(returns, rf, periods),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns, periods),
        "max_dd_bars": max_drawdown_duration(returns),
        "ulcer": ulcer_index(returns),
        "var_95": var,
        "cvar_95": cvar,
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurt() + 3.0),
    }
    if exposure is not None:
        out["exposure"] = float(exposure.abs().mean())
    if turnover is not None:
        out["turnover_per_year"] = float(turnover.sum() / max(out["years"], 1e-9))
    if trades is not None:
        out.update(trade_stats(trades))
        out["trades_per_year"] = out["n_trades"] / max(out["years"], 1e-9)
    if benchmark is not None:
        out.update(alpha_beta(returns, benchmark, rf, periods))
    return out


PERCENT_KEYS = {
    "total_return",
    "cagr",
    "vol",
    "max_drawdown",
    "exposure",
    "hit_rate",
    "avg_win",
    "avg_loss",
    "var_95",
    "cvar_95",
    "alpha",
    "tracking_error",
    "ulcer",
}


def format_table(rows: dict[str, dict], keys: list[str] | None = None) -> pd.DataFrame:
    """Turn ``{label: summary_dict}`` into a display DataFrame with % formatting."""
    df = pd.DataFrame(rows).T
    if keys:
        df = df[[k for k in keys if k in df.columns]]
    fmt = df.copy().astype(object)
    for col in df.columns:
        for idx in df.index:
            v = df.loc[idx, col]
            if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                if np.isnan(v):
                    fmt.loc[idx, col] = "–"
                elif col in PERCENT_KEYS:
                    fmt.loc[idx, col] = f"{v:.1%}"
                elif col in {"n_trades", "max_dd_bars"}:
                    fmt.loc[idx, col] = f"{int(v)}"
                else:
                    fmt.loc[idx, col] = f"{v:.2f}"
    return fmt
