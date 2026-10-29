# Methodology

This document specifies exactly what the backtests in this repository do, so
that every number in the README and `reports/report.html` can be reproduced
and challenged.

## 1. Data

| Item | Source | Notes |
|---|---|---|
| Daily OHLCV | Yahoo Finance via `yfinance`, `auto_adjust=True` | Split- and dividend-adjusted, so close-to-close returns are total returns. Cached as parquet in `data/raw/`; `data/raw/manifest.json` records the fetch date. |
| Risk-free rate | FRED `DTB3` (3-month T-bill, secondary market) | Discount-basis yield in percent; the few-bp difference to bond-equivalent yield is ignored. |
| Regime labels | Yahoo `^VIX` | Used only as a *lagged* label, never traded. |

Known limitations of free data, and how they are handled:

* **Adjusted prices are not point-in-time.** Yahoo rewrites history after each
  later dividend or split. The committed cache freezes one vintage; the event
  engine's notebook replica matches the original notebook's dollar figures to
  within $1 after rescaling for this.
* **Survivorship / selection.** The universe (`src/trendlab/universe.py`) is one
  liquid ETF per asset class, fixed *before* looking at results. NVDA is kept
  only because the original notebook used it, and is flagged as a
  hindsight-selected asset throughout.
* **ETF inception dates** limit history (QQQ 1999, TLT/IEF 2002, GLD 2004, DBC
  2006). Bitcoin history on Yahoo starts in September 2014.
* **Crypto trades 24/7**: Bitcoin bars are calendar days, annualised with 365;
  ETFs with 252.

## 2. Signals

A strategy maps prices to a **target exposure** `target[t]`, decided after
observing the close of bar `t`, using only data up to and including `t`
(enforced by perturbation tests in `tests/`).

* **EMA crossover**: `EMA_n(t) = a*P(t) + (1-a)*EMA_n(t-1)`, `a = 2/(n+1)`,
  seeded with the first price. Long while `EMA_fast > EMA_slow`, flat otherwise.
  This is the *state* form of the golden/death cross: the position depends on
  the current regime, not only on the crossing day.
* **Warm-up**: the seed keeps weight `(1-a)^t` after `t` bars. Results are
  only counted once that weight is below 1% (`ema_warmup`): about 461 bars for
  EMA200. The original notebook computed EMA200 from the first day of a three-year
  window, so for most of 2022 its "EMA200" was mostly the January 2022 price.
* Other rules: SMA crossover, hysteresis band, Faber (2007) 10-month SMA,
  time-series momentum (Moskowitz, Ooi & Pedersen 2012), Donchian breakout, and
  a volatility-targeting overlay (`w = base * min(L, sigma*/sigma_t)`).

## 3. Execution and accounting

Bar `i` spans close `i-1` to close `i`. With held exposure `w`, asset return
`g`, per-bar T-bill return `rf` and cost `c` (bp per side):

```
r_i = w_i * g_i + (1 - w_i) * rf_i - |w_i - w_{i-1}| * c / 1e4
```

| `execution` | Decision at close `t` is filled at | Use |
|---|---|---|
| `same_close` | close `t` | What the original notebook did. It needs the close both to compute the signal and to trade on it, so it is optimistic. Kept only to measure that bias. |
| `next_open` | open `t+1` (overnight leg at the old weight) | Realistic for a daily process |
| `next_close` | close `t+1` | **Default.** Conservative and needs no intraday data |

* `rf_i` accrues the calendar days since the previous bar at the yield known
  at the previous bar (`(1+y)^(days/365) - 1`), so Friday→Monday earns three days.
* Costs default to 5 bp per side for ETFs/stocks (1 bp commission + 2 bp
  half-spread + 2 bp slippage) and 15 bp for Bitcoin. `e10_costs` sweeps
  0–100 bp and estimates spreads with Corwin & Schultz (2012).
* **Window opening**: the account opens in cash at the first close after
  warm-up and immediately takes the strategy's current position, paying the
  opening cost. A rule that is already in a trend therefore starts invested,
  exactly like buy-and-hold, and both legs of every comparison share the same
  window.
* The vectorised engine (`engine.py`) is cross-checked against a bar-by-bar
  event-driven engine (`event_engine.py`), with agreement to 1e-10 in tests. The
  event engine also supports integer shares and dollar commissions, and
  contains the exact replica of the original notebook.

## 4. Metrics

All metrics are computed on per-bar simple returns, annualised with `P` (252
or 365) bars per year.

* `CAGR = (E_T/E_0)^(P/T) - 1`; `vol = std(r) * sqrt(P)`.
* `Sharpe = mean(r - rf) / std(r - rf) * sqrt(P)`, using excess returns.
* `Sortino = mean(r - rf) / sqrt(mean(min(r - rf, 0)^2)) * sqrt(P)`, where the
  downside deviation is averaged over *all* bars.
* `MaxDD = min_t(E_t / max(E_0, max_{s<=t} E_s) - 1)`; `Calmar = CAGR / |MaxDD|`.
* Alpha and beta come from an OLS of strategy excess returns on benchmark excess
  returns, reported with the alpha t-statistic, tracking error and information
  ratio.
* Trade statistics (hit rate, profit factor, holding period) are computed on
  round trips that include financing and both entry and exit costs. They
  compound exactly to the equity curve.

## 5. Statistical inference

| Question | Tool | Reference |
|---|---|---|
| Is the Sharpe ratio > 0 given skew and fat tails? | Probabilistic Sharpe ratio, `PSR = Phi[(SR-SR*) sqrt(T-1) / sqrt(1 - g3 SR + (g4-1)/4 SR^2)]` | Bailey & López de Prado (2012) |
| How long a track record is needed? | Minimum track record length | Bailey & López de Prado (2012) |
| Is the best of N configurations real? | Deflated Sharpe ratio: PSR against `SR0 = sqrt(V[SR_n]) ((1-γ) Φ⁻¹(1-1/N) + γ Φ⁻¹(1-1/(Ne)))` | Bailey & López de Prado (2014) |
| Does in-sample selection generalise? | Probability of backtest overfitting via CSCV (S = 16 blocks, 12,870 splits) | Bailey, Borwein, López de Prado & Zhu (2017) |
| Is the Sharpe *difference* vs buy-and-hold significant? | Paired stationary bootstrap (mean block 20 days) with Holm and Benjamini–Hochberg corrections across assets | Politis & Romano (1994); Ledoit & Wolf (2008) |
| Is the rule's *timing* better than chance? | Random-entry Monte Carlo that keeps trade count, holding periods and exposure | — |
| Are returns autocorrelated enough to bias √P annualisation? | Lo (2002) adjusted Sharpe | Lo (2002) |

## 6. Out-of-sample protocol

* **A priori choice**: EMA50/200 is the textbook "golden cross" and is fixed in
  advance. Every other configuration is reported as a search and counted in
  the multiple-testing corrections.
* **Walk-forward** (`validation.walk_forward`): rolling 5-year training and
  1-year test windows. Parameters are chosen by in-sample Sharpe and frozen
  for the test year. Indicators are warmed up on the full prior history, and
  switching parameters at a fold boundary pays the turnover cost. A
  perturbation test checks that future data cannot change past choices.

## 7. What this study does not do

It does not model taxes, borrow costs for shorts, intraday stops, market
impact beyond a linear bps charge, or point-in-time constituent data. Results
on a handful of ETFs over about 20–30 years remain one draw of history. The
statistical tests exist to show how wide the error bars are.
