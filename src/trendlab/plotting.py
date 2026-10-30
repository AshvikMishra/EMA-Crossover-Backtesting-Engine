"""Publication-style matplotlib charts for the trend-following study.

Design follows the principle that the data is the only loud element (Tufte,
*The Visual Display of Quantitative Information*, 1983): hairline recessive
grids, thin 2px lines, one fixed categorical palette assigned in order and never
cycled, a single y-axis per panel, and a title that states the takeaway with a
subtitle that states the setup. The palette is a colour-vision-deficiency
validated categorical set; the strategy always wears slot 1 (blue) and any
buy-and-hold benchmark a neutral grey, in every chart.

No global matplotlib state is touched at import time. Every public function
renders inside :func:`theme` (an ``rc_context``), returns the ``Figure`` and,
when ``path`` is given, saves a PNG at 150 dpi and closes the figure. All code is
Agg-safe (no ``pyplot.show``).
"""

from __future__ import annotations

import contextvars
import functools
import textwrap
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

from .metrics import drawdown_series

__all__ = [
    "BENCHMARK_COLOR",
    "CATEGORICAL",
    "STRATEGY_COLOR",
    "TOKENS",
    "bar_comparison",
    "bias_waterfall",
    "color_for",
    "cost_sensitivity_line",
    "crisis_bars",
    "equity_and_drawdown",
    "equity_curves",
    "histogram_with_marker",
    "is_benchmark",
    "metric_heatmap",
    "price_with_signals",
    "regime_bars",
    "rolling_line",
    "save_figure",
    "series_colors",
    "smile_scatter",
    "theme",
    "underwater",
    "walk_forward_chart",
]

TOKENS: dict[str, dict] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "benchmark": "#999891",
        "other": "#c3c2b7",
        "mid": "#f0efec",
        # no red-family hues: red is reserved for the claim under examination on the site
        "categorical": [
            "#2a78d6",
            "#b8860b",
            "#1baf7a",
            "#5b6b7f",
            "#6250d6",
            "#008300",
            "#0f6e6e",
            "#7a5a0e",
        ],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "benchmark": "#8a8983",
        "other": "#52514e",
        "mid": "#383835",
        "categorical": [
            "#3987e5",
            "#d95926",
            "#199e70",
            "#c98500",
            "#d55181",
            "#008300",
            "#9085e9",
            "#e66767",
        ],
    },
}
CATEGORICAL: list[str] = TOKENS["light"]["categorical"]
STRATEGY_COLOR: str = CATEGORICAL[0]
BENCHMARK_COLOR: str = TOKENS["light"]["benchmark"]
SEQUENTIAL = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
_FONTS = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "Noto Sans", "Liberation Sans"]
_BENCH_KEYS = ("buy & hold", "buy and hold", "buy&hold", "b&h", "bnh", "benchmark", "bench")
_THEME: contextvars.ContextVar[str | None] = contextvars.ContextVar("trendlab_theme", default=None)
DPI = 150


# --------------------------------------------------------------------------- style


def _tokens() -> dict:
    return TOKENS[_THEME.get() or "light"]


@functools.cache
def _font_family() -> list[str]:
    """Preferred UI sans faces that are installed and carry the glyphs we print."""
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    need = {0x2212, 0x00D7, 0x2013, 0x00B2}
    ok = []
    for name in _FONTS:
        entry = next((f for f in font_manager.fontManager.ttflist if f.name == name), None)
        if entry is None:
            continue
        try:
            if need <= set(FT2Font(entry.fname).get_charmap()):
                ok.append(name)
        except (OSError, RuntimeError):
            continue
    return ok + ["DejaVu Sans"]


def _rc(t: dict) -> dict:
    return {
        "figure.facecolor": t["surface"],
        "figure.edgecolor": t["surface"],
        "figure.dpi": 100,
        "figure.constrained_layout.use": True,
        "savefig.facecolor": t["surface"],
        "savefig.dpi": DPI,
        "axes.facecolor": t["surface"],
        "axes.edgecolor": t["axis"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": t["ink2"],
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.titlecolor": t["ink2"],
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=t["categorical"]),
        "axes.xmargin": 0.01,
        "grid.color": t["grid"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": t["axis"],
        "ytick.color": t["axis"],
        "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3,
        "ytick.major.size": 0,
        "ytick.minor.size": 0,
        "xtick.minor.size": 0,
        # concrete family names (not the generic "sans-serif") are stored on every Text at
        # creation, so a returned figure renders the same when saved outside the context
        "font.family": _font_family(),
        "font.sans-serif": _font_family(),
        "font.size": 10,
        "text.color": t["ink"],
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": t["ink2"],
        "legend.handlelength": 1.4,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 8,
        "patch.linewidth": 0,
        "image.interpolation": "nearest",
    }


@contextmanager
def theme(dark: bool = False) -> Iterator[dict]:
    """Context manager applying the trendlab style; yields the colour tokens.

    Nesting is allowed; plotting functions called inside inherit the theme.
    """
    name = "dark" if dark else "light"
    token = _THEME.set(name)
    try:
        with mpl.rc_context(_rc(TOKENS[name])):
            yield TOKENS[name]
    finally:
        _THEME.reset(token)


@contextmanager
def _styled() -> Iterator[dict]:
    with theme(dark=_THEME.get() == "dark") as t:
        yield t


def save_figure(fig: Figure, path: str | Path, dpi: int = DPI) -> Path:
    """Save ``fig`` as PNG (parents created) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _finish(fig: Figure, path: str | Path | None) -> Figure:
    if path is not None:
        save_figure(fig, path)
    return fig


def _new(figsize: tuple[float, float]) -> tuple[Figure, object]:
    return plt.subplots(figsize=figsize)


def _titles(ax, title: str | None, subtitle: str | None) -> None:
    """Takeaway title (bold, primary ink) above a setup subtitle (secondary ink)."""
    t = _tokens()
    lines = subtitle.count("\n") + 1 if subtitle else 0
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9.5,
            color=t["ink2"],
        )
    if title:
        ax.set_title(
            title,
            loc="left",
            fontsize=13,
            fontweight="bold",
            color=t["ink"],
            pad=10 + 14 * lines,
        )


# --------------------------------------------------------------------------- colours


def is_benchmark(name: str) -> bool:
    """True for names that denote buy-and-hold / a passive benchmark."""
    n = str(name).lower()
    return any(k in n for k in _BENCH_KEYS)


def series_colors(names: Sequence[str], overrides: Mapping[str, str] | None = None) -> dict:
    """Stable colour per series: benchmarks grey, others categorical slots in order.

    Past eight non-benchmark series the remainder fold into a muted "other" grey
    rather than cycling hues.
    """
    t = _tokens()
    overrides = dict(overrides or {})
    out, slot = {}, 0
    for n in names:
        if n in overrides:
            out[n] = overrides[n]
        elif is_benchmark(n):
            out[n] = t["benchmark"]
        elif slot < len(t["categorical"]):
            out[n] = t["categorical"][slot]
            slot += 1
        else:
            out[n] = t["other"]
    return out


def color_for(name: str, index: int = 0) -> str:
    """Colour of one series: benchmark grey, else categorical slot ``index``."""
    t = _tokens()
    if is_benchmark(name):
        return t["benchmark"]
    cats = t["categorical"]
    return cats[index] if index < len(cats) else t["other"]


# --------------------------------------------------------------------------- formatting


def _minus(s: str) -> str:
    return s.replace("-", "−")


def _pct(v: float, digits: int = 0) -> str:
    return _minus(f"{v:.{digits}%}")


def _fmt(fmt: str | Callable[[float], str], v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return _minus(fmt(v) if callable(fmt) else fmt.format(v))


def _pct_axis(axis, digits: int = 0) -> None:
    axis.set_major_formatter(FuncFormatter(lambda v, _: _pct(v, digits)))


def _growth_axis(ax) -> None:
    def f(v, _):
        if v >= 10:
            return f"{v:,.0f}×"
        if v >= 1:
            return f"{v:.3g}×"
        return f"{v:.2g}×"

    lo, hi = ax.dataLim.intervaly
    narrow = np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi / lo < 4
    # a 1-2-5 log locator leaves a single labelled tick when growth spans < 4x
    loc = MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]) if narrow else LogLocator(10, (1.0, 2.0, 5.0))
    ax.yaxis.set_major_locator(loc)
    ax.yaxis.set_major_formatter(FuncFormatter(f))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _date_axis(ax) -> None:
    loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, show_offset=False))


def _legend(ax, n: int, **kw):
    if n < 2:
        return None
    kw.setdefault("loc", "best")
    kw.setdefault("ncols", 1 if n > 4 else min(n, 2))
    return ax.legend(**kw)


# --------------------------------------------------------------------------- data helpers


def _is_returns(s: pd.Series) -> bool:
    """Simple returns if any value is <= 0 or every value is below 0.5 in magnitude.

    The second test catches all-positive return series (e.g. a rule sitting in
    T-bills for a whole window), which an equity curve can never resemble.
    """
    return bool(len(s)) and bool((s <= 0).any() or s.abs().max() < 0.5)


def _as_returns(s: pd.Series) -> pd.Series:
    s = pd.Series(s).astype(float).dropna()
    if _is_returns(s):
        return s
    return s.pct_change().fillna(0.0)


def _as_growth(s: pd.Series) -> pd.Series:
    """Growth of 1: equity is rebased to its first value; returns are compounded."""
    s = pd.Series(s).astype(float).dropna()
    if _is_returns(s):
        return (1.0 + s).cumprod()
    return s / s.iloc[0]


def _close(prices: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        return prices["Close"].astype(float)
    return pd.Series(prices).astype(float)


def _end_labels(ax, items: list[tuple[pd.Timestamp, float, str, str]], min_gap=0.05) -> None:
    """Dot + value at each line end, skipped entirely when ends would collide."""
    if not items or len(items) > 4:
        return
    t = _tokens()
    ax.autoscale_view()
    to_axes = ax.transAxes.inverted()
    ys = sorted(
        to_axes.transform(ax.transData.transform((mdates.date2num(x), y)))[1]
        for x, y, _, _ in items
    )
    if len(ys) > 1 and np.min(np.diff(ys)) < min_gap:
        return
    for x, y, label, color in items:
        ax.plot([x], [y], "o", ms=7, color=color, mec=t["surface"], mew=2, zorder=5)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=t["ink"],
            annotation_clip=False,
        )


def _ordered(series: Mapping[str, pd.Series], highlight) -> list[str]:
    names = list(series)
    # draw benchmarks first so the strategy sits on top
    return sorted(names, key=lambda n: (not is_benchmark(n), n in _as_list(highlight)))


def _as_list(x) -> list:
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


# --------------------------------------------------------------------------- price chart


def price_with_signals(
    prices: pd.Series | pd.DataFrame,
    fast_ma: pd.Series,
    slow_ma: pd.Series,
    exposure: pd.Series,
    title: str | None = None,
    *,
    subtitle: str | None = None,
    log: bool = True,
    labels: tuple[str, str] = ("Fast MA", "Slow MA"),
    markers: bool = True,
    path: str | Path | None = None,
) -> Figure:
    """Close price with the two moving averages, in-market shading and entry/exit markers.

    ``exposure`` is the position actually held (``BacktestResult.exposure``): its
    value at bar ``i`` earns the return from close ``i-1`` to close ``i``. That
    interval is shaded when exposure > 0, and entry/exit markers sit on the fill,
    i.e. the close *before* the first bar held with the new position. An entry is
    a move from <= 0 to > 0, an exit the reverse (so vol-scaled exposures do not
    produce a marker every day). Do not pass ``target`` (the decision) here.
    """
    with _styled() as t:
        px = _close(prices).dropna()
        exp = pd.Series(exposure).reindex(px.index)
        fig, ax = _new((10, 5.2))
        cats = t["categorical"]
        held = (exp.fillna(0.0) > 0).to_numpy()
        # interval (t_k, t_k+1] is in the market iff the bar ending at t_k+1 is held
        shade = np.append(held[1:], False).astype(float)
        ax.fill_between(
            px.index,
            0,
            shade,
            step="post",
            transform=ax.get_xaxis_transform(),
            color=cats[0],
            alpha=0.10,
            lw=0,
            label="In market",
        )
        ax.plot(px.index, px.to_numpy(), color=t["ink2"], lw=1.2, label="Price")
        ax.plot(fast_ma.index, fast_ma.to_numpy(), color=cats[1], lw=1.6, label=labels[0])
        ax.plot(slow_ma.index, slow_ma.to_numpy(), color=cats[2], lw=1.6, label=labels[1])
        if markers:
            pos = np.flatnonzero(exp.notna().to_numpy())
            change = np.diff(held[pos].astype(int))
            fill = np.maximum(pos[1:] - 1, 0)
            buys, sells = px.index[fill[change > 0]], px.index[fill[change < 0]]
            ring = {"mec": t["surface"], "mew": 1.5, "ms": 8, "ls": "none", "zorder": 6}
            ax.plot(buys, px.loc[buys], "^", color=cats[0], label="Entry", **ring)
            ax.plot(sells, px.loc[sells], "v", color=t["ink"], label="Exit", **ring)
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.4g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("Price" + (" (log scale)" if log else ""))
        _date_axis(ax)
        _legend(ax, 6, loc="upper left", ncols=3)
        _titles(ax, title, subtitle)
        return _finish(fig, path)


# --------------------------------------------------------------------------- equity & drawdown


def _plot_equity(ax, series, log, highlight, colors) -> None:
    t = _tokens()
    cmap = series_colors(list(series), colors)
    hl = _as_list(highlight)
    ends = []
    for name in _ordered(series, highlight):
        g = _as_growth(series[name])
        if g.empty:
            continue
        faded = bool(hl) and name not in hl and not is_benchmark(name)
        lw = 2.0 if (not hl or name in hl) else 1.4
        ax.plot(
            g.index, g.to_numpy(), color=cmap[name], lw=lw, alpha=0.35 if faded else 1.0, label=name
        )
        if not faded:
            ends.append((g.index[-1], float(g.iloc[-1]), f"{g.iloc[-1]:.2f}×", cmap[name]))
    if log:
        ax.set_yscale("log")
        _growth_axis(ax)
    ax.axhline(1.0, color=t["axis"], lw=0.8, zorder=1)
    ax.set_ylabel("Growth of $1" + (" (log)" if log else ""))
    order = [str(n) for n in series]
    handles, labels = ax.get_legend_handles_labels()
    pairs = sorted(zip(labels, handles, strict=False), key=lambda p: order.index(p[0]))
    if len(pairs) > 1:
        ax.legend([h for _, h in pairs], [lb for lb, _ in pairs], loc="upper left")
    _end_labels(ax, ends)


def _plot_drawdown(ax, series, colors, annotate: bool = True) -> None:
    t = _tokens()
    cmap = series_colors(list(series), colors)
    for name in _ordered(series, None):
        dd = drawdown_series(_as_returns(series[name]))
        c = cmap[name]
        ax.fill_between(dd.index, dd.to_numpy(), 0, color=c, alpha=0.10, lw=0)
        ax.plot(dd.index, dd.to_numpy(), color=c, lw=1.4, label=name)
        if annotate and len(series) <= 3 and len(dd) and dd.min() < 0:
            when = dd.idxmin()
            ax.plot([when], [dd.min()], "o", ms=6, color=c, mec=t["surface"], mew=2, zorder=5)
            ax.annotate(
                _pct(dd.min()),
                xy=(when, dd.min()),
                xytext=(6, -2),
                textcoords="offset points",
                va="top",
                fontsize=8.5,
                color=t["ink2"],
            )
    ax.axhline(0.0, color=t["axis"], lw=0.8)
    ax.set_ylabel("Drawdown")
    _pct_axis(ax.yaxis)
    lo = min(ax.get_ylim()[0], -0.01)
    ax.set_ylim(lo * 1.12, 0.02 * abs(lo))


def equity_curves(
    series: Mapping[str, pd.Series],
    log: bool = True,
    title: str | None = None,
    highlight: str | Sequence[str] | None = None,
    *,
    subtitle: str | None = None,
    colors: Mapping[str, str] | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Growth-of-$1 lines for ``{name: equity or returns}``.

    A series with any value <= 0, or with every value below 0.5 in magnitude, is
    treated as simple returns and compounded; otherwise it is equity and rebased to 1. ``highlight`` keeps the named series
    at full strength and fades the other non-benchmark lines.
    """
    with _styled():
        fig, ax = _new((10, 5.2))
        _plot_equity(ax, series, log, highlight, colors)
        _date_axis(ax)
        _titles(ax, title, subtitle)
        return _finish(fig, path)


def underwater(
    series: Mapping[str, pd.Series],
    title: str | None = None,
    *,
    subtitle: str | None = None,
    colors: Mapping[str, str] | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Drawdown-from-peak chart for ``{name: returns or equity}``; troughs are labelled."""
    with _styled():
        fig, ax = _new((10, 4.2))
        _plot_drawdown(ax, series, colors)
        _date_axis(ax)
        _legend(ax, len(series), loc="lower left")
        _titles(ax, title, subtitle)
        return _finish(fig, path)


def equity_and_drawdown(
    series: Mapping[str, pd.Series],
    log: bool = True,
    title: str | None = None,
    *,
    subtitle: str | None = None,
    highlight: str | Sequence[str] | None = None,
    colors: Mapping[str, str] | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Two stacked panels sharing the x-axis: growth of $1 on top, drawdown below."""
    with _styled():
        fig, (top, bot) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.6, 1.2]}
        )
        _plot_equity(top, series, log, highlight, colors)
        _plot_drawdown(bot, series, colors)
        top.tick_params(axis="x", length=0)
        _date_axis(bot)
        _titles(top, title, subtitle)
        return _finish(fig, path)


# --------------------------------------------------------------------------- bars


def _bar_label_fmt(percent: bool, fmt):
    if fmt is not None:
        return fmt
    return (lambda v: f"{v:.1%}") if percent else "{:.2f}"


def bias_waterfall(
    steps: Sequence[tuple[str, float]],
    title: str | None = None,
    *,
    subtitle: str | None = None,
    benchmark: float | None = None,
    benchmark_label: str = "Buy & hold",
    final_label: str | None = "Honest result",
    percent: bool = True,
    ylabel: str | None = None,
    fmt: str | Callable[[float], str] | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Waterfall of a headline metric as each bias is removed.

    ``steps[0]`` is the naive figure (full bar); each later step is drawn as a
    floating bar from the previous level to its own (red = edge lost, blue =
    gained), labelled with the new level and the change. ``final_label`` adds a
    closing full bar at the last level; ``benchmark`` draws a B&H reference line.
    """
    with _styled() as t:
        if not steps:
            raise ValueError("steps must not be empty")
        lab = _bar_label_fmt(percent, fmt)
        labels = [s[0] for s in steps]
        vals = [float(s[1]) for s in steps]
        up, down = t["categorical"][0], t["categorical"][7]
        bars: list[tuple[str, float, float, str, str]] = [
            (labels[0], 0.0, vals[0], t["categorical"][0], _fmt(lab, vals[0]))
        ]
        for i in range(1, len(vals)):
            d = vals[i] - vals[i - 1]
            delta = f"{d * 100:+.1f} pp" if percent else _fmt("{:+.2f}", d)
            text = f"{_fmt(lab, vals[i])}\n{_minus(delta)}"
            bars.append((labels[i], vals[i - 1], vals[i], up if d >= 0 else down, text))
        if final_label:
            bars.append((final_label, 0.0, vals[-1], t["categorical"][0], _fmt(lab, vals[-1])))

        fig, ax = _new((max(7.5, 1.35 * len(bars) + 2), 5.4))
        x = np.arange(len(bars))
        width = 0.42
        for i, (_, lo, hi, c, text) in enumerate(bars):
            ax.bar(i, hi - lo, bottom=lo, width=width, color=c, edgecolor=t["surface"], lw=0)
            top = max(lo, hi)
            ax.annotate(
                text,
                xy=(i, top),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=t["ink"],
                linespacing=1.2,
            )
            if i + 1 < len(bars):
                ax.plot(
                    [i + width / 2, i + 1 - width / 2], [hi, hi], color=t["axis"], lw=0.8, zorder=1
                )
        ax.axhline(0.0, color=t["axis"], lw=0.8)
        if benchmark is not None:
            ax.axhline(benchmark, color=t["benchmark"], lw=1.6, zorder=2)
            ax.annotate(
                f"{benchmark_label} {_fmt(lab, benchmark)}",
                xy=(1.0, benchmark),
                xycoords=("axes fraction", "data"),
                xytext=(0, 4),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=8.5,
                color=t["ink2"],
            )
        ax.set_xticks(x, [textwrap.fill(b[0], 14) for b in bars])
        ax.set_xlim(-0.7, len(bars) - 0.3)
        ax.tick_params(axis="x", length=0)
        if percent:
            _pct_axis(ax.yaxis)
        if ylabel:
            ax.set_ylabel(ylabel)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.12 * (hi - lo))
        _titles(ax, title, subtitle)
        return _finish(fig, path)


def _grouped(
    ax,
    df: pd.DataFrame,
    horizontal: bool,
    fmt,
    colors: Mapping[str, str] | None,
    labels: bool,
) -> None:
    t = _tokens()
    df = df.astype(float)
    cols = list(df.columns)
    cmap = series_colors(cols, colors)
    n, k = len(df), len(cols)
    group = 0.76
    size = group / k
    pos = np.arange(n)
    for j, c in enumerate(cols):
        offs = pos - group / 2 + size * (j + 0.5)
        v = df[c].to_numpy()
        kw = {"color": cmap[c], "edgecolor": t["surface"], "linewidth": 1.5, "label": str(c)}
        if horizontal:
            ax.barh(offs, np.nan_to_num(v), height=size, **kw)
        else:
            ax.bar(offs, np.nan_to_num(v), width=size, **kw)
        if not labels:
            continue
        for o, val in zip(offs, v, strict=False):
            if np.isnan(val):
                continue
            pad = 3 if val >= 0 else -3
            text = _fmt(fmt, val)
            if horizontal:
                ax.annotate(
                    text,
                    xy=(val, o),
                    xytext=(pad, 0),
                    textcoords="offset points",
                    ha="left" if val >= 0 else "right",
                    va="center",
                    fontsize=7.5,
                    color=t["ink2"],
                )
            else:
                ax.annotate(
                    text,
                    xy=(o, val),
                    xytext=(0, pad),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=7.5,
                    color=t["ink2"],
                )
    names = [str(i) for i in df.index]
    if horizontal:
        ax.set_yticks(pos, names)
        ax.invert_yaxis()
        ax.axvline(0.0, color=t["axis"], lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0)
        ax.margins(x=0.12)
    else:
        ax.set_xticks(pos, [textwrap.fill(s, 14) for s in names])
        ax.axhline(0.0, color=t["axis"], lw=0.8)
        ax.tick_params(axis="x", length=0)
        ax.margins(y=0.12)
    if k > 1:
        ax.figure.legend(loc="outside lower center", ncols=min(k, 4))


def _index_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df.index = [" · ".join(map(str, i)) for i in df.index]
    return df


def bar_comparison(
    df: pd.DataFrame,
    title: str | None = None,
    *,
    subtitle: str | None = None,
    percent: bool = True,
    fmt: str | Callable[[float], str] | None = None,
    xlabel: str | None = None,
    colors: Mapping[str, str] | None = None,
    labels: bool = True,
    path: str | Path | None = None,
) -> Figure:
    """Horizontal grouped bars: one row per asset (``df.index``), one bar per column.

    Columns whose name denotes buy-and-hold are grey; others take categorical slots.
    """
    with _styled():
        df = _index_labels(df)
        fig, ax = _new((9, max(3.2, 0.42 * len(df) * max(1, len(df.columns) / 2) + 1.6)))
        _grouped(ax, df, True, _bar_label_fmt(percent, fmt), colors, labels)
        if percent:
            _pct_axis(ax.xaxis)
        if xlabel:
            ax.set_xlabel(xlabel)
        _titles(ax, title, subtitle)
        return _finish(fig, path)


def _pick(df: pd.DataFrame, given: str | None, keys: Sequence[str], exclude=()) -> str | None:
    if given is not None:
        return given
    for c in df.columns:
        if c in exclude or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if any(k in str(c).lower() for k in keys):
            return c
    return None


def crisis_bars(
    crisis_table: pd.DataFrame,
    title: str | None = None,
    *,
    subtitle: str | None = None,
    strategy_col: str | None = None,
    bnh_col: str | None = None,
    labels: tuple[str, str] = ("Trend rule", "Buy & hold"),
    name_col: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Grouped bars of buy-and-hold vs strategy return in each named crisis.

    Rows are crises (index, or ``name_col``). Columns are auto-detected when not
    given: B&H matches ``bnh/b&h/buy/hold/bench``, strategy ``strat/trend/rule/ema``;
    otherwise the first two numeric columns are taken as (strategy, B&H).
    """
    df = crisis_table.copy()
    if name_col is None:
        name_col = next(
            (c for c in df.columns if str(c).lower() in {"crisis", "name", "period", "event"}),
            None,
        )
    if name_col is not None:
        df = df.set_index(name_col)
    bnh = _pick(df, bnh_col, ("bnh", "b&h", "buy", "hold", "bench"))
    strat = _pick(df, strategy_col, ("strat", "trend", "rule", "ema", "sma"), exclude=(bnh,))
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if strat is None or bnh is None:
        rest = [c for c in numeric if c not in (strat, bnh)]
        strat = strat or rest.pop(0)
        bnh = bnh or rest.pop(0)
    plot = df[[strat, bnh]].astype(float).dropna(how="all")
    plot.columns = list(labels)
    t = _tokens()
    colors = {labels[0]: t["categorical"][0], labels[1]: t["benchmark"]}
    return bar_comparison(
        plot,
        title,
        subtitle=subtitle,
        percent=True,
        xlabel="Return over the crisis window",
        colors=colors,
        path=path,
    )


def regime_bars(
    df: pd.DataFrame,
    title: str | None = None,
    *,
    subtitle: str | None = None,
    percent: bool = True,
    fmt: str | Callable[[float], str] | None = None,
    ylabel: str | None = None,
    colors: Mapping[str, str] | None = None,
    labels: bool = True,
    path: str | Path | None = None,
) -> Figure:
    """Vertical grouped bars: one group per regime (``df.index``), one bar per column."""
    with _styled():
        df = _index_labels(df)
        fig, ax = _new((max(7, 1.1 * len(df) * max(1, len(df.columns) / 2) + 2), 4.8))
        _grouped(ax, df, False, _bar_label_fmt(percent, fmt), colors, labels)
        if percent:
            _pct_axis(ax.yaxis)
        if ylabel:
            ax.set_ylabel(ylabel)
        _titles(ax, title, subtitle)
        return _finish(fig, path)


# --------------------------------------------------------------------------- heatmap


def _luminance(rgb) -> float:
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _diverging_cmap(t: dict) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "trendlab_div", ["#7a5a0e", "#c99a2e", t["mid"], t["categorical"][0], "#184f95"]
    )


def _tick(v) -> str:
    return str(int(v)) if isinstance(v, (float, np.floating)) and float(v).is_integer() else str(v)


def _mark_cell(data: pd.DataFrame, mark: tuple | None) -> tuple[int, int] | None:
    """Position of the (fast, slow) cell, honouring a slow x fast layout (``grid_pivot``)."""
    if mark is None:
        return None
    row, col = mark
    if str(data.index.name).lower() == "slow" or str(data.columns.name).lower() == "fast":
        row, col = col, row
    if row in data.index and col in data.columns:
        return data.index.get_loc(row), data.columns.get_loc(col)
    return None


def metric_heatmap(
    df: pd.DataFrame,
    title: str | None = None,
    fmt: str | Callable[[float], str] = "{:.2f}",
    center: float | None = None,
    mark: tuple | None = (50, 200),
    *,
    subtitle: str | None = None,
    cbar_label: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Annotated heatmap of a metric pivot (index = fast window, columns = slow window).

    Sequential blue when ``center`` is None, otherwise a blue/red diverging scale
    symmetric around ``center`` with a grey midpoint. NaN cells (e.g. fast >= slow)
    are left blank. ``mark = (fast, slow)`` outlines that cell, by default 50/200;
    when the index is named "slow" or the columns "fast" (the
    :func:`trendlab.validation.grid_pivot` layout) it is looked up as (slow, fast).
    """
    with _styled() as t:
        data = df.astype(float)
        arr = np.ma.masked_invalid(data.to_numpy())
        nr, nc = arr.shape
        fig, ax = _new((max(6.0, 0.72 * nc + 2.6), max(4.2, 0.5 * nr + 2.0)))
        finite = arr.compressed()
        vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if center is None:
            ramp = SEQUENTIAL if _THEME.get() != "dark" else SEQUENTIAL[::-1]
            cmap = LinearSegmentedColormap.from_list("trendlab_seq", ramp)
            norm = Normalize(vmin, vmax if vmax > vmin else vmin + 1e-12)
        else:
            cmap = _diverging_cmap(t)
            half = max(abs(vmax - center), abs(center - vmin)) or 1e-12
            norm = Normalize(center - half, center + half)
        cmap = cmap.with_extremes(bad=t["surface"])
        im = ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")
        size = float(np.clip(110 / max(nr, nc, 1), 6.5, 9.5))
        for i in range(nr):
            for j in range(nc):
                v = data.iat[i, j]
                if np.isnan(v):
                    continue
                dark_cell = _luminance(cmap(norm(v))) < 0.35
                ax.text(
                    j,
                    i,
                    _fmt(fmt, v),
                    ha="center",
                    va="center",
                    fontsize=size,
                    color="#ffffff" if dark_cell else "#0b0b0b",
                )
        cell = _mark_cell(data, mark)
        if cell is not None:
            i, j = cell
            ax.add_patch(
                Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=t["ink"], lw=2.2, zorder=4)
            )
        ax.set_xticks(range(nc), [_tick(c) for c in data.columns])
        ax.set_yticks(range(nr), [_tick(r) for r in data.index])
        ax.set_xlabel(xlabel or (data.columns.name or "Slow window"))
        ax.set_ylabel(ylabel or (data.index.name or "Fast window"))
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, aspect=30)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=8.5, labelcolor=t["ink2"])
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt(fmt, v)))
        if cbar_label:
            cb.set_label(cbar_label, color=t["ink2"])
        _titles(ax, title, subtitle)
        return _finish(fig, path)


# --------------------------------------------------------------------------- scatter & lines


def _coefs(coefs) -> tuple[float, float, float]:
    """(intercept, linear, quadratic) from a sequence in increasing power or a mapping."""
    if isinstance(coefs, Mapping) or isinstance(coefs, pd.Series):
        m = {str(k).lower(): float(v) for k, v in dict(coefs).items()}
        pick = {
            0: ("alpha", "const", "intercept", "a", "c0"),
            1: ("beta", "x", "linear", "b", "c1", "mkt"),
            2: ("gamma", "x2", "quadratic", "c", "c2", "mkt2", "x^2"),
        }
        out = []
        for p in range(3):
            key = next((k for k in pick[p] if k in m), None)
            if key is None:
                raise KeyError(f"cannot find power-{p} coefficient in {list(m)}")
            out.append(m[key])
        return out[0], out[1], out[2]
    a, b, c = (float(v) for v in coefs)
    return a, b, c


def smile_scatter(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    coefs=None,
    title: str | None = None,
    *,
    subtitle: str | None = None,
    xlabel: str = "Buy-and-hold quarterly return",
    ylabel: str = "Trend-rule quarterly return",
    percent: bool = True,
    path: str | Path | None = None,
) -> Figure:
    """Quarterly strategy-vs-asset scatter with the quadratic fit ``y = a + b x + c x^2``.

    ``coefs`` is ``(a, b, c)`` in increasing power, or a mapping with keys such as
    alpha/beta/gamma or const/x/x2; when None it is fitted by least squares. A
    positive ``c`` is the "trend-following smile" (long-option-like payoff).
    """
    with _styled() as t:
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        if coefs is None:
            c2, c1, c0 = np.polyfit(xs, ys, 2)
            a, b, c = float(c0), float(c1), float(c2)
        else:
            a, b, c = _coefs(coefs)
        fig, ax = _new((7.2, 5.6))
        ax.axhline(0.0, color=t["axis"], lw=0.8, zorder=1)
        ax.axvline(0.0, color=t["axis"], lw=0.8, zorder=1)
        ax.scatter(
            xs,
            ys,
            s=42,
            color=t["categorical"][0],
            edgecolor=t["surface"],
            linewidths=1.5,
            alpha=0.85,
            zorder=3,
            label="Quarter",
        )
        grid = np.linspace(xs.min(), xs.max(), 200) if xs.size else np.array([0.0])
        ax.plot(grid, a + b * grid + c * grid**2, color=t["categorical"][1], lw=2, zorder=4)
        lo, hi = (xs.min(), xs.max()) if xs.size else (-1, 1)
        ax.plot([lo, hi], [lo, hi], color=t["benchmark"], lw=1.0, zorder=2, label="y = x")
        eq = _minus(f"fit: y = {a:.3f} {'+' if b >= 0 else '-'} {abs(b):.2f}x ")
        eq += _minus(f"{'+' if c >= 0 else '-'} {abs(c):.2f}x²")
        ax.annotate(
            eq,
            xy=(0.02, 0.97),
            xycoords="axes fraction",
            va="top",
            fontsize=9,
            color=t["ink"],
        )
        ax.grid(axis="both")
        if percent:
            _pct_axis(ax.xaxis)
            _pct_axis(ax.yaxis)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower right")
        _titles(ax, title, subtitle)
        return _finish(fig, path)


def rolling_line(
    series: Mapping[str, pd.Series],
    title: str | None = None,
    ylabel: str | None = None,
    hline: float | None = 0.0,
    *,
    subtitle: str | None = None,
    percent: bool = False,
    fmt: str | Callable[[float], str] = "{:.2f}",
    colors: Mapping[str, str] | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Lines for rolling statistics ``{name: Series}`` with an optional reference ``hline``."""
    with _styled() as t:
        fig, ax = _new((10, 4.4))
        cmap = series_colors(list(series), colors)
        if hline is not None:
            ax.axhline(hline, color=t["axis"], lw=1.0, zorder=1)
        ends = []
        for name in _ordered(series, None):
            s = pd.Series(series[name]).astype(float).dropna()
            ax.plot(s.index, s.to_numpy(), color=cmap[name], lw=1.8, label=name)
            if len(s):
                lbl = _pct(s.iloc[-1], 1) if percent else _fmt(fmt, s.iloc[-1])
                ends.append((s.index[-1], float(s.iloc[-1]), lbl, cmap[name]))
        handles, labels = ax.get_legend_handles_labels()
        order = [str(n) for n in series]
        pairs = sorted(zip(labels, handles, strict=False), key=lambda p: order.index(p[0]))
        if len(pairs) > 1:
            ax.legend([h for _, h in pairs], [lb for lb, _ in pairs], loc="best")
        if percent:
            _pct_axis(ax.yaxis)
        if ylabel:
            ax.set_ylabel(ylabel)
        _date_axis(ax)
        _end_labels(ax, ends)
        _titles(ax, title, subtitle)
        return _finish(fig, path)


def histogram_with_marker(
    null_samples: Sequence[float] | np.ndarray | pd.Series,
    actual: float,
    title: str | None = None,
    *,
    subtitle: str | None = None,
    xlabel: str = "Statistic under the null",
    actual_label: str = "Actual",
    bins: int = 50,
    p_value: float | None = None,
    alternative: str = "greater",
    fmt: str | Callable[[float], str] = "{:.2f}",
    path: str | Path | None = None,
) -> Figure:
    """Histogram of a bootstrap / Monte-Carlo null with the observed statistic marked.

    Unless ``p_value`` is given, the empirical p-value ``(1 + #{null >= actual}) / (n + 1)``
    is shown (``alternative="less"`` flips the inequality).
    """
    if alternative not in ("greater", "less"):
        raise ValueError("alternative must be 'greater' or 'less'")
    with _styled() as t:
        null = np.asarray(null_samples, dtype=float)
        null = null[np.isfinite(null)]
        if p_value is None and null.size:
            hits = (null <= actual) if alternative == "less" else (null >= actual)
            p_value = (1 + int(hits.sum())) / (null.size + 1)
        fig, ax = _new((8.5, 4.6))
        ax.hist(
            null,
            bins=bins,
            color=t["benchmark"],
            alpha=0.55,
            edgecolor=t["surface"],
            linewidth=1.0,
            label="Null distribution",
        )
        ax.axvline(actual, color=t["categorical"][0], lw=2.2, label=actual_label, zorder=4)
        text = f"{actual_label} {_fmt(fmt, actual)}"
        if p_value is not None:
            text += f"\np = {p_value:.3f}"
        lo, hi = ax.get_xlim()
        right = actual < lo + 0.7 * (hi - lo)
        ax.annotate(
            text,
            xy=(actual, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(6 if right else -6, -4),
            textcoords="offset points",
            ha="left" if right else "right",
            va="top",
            fontsize=9,
            color=t["ink"],
        )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        _titles(ax, title, subtitle)
        return _finish(fig, path)


def cost_sensitivity_line(
    df: pd.DataFrame,
    title: str | None = None,
    *,
    subtitle: str | None = None,
    xlabel: str = "Cost per side (bps)",
    ylabel: str | None = None,
    percent: bool = False,
    fmt: str | Callable[[float], str] = "{:.2f}",
    hline: float | None = 0.0,
    colors: Mapping[str, str] | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Metric vs trading-cost lines: ``df.index`` = bps, one column per asset."""
    with _styled() as t:
        data = df.astype(float).sort_index()
        cols = list(data.columns)
        cmap = series_colors(cols, colors)
        fig, ax = _new((9, 4.8))
        if hline is not None:
            ax.axhline(hline, color=t["axis"], lw=1.0, zorder=1)
        xs = data.index.to_numpy(dtype=float)
        for c in cols:
            ax.plot(
                xs,
                data[c].to_numpy(),
                color=cmap[c],
                lw=1.8,
                marker="o",
                ms=6,
                mec=t["surface"],
                mew=1.5,
                label=str(c),
            )
        if len(cols) <= 4 and len(xs):
            ys = sorted(data.iloc[-1].dropna().to_numpy())
            span = np.ptp(ax.get_ylim())
            if len(ys) < 2 or np.min(np.diff(ys)) > 0.05 * span:
                for c in cols:
                    v = data[c].iloc[-1]
                    if np.isfinite(v):
                        lbl = _pct(v, 1) if percent else _fmt(fmt, v)
                        ax.annotate(
                            lbl,
                            xy=(xs[-1], v),
                            xytext=(7, 0),
                            textcoords="offset points",
                            va="center",
                            fontsize=9,
                            color=t["ink"],
                            annotation_clip=False,
                        )
        if percent:
            _pct_axis(ax.yaxis)
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", length=0)
        _legend(ax, len(cols), ncols=len(cols) if len(cols) <= 4 else 2)
        _titles(ax, title, subtitle)
        return _finish(fig, path)


# --------------------------------------------------------------------------- walk-forward


def _fold_label(row: pd.Series, label_col: str | None) -> str:
    if label_col is not None:
        return str(row[label_col])
    if "fast" in row.index and "slow" in row.index:
        fast, slow = pd.to_numeric(row[["fast", "slow"]], errors="coerce")
        if np.isfinite(fast) and np.isfinite(slow):
            return f"{fast:g}/{slow:g}"
    for key in ("params", "label", "best", "chosen"):
        if key in row.index:
            return str(row[key])
    return ""


def walk_forward_chart(
    oos_equity: pd.Series,
    bnh_equity: pd.Series,
    folds: pd.DataFrame,
    title: str | None = None,
    *,
    subtitle: str | None = None,
    log: bool = True,
    strategy_label: str = "Walk-forward (out of sample)",
    bnh_label: str = "Buy & hold",
    start_col: str | None = None,
    label_col: str | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Stitched out-of-sample equity vs B&H with fold boundaries and the chosen params.

    ``folds`` needs a test-window start column (``test_start``/``oos_start``/
    ``start`` or ``start_col``); labels come from finite ``fast``/``slow`` columns,
    else ``params``/``label``/``best``/``chosen`` or ``label_col``. Alternate folds
    are lightly banded. ``bnh_equity`` is clipped to the out-of-sample span so both
    curves start at $1 on the same bar.
    """
    with _styled() as t:
        fig, ax = _new((10, 5.2))
        oos = pd.Series(oos_equity).dropna()
        # both lines start at $1 on the first out-of-sample bar
        bnh = pd.Series(bnh_equity).dropna().loc[oos.index[0] : oos.index[-1]]
        series = {strategy_label: oos, bnh_label: bnh}
        colors = {strategy_label: t["categorical"][0], bnh_label: t["benchmark"]}
        if start_col is None:
            start_col = next(
                (c for c in ("test_start", "oos_start", "start") if c in folds.columns), None
            )
        if start_col is None:
            raise KeyError("folds needs a test_start / oos_start / start column")
        end_col = next(
            (c for c in ("test_end", "oos_end", "end") if c in folds.columns and c != start_col),
            None,
        )
        f = folds.sort_values(start_col).reset_index(drop=True)
        starts = pd.to_datetime(f[start_col])
        last = oos.index[-1]
        ends = pd.to_datetime(f[end_col]) if end_col else starts.shift(-1).fillna(last)
        for i, (s0, s1) in enumerate(zip(starts, ends, strict=False)):
            if i % 2 == 1:
                ax.axvspan(s0, s1, color=t["ink"], alpha=0.035, lw=0, zorder=0)
            ax.axvline(s0, color=t["grid"], lw=1.0, zorder=1)
        _plot_equity(ax, series, log, strategy_label, colors)
        many = len(f) > 10
        for i, s0 in enumerate(starts):
            lbl = _fold_label(f.iloc[i], label_col)
            if not lbl:
                continue
            mid = s0 + (ends.iloc[i] - s0) / 2
            ax.annotate(
                lbl,
                xy=(mid, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -4),
                textcoords="offset points",
                ha="center",
                va="top",
                rotation=90 if many else 0,
                fontsize=7.5 if many else 8.5,
                color=t["muted"],
            )
        leg = ax.get_legend()
        if leg is not None:
            leg.set_loc("lower right")
        _date_axis(ax)
        _titles(ax, title, subtitle)
        return _finish(fig, path)
