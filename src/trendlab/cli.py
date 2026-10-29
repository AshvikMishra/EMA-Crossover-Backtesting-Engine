"""Command-line interface: ``trendlab {run,backtest,fetch,list}``."""

from __future__ import annotations

import argparse
import sys


def _cmd_run(args) -> int:
    from .experiments import run_experiments

    run_experiments(args.experiments or None, quick=args.quick, out_dir=args.out)
    return 0


def _cmd_list(_args) -> int:
    import importlib

    from .experiments import EXPERIMENTS

    for name in EXPERIMENTS:
        mod = importlib.import_module(f"trendlab.experiments.{name}")
        print(f"{name:28s} {mod.TITLE}")
    return 0


def _cmd_fetch(args) -> int:
    from .data import load_fred, load_prices
    from .universe import ALL_TICKERS

    tickers = args.tickers or ALL_TICKERS
    for t in tickers:
        df = load_prices(t, refresh=args.refresh, offline=False)
        print(f"{t:8s} {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} bars)")
    load_fred("DTB3", refresh=args.refresh, offline=False)
    print("FRED DTB3 ok")
    return 0


def _cmd_backtest(args) -> int:
    from .experiments.common import Context
    from .metrics import format_table
    from .strategies import MACrossover

    ctx = Context()
    strat = MACrossover(fast=args.fast, slow=args.slow, kind=args.kind, band=args.band)
    overrides = {"execution": args.execution}
    if args.start:
        overrides["start"] = args.start
    if args.end:
        overrides["end"] = args.end
        ctx.end = args.end
    if args.bps is not None:
        from .engine import CostModel

        overrides["costs"] = CostModel.flat(args.bps)
    res, bnh = ctx.pair(args.ticker, strat, **overrides)
    keys = [
        "start",
        "end",
        "total_return",
        "cagr",
        "vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "exposure",
        "n_trades",
        "hit_rate",
    ]
    table = format_table({res.name: res.summary(bnh), "Buy & Hold": bnh.summary()}, keys)
    print(f"{args.ticker}  execution={args.execution}  costs={res.config.costs.bps:g}bp/side")
    print(table.T.to_string())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trendlab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run research experiments and build reports/report.html")
    r.add_argument("experiments", nargs="*", help="e.g. e01 e03 (default: all)")
    r.add_argument("--quick", action="store_true", help="small grids / few simulations")
    r.add_argument("--out", default=None, help="output directory (default: reports/)")
    r.set_defaults(func=_cmd_run)

    sub.add_parser("list", help="list experiments").set_defaults(func=_cmd_list)

    f = sub.add_parser("fetch", help="download / refresh cached market data")
    f.add_argument("tickers", nargs="*")
    f.add_argument("--refresh", action="store_true")
    f.set_defaults(func=_cmd_fetch)

    b = sub.add_parser("backtest", help="backtest one MA crossover against buy-and-hold")
    b.add_argument("ticker")
    b.add_argument("--fast", type=int, default=50)
    b.add_argument("--slow", type=int, default=200)
    b.add_argument("--kind", choices=["ema", "sma"], default="ema")
    b.add_argument("--band", type=float, default=0.0, help="hysteresis band, e.g. 0.02")
    b.add_argument(
        "--execution", choices=["next_close", "next_open", "same_close"], default="next_close"
    )
    b.add_argument("--bps", type=float, default=None, help="cost per side in bp")
    b.add_argument("--start")
    b.add_argument("--end")
    b.set_defaults(func=_cmd_backtest)

    args = p.parse_args(argv)
    if args.cmd == "backtest" and args.fast >= args.slow:
        b.error("--fast must be smaller than --slow")
    if getattr(args, "out", None) is None and args.cmd == "run":
        from .experiments.common import REPORTS_DIR

        args.out = REPORTS_DIR
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"trendlab: error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
