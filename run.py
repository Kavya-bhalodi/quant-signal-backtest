"""Run the full study: per-signal backtests, train/test split, IC decay,
cost sensitivity, and a train-fitted combination evaluated out of sample.

    python run.py                 # real data via yfinance (cached in data/)
    python run.py --synthetic     # random-walk null test, no internet needed
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import information_coefficient, metrics, run_backtest, to_weights
from src.data import load_prices, synthetic_prices
from src.signals import SIGNALS

IC_HORIZONS = [1, 5, 10, 21, 63]
COST_GRID = [0, 2, 5, 10, 20]


def md_table(df: pd.DataFrame, fmt="{:.3f}") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--split", default="2019-01-01", help="first date of the test period")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--delay", type=int, default=1)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    prices = synthetic_prices() if args.synthetic else load_prices(start=args.start, end=args.end)
    warmup = prices.index[252]  # longest lookback; earlier P&L is meaningless
    train = slice(warmup, pd.Timestamp(args.split) - pd.Timedelta(days=1))
    test = slice(pd.Timestamp(args.split), None)
    print(f"{prices.shape[1]} assets, {prices.index[0].date()} to {prices.index[-1].date()}, "
          f"train {warmup.date()}..{args.split}, test {args.split}..")

    weights, results, rows, ic_rows, cost_rows = {}, {}, [], {}, {}
    for name, fn in SIGNALS.items():
        sig = fn(prices)
        weights[name] = to_weights(sig)
        bt = run_backtest(weights[name], prices, args.cost_bps, args.delay)
        results[name] = bt
        for period, sl in [("train", train), ("test", test)]:
            m = metrics(bt["net"].loc[sl], bt["turnover"])
            rows.append({"signal": name, "period": period, **m})
        ic_rows[name] = {f"{h}d": information_coefficient(sig, prices, h).loc[train].mean()
                         for h in IC_HORIZONS}
        cost_rows[name] = {f"{c}bps": metrics(
            run_backtest(weights[name], prices, c, args.delay)["net"].loc[train])["sharpe"]
            for c in COST_GRID}

    # Combination: weight each signal by its TRAIN-period net Sharpe (floored at 0),
    # then evaluate on the untouched test period.
    train_sharpe = {r["signal"]: r["sharpe"] for r in rows if r["period"] == "train"}
    alpha = {k: max(v, 0.0) for k, v in train_sharpe.items()}
    total = sum(alpha.values())
    alpha = {k: (v / total if total > 0 else 1 / len(alpha)) for k, v in alpha.items()}
    combo_w = sum(alpha[k] * weights[k] for k in weights)
    combo_w = combo_w.div(combo_w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    results["combined"] = run_backtest(combo_w, prices, args.cost_bps, args.delay)
    for period, sl in [("train", train), ("test", test)]:
        rows.append({"signal": "combined", "period": period,
                     **metrics(results["combined"]["net"].loc[sl], results["combined"]["turnover"])})

    perf = pd.DataFrame(rows).set_index(["signal", "period"])
    perf.to_csv(out / "metrics.csv")
    sharpe_tbl = perf["sharpe"].unstack()[["train", "test"]]
    sharpe_tbl["decay"] = sharpe_tbl["test"] - sharpe_tbl["train"]
    sharpe_tbl.index.name = "signal"
    ic_tbl = pd.DataFrame(ic_rows).T
    ic_tbl.index.name = "signal"
    cost_tbl = pd.DataFrame(cost_rows).T
    cost_tbl.index.name = "signal"

    # Plots
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, bt in results.items():
        bt["net"].loc[warmup:].cumsum().plot(ax=ax, label=name, lw=2 if name == "combined" else 1)
    ax.axvline(pd.Timestamp(args.split), color="k", ls="--", lw=1)
    ax.text(pd.Timestamp(args.split), ax.get_ylim()[1], " test period", va="top")
    ax.set_title(f"Cumulative net return ({args.cost_bps:g} bps costs, delay {args.delay}d)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "equity_curves.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ic_tbl.T.plot(ax=ax, marker="o")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("forward horizon"); ax.set_ylabel("mean rank IC (train)")
    ax.set_title("Signal decay: IC vs forecast horizon"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "ic_decay.png", dpi=120); plt.close(fig)

    # Report
    data_note = ("SYNTHETIC random-walk data (null test, expect Sharpe ~ 0)" if args.synthetic
                 else f"{prices.shape[1]} US large caps via yfinance")
    report = f"""# Results

Data: {data_note}. Costs {args.cost_bps:g} bps per unit turnover, execution delay {args.delay} day(s).
Train: {warmup.date()} to {args.split} | Test: {args.split} to {prices.index[-1].date()}

## Out-of-sample Sharpe decay (net of costs)
{md_table(sharpe_tbl)}

Combination weights (fitted on train only): {", ".join(f"{k}={v:.2f}" for k, v in alpha.items())}

## Full metrics
{md_table(perf.reset_index().assign(idx=lambda d: d.signal + " / " + d.period).set_index("idx").drop(columns=["signal", "period"]).rename_axis("signal / period"))}

## Mean rank IC by forecast horizon (train)
{md_table(ic_tbl, "{:.4f}")}

## Cost sensitivity: train Sharpe at different cost levels
{md_table(cost_tbl)}

![equity](equity_curves.png)
![ic](ic_decay.png)
"""
    (out / "report.md").write_text(report)
    print(sharpe_tbl.round(3).to_string())
    print(f"\nWrote {out}/report.md, metrics.csv, equity_curves.png, ic_decay.png")


if __name__ == "__main__":
    main()
