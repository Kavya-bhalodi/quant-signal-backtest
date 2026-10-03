"""Vectorised long-short backtest with execution delay and transaction costs."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_weights(signal: pd.DataFrame) -> pd.DataFrame:
    """Rank cross-sectionally, demean, scale to gross exposure 1.

    Result is dollar-neutral (weights sum to 0) with sum(|w|) = 1 each day.
    Ranking makes the portfolio robust to outliers in the raw signal.
    """
    ranks = signal.rank(axis=1)
    demeaned = ranks.sub(ranks.mean(axis=1), axis=0)
    gross = demeaned.abs().sum(axis=1).replace(0, np.nan)
    return demeaned.div(gross, axis=0).fillna(0.0)


def run_backtest(weights: pd.DataFrame, prices: pd.DataFrame,
                 cost_bps=5.0, delay=1) -> pd.DataFrame:
    """Daily P&L of a weight schedule.

    Timing: weights are computed from data up to close t. With delay=1 they
    are traded at close t+1 and first earn the return from t+1 to t+2. This is
    deliberately conservative: it avoids assuming we can trade at the same
    close we used to compute the signal.

    Costs: cost_bps per unit of turnover (one-way), charged on the trade date.
    """
    returns = prices.pct_change(fill_method=None)
    held = weights.shift(1 + delay).reindex(returns.index).fillna(0.0)
    gross = (held * returns.fillna(0.0)).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "net": net, "turnover": turnover})


def metrics(pnl: pd.Series, turnover: pd.Series | None = None) -> dict:
    """Annualised performance statistics for a daily return series."""
    pnl = pnl.dropna()
    n_years = len(pnl) / TRADING_DAYS
    ann_ret = pnl.mean() * TRADING_DAYS
    ann_vol = pnl.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    equity = pnl.cumsum()
    max_dd = (equity - equity.cummax()).min()
    out = {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "t_stat": sharpe * np.sqrt(n_years),  # H0: true Sharpe = 0
        "max_drawdown": max_dd,
        "hit_rate": (pnl > 0).mean(),
    }
    if turnover is not None:
        out["daily_turnover"] = turnover.loc[pnl.index].mean()
    return out


def information_coefficient(signal: pd.DataFrame, prices: pd.DataFrame,
                            horizon: int) -> pd.Series:
    """Daily Spearman rank correlation between signal at t and the forward
    return from t to t+horizon, across stocks. Mean IC by horizon shows how
    quickly a signal's predictive power decays."""
    fwd = prices.shift(-horizon) / prices - 1
    mask = signal.notna() & fwd.notna()
    s = signal.where(mask).rank(axis=1)
    f = fwd.where(mask).rank(axis=1)
    return s.corrwith(f, axis=1)
