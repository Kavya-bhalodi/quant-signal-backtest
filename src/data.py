"""Price data loading: real (yfinance, cached to disk) or synthetic (random walk)."""
from pathlib import Path

import numpy as np
import pandas as pd

# Large-cap US stocks that have traded continuously since at least 2010.
# NOTE: this list is chosen with hindsight (these firms survived and grew),
# so results carry survivorship bias. See README "Limitations".
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "JPM", "JNJ", "XOM", "PG", "V",
    "MA", "HD", "CVX", "KO", "PEP", "MRK", "PFE", "ABT", "WMT", "BAC",
    "WFC", "C", "GS", "MS", "INTC", "CSCO", "ORCL", "IBM", "QCOM", "TXN",
    "ADBE", "CRM", "NKE", "MCD", "SBUX", "DIS", "CMCSA", "T", "VZ", "UNH",
    "LLY", "BMY", "AMGN", "GILD", "MDT", "CAT", "DE", "HON", "MMM", "GE",
    "BA", "LMT", "UPS", "COST", "TGT", "LOW", "AXP", "BLK", "SPG", "NEE",
    "DUK", "SO",
]


def load_prices(tickers=UNIVERSE, start="2010-01-01", end="2025-12-31",
                cache_dir="data") -> pd.DataFrame:
    """Daily split/dividend-adjusted close prices, one column per ticker.

    Downloads once via yfinance and caches to CSV so reruns are reproducible.
    """
    cache = Path(cache_dir) / f"prices_{start}_{end}.csv"
    if cache.exists():
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        raw = yf.download(list(tickers), start=start, end=end,
                          auto_adjust=True, progress=False)
        prices = raw["Close"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache)
    return clean_prices(prices)


def clean_prices(prices: pd.DataFrame, max_missing=0.2, ffill_limit=5) -> pd.DataFrame:
    """Drop tickers with too much missing data; forward-fill short gaps only."""
    prices = prices.sort_index()
    prices = prices.loc[:, prices.isna().mean() <= max_missing]
    return prices.ffill(limit=ffill_limit)


def synthetic_prices(n_assets=50, n_days=3500, seed=0, start="2010-01-01") -> pd.DataFrame:
    """Independent geometric random walks: there is NO predictability by construction.

    Used as a null test: every signal should show Sharpe ~ 0 here. A clearly
    positive Sharpe on this data would indicate look-ahead bias in the code.
    """
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.01, 0.03, n_assets)
    rets = rng.standard_normal((n_days, n_assets)) * vols
    idx = pd.bdate_range(start, periods=n_days)
    cols = [f"SYN{i:02d}" for i in range(n_assets)]
    # Compound simple returns (not exp of log-returns) so every asset has an
    # arithmetic mean return of exactly zero; otherwise high-vol assets get a
    # +sigma^2/2 drift, which would fake a (negative) low-volatility "signal".
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=cols)
