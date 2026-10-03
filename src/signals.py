"""Cross-sectional signals. Every value at date t uses prices up to the close of t only."""
import pandas as pd


def momentum(prices: pd.DataFrame, lookback=252, skip=21) -> pd.DataFrame:
    """12-1 month momentum: return from t-252 to t-21 (skips the last month,
    which is dominated by short-term reversal)."""
    return prices.shift(skip) / prices.shift(lookback) - 1


def reversal(prices: pd.DataFrame, lookback=5) -> pd.DataFrame:
    """Short-term reversal: last week's losers are expected to bounce."""
    return -(prices / prices.shift(lookback) - 1)


def low_volatility(prices: pd.DataFrame, window=60) -> pd.DataFrame:
    """Low-volatility anomaly: prefer stocks with lower recent realised volatility."""
    return -prices.pct_change(fill_method=None).rolling(window).std()


SIGNALS = {
    "momentum_12_1": momentum,
    "reversal_5d": reversal,
    "low_vol_60d": low_volatility,
}
