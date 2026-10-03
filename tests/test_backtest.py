import numpy as np
import pandas as pd

from src.backtest import information_coefficient, metrics, run_backtest, to_weights
from src.data import synthetic_prices
from src.signals import SIGNALS


def test_weights_dollar_neutral_unit_gross():
    p = synthetic_prices(n_days=400)
    w = to_weights(SIGNALS["reversal_5d"](p)).iloc[10:]
    assert np.allclose(w.sum(axis=1), 0, atol=1e-12)
    assert np.allclose(w.abs().sum(axis=1), 1)


def test_cheating_signal_is_neutralised_by_execution_delay():
    """A signal built from TOMORROW's return would be wildly profitable if the
    backtest leaked future data. With correct timing it must earn ~nothing."""
    p = synthetic_prices(seed=3)
    tomorrow = p.pct_change(fill_method=None).shift(-1)
    bt = run_backtest(to_weights(tomorrow), p, cost_bps=0, delay=1)
    assert abs(metrics(bt["gross"].iloc[5:])["sharpe"]) < 1.0


def test_same_cheat_without_any_lag_is_caught():
    """Sanity check that the test above has teeth: holding the cheat weights on
    the very day of the return it peeked at gives an absurd Sharpe."""
    p = synthetic_prices(seed=3)
    r = p.pct_change(fill_method=None)
    w = to_weights(r)  # weights at t use return at t
    pnl = (w * r).sum(axis=1).iloc[5:]
    assert metrics(pnl)["sharpe"] > 10


def test_real_signals_have_no_edge_on_random_walks():
    p = synthetic_prices(seed=7)
    for name, fn in SIGNALS.items():
        s = metrics(run_backtest(to_weights(fn(p)), p, 0)["gross"].iloc[253:])["sharpe"]
        assert abs(s) < 1.2, name  # ~4 standard errors at this sample length


def test_costs_equal_turnover_times_rate():
    p = synthetic_prices(n_days=500)
    w = to_weights(SIGNALS["reversal_5d"](p))
    bt = run_backtest(w, p, cost_bps=10)
    assert np.allclose(bt["gross"] - bt["net"], bt["turnover"] * 10 / 1e4)


def test_ic_of_perfect_forecast_is_one():
    p = synthetic_prices(n_days=300)
    fwd = p.shift(-5) / p - 1
    ic = information_coefficient(fwd, p, horizon=5).dropna()
    assert np.allclose(ic, 1.0)
