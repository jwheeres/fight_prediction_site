"""Null test: strategy run against synthetic random-walk data.

Per CLAUDE.md: "Run the strategy against synthetic random-walk data.
Expectancy must come out near zero. If random data shows real edge, there is
lookahead bias somewhere." A random walk has no real mean-reversion
structure, so this strategy — which relies entirely on RSI/SMA mean
reversion — should show no consistent positive edge on it.
"""

import numpy as np
import pandas as pd

from trading.config import StrategyConfig
from trading.strategy import simulate_over_series

# "Near zero" per CLAUDE.md, not "positive": a small negative expectancy is
# an honest null result (natural cost of trading noise), a strongly positive
# one is the lookahead-bias red flag this test exists to catch.
MAX_ALLOWED_EXPECTANCY = 0.05
MIN_TRADES_FOR_SIGNAL = 15


def _synthetic_ohlcv(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    close = pd.Series(100.0 * np.exp(np.cumsum(returns)))
    open_ = close.shift(1).fillna(close.iloc[0])
    wiggle = np.abs(rng.normal(0, 0.002, n))
    high = pd.concat([open_, close], axis=1).max(axis=1) * (1 + wiggle)
    low = pd.concat([open_, close], axis=1).min(axis=1) * (1 - wiggle)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    df.index = pd.date_range("2015-01-01", periods=n, freq="B")
    return df


def test_expectancy_near_zero_on_random_walk():
    cfg = StrategyConfig()  # default (real) strategy parameters, no cheating
    df = _synthetic_ohlcv(n=1500, seed=42)

    trades = simulate_over_series(df, cfg)
    assert len(trades) >= MIN_TRADES_FOR_SIGNAL, (
        f"only {len(trades)} trades generated; test needs enough trades for "
        "expectancy to be a meaningful signal, not noise"
    )

    expectancy = sum(t.pnl_pct for t in trades) / len(trades)
    assert expectancy < MAX_ALLOWED_EXPECTANCY, (
        f"expectancy {expectancy:.4f} is suspiciously positive on random-walk "
        "data — check for lookahead bias"
    )
