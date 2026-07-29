"""No-lookahead guarantees.

Two things are checked: (1) every indicator value at index t is unchanged
if the series is truncated to [:t+1] — i.e. nothing about the value at t
depends on rows after t; (2) simulate_over_series never fills an entry at
the same bar index that produced the entry signal.
"""

import numpy as np
import pandas as pd
import pytest

from trading.config import StrategyConfig
from trading.indicators import atr, rsi, sma
from trading.strategy import compute_indicators, simulate_over_series


def _synthetic_close(n: int, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    return pd.Series(100.0 * np.exp(np.cumsum(returns)))


@pytest.mark.parametrize("check_index", [10, 25, 40])
def test_sma_no_lookahead(check_index):
    close = _synthetic_close(50)
    full = sma(close, window=5)
    truncated = sma(close.iloc[: check_index + 1], window=5)
    got_full = full.iloc[check_index]
    got_truncated = truncated.iloc[check_index]
    if pd.isna(got_full):
        assert pd.isna(got_truncated)
    else:
        assert got_full == pytest.approx(got_truncated)


@pytest.mark.parametrize("check_index", [10, 25, 40])
def test_rsi_no_lookahead(check_index):
    close = _synthetic_close(50)
    full = rsi(close, period=3)
    truncated = rsi(close.iloc[: check_index + 1], period=3)
    got_full = full.iloc[check_index]
    got_truncated = truncated.iloc[check_index]
    if pd.isna(got_full):
        assert pd.isna(got_truncated)
    else:
        assert got_full == pytest.approx(got_truncated)


@pytest.mark.parametrize("check_index", [10, 25, 40])
def test_atr_no_lookahead(check_index):
    close = _synthetic_close(50)
    high = close * 1.01
    low = close * 0.99
    full = atr(high, low, close, period=4)
    truncated = atr(
        high.iloc[: check_index + 1], low.iloc[: check_index + 1], close.iloc[: check_index + 1], period=4
    )
    got_full = full.iloc[check_index]
    got_truncated = truncated.iloc[check_index]
    if pd.isna(got_full):
        assert pd.isna(got_truncated)
    else:
        assert got_full == pytest.approx(got_truncated)


def _synthetic_ohlcv(n: int, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = _synthetic_close(n, seed=seed)
    open_ = close.shift(1).fillna(close.iloc[0])
    wiggle = np.abs(rng.normal(0, 0.002, n))
    high = pd.concat([open_, close], axis=1).max(axis=1) * (1 + wiggle)
    low = pd.concat([open_, close], axis=1).min(axis=1) * (1 - wiggle)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    df.index = pd.date_range("2020-01-01", periods=n, freq="B")
    return df


def test_entries_fill_on_bar_after_signal():
    cfg = StrategyConfig(sma_regime=20, rsi_period=2, rsi_threshold=40, exit_sma=5, time_stop_bars=5, atr_period=5)
    df = _synthetic_ohlcv(200)
    indicators = compute_indicators(df, cfg)
    trades = simulate_over_series(df, cfg)

    assert trades, "expected at least one trade with a loose threshold over 200 bars"

    signal_index_by_date = {date: i for i, date in enumerate(indicators.index)}
    for trade in trades:
        entry_pos = signal_index_by_date[trade.entry_date]
        signal_pos = entry_pos - 1
        assert bool(indicators["entry_signal"].iloc[signal_pos]) is True
        # The fill uses the open of the bar AFTER the signal bar, never the
        # signal bar's own close/open.
        assert trade.entry_price == pytest.approx(float(indicators["open"].iloc[entry_pos]))
        assert trade.entry_date != indicators.index[signal_pos]
