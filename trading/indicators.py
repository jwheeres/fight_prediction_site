"""Pure indicator functions: SMA, RSI, ATR.

Every function here is backward-looking only (pandas `.rolling()`, never
`.shift(-1)` or similar), which is exactly what tests/test_no_lookahead.py
checks: truncating a series to `[:t+1]` must not change the indicator value
already computed at index t.
"""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    """Simple (non-Wilder) RSI: rolling mean of gains vs rolling mean of losses.

    Wilder's smoothing is the "textbook" RSI, but at period=2 (the value this
    strategy uses) the difference from a plain rolling mean is negligible and
    the plain version is simpler to reason about and test.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 and avg_gain > 0 means RS -> inf, i.e. RSI -> 100.
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    # avg_loss == 0 and avg_gain == 0 (flat price) is conventionally RSI 50.
    result = result.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return result


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range via a simple rolling mean of true range."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()
