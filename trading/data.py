"""Market data loading via yfinance.

Isolated in one module deliberately: this is the only place that knows where
price data comes from. Swapping the source later (a different vendor, or
eventually quotes pulled through the Robinhood MCP connection) should only
touch this file.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


def fetch_daily_ohlcv(symbols: tuple[str, ...], lookback_days: int = 400) -> dict[str, pd.DataFrame]:
    """One DataFrame per symbol, columns: open, high, low, close, volume.

    `lookback_days` defaults to 400 calendar days so a 200-day SMA has room
    to warm up. Symbols that fail to download are silently omitted; callers
    should check for missing keys rather than assume every symbol returned.
    """
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        history = yf.Ticker(symbol).history(period=f"{lookback_days}d", auto_adjust=False)
        if history.empty:
            continue
        history = history.rename(columns=_COLUMN_MAP)
        result[symbol] = history[["open", "high", "low", "close", "volume"]]
    return result


def fetch_market_context(sma_regime: int = 200) -> dict:
    """VIX level and whether SPY is above its own regime SMA."""
    spy_history = fetch_daily_ohlcv(("SPY",), lookback_days=sma_regime + 50).get("SPY")
    vix_history = fetch_daily_ohlcv(("^VIX",), lookback_days=5).get("^VIX")

    spy_above_200sma = None
    if spy_history is not None and len(spy_history) >= sma_regime:
        spy_sma = spy_history["close"].rolling(window=sma_regime).mean().iloc[-1]
        spy_above_200sma = bool(spy_history["close"].iloc[-1] > spy_sma)

    vix = None
    if vix_history is not None and not vix_history.empty:
        vix = float(vix_history["close"].iloc[-1])

    return {"vix": vix, "spy_above_200sma": spy_above_200sma}
