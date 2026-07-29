"""Entry/exit rules and a minimal single-symbol backtest loop.

Execution timing (CLAUDE.md, non-negotiable): a signal is computed on the
close of day T; any resulting order is dated for the **open of day T+1**.
Nothing in this module ever reads a fill price from the same bar that
produced the signal — that's what tests/test_no_lookahead.py checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading.config import StrategyConfig
from trading.indicators import atr as atr_indicator
from trading.indicators import rsi as rsi_indicator
from trading.indicators import sma


@dataclass(frozen=True)
class IndicatorSnapshot:
    date: pd.Timestamp
    close: float
    sma_regime: float
    sma_exit: float
    rsi: float
    atr: float


@dataclass(frozen=True)
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    bars_held: int
    exit_reason: str

    @property
    def pnl_pct(self) -> float:
        return (self.exit_price - self.entry_price) / self.entry_price


def compute_indicators(ohlcv: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Return ohlcv with sma_regime, sma_exit, rsi, atr, entry_signal columns.

    Expects columns: open, high, low, close. Purely vectorized/backward-looking.
    """
    out = ohlcv.copy()
    out["sma_regime"] = sma(out["close"], cfg.sma_regime)
    out["sma_exit"] = sma(out["close"], cfg.exit_sma)
    out["rsi"] = rsi_indicator(out["close"], cfg.rsi_period)
    out["atr"] = atr_indicator(out["high"], out["low"], out["close"], cfg.atr_period)

    warmed_up = out[["sma_regime", "rsi", "atr"]].notna().all(axis=1)
    out["entry_signal"] = warmed_up & (out["close"] > out["sma_regime"]) & (
        out["rsi"] < cfg.rsi_threshold
    )
    return out


def evaluate_signals(ohlcv: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Public entry point used by paper_runner: indicator-augmented frame."""
    return compute_indicators(ohlcv, cfg)


def snapshot_at(df_with_indicators: pd.DataFrame, idx: int) -> IndicatorSnapshot:
    row = df_with_indicators.iloc[idx]
    return IndicatorSnapshot(
        date=df_with_indicators.index[idx],
        close=float(row["close"]),
        sma_regime=float(row["sma_regime"]) if pd.notna(row["sma_regime"]) else float("nan"),
        sma_exit=float(row["sma_exit"]) if pd.notna(row["sma_exit"]) else float("nan"),
        rsi=float(row["rsi"]) if pd.notna(row["rsi"]) else float("nan"),
        atr=float(row["atr"]) if pd.notna(row["atr"]) else float("nan"),
    )


def exit_decision(
    close: float,
    sma_exit: float,
    entry_price: float,
    entry_atr: float,
    bars_held: int,
    cfg: StrategyConfig,
) -> tuple[bool, str]:
    """Should an open position exit, evaluated on today's close?

    CLAUDE.md lists three exit conditions without specifying a tie-break
    priority for same-day ties. This checks the disaster stop first since it
    exists purely for capital protection — on a day where both a "healthy"
    exit and the disaster stop would trigger, protecting capital wins.
    """
    disaster_level = entry_price - (cfg.atr_stop_mult * entry_atr)
    if close < disaster_level:
        return True, "disaster_stop"
    if close > sma_exit:
        return True, "primary_sma_exit"
    if bars_held >= cfg.time_stop_bars:
        return True, "time_stop"
    return False, ""


def simulate_over_series(ohlcv: pd.DataFrame, cfg: StrategyConfig) -> list[Trade]:
    """Minimal single-symbol backtest: at most one open position at a time.

    Used only by tests (the null test and no-lookahead test) — not a general
    backtest framework. Fills use the open of the bar *after* the signal bar,
    per the execution-timing rule.
    """
    df = compute_indicators(ohlcv, cfg)
    trades: list[Trade] = []

    in_position = False
    entry_fill_idx = -1
    entry_price = float("nan")
    entry_atr = float("nan")

    n = len(df)
    for i in range(n):
        if in_position:
            bars_held = i - entry_fill_idx
            close = df["close"].iloc[i]
            sma_exit = df["sma_exit"].iloc[i]
            should_exit, reason = exit_decision(
                close=close,
                sma_exit=sma_exit,
                entry_price=entry_price,
                entry_atr=entry_atr,
                bars_held=bars_held,
                cfg=cfg,
            )
            if should_exit and i + 1 < n:
                exit_price = float(df["open"].iloc[i + 1])
                trades.append(
                    Trade(
                        entry_date=df.index[entry_fill_idx],
                        exit_date=df.index[i + 1],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        bars_held=bars_held,
                        exit_reason=reason,
                    )
                )
                in_position = False
            continue

        if bool(df["entry_signal"].iloc[i]) and i + 1 < n:
            # Signal fires on the close of bar i; the fill (and the trade's
            # "entry") happens on the open of bar i+1 — never the same bar.
            entry_fill_idx = i + 1
            entry_price = float(df["open"].iloc[entry_fill_idx])
            entry_atr = float(df["atr"].iloc[i])
            in_position = True

    return trades
