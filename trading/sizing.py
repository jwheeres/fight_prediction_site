"""Position sizing.

Deliberately simple per CLAUDE.md: "Sizing is deliberately deferred — do not
tune it until 100+ live trades show a real drawdown profile." This is the
one formula the spec gives; no additional cleverness belongs here.
"""

from __future__ import annotations

import math

from trading.config import RiskConfig


def position_size(equity: float, atr: float, price: float, risk_cfg: RiskConfig) -> int:
    """Shares to buy, given equity, ATR at entry, and current price.

    shares = (equity * risk_pct) / (atr_stop_mult * atr), capped so no single
    position exceeds max_position_notional_pct of equity. Returns 0 (rather
    than raising) for degenerate inputs — a paper run should log a
    zero-size/skip, not crash on a bad ATR read.
    """
    if equity <= 0 or price <= 0 or atr <= 0:
        return 0

    risk_dollars = equity * risk_cfg.risk_per_trade_pct
    shares_by_risk = risk_dollars / (risk_cfg.atr_stop_mult * atr)

    max_notional = equity * risk_cfg.max_position_notional_pct
    shares_by_notional = max_notional / price

    shares = min(shares_by_risk, shares_by_notional)
    return max(0, math.floor(shares))
