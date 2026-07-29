"""Kill-switch checks.

Per CLAUDE.md: "build these before the strategy logic" and "Any supervisor
layer gets kill authority only — it may halt trading, never initiate."
Every check here can only turn an order OFF; none of them can create one.

Each check returns (allowed, reason) so paper_runner can log *why* an
intended order was rejected, matching the "signals generated but NOT taken,
and why" logging requirement.
"""

from __future__ import annotations

from trading.config import RiskConfig


def check_drawdown_halt(
    current_equity: float, peak_equity: float, risk_cfg: RiskConfig
) -> tuple[bool, str]:
    if peak_equity <= 0:
        return True, ""
    drawdown = (peak_equity - current_equity) / peak_equity
    if drawdown >= risk_cfg.max_drawdown_halt_pct:
        return False, (
            f"drawdown {drawdown:.1%} >= halt threshold "
            f"{risk_cfg.max_drawdown_halt_pct:.1%}"
        )
    return True, ""


def check_order_cap(orders_placed_today: int, risk_cfg: RiskConfig) -> tuple[bool, str]:
    if orders_placed_today >= risk_cfg.max_orders_per_day:
        return False, (
            f"orders today {orders_placed_today} >= daily cap "
            f"{risk_cfg.max_orders_per_day}"
        )
    return True, ""


def check_position_cap(
    current_position_count: int, risk_cfg: RiskConfig
) -> tuple[bool, str]:
    if current_position_count >= risk_cfg.max_concurrent_positions:
        return False, (
            f"open positions {current_position_count} >= max "
            f"{risk_cfg.max_concurrent_positions}"
        )
    return True, ""


def check_notional_cap(
    order_notional: float, equity: float, risk_cfg: RiskConfig
) -> tuple[bool, str]:
    if equity <= 0:
        return False, "equity <= 0"
    max_notional = equity * risk_cfg.max_position_notional_pct
    if order_notional > max_notional:
        return False, (
            f"order notional {order_notional:.2f} > cap {max_notional:.2f} "
            f"({risk_cfg.max_position_notional_pct:.0%} of equity)"
        )
    return True, ""


def reconcile_positions() -> tuple[bool, str]:
    """Reconcile local paper state against the broker before each session.

    CLAUDE.md requires this against real Robinhood positions, which needs the
    MCP connection this phase explicitly excludes. Rather than silently
    no-op, this returns an explicit "not reconciled" result so paper_runner
    logs the gap instead of hiding it. Replace with a real MCP-backed check
    when the broker connection is wired up.
    """
    return False, "reconciliation skipped — no MCP connection yet"
