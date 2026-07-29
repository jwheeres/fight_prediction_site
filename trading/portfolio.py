"""Paper portfolio state.

This is *paper* state only — a local record of what a live account would
look like if orders had actually been sent. It is explicitly not reconciled
against the broker (see risk.reconcile_positions); once the MCP connection
exists, this file is where that reconciliation would plug in.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Position:
    symbol: str
    shares: int
    entry_price: float
    entry_atr: float
    entry_date: str  # ISO date string; JSON-friendly
    bars_held: int = 0


@dataclass
class PortfolioState:
    cash: float
    peak_equity: float
    positions: dict[str, Position] = field(default_factory=dict)
    orders_today: int = 0
    last_run_date: str | None = None

    def equity(self, last_prices: dict[str, float]) -> float:
        holdings_value = sum(
            pos.shares * last_prices.get(pos.symbol, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.cash + holdings_value


def new_portfolio_state(starting_equity: float) -> PortfolioState:
    return PortfolioState(cash=starting_equity, peak_equity=starting_equity)


def load_portfolio_state(path: Path, starting_equity: float) -> PortfolioState:
    if not path.exists():
        return new_portfolio_state(starting_equity)

    raw = json.loads(path.read_text())
    positions = {
        symbol: Position(**pos_dict) for symbol, pos_dict in raw.get("positions", {}).items()
    }
    return PortfolioState(
        cash=raw["cash"],
        peak_equity=raw["peak_equity"],
        positions=positions,
        orders_today=raw.get("orders_today", 0),
        last_run_date=raw.get("last_run_date"),
    )


def save_portfolio_state(state: PortfolioState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    path.write_text(json.dumps(payload, indent=2, default=str))
