"""JSONL logging for signals, intended orders, and skipped signals.

Matches CLAUDE.md's logging requirements: full indicator state at signal
time, signals generated but not taken (and why), and market context on every
run. One append-only file per day under PathsConfig.log_dir.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, log_dir: Path, run_date: str | None = None):
        self.log_dir = log_dir
        self.run_date = run_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"paper_{self.run_date}.jsonl"

    def _write(self, record_type: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": record_type,
            **payload,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_market_context(self, context: dict[str, Any]) -> None:
        self._write("market_context", context)

    def log_signal(self, symbol: str, snapshot: Any) -> None:
        payload = {"symbol": symbol}
        payload.update(asdict(snapshot) if hasattr(snapshot, "__dataclass_fields__") else snapshot)
        self._write("signal", payload)

    def log_intended_order(
        self,
        symbol: str,
        side: str,
        shares: int,
        intended_price: float,
        allowed: bool,
        reason: str = "",
    ) -> None:
        self._write(
            "intended_order",
            {
                "symbol": symbol,
                "side": side,
                "shares": shares,
                "intended_price": intended_price,
                "allowed": allowed,
                "reason": reason,
            },
        )

    def log_skipped_signal(self, symbol: str, reason: str) -> None:
        self._write("skipped_signal", {"symbol": symbol, "reason": reason})

    def log_note(self, message: str) -> None:
        self._write("note", {"message": message})
