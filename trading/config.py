"""Configuration for the mean-reversion trading system.

Plain dataclasses, no YAML/JSON parsing — see CLAUDE.md's "boring, readable
code" preference. Every field is overridable via constructor args so tests
(e.g. the null test) can inject a synthetic universe or tighter tolerances
without touching disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_UNIVERSE = (
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "SPY", "QQQ", "IWM",
)


@dataclass(frozen=True)
class StrategyConfig:
    sma_regime: int = 200
    rsi_period: int = 2
    rsi_threshold: float = 15.0
    exit_sma: int = 5
    time_stop_bars: int = 10
    atr_period: int = 14
    atr_stop_mult: float = 3.0


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float = 0.01
    max_position_notional_pct: float = 0.20
    max_concurrent_positions: int = 5
    max_orders_per_day: int = 5
    max_drawdown_halt_pct: float = 0.20


@dataclass(frozen=True)
class PathsConfig:
    log_dir: Path = REPO_ROOT / "logs"
    portfolio_state_file: Path = REPO_ROOT / "logs" / "portfolio_state.json"


@dataclass(frozen=True)
class Config:
    universe: tuple[str, ...] = DEFAULT_UNIVERSE
    starting_equity: float = 100_000.0
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def load_config() -> Config:
    """Single entry point for runtime config.

    Returns the default Config. Kept as a function (rather than importing the
    module-level dataclass directly) so a future source of overrides — env
    vars, a config file — has one place to plug in without changing callers.
    """
    return Config()
