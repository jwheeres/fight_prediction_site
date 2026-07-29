"""Paper-mode daily runner.

Logs every signal, every intended order, and every skipped signal. Sends
nothing anywhere: there is no broker/MCP import in this module or anything
it calls. Run it once per day, same cadence CLAUDE.md describes for the
eventual live "Execute trades + log" role — minus the actual execution.

    python -m trading.paper_runner
"""

from __future__ import annotations

from trading import data, risk, sizing, strategy
from trading.config import load_config
from trading.logging_utils import RunLogger
from trading.portfolio import Position, load_portfolio_state, save_portfolio_state


def run() -> None:
    cfg = load_config()

    ohlcv = data.fetch_daily_ohlcv(cfg.universe, lookback_days=cfg.strategy.sma_regime + 60)
    if not ohlcv:
        print("No market data fetched; aborting run without touching portfolio state.")
        return

    run_date = str(max(df.index[-1] for df in ohlcv.values()).date())
    logger = RunLogger(cfg.paths.log_dir, run_date=run_date)

    context = data.fetch_market_context(cfg.strategy.sma_regime)
    logger.log_market_context(context)

    reconciled, reconcile_reason = risk.reconcile_positions()
    logger.log_note(f"reconcile_positions: allowed={reconciled} reason={reconcile_reason}")

    state = load_portfolio_state(cfg.paths.portfolio_state_file, cfg.starting_equity)
    if state.last_run_date != run_date:
        state.orders_today = 0

    last_prices = {symbol: float(df["close"].iloc[-1]) for symbol, df in ohlcv.items()}
    current_equity = state.equity(last_prices)
    state.peak_equity = max(state.peak_equity, current_equity)

    drawdown_ok, drawdown_reason = risk.check_drawdown_halt(
        current_equity, state.peak_equity, cfg.risk
    )
    if not drawdown_ok:
        logger.log_note(f"TRADING HALTED: {drawdown_reason}")

    intended_buys = 0
    intended_sells = 0
    skipped = 0

    for symbol in cfg.universe:
        df = ohlcv.get(symbol)
        if df is None or len(df) < cfg.strategy.sma_regime:
            logger.log_skipped_signal(symbol, "insufficient price history")
            skipped += 1
            continue

        signals = strategy.evaluate_signals(df, cfg.strategy)
        snapshot = strategy.snapshot_at(signals, -1)
        logger.log_signal(symbol, snapshot)

        position = state.positions.get(symbol)

        if position is not None:
            # Exits are always evaluated, even while trading is halted:
            # halting fails safe by blocking new entries, never by blocking
            # a position from closing.
            should_exit, exit_reason = strategy.exit_decision(
                close=snapshot.close,
                sma_exit=snapshot.sma_exit,
                entry_price=position.entry_price,
                entry_atr=position.entry_atr,
                bars_held=position.bars_held,
                cfg=cfg.strategy,
            )
            if should_exit:
                logger.log_intended_order(
                    symbol, "SELL", position.shares, snapshot.close,
                    allowed=True, reason=exit_reason,
                )
                intended_sells += 1
                state.cash += position.shares * snapshot.close
                del state.positions[symbol]
            else:
                position.bars_held += 1
            continue

        if not bool(signals["entry_signal"].iloc[-1]):
            continue

        if not drawdown_ok:
            logger.log_skipped_signal(symbol, f"trading halted: {drawdown_reason}")
            skipped += 1
            continue

        order_cap_ok, order_cap_reason = risk.check_order_cap(state.orders_today, cfg.risk)
        position_cap_ok, position_cap_reason = risk.check_position_cap(
            len(state.positions), cfg.risk
        )
        if not order_cap_ok or not position_cap_ok:
            reason = order_cap_reason or position_cap_reason
            logger.log_skipped_signal(symbol, reason)
            skipped += 1
            continue

        shares = sizing.position_size(current_equity, snapshot.atr, snapshot.close, cfg.risk)
        if shares <= 0:
            logger.log_skipped_signal(symbol, "sizing produced zero shares")
            skipped += 1
            continue

        notional = shares * snapshot.close
        notional_ok, notional_reason = risk.check_notional_cap(notional, current_equity, cfg.risk)
        if not notional_ok:
            logger.log_skipped_signal(symbol, notional_reason)
            skipped += 1
            continue

        logger.log_intended_order(
            symbol, "BUY", shares, snapshot.close, allowed=True, reason="entry_signal"
        )
        intended_buys += 1
        state.cash -= notional
        state.orders_today += 1
        state.positions[symbol] = Position(
            symbol=symbol,
            shares=shares,
            entry_price=snapshot.close,
            entry_atr=snapshot.atr,
            entry_date=run_date,
            bars_held=0,
        )

    state.last_run_date = run_date
    save_portfolio_state(state, cfg.paths.portfolio_state_file)

    print(f"Paper run {run_date}: equity=${current_equity:,.2f} "
          f"buys={intended_buys} sells={intended_sells} skipped={skipped} "
          f"open_positions={len(state.positions)}")
    print(f"Log: {logger.path}")


if __name__ == "__main__":
    run()
