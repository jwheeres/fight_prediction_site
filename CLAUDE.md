# Mean-Reversion Trading System — Project Context

## What this is

A systematic swing-trading system for sector ETFs, executing via Robinhood's
Trading MCP. Personal experiment. The goal is to learn how a real system behaves
in live markets, not to generate income.

## Architecture — read this before proposing changes

**No LLM in the execution loop.** The live trader is plain, deterministic Python.
It speaks MCP to Robinhood directly (MCP is JSON-RPC over HTTP with a tool schema
— it does not require a model on the client side).

Three separate roles, deliberately not merged:

| Role | Tool | Cadence |
|---|---|---|
| Write/revise the strategy code | Claude Code | On demand |
| Execute trades + log | Plain Python on a VPS, cron | Daily |
| Review logs, propose changes | Claude Code, headless | Weekly |

Any supervisor layer gets **kill authority only** — it may halt trading, never
initiate. Halting fails safe; initiating fails dangerous.

## Universe

XLF, XLE, XLK, XLV, XLI, XLP, XLY, XLU, SPY, QQQ, IWM

Sector/index ETFs specifically — chosen because they have **no earnings dates**
(deleting the single largest blowup risk), no single-name disaster risk, no
delistings or corporate actions, and tight spreads.

Known tradeoff, accepted: these are *more* correlated with each other than a
basket of individual names would be. They go oversold together and fall together.
This is why position limits matter more than they would otherwise.

## Strategy rules

**Entry** — both conditions, evaluated on the daily close:
- Close > 200-day SMA (regime filter — NOT OPTIONAL; without it this is just
  buying things on their way down)
- RSI(2) < threshold (default 15; see Open Questions)

**Exit** — first to trigger:
- Close > 5-day SMA (primary, typically 2–6 days)
- 10-bar time stop
- Disaster stop: close < entry − (3 × ATR(14) at entry)

**Execution timing:** signals fire on the close of day T, orders fill at the open
of day T+1. Never evaluate a fill on the same bar that produced the signal.

**Stops are evaluated on the CLOSE only.** Never place a resting stop order in
the market. Two reasons: intraday wicks cannot trigger a close-based rule, and a
resting stop becomes a market order during a flush and fills at a terrible price.
The stop condition lives in code; the order is sent only when the condition is met.

## Position sizing and limits

- Risk 1% of equity per position (risk = loss if the stop hits, NOT capital deployed)
- Shares = (equity × 0.01) / (3 × ATR)
- Cap any single position at 20% of equity notional — without this, low-ATR
  periods produce absurd leverage
- Max 5 concurrent positions
- Sizing is deliberately **deferred** — do not tune it until 100+ live trades
  show a real drawdown profile

## Kill switches — build these before the strategy logic

- Halt all trading at a defined account drawdown
- Cap orders per day (catches runaway loops)
- Reject any order that would exceed position count or notional caps
- Reconcile local position state against Robinhood before every session; halt on
  mismatch rather than guessing

## Logging requirements

Improvement is impossible without these. Log every one:

- **Intended price vs actual fill**, every order — this is the real slippage
  number and the one thing a backtest fundamentally cannot tell you
- Signals generated but NOT taken, and why (no slot / no capital)
- Full indicator state at signal time (RSI, both SMAs, ATR, close)
- Market context: VIX, SPY vs its own 200-day — enables slicing by regime
- Every order request and the raw MCP response

## Testing requirements

Non-negotiable for anything touching order logic:

1. **Null test.** Run the strategy against synthetic random-walk data.
   Expectancy must come out near zero. If random data shows real edge, there is
   lookahead bias somewhere. (A prior run of this engine on a random walk gave a
   60.5% win rate and −18.4% total return — a useful reminder that win rate alone
   means nothing.)
2. **No lookahead.** Assert that no fill price is drawn from the same bar as its
   signal.
3. **Paper/dry-run mode** that logs intended orders without sending them. Default
   to this; live trading requires an explicit flag.

## Judging results

- **Expectancy, not win rate.** Mean reversion has a fat left tail: steady small
  wins, occasional clustered losses. A 70% win rate can accompany a losing system.
- Compare against buy-and-hold SPY. If it doesn't beat that risk-adjusted, the
  honest conclusion is to own the index.
- Under ~30 trades/year, skill and luck are indistinguishable regardless of results.
- Count every parameter variation tried. Selecting the best of N trials inflates
  apparent performance (see: False Strategy Theorem, Deflated Sharpe Ratio).
  Prefer a parameter where *neighbouring* values also work over an isolated spike.

## Open questions

- RSI entry threshold: test 5/10/15/20/25. Empirical, not something to reason out.
  Pick for robustness across neighbours, not peak backtest score.

## Explicitly out of scope

- Individual stocks (earnings gap risk)
- Ag futures / commodity futures (Robinhood does not offer them)
- Intraday or day trading (no latency edge vs HFT; spread costs scale with crossings)
- Any design where a language model decides trades

## Working style for this repo

- Plan before writing code for anything non-trivial
- One change at a time; state the mechanistic reason, not "the numbers improved"
- Prefer boring, readable code over clever code — this handles money
- Always show what a change does to the null test and the paper-mode output
