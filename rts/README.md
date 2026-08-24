# AI vs AI — a small RTS you watch instead of play

Two AI commanders fight over an Age-of-Empires-shaped map: gather food, wood
and gold, build houses and barracks, train spearmen, archers and knights, and
try to knock the other one's town centre down. You do not play. You watch it in
a browser, with each agent's reasoning and its last set of orders on screen next
to the map.

A commander is either a scripted bot (deterministic, no network) or a Claude
model. They are interchangeable, so you can run `llm` vs `llm`, `llm` vs a bot,
or bot vs bot as a balance test.

## Run it

```bash
python -m rts serve                         # two scripted bots, opens on :8765
python -m rts serve --red llm --blue turtle # Claude against a heuristic bot
python -m rts serve --red llm --blue llm    # the one you actually want
```

Then open <http://127.0.0.1:8765>. Pause, change speed (0.5×–8×) and start a
new match from the header. When a match ends a new one begins after 12 seconds,
so it will happily run all day.

LLM commanders need `ANTHROPIC_API_KEY`. Without one they say so on screen and
fall back to the scripted bot — the match never stops because an agent failed.

```bash
export ANTHROPIC_API_KEY=sk-...
python -m rts serve --red llm --blue llm \
  --red-persona "a paranoid general who trusts nothing and turtles" \
  --blue-persona "an over-caffeinated aggressor who cannot sit still"
```

Useful flags: `--model` (default `claude-sonnet-5`), `--think-interval`
(ticks between agent turns; 10 ticks = 1 second, default 50), `--max-calls`
(cap API calls per agent per match, then fall back to scripted),
`--speed`, `--seed`, `--port`, `--no-autorestart`.

**Cost.** Each agent turn is one API call. At the default 5-second interval a
10-minute match is roughly 120 calls per agent. Raise `--think-interval` or set
`--max-calls` before leaving it streaming overnight.

## What the agent actually does

Commanders issue *macro* orders only — six verbs, at most six per turn. Unit
movement, gathering, targeting and combat are handled by the engine. An agent
that had to steer forty units individually would spend its entire turn on
bookkeeping instead of on strategy.

```
train   {"cmd":"train","unit":"villager|spearman|archer|knight","count":1-10}
build   {"cmd":"build","building":"house|barracks|stable|tower|town_center"}
assign  {"cmd":"assign","resource":"food|wood|gold","count":1-20}
attack  {"cmd":"attack"}            send the whole army at the enemy base
defend  {"cmd":"defend"}            pull the army home
say     {"cmd":"say","text":"..."}  trash talk for the stream
```

Every turn it gets a situation report of about 2 KB: its own economy in detail,
a scouting-level summary of the enemy, and — the part that makes it improve
within a match — **the result of every order it gave last turn**:

```json
"result_of_your_last_orders": [
  "training 2x spearman",
  "rejected build stable: need 50 gold",
  "rejected train knight: population capped at 25 — build houses"
]
```

Rejections explain exactly what was missing, so an agent that asks for something
it cannot afford finds out why and can react. That feedback loop matters far
more than the size of the vocabulary.

The system prompt's cost and counter tables are **generated from
`config.py`**, so re-balancing the game cannot leave the agent playing a
version that no longer exists.

## Rules

| | cost | notes |
|---|---|---|
| villager | 50 food | gathers, builds |
| spearman | 35 food 25 wood | 2.5× vs knights |
| archer | 35 wood 25 gold | 1.6× vs spearmen, range 4.5 |
| knight | 60 food 45 gold | 1.6× vs archers, fast, 2 pop |
| house | 30 wood | +5 population |
| barracks | 150 wood | trains spearmen and archers |
| stable | 175 wood 50 gold | trains knights |
| tower | 125 wood 40 gold | shoots for 10 |
| town centre | 275 wood | +5 pop, resource drop-off, 1500 hp |

Military units deal **double damage to buildings**, which is what makes
attacking viable at all. You lose when you have no town centre and no military
units left; if the twenty-minute clock runs out, the higher score wins.

Two mechanics exist specifically because their absence produced bad games:

- **Attack waves are committed.** `attack` sends the units you have *now*.
  Units trained afterwards stay home. Without this a bot feeds reinforcements
  across the map one at a time and loses a game it was winning.
- **Waves move together.** A unit will not get more than 5 tiles ahead of the
  median of its own wave, so knights don't arrive alone and die before the
  spearmen turn up.

## Headless mode

No server, no UI, as fast as the CPU allows — this is how the balance above
was checked.

```bash
python -m rts headless --red rush --blue balanced --matches 20
python -m rts headless --red boom --blue turtle --matches 50 --json > results.jsonl
```

Same seed, same match, every time. Current scripted bot win rates over 120
matches (every ordered pairing, 10 seeds each):

| bot | wins | style |
|---|---|---|
| balanced | 83% | barracks then stable, attacks at a 1.25× power edge |
| rush | 53% | 11 villagers, two barracks, attacks whenever it is even |
| boom | 38% | 22 villagers, knights, attacks at a 1.5× edge |
| turtle | 25% | towers, waits for a 1.9× edge |

Median match: 4 minutes. 8% reach the clock. Read that table as *no strategy is
dominant or hopeless*, not as a claim about which is objectively best — it is
four hand-written heuristics on one map layout.

## Design notes and limitations

- **The simulation never waits for an agent.** LLM calls run on a background
  thread; orders are applied whenever they arrive, which may be several seconds
  and a few hundred ticks later. A stream that freezes every turn is unwatchable.
  A commander that is still thinking is simply not asked again.
- **Deterministic.** All randomness goes through one seeded RNG and entities are
  iterated in id order, so a seed reproduces a match exactly. That is the only
  reason headless balance testing is worth anything.
- **No pathfinding.** Units move in straight lines; buildings do not block
  movement. It reads fine at this zoom level and keeps the loop honest.
- **No fog of war.** The enemy section of the report is deliberately coarse
  (counts and a rough position, no exact coordinates) rather than hidden.
- **The map is mirror-symmetric.** Resources are scattered on one half and
  mirrored, so the seed does not decide the match. There's a test for it.
- **Cost of the sim** is O(units²) per tick for target selection. Fine at 1×
  with two 80-pop armies; 8× on a full late-game map will start to lag.
- Snapshots are ~7 KB at 10 Hz per spectator. Local viewing is the design point,
  not a thousand concurrent viewers.

## Files

```
config.py      all balance numbers, one file
engine.py      the deterministic simulation
orders.py      the six-verb command vocabulary and its validation
view.py        the situation report an agent reasons over
commanders.py  scripted bots + the Claude commander (async, with fallback)
match.py       world + commanders + snapshots for the browser
server.py      stdlib HTTP server, SSE stream, pause/speed/restart
static/        the spectator page (one file, no dependencies)
__main__.py    serve / headless
```

No third-party dependencies — stdlib Python and one HTML file. Tests live in
`tests/test_rts_engine.py` and `tests/test_rts_agents.py`; none of them need
network access.
