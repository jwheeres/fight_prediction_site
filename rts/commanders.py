"""The things that play the game.

A commander is asked for orders every few seconds and answers with a list of
macro commands plus a line of trash talk. Two implementations ship here:

  ScriptedCommander -- deterministic heuristics, no network, instant. Useful as
      a baseline, as a balance-testing opponent, and as the fallback when an
      LLM call fails mid-match.
  LLMCommander -- asks a Claude model. The call happens on a background thread
      and the simulation never waits for it: orders are applied whenever they
      arrive. A stream that freezes for four seconds every turn is unwatchable.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import config as cfg
from .orders import SCHEMA_HELP

# ANTHROPIC_BASE_URL is honoured so this works behind a gateway or proxy.
ANTHROPIC_BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
ANTHROPIC_URL = f"{ANTHROPIC_BASE}/v1/messages"
DEFAULT_MODEL = os.environ.get("RTS_MODEL", "claude-sonnet-5")


@dataclass
class Decision:
    orders: list[dict]
    talk: str = ""
    plan: str = ""
    source: str = "script"       # script | llm | fallback
    latency_ms: int = 0
    error: str = ""
    results: list[str] = field(default_factory=list)


class Commander:
    """Base class. ``request`` starts a turn, ``poll`` collects it when ready."""

    def __init__(self, name: str):
        self.name = name
        self.kind = "commander"

    def request(self, view: dict) -> None:
        raise NotImplementedError

    def poll(self) -> Decision | None:
        raise NotImplementedError

    @property
    def busy(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Scripted
# ---------------------------------------------------------------------------

PERSONALITIES = {
    "rush": {
        "target_villagers": 11,
        "attack_power": 45,
        "attack_ratio": 1.0,
        "comp": ["spearman", "spearman", "archer"],
        "barracks": 2,
        "towers": 0,
        "talk": ["No time for economy.", "Barracks first, questions later.",
                 "Pressure early, pressure always.", "Your villagers look soft."],
    },
    "boom": {
        "target_villagers": 22,
        "attack_power": 140,
        "attack_ratio": 1.5,
        "comp": ["archer", "knight", "knight"],
        "barracks": 2,
        "towers": 1,
        "talk": ["Economy compounds. Aggression doesn't.",
                 "I'll be fine in ten minutes.", "Let them poke. I'm mining gold.",
                 "Every villager is a soldier I can afford later."],
    },
    "turtle": {
        "target_villagers": 16,
        "attack_power": 165,
        "attack_ratio": 1.9,
        "comp": ["spearman", "archer", "archer"],
        "barracks": 2,
        "towers": 3,
        "talk": ["Come and get it.", "Towers don't get tired.",
                 "I win by not losing.", "That was a lot of units you just spent."],
    },
    "balanced": {
        "target_villagers": 14,
        "attack_power": 90,
        "attack_ratio": 1.25,
        "comp": ["spearman", "archer", "knight"],
        "barracks": 2,
        "towers": 1,
        "talk": ["Steady does it.", "Trading evenly is fine by me.",
                 "Nothing fancy, just enough.", "Timing is everything."],
    },
}


class ScriptedCommander(Commander):
    def __init__(self, name: str, personality: str = "balanced", seed: int = 0):
        super().__init__(name)
        self.kind = f"script:{personality}"
        self.personality = personality
        self.profile = PERSONALITIES.get(personality, PERSONALITIES["balanced"])
        self._pending: Decision | None = None
        self._comp_index = 0
        self._talk_index = seed % max(len(self.profile["talk"]), 1)

    def request(self, view: dict) -> None:
        self._pending = self.decide(view)

    def poll(self) -> Decision | None:
        decision, self._pending = self._pending, None
        return decision

    def decide(self, view: dict) -> Decision:
        you = view["you"]
        res = you["resources"]
        pop = you["population"]
        buildings = you["buildings"]
        villagers = sum(v for k, v in you["villagers"].items())
        profile = self.profile
        orders: list[dict] = []

        def have(kind: str) -> int:
            return buildings.get(kind, 0)

        def pending(kind: str) -> int:
            return buildings.get(f"{kind} (under construction)", 0)

        # 1. Never get housed. Everything else is downstream of population.
        if pop["cap"] - pop["used"] <= 2 and pop["cap"] < cfg.POP_MAX and pending("house") == 0:
            orders.append({"cmd": "build", "building": "house"})

        # 2. Economy up to the personality's target.
        if villagers < profile["target_villagers"]:
            orders.append({"cmd": "train", "unit": "villager", "count": 2})

        # 3. Military production buildings. Army size is capped by how many
        #    places you can train from, not by resources -- a bot sitting on
        #    600 wood with one barracks is losing to one with three.
        barracks = have("barracks") + pending("barracks")
        stables = have("stable") + pending("stable")
        surplus = res["wood"] > 320 and villagers >= profile["target_villagers"] * 0.7
        if barracks == 0:
            orders.append({"cmd": "build", "building": "barracks"})
        elif "knight" in profile["comp"] and stables == 0 and res["wood"] >= 175:
            orders.append({"cmd": "build", "building": "stable"})
        elif barracks < profile["barracks"] or (surplus and barracks < 4):
            orders.append({"cmd": "build", "building": "barracks"})

        towers = have("tower") + pending("tower")
        if towers < profile["towers"] and res["wood"] >= 200:
            orders.append({"cmd": "build", "building": "tower"})

        # 4. Army. Cycle the composition so we don't get hard-countered.
        if have("barracks") or have("stable"):
            unit = profile["comp"][self._comp_index % len(profile["comp"])]
            self._comp_index += 1
            if unit == "knight" and not have("stable"):
                unit = "archer"
            orders.append({"cmd": "train", "unit": unit, "count": 2})

        # 5. Keep the resource that's blocking us staffed.
        jobs = you["villagers"]
        if res["wood"] < 120 and jobs.get("wood", 0) < villagers * 0.5:
            orders.append({"cmd": "assign", "resource": "wood", "count": 2})
        elif res["gold"] < 80 and jobs.get("gold", 0) < 4 and "knight" in profile["comp"]:
            orders.append({"cmd": "assign", "resource": "gold", "count": 2})
        elif res["food"] < 120 and jobs.get("food", 0) < 3:
            orders.append({"cmd": "assign", "resource": "food", "count": 2})

        # 6. Fight or hold.
        talk = ""
        attacking = you["posture"] == "attack"
        # Two cautious bots will otherwise wait each other out to the clock.
        # Past the halfway mark, everybody gets braver.
        patience = 1.0 if view["tick"] < cfg.MATCH_TICK_LIMIT * 0.4 else 0.7
        power, enemy_power = you["army_power"], view["enemy"]["army_power"]
        if attacking and you["base_under_attack"]:
            orders.append({"cmd": "defend"})
            talk = "Back home, they're in my base."
        elif attacking and power < profile["attack_power"] * 0.45:
            # The push died. Without this the bot feeds new units in one at a
            # time forever, which is how you lose a game you were winning.
            orders.append({"cmd": "defend"})
            talk = "Push failed. Regrouping."
        elif attacking and enemy_power > power * 2.2:
            orders.append({"cmd": "defend"})
            talk = "That's more than I want to fight."
        elif not attacking and power >= profile["attack_power"] * patience \
                and power >= enemy_power * (profile["attack_ratio"] * patience):
            orders.append({"cmd": "attack"})
            talk = profile["talk"][self._talk_index % len(profile["talk"])]
            self._talk_index += 1

        return Decision(orders=orders[:cfg.MAX_ORDERS_PER_TURN], talk=talk,
                        plan=f"{self.personality} heuristics", source="script")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _costs_table() -> str:
    lines = ["UNITS (cost / train time / hp / attack / range):"]
    for kind, s in cfg.UNITS.items():
        cost = ", ".join(f"{v} {k}" for k, v in s["cost"].items())
        lines.append(f"  {kind:9} {cost:24} {s['train_ticks']//cfg.TICKS_PER_SECOND}s  "
                     f"hp {s['hp']:3}  atk {s['attack']:2}  range {s['range']}  "
                     f"pop {s['pop']}  built at {s['trained_at']}")
    lines.append("BUILDINGS (cost / build time):")
    for kind, s in cfg.BUILDINGS.items():
        cost = ", ".join(f"{v} {k}" for k, v in s["cost"].items())
        extra = []
        if s["pop_provided"]:
            extra.append(f"+{s['pop_provided']} pop")
        if s["attack"]:
            extra.append(f"shoots for {s['attack']}")
        if s["dropoff"]:
            extra.append("resource drop-off")
        lines.append(f"  {kind:12} {cost:22} {s['build_ticks']//cfg.TICKS_PER_SECOND}s  "
                     f"{', '.join(extra)}")
    lines.append("COUNTERS (damage multipliers):")
    for atk, table in cfg.ATTACK_BONUS.items():
        for target, mult in table.items():
            lines.append(f"  {atk} deals {mult}x to {target}")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are an AI commander playing a real-time strategy game \
live on stream. You control one team. Your opponent is another AI.

HOW THE GAME WORKS
You do not control individual units. You issue macro orders every few seconds \
and the game handles movement, gathering and fighting for you. Villagers gather \
food, wood and gold and construct buildings. Military units defend your base \
until you order an attack.

You lose when you have no town centre and no military units left. If the clock \
runs out, the higher score wins (score counts resources gathered, army and \
buildings owned, and kills).

{costs}

POPULATION
Each town centre and house gives +5 population. Running into the population cap \
is the single most common way to lose — build houses before you need them.

YOUR ORDERS
Reply with JSON only. No prose outside the JSON, no markdown fences.

{{"plan": "one short sentence on what you're doing and why",
  "talk": "one line of commentary for the audience, in character",
  "orders": [ ... up to {max_orders} orders ... ]}}

Valid orders:
{schema}

The result of every order you issued last turn comes back to you in \
"result_of_your_last_orders". Read it. If something was rejected it tells you \
exactly what was missing.

STYLE
You are {persona}
Keep "talk" to one short punchy sentence. You are being watched."""

DEFAULT_PERSONA = ("a confident, slightly smug commander who explains their "
                   "thinking in as few words as possible.")


def _extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in fences or a sentence. Dig it out."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, depth = text.find("{"), 0
    if start == -1:
        raise ValueError("no JSON object in response")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object in response")


class LLMCommander(Commander):
    """Asks a Claude model for orders, on a background thread."""

    def __init__(self, name: str, persona: str = DEFAULT_PERSONA,
                 model: str = DEFAULT_MODEL, api_key: str | None = None,
                 fallback: Commander | None = None, timeout: float = 25.0,
                 max_calls: int | None = None):
        super().__init__(name)
        self.kind = f"llm:{model}"
        self.model = model
        self.persona = persona
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.fallback = fallback or ScriptedCommander(name, "balanced")
        self.max_calls = max_calls
        self.calls = 0
        self.system = SYSTEM_PROMPT.format(
            costs=_costs_table(), schema=SCHEMA_HELP,
            max_orders=cfg.MAX_ORDERS_PER_TURN, persona=persona)
        self._lock = threading.Lock()
        self._result: Decision | None = None
        self._thread: threading.Thread | None = None
        self._history: list[dict] = []

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def request(self, view: dict) -> None:
        if self.busy:
            return  # still thinking about the last turn; skip this one
        if not self.api_key or (self.max_calls is not None and self.calls >= self.max_calls):
            self.fallback.request(view)
            decision = self.fallback.poll() or Decision(orders=[])
            decision.source = "fallback"
            decision.error = ("no ANTHROPIC_API_KEY set" if not self.api_key
                              else f"call budget of {self.max_calls} spent")
            with self._lock:
                self._result = decision
            return
        self.calls += 1
        self._thread = threading.Thread(target=self._think, args=(view,), daemon=True)
        self._thread.start()

    def poll(self) -> Decision | None:
        with self._lock:
            decision, self._result = self._result, None
        return decision

    # -- internals ----------------------------------------------------------

    def _think(self, view: dict) -> None:
        started = time.time()
        try:
            text = self._call_api(json.dumps(view, indent=1))
            parsed = _extract_json(text)
            orders = parsed.get("orders") or []
            if not isinstance(orders, list):
                orders = []
            decision = Decision(
                orders=orders[:cfg.MAX_ORDERS_PER_TURN],
                talk=str(parsed.get("talk", ""))[:200],
                plan=str(parsed.get("plan", ""))[:300],
                source="llm",
                latency_ms=int((time.time() - started) * 1000),
            )
            # Keep a short rolling memory so the commander has continuity.
            self._history.append({"role": "assistant", "content": json.dumps(
                {"plan": decision.plan, "orders": decision.orders})})
            self._history = self._history[-6:]
        except Exception as exc:  # noqa: BLE001 - a dead agent must not kill the match
            self.fallback.request(view)
            decision = self.fallback.poll() or Decision(orders=[])
            decision.source = "fallback"
            decision.error = f"{type(exc).__name__}: {exc}"[:200]
            decision.latency_ms = int((time.time() - started) * 1000)
        with self._lock:
            self._result = decision

    def _call_api(self, user_content: str) -> str:
        messages = self._history + [{"role": "user", "content": user_content}]
        payload = json.dumps({
            "model": self.model,
            "max_tokens": 800,
            "system": self.system,
            "messages": messages,
        }).encode()
        request = urllib.request.Request(
            ANTHROPIC_URL, data=payload, method="POST",
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            })
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read()[:200].decode(errors='replace')}") from exc
        return "".join(block.get("text", "") for block in body.get("content", []))


def make_commander(spec: str, name: str, seed: int = 0,
                   persona: str | None = None, model: str = DEFAULT_MODEL,
                   max_calls: int | None = None) -> Commander:
    """Build a commander from a CLI-friendly spec string.

    ``rush`` / ``boom`` / ``turtle`` / ``balanced``  -> scripted
    ``llm`` or ``llm:<persona words>``               -> Claude
    """
    spec = (spec or "balanced").strip()
    if spec.startswith("llm"):
        _, _, inline_persona = spec.partition(":")
        return LLMCommander(
            name,
            persona=persona or inline_persona.strip() or DEFAULT_PERSONA,
            model=model,
            fallback=ScriptedCommander(name, "balanced", seed),
            max_calls=max_calls,
        )
    return ScriptedCommander(name, spec, seed)
