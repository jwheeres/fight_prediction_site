"""Wires a World to two commanders and produces snapshots for the spectator.

The important detail is that thinking is asynchronous. Every
``think_interval`` ticks each commander is *asked* for orders; its answer is
applied whenever it turns up, which for an LLM might be several seconds and a
few hundred ticks later. The simulation never blocks on an agent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import config as cfg
from .commanders import Commander, Decision
from .engine import World
from .orders import apply_orders
from .view import build_view

TEAM_COLORS = ("#ff4d4d", "#4da6ff")
KIND_INDEX = ["villager", "spearman", "archer", "knight",
              "town_center", "house", "barracks", "stable", "tower"]
NODE_INDEX = ["food", "wood", "gold"]


class Match:
    def __init__(self, commanders: list[Commander], seed: int = 0,
                 think_interval: int = cfg.THINK_INTERVAL_TICKS,
                 record_path: str | None = None):
        self.world = World(seed=seed)
        self.commanders = commanders
        self.think_interval = max(5, think_interval)
        self.last_results: list[list[str]] = [["match started"], ["match started"]]
        self.last_decision: list[Decision | None] = [None, None]
        self.chat: list[dict] = []
        self.started_at = time.time()
        self._recorder = open(record_path, "w", encoding="utf-8") if record_path else None

        for team, commander in zip(self.world.teams, commanders):
            self.world.log(f"{team.name} commanded by {commander.name} ({commander.kind})")

    # -- driving ------------------------------------------------------------

    def step(self) -> None:
        if self.world.finished:
            return

        for idx, commander in enumerate(self.commanders):
            # Stagger the two agents by half an interval so their thinking
            # (and their API calls) don't land on the same tick.
            offset = idx * (self.think_interval // 2)
            if (self.world.tick + offset) % self.think_interval == 0:
                commander.request(build_view(self.world, idx, self.last_results[idx]))

            decision = commander.poll()
            if decision is not None:
                self._apply(idx, decision)

        self.world.step()
        if self._recorder is not None and self.world.tick % 5 == 0:
            self._recorder.write(json.dumps(self.snapshot()) + "\n")

    def _apply(self, idx: int, decision: Decision) -> None:
        decision.results = apply_orders(self.world, idx, decision.orders)
        self.last_results[idx] = decision.results
        self.last_decision[idx] = decision
        team = self.world.teams[idx]
        if decision.talk:
            self.chat.append({"tick": self.world.tick, "team": team.name,
                              "color": TEAM_COLORS[idx], "text": decision.talk})
            del self.chat[:-40]
        if decision.error:
            self.world.log(f"{team.name} agent fell back: {decision.error}")

    def run(self, max_ticks: int = cfg.MATCH_TICK_LIMIT) -> World:
        """Run flat out with no pacing. Used for headless batches and tests."""
        while not self.world.finished and self.world.tick < max_ticks:
            self.step()
        self.close()
        return self.world

    def close(self) -> None:
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None

    # -- snapshots ----------------------------------------------------------

    def snapshot(self) -> dict:
        world = self.world
        entities = []
        for e in sorted(world.entities.values(), key=lambda e: e.id):
            entities.append({
                "i": e.id,
                "t": e.team,
                "k": KIND_INDEX.index(e.kind),
                "x": round(e.x, 1),
                "y": round(e.y, 1),
                "h": round(e.hp / e.max_hp, 2),
                "b": 1 if e.is_building else 0,
                "c": 1 if e.complete else 0,
            })

        nodes = [{"x": round(n.x, 1), "y": round(n.y, 1),
                  "k": NODE_INDEX.index(n.kind),
                  "a": round(n.amount / cfg.NODE_AMOUNTS[n.kind], 2)}
                 for n in world.nodes.values()]

        teams = []
        for idx, team in enumerate(world.teams):
            army: dict[str, int] = {}
            for u in world.units_of(idx):
                army[u.kind] = army.get(u.kind, 0) + 1
            teams.append({
                "name": team.name,
                "color": TEAM_COLORS[idx],
                "resources": {k: int(v) for k, v in team.resources.items()},
                "pop": world.pop_used(idx),
                "cap": world.pop_cap(idx),
                "score": world.score(idx),
                "army_power": world.army_power(idx),
                "posture": team.posture,
                "units": army,
                "killed": team.units_killed,
                "lost": team.units_lost,
            })

        agents = []
        for idx, commander in enumerate(self.commanders):
            decision = self.last_decision[idx]
            agents.append({
                "name": commander.name,
                "kind": commander.kind,
                "thinking": commander.busy,
                "plan": decision.plan if decision else "",
                "talk": decision.talk if decision else "",
                "orders": decision.orders if decision else [],
                "results": decision.results if decision else [],
                "source": decision.source if decision else "",
                "latency_ms": decision.latency_ms if decision else 0,
                "error": decision.error if decision else "",
            })

        seconds = world.tick // cfg.TICKS_PER_SECOND
        return {
            "tick": world.tick,
            "clock": f"{seconds // 60}:{seconds % 60:02d}",
            "map": {"w": cfg.MAP_WIDTH, "h": cfg.MAP_HEIGHT},
            "finished": world.finished,
            "winner": world.winner,
            "reason": world.finish_reason,
            "entities": entities,
            "nodes": nodes,
            "teams": teams,
            "agents": agents,
            "chat": self.chat[-14:],
            "events": [{"tick": t, "text": x} for t, x in world.events[-10:]],
        }


def run_headless(spec_a: str, spec_b: str, seed: int = 0, quiet: bool = False,
                 think_interval: int = cfg.THINK_INTERVAL_TICKS) -> dict:
    """One match, as fast as the CPU allows. Returns a small result record."""
    from .commanders import make_commander

    commanders = [make_commander(spec_a, "RED", seed), make_commander(spec_b, "BLUE", seed + 1)]
    match = Match(commanders, seed=seed, think_interval=think_interval)
    world = match.run()
    result = {
        "seed": seed,
        "red": spec_a,
        "blue": spec_b,
        "winner": None if world.winner is None else world.teams[world.winner].name,
        "reason": world.finish_reason,
        "ticks": world.tick,
        "score": [world.score(0), world.score(1)],
    }
    if not quiet:
        print(f"seed {seed:>4}  {spec_a:>8} vs {spec_b:<8}  "
              f"winner {str(result['winner']):>6}  "
              f"{world.tick // cfg.TICKS_PER_SECOND // 60}m  ({world.finish_reason})")
    return result


def record_path_for(directory: str) -> str:
    Path(directory).mkdir(parents=True, exist_ok=True)
    return str(Path(directory) / f"match-{int(time.time())}.jsonl")
