"""Builds the compact situation report a commander reasons over.

Two rules shaped this:
  * It has to fit comfortably in a prompt, so it is a summary, not a dump of
    every entity. Roughly 30 lines of JSON regardless of army size.
  * The enemy section is a scouting-style summary (counts and a rough position)
    rather than exact state. This is a perfect-information game with imperfect
    *resolution*, which is a reasonable stand-in for fog of war and keeps the
    agent from trying to micro against exact coordinates.
"""

from __future__ import annotations

import math

from . import config as cfg
from .engine import World


def _clock(tick: int) -> str:
    seconds = tick // cfg.TICKS_PER_SECOND
    return f"{seconds // 60}:{seconds % 60:02d}"


def _army_counts(world: World, team: int) -> dict[str, int]:
    counts = {k: 0 for k in cfg.MILITARY}
    for u in world.units_of(team):
        if u.kind in counts:
            counts[u.kind] += 1
    return {k: v for k, v in counts.items() if v}


def _building_counts(world: World, team: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for b in world.buildings_of(team):
        key = b.kind if b.complete else f"{b.kind} (under construction)"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _army_centroid(world: World, team: int) -> tuple[float, float] | None:
    army = [u for u in world.units_of(team) if u.kind in cfg.MILITARY]
    if not army:
        return None
    return (sum(u.x for u in army) / len(army), sum(u.y for u in army) / len(army))


def _describe_position(world: World, team: int, point: tuple[float, float] | None) -> str:
    if point is None:
        return "no army"
    x, _ = point
    home = world.teams[team].base_x
    enemy_home = world.teams[1 - team].base_x
    frac = abs(x - home) / max(abs(enemy_home - home), 1e-6)
    if frac < 0.25:
        return "at home"
    if frac < 0.6:
        return "in the middle of the map"
    if frac < 0.85:
        return "approaching the enemy base"
    return "inside the enemy base"


def _villager_jobs(world: World, team: int) -> dict[str, int]:
    jobs = {"food": 0, "wood": 0, "gold": 0, "building": 0}
    for v in world.units_of(team, "villager"):
        jobs["building" if v.job == "build" else v.job] += 1
    return jobs


def _nodes_near(world: World, x: float, y: float, radius: float = 22.0) -> dict[str, int]:
    counts = {k: 0 for k in cfg.RESOURCE_KINDS}
    for n in world.nodes.values():
        if math.hypot(n.x - x, n.y - y) <= radius:
            counts[n.kind] += 1
    return counts


def build_view(world: World, team_idx: int, last_results: list[str]) -> dict:
    me = world.teams[team_idx]
    them = world.teams[1 - team_idx]
    my_army = _army_centroid(world, team_idx)
    their_army = _army_centroid(world, 1 - team_idx)
    tcs = world.buildings_of(them.index, "town_center")

    return {
        "tick": world.tick,
        "clock": _clock(world.tick),
        "time_remaining": _clock(max(0, cfg.MATCH_TICK_LIMIT - world.tick)),
        "you": {
            "team": me.name,
            "resources": {k: int(v) for k, v in me.resources.items()},
            "population": {"used": world.pop_used(team_idx),
                           "cap": world.pop_cap(team_idx)},
            "villagers": _villager_jobs(world, team_idx),
            "army": _army_counts(world, team_idx) or "none",
            "army_power": world.army_power(team_idx),
            "army_position": _describe_position(world, team_idx, my_army),
            "buildings": _building_counts(world, team_idx),
            "training_now": [k for b in world.buildings_of(team_idx)
                             for k in b.train_queue][:8],
            "posture": me.posture,
            "base_under_attack": world.under_attack(team_idx),
            "units_lost": me.units_lost,
        },
        "enemy": {
            "team": them.name,
            "army": _army_counts(world, them.index) or "none",
            "army_power": world.army_power(them.index),
            "army_position": _describe_position(world, them.index, their_army),
            "villagers": len(world.units_of(them.index, "villager")),
            "buildings": _building_counts(world, them.index),
            "town_centers_remaining": len(tcs),
        },
        "map": {
            "your_base": [round(me.base_x), round(me.base_y)],
            "enemy_base": [round(them.base_x), round(them.base_y)],
            "resource_nodes_near_your_base": _nodes_near(world, me.base_x, me.base_y),
        },
        "result_of_your_last_orders": last_results,
    }
