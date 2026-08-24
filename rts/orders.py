"""The command vocabulary a commander may issue.

Deliberately tiny and macro-level. Every order returns a short human-readable
result string, and those strings are fed back to the commander on its next
turn -- so an agent that asks for a stable it cannot afford finds out why,
and can react. That feedback loop matters more than the size of the vocabulary.
"""

from __future__ import annotations

import math
from typing import Any

from . import config as cfg
from .engine import Entity, World

VALID_COMMANDS = ("train", "build", "assign", "attack", "defend", "say")

SCHEMA_HELP = """\
train   {"cmd":"train","unit":"villager|spearman|archer|knight","count":1-10}
build   {"cmd":"build","building":"house|barracks|stable|tower|town_center"}
assign  {"cmd":"assign","resource":"food|wood|gold","count":1-20}
attack  {"cmd":"attack"}            send the whole army at the enemy base
defend  {"cmd":"defend"}            pull the army home
say     {"cmd":"say","text":"..."}  trash talk for the stream"""


def _count(value, default: int, high: int) -> int:
    """Models write ``"count": "a few"`` often enough to matter. Never raise."""
    try:
        return max(1, min(int(value), high))
    except (TypeError, ValueError):
        return default


def _afford(team, cost: dict[str, float]) -> bool:
    return all(team.resources.get(k, 0) >= v for k, v in cost.items())


def _pay(team, cost: dict[str, float]) -> None:
    for k, v in cost.items():
        team.resources[k] -= v


def _missing(team, cost: dict[str, float]) -> str:
    short = [f"{int(v - team.resources.get(k, 0))} {k}"
             for k, v in cost.items() if team.resources.get(k, 0) < v]
    return " and ".join(short)


def _find_site(world: World, team_idx: int, kind: str, forward: bool) -> tuple[float, float]:
    """Spiral out from home for a clear patch of ground.

    ``forward`` biases toward the middle of the map, which is where you want
    towers and not where you want houses.
    """
    team = world.teams[team_idx]
    cx, cy = team.base_x, team.base_y
    if forward:
        toward = cfg.MAP_WIDTH / 2
        cx += math.copysign(min(11.0, abs(toward - cx)), toward - cx)

    size = cfg.BUILDINGS[kind]["size"]
    for ring in range(2, 30):
        for i in range(12):
            angle = 2 * math.pi * i / 12 + ring * 0.4
            x = cx + ring * 1.6 * math.cos(angle)
            y = cy + ring * 1.6 * math.sin(angle)
            if not (2 < x < cfg.MAP_WIDTH - 2 and 2 < y < cfg.MAP_HEIGHT - 2):
                continue
            clear = all(math.hypot(b.x - x, b.y - y) > size + b.stats["size"]
                        for b in world.entities.values() if b.is_building)
            clear = clear and all(math.hypot(n.x - x, n.y - y) > size + 1.5
                                  for n in world.nodes.values())
            if clear:
                return x, y
    return cx, cy


def _pick_builder(world: World, team_idx: int) -> Entity | None:
    """A villager that is free to start a new site, or None.

    Deliberately refuses to pull a villager off a half-finished building. An
    agent that orders a house every turn would otherwise reassign the same
    workers over and over and never finish anything -- the economy stalls and
    nothing explains why.
    """
    free = [v for v in world.units_of(team_idx, "villager") if v.job != "build"]
    if not free:
        return None
    # Prefer whoever is carrying least -- interrupting a full load wastes a trip.
    return min(free, key=lambda v: (v.carrying, v.id))


def apply_order(world: World, team_idx: int, order: dict[str, Any]) -> str:
    """Validate and execute a single order. Never raises on bad input."""
    if not isinstance(order, dict):
        return f"rejected: order must be an object, got {type(order).__name__}"
    cmd = str(order.get("cmd", "")).lower().strip()
    team = world.teams[team_idx]

    if cmd not in VALID_COMMANDS:
        return f"rejected: unknown command {cmd!r} (valid: {', '.join(VALID_COMMANDS)})"

    if cmd == "say":
        return "said it"

    if cmd == "train":
        unit = str(order.get("unit", "")).lower()
        if unit not in cfg.UNITS:
            return f"rejected train: no such unit {unit!r}"
        count = _count(order.get("count", 1), 1, 10)
        stats = cfg.UNITS[unit]
        source_kind = stats["trained_at"]
        sources = [b for b in world.buildings_of(team_idx, source_kind) if b.complete]
        if not sources:
            return f"rejected train {unit}: no completed {source_kind}"
        queued = 0
        for _ in range(count):
            if world.pop_used(team_idx) + stats["pop"] > world.pop_cap(team_idx):
                break
            if not _afford(team, stats["cost"]):
                break
            _pay(team, stats["cost"])
            min(sources, key=lambda b: (len(b.train_queue), b.id)).train_queue.append(unit)
            queued += 1
        if queued == count:
            return f"training {count}x {unit}"
        if queued:
            return (f"training {queued}x {unit} (asked for {count}; "
                    f"ran out of resources or population)")
        if world.pop_used(team_idx) + stats["pop"] > world.pop_cap(team_idx):
            return (f"rejected train {unit}: population capped at "
                    f"{world.pop_cap(team_idx)} — build houses")
        return f"rejected train {unit}: need {_missing(team, stats['cost'])}"

    if cmd == "build":
        kind = str(order.get("building", "")).lower()
        if kind not in cfg.BUILDINGS:
            return f"rejected build: no such building {kind!r}"
        stats = cfg.BUILDINGS[kind]
        if not _afford(team, stats["cost"]):
            return f"rejected build {kind}: need {_missing(team, stats['cost'])}"
        builder = _pick_builder(world, team_idx)
        if builder is None:
            if world.units_of(team_idx, "villager"):
                return (f"rejected build {kind}: every villager is already "
                        f"building something — wait, or train more villagers")
            return f"rejected build {kind}: no villagers left"
        _pay(team, stats["cost"])
        x, y = _find_site(world, team_idx, kind, forward=(kind in ("tower", "town_center")))
        site = world.spawn_building(kind, team_idx, x, y)
        builder.job = "build"
        builder.build_target = site.id
        return f"started building a {kind}"

    if cmd == "assign":
        resource = str(order.get("resource", "")).lower()
        if resource not in cfg.RESOURCE_KINDS:
            return f"rejected assign: no such resource {resource!r}"
        count = _count(order.get("count", 1), 1, 20)
        villagers = world.units_of(team_idx, "villager")
        movable = [v for v in villagers if v.job != resource and v.job != "build"]
        if not movable:
            return f"rejected assign: no villagers available to move to {resource}"
        # Pull from whichever job is most crowded, so the eco stays balanced.
        crowding: dict[str, int] = {}
        for v in villagers:
            crowding[v.job] = crowding.get(v.job, 0) + 1
        movable.sort(key=lambda v: (-crowding.get(v.job, 0), v.id))
        moved = 0
        for v in movable[:count]:
            v.job = resource
            v.node_target = None
            moved += 1
        return f"{moved} villagers now on {resource}"

    if cmd == "attack":
        enemy = world.teams[1 - team_idx]
        army = [u for u in world.units_of(team_idx) if u.kind in cfg.MILITARY]
        if not army:
            return "rejected attack: you have no military units"
        tcs = world.buildings_of(enemy.index, "town_center")
        target = (tcs[0].x, tcs[0].y) if tcs else (enemy.base_x, enemy.base_y)
        team.posture = "attack"
        team.attack_target = target
        team.attack_squad = {u.id for u in army}
        for u in army:
            u.target_id = None
        world.log(f"{team.name} attacks with {len(army)} units")
        return f"attacking with {len(army)} units"

    # defend
    team.posture = "defend"
    team.attack_target = None
    team.attack_squad = set()
    for u in world.units_of(team_idx):
        u.target_id = None
    return "army pulled back to defend"


def apply_orders(world: World, team_idx: int, orders: list[Any]) -> list[str]:
    results: list[str] = []
    for order in (orders or [])[:cfg.MAX_ORDERS_PER_TURN]:
        results.append(apply_order(world, team_idx, order))
    if not results:
        results.append("no orders issued")
    return results
