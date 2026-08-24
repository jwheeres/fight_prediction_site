"""Deterministic tick-based RTS simulation.

Design notes:
  * Everything random goes through ``World.rng`` so a seed reproduces a match
    exactly. That is what makes headless balance testing worth anything.
  * Commanders only ever issue *macro* orders (train this, build that, attack).
    Per-unit micro is handled here, deterministically. An agent that had to
    steer 40 units individually would spend its whole turn on bookkeeping.
  * There is no pathfinding. Units move in straight lines and buildings do not
    block. For a spectator toy this reads fine and keeps the loop honest.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable

from . import config as cfg

TEAM_NAMES = ("RED", "BLUE")


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


@dataclass
class Node:
    """A gatherable resource pile."""

    id: int
    kind: str
    x: float
    y: float
    amount: float

    @property
    def depleted(self) -> bool:
        return self.amount <= 0


@dataclass
class Entity:
    """A unit or a building. One class, because the differences are small."""

    id: int
    team: int
    kind: str
    x: float
    y: float
    hp: float
    max_hp: float
    is_building: bool = False

    # buildings
    build_progress: float = 0.0     # counts up to build_ticks
    build_ticks: float = 0.0
    train_queue: list[str] = field(default_factory=list)
    train_progress: float = 0.0

    # villagers
    job: str = "food"               # a resource kind, or "build"
    build_target: int | None = None
    carrying: float = 0.0
    carry_kind: str | None = None
    node_target: int | None = None

    # military
    target_id: int | None = None
    cooldown: float = 0.0

    @property
    def complete(self) -> bool:
        return (not self.is_building) or self.build_progress >= self.build_ticks

    @property
    def stats(self) -> dict:
        return cfg.BUILDINGS[self.kind] if self.is_building else cfg.UNITS[self.kind]


@dataclass
class Team:
    index: int
    name: str
    resources: dict[str, float]
    base_x: float
    base_y: float
    posture: str = "defend"                 # "defend" | "attack"
    attack_target: tuple[float, float] | None = None
    # The units committed when the attack order was given. Units trained
    # afterwards stay home instead of walking across the map alone and dying
    # one at a time, which is the classic way a winning push becomes a loss.
    attack_squad: set[int] = field(default_factory=set)
    gathered: dict[str, float] = field(default_factory=lambda: {"food": 0.0, "wood": 0.0, "gold": 0.0})
    units_lost: int = 0
    units_killed: int = 0


class World:
    """The whole game state, plus the tick that advances it."""

    def __init__(self, seed: int = 0, team_names: Iterable[str] = TEAM_NAMES):
        self.rng = random.Random(seed)
        self.seed = seed
        self.tick = 0
        self.next_id = 1
        self.entities: dict[int, Entity] = {}
        self.nodes: dict[int, Node] = {}
        self.events: list[tuple[int, str]] = []   # (tick, text) for the stream feed
        self.winner: int | None = None
        # Recomputed once per tick: how far the middle of each attack wave
        # still has to travel, and each team's entities.
        self._squad_front: list[float | None] = [None, None]
        self._by_team: list[list[Entity]] = [[], []]
        self.finished = False
        self.finish_reason = ""

        names = list(team_names)
        self.teams = [
            Team(0, names[0], dict(cfg.STARTING_RESOURCES),
                 cfg.BASE_MARGIN, cfg.MAP_HEIGHT / 2),
            Team(1, names[1], dict(cfg.STARTING_RESOURCES),
                 cfg.MAP_WIDTH - cfg.BASE_MARGIN, cfg.MAP_HEIGHT / 2),
        ]
        self._generate_map()
        self._place_starting_units()

    # -- setup --------------------------------------------------------------

    def _new_id(self) -> int:
        self.next_id += 1
        return self.next_id - 1

    def _generate_map(self) -> None:
        """Scatter resources on the left half, then mirror. Symmetry keeps the
        match about the commanders rather than about the dice."""
        left: list[tuple[str, float, float]] = []

        def scatter(kind: str, count: float, cx: float, cy: float, spread: float) -> None:
            for _ in range(int(count)):
                x = min(max(self.rng.gauss(cx, spread), 2.0), cfg.MAP_WIDTH / 2 - 2)
                y = min(max(self.rng.gauss(cy, spread), 2.0), cfg.MAP_HEIGHT - 2)
                left.append((kind, x, y))

        base = self.teams[0]
        scatter("food", 5, base.base_x + 6, base.base_y, 5)
        scatter("wood", 12, base.base_x + 4, base.base_y - 10, 7)
        scatter("wood", 10, base.base_x + 4, base.base_y + 10, 7)
        scatter("gold", 3, base.base_x + 10, base.base_y + 4, 4)
        # contested middle
        scatter("gold", 3, cfg.MAP_WIDTH / 2 - 8, cfg.MAP_HEIGHT / 2, 9)
        scatter("food", 3, cfg.MAP_WIDTH / 2 - 6, cfg.MAP_HEIGHT / 2, 10)

        for kind, x, y in left:
            amount = cfg.NODE_AMOUNTS[kind]
            for mx in (x, cfg.MAP_WIDTH - x):
                nid = self._new_id()
                self.nodes[nid] = Node(nid, kind, mx, y, amount)

    def _place_starting_units(self) -> None:
        for team in self.teams:
            self.spawn_building("town_center", team.index, team.base_x, team.base_y,
                                complete=True)
            for i in range(cfg.STARTING_VILLAGERS):
                angle = 2 * math.pi * i / cfg.STARTING_VILLAGERS
                unit = self.spawn_unit("villager", team.index,
                                       team.base_x + 3 * math.cos(angle),
                                       team.base_y + 3 * math.sin(angle))
                unit.job = "food" if i < 2 else "wood"

    # -- spawning -----------------------------------------------------------

    def spawn_unit(self, kind: str, team: int, x: float, y: float) -> Entity:
        stats = cfg.UNITS[kind]
        ent = Entity(self._new_id(), team, kind, x, y, stats["hp"], stats["hp"])
        self.entities[ent.id] = ent
        return ent

    def spawn_building(self, kind: str, team: int, x: float, y: float,
                       complete: bool = False) -> Entity:
        stats = cfg.BUILDINGS[kind]
        ent = Entity(self._new_id(), team, kind, x, y,
                     stats["hp"] if complete else stats["hp"] * 0.1, stats["hp"],
                     is_building=True)
        ent.build_ticks = stats["build_ticks"]
        ent.build_progress = stats["build_ticks"] if complete else 0.0
        self.entities[ent.id] = ent
        return ent

    # -- queries ------------------------------------------------------------

    def team_entities(self, team: int) -> list[Entity]:
        return [e for e in self.entities.values() if e.team == team]

    def units_of(self, team: int, kind: str | None = None) -> list[Entity]:
        return [e for e in self.entities.values()
                if e.team == team and not e.is_building
                and (kind is None or e.kind == kind)]

    def buildings_of(self, team: int, kind: str | None = None) -> list[Entity]:
        return [e for e in self.entities.values()
                if e.team == team and e.is_building
                and (kind is None or e.kind == kind)]

    def pop_used(self, team: int) -> int:
        return sum(cfg.UNITS[e.kind]["pop"] for e in self.units_of(team)) + \
            sum(cfg.UNITS[k]["pop"] for b in self.buildings_of(team) for k in b.train_queue)

    def pop_cap(self, team: int) -> int:
        cap = sum(cfg.BUILDINGS[b.kind]["pop_provided"]
                  for b in self.buildings_of(team) if b.complete)
        return min(cap, cfg.POP_MAX)

    def army_power(self, team: int) -> float:
        """A single number for 'how scary is this army', for the agent's view."""
        total = 0.0
        for e in self.units_of(team):
            if e.kind in cfg.MILITARY:
                s = cfg.UNITS[e.kind]
                total += s["attack"] * (e.hp / e.max_hp) * (1 + s["hp"] / 100)
        return round(total, 1)

    def score(self, team: int) -> float:
        t = self.teams[team]
        eco = sum(t.gathered.values())
        army = sum(sum(cfg.UNITS[e.kind]["cost"].values()) for e in self.units_of(team))
        built = sum(sum(cfg.BUILDINGS[b.kind]["cost"].values())
                    for b in self.buildings_of(team) if b.complete)
        return round(eco * 0.5 + army + built + t.units_killed * 25, 1)

    def log(self, text: str) -> None:
        self.events.append((self.tick, text))
        if len(self.events) > 400:
            del self.events[:-400]

    # -- the tick -----------------------------------------------------------

    def step(self) -> None:
        if self.finished:
            return
        self.tick += 1
        self._update_indexes()

        # Stable iteration order by id keeps the sim reproducible.
        for ent in sorted(self.entities.values(), key=lambda e: e.id):
            if ent.id not in self.entities:
                continue  # died earlier this tick
            if ent.is_building:
                self._step_building(ent)
            elif ent.kind == "villager":
                self._step_villager(ent)
            else:
                self._step_military(ent)

        self._prune_squads()
        self._check_end_conditions()

    def _update_indexes(self) -> None:
        self._by_team = [[], []]
        for ent in self.entities.values():
            self._by_team[ent.team].append(ent)

        for team in self.teams:
            target = team.attack_target
            members = [self.entities[i] for i in team.attack_squad
                       if i in self.entities]
            if target is None or not members:
                self._squad_front[team.index] = None
                continue
            # Median rather than mean distance-to-target: one straggler pinned
            # down at home must not freeze the whole wave in place.
            distances = sorted(_dist(m.x, m.y, target[0], target[1]) for m in members)
            self._squad_front[team.index] = distances[len(distances) // 2]

    # -- buildings ----------------------------------------------------------

    def _step_building(self, b: Entity) -> None:
        if not b.complete:
            return
        stats = cfg.BUILDINGS[b.kind]
        if stats["attack"]:
            self._attack_step(b, stats["attack"], stats["range"], armor_piercing=False)
        if not b.train_queue:
            return
        kind = b.train_queue[0]
        b.train_progress += 1
        if b.train_progress >= cfg.UNITS[kind]["train_ticks"]:
            b.train_queue.pop(0)
            b.train_progress = 0.0
            angle = self.rng.uniform(0, 2 * math.pi)
            unit = self.spawn_unit(kind, b.team,
                                   b.x + 3.5 * math.cos(angle),
                                   b.y + 3.5 * math.sin(angle))
            if kind == "villager":
                unit.job = self._least_worked_resource(b.team)

    def _least_worked_resource(self, team: int) -> str:
        counts = {k: 0 for k in cfg.RESOURCE_KINDS}
        for v in self.units_of(team, "villager"):
            if v.job in counts:
                counts[v.job] += 1
        return min(cfg.RESOURCE_KINDS, key=lambda k: (counts[k], k))

    # -- villagers ----------------------------------------------------------

    def _step_villager(self, v: Entity) -> None:
        # A villager only fights back if something is right on top of it.
        if self._enemy_within(v, 1.5) is not None:
            self._attack_step(v, cfg.UNITS["villager"]["attack"], 1.0)
            return

        if v.job == "build":
            self._step_builder(v)
            return

        if v.carrying >= cfg.CARRY_CAPACITY:
            drop = self._nearest(self.buildings_of(v.team), v,
                                 pred=lambda b: b.complete and cfg.BUILDINGS[b.kind]["dropoff"])
            if drop is None:
                v.carrying = 0.0
                return
            if self._move_toward(v, drop.x, drop.y, 2.5):
                team = self.teams[v.team]
                team.resources[v.carry_kind] += v.carrying
                team.gathered[v.carry_kind] += v.carrying
                v.carrying = 0.0
            return

        node = self.nodes.get(v.node_target or -1)
        if node is None or node.depleted or node.kind != v.job:
            node = self._nearest(self.nodes.values(), v,
                                 pred=lambda n: n.kind == v.job and not n.depleted)
            if node is None:  # that resource is gone from the map entirely
                node = self._nearest(self.nodes.values(), v, pred=lambda n: not n.depleted)
                if node is None:
                    return
                v.job = node.kind
            v.node_target = node.id

        if self._move_toward(v, node.x, node.y, cfg.GATHER_RANGE):
            take = min(cfg.GATHER_RATE, node.amount)
            node.amount -= take
            v.carrying += take
            v.carry_kind = node.kind
            if node.depleted:
                self.nodes.pop(node.id, None)
                v.node_target = None

    def _step_builder(self, v: Entity) -> None:
        site = self.entities.get(v.build_target or -1)
        if site is None or not site.is_building or site.complete:
            v.job = self._least_worked_resource(v.team)
            v.build_target = None
            return
        if self._move_toward(v, site.x, site.y, cfg.BUILD_RANGE):
            site.build_progress += cfg.BUILD_RATE
            frac = min(site.build_progress / site.build_ticks, 1.0)
            site.hp = max(site.hp, site.max_hp * (0.1 + 0.9 * frac))
            if site.complete:
                self.log(f"{self.teams[v.team].name} finished a {site.kind}")
                v.build_target = None
                # Adopt another stalled site before going back to gathering,
                # so a building whose builder was killed still gets finished.
                orphan = self._nearest(
                    self.buildings_of(v.team), v,
                    pred=lambda b: not b.complete and not any(
                        o.build_target == b.id for o in self.units_of(v.team, "villager")))
                if orphan is not None:
                    v.build_target = orphan.id
                else:
                    v.job = self._least_worked_resource(v.team)

    # -- military -----------------------------------------------------------

    def _step_military(self, u: Entity) -> None:
        stats = cfg.UNITS[u.kind]
        team = self.teams[u.team]

        target = self.entities.get(u.target_id or -1)
        if target is None or target.team == u.team:
            target = self._enemy_within(u, cfg.AGGRO_RANGE)
            u.target_id = target.id if target else None

        if target is not None:
            self._attack_step(u, stats["attack"], stats["range"])
            return

        if team.posture == "attack" and team.attack_target is not None \
                and u.id in team.attack_squad:
            tx, ty = team.attack_target
            front = self._squad_front[u.team]
            if front is not None and _dist(u.x, u.y, tx, ty) < front - cfg.SQUAD_COHESION:
                return  # too far out in front; wait for the rest of the wave
            if self._move_toward(u, tx, ty, 2.0):
                # Arrived and nothing in aggro range: pick off the nearest
                # enemy anywhere, so a won push finishes the job.
                enemy = self._nearest(self._by_team[1 - u.team], u)
                u.target_id = enemy.id if enemy else None
            return

        # Defending: hold a loose ring around home so the army isn't a single dot.
        angle = (u.id * 2.399963)  # golden-angle spread, deterministic
        radius = 5.0 + (u.id % 4) * 1.5
        self._move_toward(u, team.base_x + radius * math.cos(angle),
                          team.base_y + radius * math.sin(angle), 1.5)

    def _attack_step(self, attacker: Entity, attack: float, reach: float,
                     armor_piercing: bool = False) -> None:
        target = self.entities.get(attacker.target_id or -1)
        if target is None or target.team == attacker.team:
            target = self._enemy_within(attacker, max(reach, cfg.AGGRO_RANGE)
                                        if not attacker.is_building else reach)
            attacker.target_id = target.id if target else None
            if target is None:
                return

        attacker.cooldown = max(0.0, attacker.cooldown - 1)
        if not self._move_toward(attacker, target.x, target.y, reach):
            return
        if attacker.cooldown > 0:
            return

        bonus = cfg.ATTACK_BONUS.get(attacker.kind, {}).get(target.kind, 1.0)
        if target.is_building and not attacker.is_building and attacker.kind != "villager":
            bonus *= cfg.BUILDING_DAMAGE_MULT
        armor = 0 if armor_piercing else target.stats.get("armor", 0)
        damage = max(1.0, attack * bonus - armor)
        target.hp -= damage
        attacker.cooldown = cfg.ATTACK_COOLDOWN
        if target.hp <= 0:
            self._kill(target, attacker.team)
            attacker.target_id = None

    def _kill(self, ent: Entity, killer_team: int) -> None:
        self.entities.pop(ent.id, None)
        self.teams[ent.team].units_lost += 1
        self.teams[killer_team].units_killed += 1
        if ent.is_building:
            self.log(f"{self.teams[killer_team].name} destroyed a "
                     f"{self.teams[ent.team].name} {ent.kind}")
        # Anything that was chasing the corpse needs a new idea.
        for other in self.entities.values():
            if other.target_id == ent.id:
                other.target_id = None

    # -- helpers ------------------------------------------------------------

    def _move_toward(self, ent: Entity, tx: float, ty: float, within: float) -> bool:
        """Step toward a point. Returns True once inside ``within`` tiles."""
        d = _dist(ent.x, ent.y, tx, ty)
        if d <= within:
            return True
        if ent.is_building:
            return False  # towers shoot what comes to them
        speed = cfg.UNITS[ent.kind]["speed"]
        step = min(speed, d)
        ent.x += (tx - ent.x) / d * step
        ent.y += (ty - ent.y) / d * step
        return _dist(ent.x, ent.y, tx, ty) <= within

    def _nearest(self, candidates, origin, pred=None):
        best, best_d = None, float("inf")
        for c in candidates:
            if pred is not None and not pred(c):
                continue
            d = _dist(origin.x, origin.y, c.x, c.y)
            if d < best_d or (d == best_d and best is not None and c.id < best.id):
                best, best_d = c, d
        return best

    def _enemy_within(self, ent: Entity, radius: float) -> Entity | None:
        best = self._nearest(self._by_team[1 - ent.team], ent)
        if best is None:
            return None
        return best if _dist(ent.x, ent.y, best.x, best.y) <= radius else None

    def under_attack(self, team: int) -> bool:
        t = self.teams[team]
        return any(e.team != team and not e.is_building
                   and _dist(e.x, e.y, t.base_x, t.base_y) < cfg.DEFEND_RADIUS + 6
                   for e in self.entities.values())

    # -- end of match -------------------------------------------------------

    def _prune_squads(self) -> None:
        for team in self.teams:
            if team.posture != "attack":
                continue
            team.attack_squad &= set(self.entities)
            if not team.attack_squad:
                team.posture = "defend"
                team.attack_target = None
                self.log(f"{team.name}'s attack wave was wiped out")

    def _check_end_conditions(self) -> None:
        alive = []
        for team in self.teams:
            has_tc = any(self.buildings_of(team.index, "town_center"))
            has_army = any(u.kind in cfg.MILITARY for u in self.units_of(team.index))
            alive.append(has_tc or has_army)

        if not alive[0] and not alive[1]:
            self._finish(None, "mutual annihilation")
        elif not alive[0]:
            self._finish(1, f"{self.teams[0].name} razed")
        elif not alive[1]:
            self._finish(0, f"{self.teams[1].name} razed")
        elif self.tick >= cfg.MATCH_TICK_LIMIT:
            s0, s1 = self.score(0), self.score(1)
            if s0 == s1:
                self._finish(None, "time limit, dead even")
            else:
                self._finish(0 if s0 > s1 else 1, f"time limit, score {s0} vs {s1}")

    def _finish(self, winner: int | None, reason: str) -> None:
        self.finished = True
        self.winner = winner
        self.finish_reason = reason
        name = "Nobody" if winner is None else self.teams[winner].name
        self.log(f"GAME OVER — {name} wins ({reason})")
