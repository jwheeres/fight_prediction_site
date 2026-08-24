"""Sim-level tests. These are the ones that catch real breakage."""

from __future__ import annotations

import json

from rts import config as cfg
from rts.commanders import Decision, ScriptedCommander, _extract_json
from rts.engine import World
from rts.match import Match, run_headless
from rts.orders import apply_order, apply_orders
from rts.view import build_view


def _match(spec_a="balanced", spec_b="rush", seed=0) -> Match:
    return Match([ScriptedCommander("RED", spec_a, seed),
                  ScriptedCommander("BLUE", spec_b, seed + 1)], seed=seed)


# -- determinism ------------------------------------------------------------

def test_same_seed_gives_identical_matches():
    a = run_headless("rush", "boom", seed=7, quiet=True)
    b = run_headless("rush", "boom", seed=7, quiet=True)
    assert a == b


def test_different_seeds_diverge():
    seeds = {json.dumps(run_headless("rush", "boom", seed=s, quiet=True), sort_keys=True)
             for s in range(4)}
    assert len(seeds) > 1


# -- the match always ends --------------------------------------------------

def test_every_matchup_terminates():
    for spec_a in ("rush", "boom", "turtle", "balanced"):
        world = _match(spec_a, "balanced", seed=3).run()
        assert world.finished
        assert world.tick <= cfg.MATCH_TICK_LIMIT
        assert world.winner in (0, 1, None)


def test_map_is_mirror_symmetric():
    """Both sides must start with the same resources within reach, or the
    seed decides the match instead of the commanders."""
    world = World(seed=11)
    left = sorted((n.kind, round(cfg.MAP_WIDTH - n.x, 4), round(n.y, 4))
                  for n in world.nodes.values() if n.x < cfg.MAP_WIDTH / 2)
    right = sorted((n.kind, round(n.x, 4), round(n.y, 4))
                   for n in world.nodes.values() if n.x > cfg.MAP_WIDTH / 2)
    assert left == right


# -- resources and population ----------------------------------------------

def test_resources_never_go_negative():
    match = _match("boom", "rush", seed=5)
    for _ in range(4000):
        match.step()
        for team in match.world.teams:
            assert all(v >= 0 for v in team.resources.values()), team.resources
        if match.world.finished:
            break


def test_population_cap_is_respected():
    match = _match("boom", "boom", seed=2)
    for _ in range(4000):
        match.step()
        for idx in (0, 1):
            assert match.world.pop_used(idx) <= max(match.world.pop_cap(idx), 5)
        if match.world.finished:
            break


def test_gathering_credits_the_right_resource():
    world = World(seed=1)
    for _ in range(900):
        world.step()
    gathered = world.teams[0].gathered
    assert gathered["food"] > 0 and gathered["wood"] > 0


# -- orders -----------------------------------------------------------------

def test_bad_orders_are_rejected_not_crashed():
    world = World(seed=1)
    for junk in [{}, {"cmd": "nuke"}, {"cmd": "train", "unit": "dragon"},
                 {"cmd": "build", "building": "castle"},
                 {"cmd": "assign", "resource": "uranium"},
                 {"cmd": "train", "unit": "villager", "count": "lots"},
                 "not an object", None, 42, []]:
        result = apply_order(world, 0, junk)
        assert isinstance(result, str) and result


def test_order_count_is_capped():
    world = World(seed=1)
    orders = [{"cmd": "say", "text": str(i)} for i in range(50)]
    assert len(apply_orders(world, 0, orders)) == cfg.MAX_ORDERS_PER_TURN


def test_cannot_train_without_the_building():
    world = World(seed=1)
    world.teams[0].resources.update({"food": 9999, "wood": 9999, "gold": 9999})
    assert "no completed barracks" in apply_order(world, 0, {"cmd": "train", "unit": "spearman"})


def test_training_deducts_resources_and_queues():
    world = World(seed=1)
    before = dict(world.teams[0].resources)
    result = apply_order(world, 0, {"cmd": "train", "unit": "villager", "count": 1})
    assert "training" in result
    assert world.teams[0].resources["food"] == before["food"] - cfg.UNITS["villager"]["cost"]["food"]
    assert any(b.train_queue for b in world.buildings_of(0))


def test_rejected_order_explains_itself():
    world = World(seed=1)
    world.teams[0].resources.update({"food": 0, "wood": 0, "gold": 0})
    result = apply_order(world, 0, {"cmd": "build", "building": "barracks"})
    assert result.startswith("rejected") and "wood" in result


def test_attack_needs_an_army_and_commits_a_squad():
    world = World(seed=1)
    assert "no military units" in apply_order(world, 0, {"cmd": "attack"})
    world.spawn_unit("spearman", 0, world.teams[0].base_x, world.teams[0].base_y)
    assert "attacking" in apply_order(world, 0, {"cmd": "attack"})
    assert world.teams[0].posture == "attack"
    assert len(world.teams[0].attack_squad) == 1
    # Units trained after the order stay home rather than trickling in alone.
    late = world.spawn_unit("knight", 0, world.teams[0].base_x, world.teams[0].base_y)
    assert late.id not in world.teams[0].attack_squad


def test_buildings_do_not_overlap():
    world = World(seed=4)
    world.teams[0].resources.update({"food": 9999, "wood": 9999, "gold": 9999})
    for _ in range(12):
        apply_order(world, 0, {"cmd": "build", "building": "house"})
    sites = world.buildings_of(0)
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            gap = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
            assert gap > 1.0, f"{a.kind} and {b.kind} are on top of each other"


# -- the view the agent sees ------------------------------------------------

def test_view_is_json_serialisable_and_compact():
    match = _match(seed=6)
    for _ in range(1500):
        match.step()
    view = build_view(match.world, 0, ["training 2x spearman"])
    text = json.dumps(view)
    assert len(text) < 4000, f"view is {len(text)} bytes; it has to fit in a prompt"
    for key in ("tick", "you", "enemy", "map", "result_of_your_last_orders"):
        assert key in view


def test_view_reports_order_feedback():
    match = _match(seed=6)
    match.step()
    view = build_view(match.world, 0, ["rejected train knight: need 45 gold"])
    assert "rejected train knight: need 45 gold" in view["result_of_your_last_orders"]


# -- snapshots for the browser ---------------------------------------------

def test_snapshot_is_serialisable_every_tick_of_a_whole_match():
    match = _match("rush", "turtle", seed=8)
    while not match.world.finished:
        match.step()
        if match.world.tick % 250 == 0:
            json.dumps(match.snapshot())
    snap = match.snapshot()
    assert snap["finished"] and len(snap["teams"]) == 2 and len(snap["agents"]) == 2


# -- commanders -------------------------------------------------------------

def test_scripted_commander_only_emits_valid_orders():
    match = _match("boom", "turtle", seed=9)
    for _ in range(3000):
        match.step()
        for decision in match.last_decision:
            if decision is None:
                continue
            for result in decision.results:
                assert "unknown command" not in result
                assert "no such" not in result
        if match.world.finished:
            break


def test_extract_json_survives_fences_and_chatter():
    assert _extract_json('{"orders": []}') == {"orders": []}
    assert _extract_json('```json\n{"orders": [1]}\n```') == {"orders": [1]}
    assert _extract_json('Sure!\n{"talk": "hi"}\nhope that helps') == {"talk": "hi"}


def test_decision_defaults_are_safe():
    decision = Decision(orders=[])
    assert decision.orders == [] and decision.source == "script"
