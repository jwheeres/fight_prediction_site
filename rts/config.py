"""Game constants for the RTS sandbox.

Everything tunable lives here so balance changes are a one-file diff.
Distances are in tiles, durations in ticks. The sim runs at TICKS_PER_SECOND,
so a "cost 90 ticks" train time is 9 seconds of wall clock at normal speed.
"""

from __future__ import annotations

TICKS_PER_SECOND = 10

# --- map -------------------------------------------------------------------

MAP_WIDTH = 96
MAP_HEIGHT = 64
BASE_MARGIN = 14  # town centres sit this far in from the left/right edges

# --- economy ---------------------------------------------------------------

CARRY_CAPACITY = 12
GATHER_RATE = 0.5          # resource units per tick while gathering
GATHER_RANGE = 1.2         # how close a villager must be to work a node
BUILD_RATE = 1.0           # build progress per tick per adjacent villager
BUILD_RANGE = 2.0

STARTING_RESOURCES = {"food": 200, "wood": 200, "gold": 100}
STARTING_VILLAGERS = 4

# --- population ------------------------------------------------------------

POP_MAX = 80

# --- combat ----------------------------------------------------------------

AGGRO_RANGE = 7.0          # military units engage anything this close
DEFEND_RADIUS = 12.0       # how far from home a defending army will wander
ATTACK_COOLDOWN = 10       # ticks between swings
SQUAD_COHESION = 5.0       # an attacking unit won't get further ahead of its
                           # squad's centre than this -- otherwise fast units
                           # arrive alone and die before the rest turn up

# Military units hit buildings harder than they hit each other. Without this
# the defender always wins: a push can beat the enemy army and still not have
# time to knock anything down before reinforcements arrive.
BUILDING_DAMAGE_MULT = 2.0

# Damage multipliers: ATTACK_BONUS[attacker][defender]
ATTACK_BONUS = {
    "spearman": {"knight": 2.5},
    "archer": {"spearman": 1.6},
    "knight": {"archer": 1.6, "villager": 1.5},
}

UNITS = {
    "villager": {
        "hp": 28, "attack": 3, "range": 1.0, "speed": 0.34, "armor": 0,
        "cost": {"food": 50}, "train_ticks": 60, "pop": 1,
        "trained_at": "town_center",
    },
    "spearman": {
        "hp": 48, "attack": 7, "range": 1.1, "speed": 0.32, "armor": 1,
        "cost": {"food": 35, "wood": 25}, "train_ticks": 85, "pop": 1,
        "trained_at": "barracks",
    },
    "archer": {
        "hp": 32, "attack": 8, "range": 4.5, "speed": 0.31, "armor": 0,
        "cost": {"wood": 35, "gold": 25}, "train_ticks": 100, "pop": 1,
        "trained_at": "barracks",
    },
    "knight": {
        "hp": 105, "attack": 14, "range": 1.2, "speed": 0.50, "armor": 2,
        "cost": {"food": 60, "gold": 45}, "train_ticks": 155, "pop": 2,
        "trained_at": "stable",
    },
}

BUILDINGS = {
    "town_center": {
        "hp": 1500, "cost": {"wood": 275}, "build_ticks": 220, "size": 3.0,
        "pop_provided": 5, "dropoff": True, "attack": 0, "range": 0,
    },
    "house": {
        "hp": 180, "cost": {"wood": 30}, "build_ticks": 55, "size": 1.6,
        "pop_provided": 5, "dropoff": False, "attack": 0, "range": 0,
    },
    "barracks": {
        "hp": 420, "cost": {"wood": 150}, "build_ticks": 130, "size": 2.4,
        "pop_provided": 0, "dropoff": False, "attack": 0, "range": 0,
    },
    "stable": {
        "hp": 420, "cost": {"wood": 175, "gold": 50}, "build_ticks": 150, "size": 2.4,
        "pop_provided": 0, "dropoff": False, "attack": 0, "range": 0,
    },
    "tower": {
        "hp": 400, "cost": {"wood": 125, "gold": 40}, "build_ticks": 150, "size": 1.6,
        "pop_provided": 0, "dropoff": False, "attack": 10, "range": 7.0,
    },
}

MILITARY = ("spearman", "archer", "knight")
RESOURCE_KINDS = ("food", "wood", "gold")

# Starting node amounts. Food is generous so games are decided by fighting,
# not by an economy quietly starving out off-screen.
NODE_AMOUNTS = {"food": 320, "wood": 260, "gold": 200}

# --- match -----------------------------------------------------------------

MATCH_TICK_LIMIT = 12000   # 20 minutes of game time, then judged on score
THINK_INTERVAL_TICKS = 50  # how often a commander is asked for orders (5s)
MAX_ORDERS_PER_TURN = 6
