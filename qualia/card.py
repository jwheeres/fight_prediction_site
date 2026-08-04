"""Build the fight card: merge model predictions with live market odds.

For each fight we compute the model's win probability from fighter stats, look
up the market consensus from the odds service, then derive the "edge" (does the
model disagree with the market, and which way), a confidence label, and the
predicted winner. This is the payload the frontend renders.
"""

from __future__ import annotations

import json
from pathlib import Path

from qualia import model, odds

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CARD_FILE = DATA_DIR / "card.json"
FIGHTERS_FILE = DATA_DIR / "fighters.json"

# How far model and market must diverge before we flag a real edge.
_EDGE_THRESHOLD = 0.08     # 8 percentage points
_HIGH_CONF_MODEL = 0.65    # model strongly favors one side
_TOSS_UP_BAND = 0.06       # model within 6pts of 50/50


def _load(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _confidence(model_prob_winner: float) -> str:
    lead = abs(model_prob_winner - 0.5)
    if model_prob_winner >= _HIGH_CONF_MODEL:
        return "high"
    if lead <= _TOSS_UP_BAND:
        return "toss"
    return "lean"


def build_card(force_refresh: bool = False) -> dict:
    card = _load(CARD_FILE)
    fighters = _load(FIGHTERS_FILE)
    odds_data = odds.build_odds_index(force_refresh=force_refresh)
    odds_index = odds_data["index"]

    fights_out = []
    high_conf_count = 0

    for fight in card["fights"]:
        name_a, name_b = fight["fighter_a"], fight["fighter_b"]
        stats_a = fighters.get(name_a, {})
        stats_b = fighters.get(name_b, {})

        model_a = model.win_probability(stats_a, stats_b)
        model_b = 1.0 - model_a

        # Market lookup (by fighter A; the row carries both sides).
        row = odds_index.get(odds.normalize_name(name_a))
        if row:
            if row["side"] == "a":
                market_a, market_b = row["market_prob_a"], row["market_prob_b"]
                odds_a, odds_b = row["odds_a"], row["odds_b"]
            else:
                market_a, market_b = row["market_prob_b"], row["market_prob_a"]
                odds_a, odds_b = row["odds_b"], row["odds_a"]
            books = row["books"]
        else:
            market_a = market_b = None
            odds_a = odds_b = None
            books = 0

        predicted_side = "a" if model_a >= model_b else "b"
        model_prob_winner = max(model_a, model_b)
        confidence = _confidence(model_prob_winner)
        if confidence == "high":
            high_conf_count += 1

        # Edge: compare the model's pick probability to the market's for the
        # SAME fighter. Positive edge_value => model likes them more than market.
        edge_type = None
        edge_value = None
        if market_a is not None:
            model_pick = model_a if predicted_side == "a" else model_b
            market_pick = market_a if predicted_side == "a" else market_b
            edge_value = round(model_pick - market_pick, 4)
            if edge_value >= _EDGE_THRESHOLD:
                edge_type = "model"    # model sees value the market doesn't
            elif edge_value <= -_EDGE_THRESHOLD:
                edge_type = "market"   # market is more confident than the model

        fights_out.append({
            "weight_class": fight["weight_class"],
            "fighter_a": name_a,
            "fighter_b": name_b,
            "predicted": predicted_side,
            "model_prob_a": round(model_a * 100, 1),
            "model_prob_b": round(model_b * 100, 1),
            "market_prob_a": round(market_a * 100, 1) if market_a is not None else None,
            "market_prob_b": round(market_b * 100, 1) if market_b is not None else None,
            "odds_a": odds_a,
            "odds_b": odds_b,
            "books": books,
            "edge_type": edge_type,
            "edge_value": edge_value,
            "confidence": confidence,
            "main_event": fight.get("main_event", False),
        })

    return {
        "event": card["event"],
        "event_date": card["event_date"],
        "is_ppv": card.get("is_ppv", False),
        "odds_source": odds_data["source"],
        "odds_fetched_at": odds_data["fetched_at"],
        "model_kind": "baseline",  # honest: not a trained model yet
        "summary": {
            "predictions": len(fights_out),
            "events": 1,
            "high_confidence": high_conf_count,
        },
        "fights": fights_out,
    }
