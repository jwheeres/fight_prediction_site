"""Build the fight card.

Two modes:
  * LIVE  — when the odds service returns real market data, the card is built
    *from that feed*: the fights shown are the real upcoming MMA matchups the
    sportsbooks are pricing right now. The model runs on any fighter we have
    stats for; fighters we don't have data on show market-only (model n/a).
  * DEMO  — when there are no live odds (no key / cold with no cache), fall back
    to the bundled static card (data/card.json) so the page still renders.

For each fight we derive the market consensus, the model's win probability, the
"edge" (model vs. market disagreement), a confidence label, and the predicted
winner. This is the payload the frontend renders.
"""

from __future__ import annotations

import json
from pathlib import Path

from qualia import model, odds

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CARD_FILE = DATA_DIR / "card.json"
# Real career stats for ~2,700 fighters (scripts/build_fighter_data.py); falls
# back to the small hand-seeded table if the real one isn't present.
FIGHTERS_FILE = DATA_DIR / "fighters_stats.json"
_FALLBACK_FIGHTERS_FILE = DATA_DIR / "fighters.json"

_EDGE_THRESHOLD = 0.08     # model vs market gap (8 pts) before we flag an edge
_HIGH_CONF_MODEL = 0.65    # model strongly favors one side
_TOSS_UP_BAND = 0.06       # model within 6 pts of 50/50
_MAX_LIVE_FIGHTS = 30      # cap the live board to a sane length


def _load(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _fighter_stats() -> dict:
    """Normalized-name -> stats, so odds-feed names match our stat table."""
    path = FIGHTERS_FILE if FIGHTERS_FILE.exists() else _FALLBACK_FIGHTERS_FILE
    raw = _load(path)
    return {odds.normalize_name(k): v for k, v in raw.items() if not k.startswith("_")}


def _confidence(model_prob_winner: float) -> str:
    if model_prob_winner >= _HIGH_CONF_MODEL:
        return "high"
    if abs(model_prob_winner - 0.5) <= _TOSS_UP_BAND:
        return "toss"
    return "lean"


def _score_fight(name_a: str, name_b: str, stats: dict,
                 market_a, market_b) -> dict:
    """Model + edge + confidence for one fight. market_* are 0-1 or None.

    Returns the per-fight fields shared by both build modes. When we have no
    stats for a fighter, the model side is None (market-only card).
    """
    sa = stats.get(odds.normalize_name(name_a))
    sb = stats.get(odds.normalize_name(name_b))

    if sa and sb:
        model_a = model.win_probability(sa, sb)
        model_b = 1.0 - model_a
        predicted = "a" if model_a >= model_b else "b"
        confidence = _confidence(max(model_a, model_b))
        edge_type = edge_value = None
        if market_a is not None:
            model_pick = model_a if predicted == "a" else model_b
            market_pick = market_a if predicted == "a" else market_b
            edge_value = round(model_pick - market_pick, 4)
            if edge_value >= _EDGE_THRESHOLD:
                edge_type = "model"
            elif edge_value <= -_EDGE_THRESHOLD:
                edge_type = "market"
        return {
            "predicted": predicted,
            "model_prob_a": round(model_a * 100, 1),
            "model_prob_b": round(model_b * 100, 1),
            "confidence": confidence,
            "edge_type": edge_type,
            "edge_value": edge_value,
            "has_model": True,
        }

    # No model data: fall back to the market favorite for the "pick", no edge.
    if market_a is not None:
        predicted = "a" if market_a >= market_b else "b"
    else:
        predicted = "a"
    return {
        "predicted": predicted,
        "model_prob_a": None,
        "model_prob_b": None,
        "confidence": None,
        "edge_type": None,
        "edge_value": None,
        "has_model": False,
    }


def _build_from_feed(odds_data: dict) -> dict:
    """Card built from the live odds feed — the real upcoming matchups."""
    stats = _fighter_stats()
    events = sorted(
        odds_data["events"], key=lambda e: e.get("commence_time") or ""
    )[:_MAX_LIVE_FIGHTS]

    fights_out = []
    high_conf = 0
    for ev in events:
        a, b = ev["fighter_a"], ev["fighter_b"]
        market_a, market_b = ev["market_prob_a"], ev["market_prob_b"]
        scored = _score_fight(a, b, stats, market_a, market_b)
        if scored["confidence"] == "high":
            high_conf += 1
        fights_out.append({
            "weight_class": None,
            "commence_time": ev.get("commence_time"),
            "fighter_a": a,
            "fighter_b": b,
            "market_prob_a": round(market_a * 100, 1) if market_a is not None else None,
            "market_prob_b": round(market_b * 100, 1) if market_b is not None else None,
            "odds_a": ev.get("odds_a"),
            "odds_b": ev.get("odds_b"),
            "books": ev.get("books", 0),
            "main_event": False,
            **scored,
        })

    first_dt = events[0].get("commence_time") if events else None
    event_date = (first_dt or "")[:10] or None
    return {
        "event": "Upcoming Fights",
        "event_date": event_date,
        "is_ppv": False,
        "live_feed": True,
        "odds_source": odds_data["source"],
        "odds_fetched_at": odds_data["fetched_at"],
        "model_kind": model.MODEL_KIND,
        "model_accuracy": model.MODEL_METRICS.get("accuracy"),
        "summary": {
            "predictions": len(fights_out),
            "events": len({(f.get("commence_time") or "")[:10] for f in fights_out}),
            "high_confidence": high_conf,
        },
        "fights": fights_out,
    }


def _build_static(force_refresh: bool) -> dict:
    """Demo card from data/card.json — used only when there are no live odds."""
    card = _load(CARD_FILE)
    stats = _fighter_stats()
    odds_data = odds.build_odds_index(force_refresh=force_refresh)
    index = odds_data["index"]

    fights_out = []
    high_conf = 0
    for fight in card["fights"]:
        a, b = fight["fighter_a"], fight["fighter_b"]
        row = index.get(odds.normalize_name(a))
        if row:
            if row["side"] == "a":
                market_a, market_b = row["market_prob_a"], row["market_prob_b"]
                odds_a, odds_b = row["odds_a"], row["odds_b"]
            else:
                market_a, market_b = row["market_prob_b"], row["market_prob_a"]
                odds_a, odds_b = row["odds_b"], row["odds_a"]
            books = row["books"]
        else:
            market_a = market_b = odds_a = odds_b = None
            books = 0

        scored = _score_fight(a, b, stats, market_a, market_b)
        if scored["confidence"] == "high":
            high_conf += 1
        fights_out.append({
            "weight_class": fight["weight_class"],
            "commence_time": None,
            "fighter_a": a,
            "fighter_b": b,
            "market_prob_a": round(market_a * 100, 1) if market_a is not None else None,
            "market_prob_b": round(market_b * 100, 1) if market_b is not None else None,
            "odds_a": odds_a,
            "odds_b": odds_b,
            "books": books,
            "main_event": fight.get("main_event", False),
            **scored,
        })

    return {
        "event": card["event"],
        "event_date": card["event_date"],
        "is_ppv": card.get("is_ppv", False),
        "live_feed": False,
        "odds_source": odds_data["source"],
        "odds_fetched_at": odds_data["fetched_at"],
        "model_kind": model.MODEL_KIND,
        "model_accuracy": model.MODEL_METRICS.get("accuracy"),
        "summary": {"predictions": len(fights_out), "events": 1, "high_confidence": high_conf},
        "fights": fights_out,
    }


def build_card(force_refresh: bool = False) -> dict:
    """Live feed when we have real odds, otherwise the demo card."""
    odds_data = odds.get_odds(force_refresh=force_refresh)
    if odds_data["source"] in ("live", "cached") and odds_data["events"]:
        return _build_from_feed(odds_data)
    return _build_static(force_refresh)
