#!/usr/bin/env python3
"""Automated post-event scoring.

Replaces the old Manus `score_and_post.py` (which hardcoded picks and whose
POST always 404/502'd). This version:

  1. Loads the current card's MODEL predictions (from qualia.card).
  2. Gets actual fight results — from The Odds API scores endpoint when
     ODDS_API_KEY is set, otherwise from a local results file
     (data/results_input.json).
  3. Scores the model's predictions against the results.
  4. POSTs the scored payload to the app's WORKING endpoint so the leaderboard
     updates (the model runs as the predictor "Qualia Model").

Usage:
    APP_BASE_URL=http://localhost:5000 python scripts/score_event.py
    # live results:
    ODDS_API_KEY=xxx APP_BASE_URL=https://your-app python scripts/score_event.py
    # local results file:
    python scripts/score_event.py --results data/results_input.json

Results file format (list of {"fighter_a","fighter_b","winner"}):
    [{"fighter_a": "Islam Makhachev", "fighter_b": "Ian Machado Garry",
      "winner": "Islam Makhachev"}]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qualia import card as card_service  # noqa: E402
from qualia import odds as odds_service  # noqa: E402

PREDICTOR_NAME = "Qualia Model"
SCORES_URL = f"{odds_service.ODDS_API_BASE}/sports/{odds_service.SPORT_KEY}/scores"


def fetch_live_results() -> list[dict]:
    """Best-effort: completed fights from The Odds API scores endpoint."""
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        return []
    try:
        resp = requests.get(SCORES_URL, params={"apiKey": key, "daysFrom": 3}, timeout=12)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  live scores fetch failed ({exc}); will try a results file")
        return []

    results = []
    for ev in resp.json():
        if not ev.get("completed"):
            continue
        scores = ev.get("scores") or []
        if len(scores) != 2:
            continue
        # Winner = higher score (MMA scores are typically 1/0 for win/loss).
        try:
            winner = max(scores, key=lambda s: float(s.get("score", 0)))["name"]
        except (ValueError, TypeError):
            continue
        results.append({"fighter_a": ev.get("home_team"), "fighter_b": ev.get("away_team"), "winner": winner})
    return results


def load_file_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, help="Local results JSON file")
    parser.add_argument("--base-url", default=os.getenv("APP_BASE_URL", "http://localhost:5000"))
    parser.add_argument("--dry-run", action="store_true", help="Score but don't POST")
    args = parser.parse_args()

    card = card_service.build_card()
    predictions = {
        odds_service.normalize_name(f["fighter_a"]): (
            f["fighter_a"] if f["predicted"] == "a" else f["fighter_b"]
        )
        for f in card["fights"]
    }

    results = fetch_live_results()
    source = "odds-api-scores"
    if not results and args.results:
        results = load_file_results(args.results)
        source = str(args.results)
    if not results:
        print("No results available (no live scores and no --results file). Nothing to score.")
        return 1

    picks = []
    correct = 0
    for r in results:
        key = odds_service.normalize_name(r["fighter_a"])
        predicted = predictions.get(key)
        if not predicted:
            continue
        is_correct = odds_service.normalize_name(predicted) == odds_service.normalize_name(r["winner"])
        correct += 1 if is_correct else 0
        picks.append({"name": PREDICTOR_NAME, "correct": is_correct,
                      "fight": f'{r["fighter_a"]} vs {r["fighter_b"]}', "predicted": predicted, "winner": r["winner"]})

    total = len(picks)
    accuracy = round(correct / total * 100, 1) if total else 0.0
    payload = {
        "event": card["event"],
        "event_date": card["event_date"],
        "source": source,
        "summary": {"scored": total, "correct": correct, "accuracy_pct": accuracy},
        "predictor_picks": picks,
    }

    print(f"Scored {total} fights: {correct} correct ({accuracy}%) for '{PREDICTOR_NAME}'")
    if args.dry_run:
        print(json.dumps(payload["summary"], indent=2))
        return 0

    url = f"{args.base_url.rstrip('/')}/api/scheduled/score-results"
    headers = {}
    secret = os.getenv("SCORING_SECRET", "").strip()
    if secret:
        headers["X-Scoring-Secret"] = secret
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        print(f"POST {url} -> {resp.status_code}: {resp.text[:200]}")
        return 0 if resp.status_code in (200, 201, 202) else 2
    except Exception as exc:
        print(f"POST failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
