"""Per-user fight picks, stored flat-file in data/picks.json.

Structure:  { event_name: { username: { fight_key: predicted_fighter } } }

A pick is a user saying who wins a given bout on a given card. When the event
is later scored (actual winners known), grade_users() turns the stored picks
into leaderboard awards, graded the same way the model's picks are.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from qualia import db

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PICKS_FILE = DATA_DIR / "picks.json"

_LOCK = threading.Lock()


def _norm(name: str) -> str:
    """Loose name match for grading (spelling/casing differ across feeds)."""
    return " ".join((name or "").lower().split())


def _read() -> dict:
    if db.enabled():
        return db.get(PICKS_FILE.stem, {})
    if not PICKS_FILE.exists():
        return {}
    try:
        return json.loads(PICKS_FILE.read_text())
    except Exception:
        return {}


def _write(data: dict) -> None:
    if db.enabled():
        db.set(PICKS_FILE.stem, data)
        return
    PICKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PICKS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(PICKS_FILE)


def set_picks(user: str, event: str, picks: dict[str, str]) -> dict[str, str]:
    """Merge {fight_key: predicted_fighter} into a user's picks for an event."""
    with _LOCK:
        data = _read()
        event_picks = data.setdefault(event, {})
        user_picks = event_picks.setdefault(user, {})
        for fight, predicted in picks.items():
            if predicted:
                user_picks[fight] = predicted
        _write(data)
        return dict(user_picks)


def get_picks(user: str, event: str) -> dict[str, str]:
    return dict(_read().get(event, {}).get(user, {}))


def all_picks_for_event(event: str) -> dict[str, dict]:
    return dict(_read().get(event, {}))


def grade_users(event: str, results: list[dict]) -> list[dict]:
    """Grade every stored user pick for `event` against actual results.

    `results` is a list of {"fight": <same key users picked on>, "winner": name}.
    Returns predictor-pick dicts ({name, correct, ...}) ready for the leaderboard
    — one per (user, graded fight). Fights a user didn't pick are skipped.
    """
    winners = {r["fight"]: r["winner"] for r in results if r.get("fight") and r.get("winner")}
    awards: list[dict] = []
    for user, user_picks in all_picks_for_event(event).items():
        for fight, predicted in user_picks.items():
            if fight not in winners:
                continue
            awards.append({
                "name": user,
                "correct": _norm(predicted) == _norm(winners[fight]),
                "fight": fight,
                "predicted": predicted,
                "winner": winners[fight],
            })
    return awards
