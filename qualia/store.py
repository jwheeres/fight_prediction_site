"""Tiny JSON-file persistence for the leaderboard and scored results.

Deliberately a flat file, not a database — it's enough for a single-instance
app and keeps the whole thing dependency-free and easy to inspect. Swap for a
real DB (Postgres, SQLite) when there are concurrent writers.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from qualia import db

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEADERBOARD_FILE = DATA_DIR / "leaderboard.json"
RESULTS_FILE = DATA_DIR / "results.json"

_LOCK = threading.Lock()


def _read(path: Path, default):
    if db.enabled():
        return db.get(path.stem, default)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write(path: Path, payload) -> None:
    if db.enabled():
        db.set(path.stem, payload)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)  # atomic on POSIX


def get_leaderboard() -> list[dict]:
    board = _read(LEADERBOARD_FILE, [])
    board.sort(key=lambda p: (-p.get("points", 0), -p.get("correct", 0)))
    return board


def save_leaderboard(board: list[dict]) -> None:
    with _LOCK:
        _write(LEADERBOARD_FILE, board)


def ensure_predictor(name: str, **extra) -> dict:
    """Add a zeroed leaderboard entry for `name` if absent (e.g. on signup),
    so a new predictor shows on the board before their first scored pick."""
    with _LOCK:
        board = _read(LEADERBOARD_FILE, [])
        for entry in board:
            if entry.get("name") == name:
                return entry
        entry = {"name": name, "points": 0, "correct": 0, "total": 0,
                 "streak": 0, "best_streak": 0, **extra}
        board.append(entry)
        _write(LEADERBOARD_FILE, board)
        return entry


def ensure_seed() -> None:
    """Ensure the model is on the board. In file mode the committed
    leaderboard.json already seeds it; in DB mode the table starts empty, so
    this puts 'Qualia Model' there on first run. Idempotent."""
    ensure_predictor("Qualia Model", is_model=True)


def get_predictor(name: str) -> dict | None:
    for entry in get_leaderboard():
        if entry.get("name") == name:
            return entry
    return None


def get_rank(name: str) -> int | None:
    """1-based rank on the (points-sorted) leaderboard, or None if absent."""
    for i, entry in enumerate(get_leaderboard(), start=1):
        if entry.get("name") == name:
            return i
    return None


def get_results() -> list[dict]:
    return _read(RESULTS_FILE, [])


def record_scored_event(payload: dict) -> dict:
    """Persist a scored event and award leaderboard points.

    This is the endpoint the old Manus scoring task could never reach (it kept
    getting 404/502). Here it actually lands: we store the event's results and,
    for any predictor picks included, award +10 per correct pick and update
    their record/streak.
    """
    with _LOCK:
        results = _read(RESULTS_FILE, [])
        # De-dupe by event name so re-running a score doesn't double count.
        results = [r for r in results if r.get("event") != payload.get("event")]
        results.append(payload)
        _write(RESULTS_FILE, results)

        board = {p["name"]: p for p in _read(LEADERBOARD_FILE, [])}
        for pick in payload.get("predictor_picks", []):
            name = pick.get("name")
            if not name:
                continue
            entry = board.setdefault(
                name, {"name": name, "points": 0, "correct": 0, "total": 0, "streak": 0, "best_streak": 0}
            )
            entry["total"] += 1
            if pick.get("correct"):
                entry["points"] += 10
                entry["correct"] += 1
                entry["streak"] += 1
                entry["best_streak"] = max(entry["best_streak"], entry["streak"])
            else:
                entry["streak"] = 0
        new_board = list(board.values())
        _write(LEADERBOARD_FILE, new_board)

    return {"stored_event": payload.get("event"), "leaderboard_size": len(new_board)}
