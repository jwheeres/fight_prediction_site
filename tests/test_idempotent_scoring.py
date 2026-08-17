"""Scoring is idempotent: re-running the same event can't inflate the board.

This is what makes a daily auto-scoring cron safe."""

import pytest

from qualia import store


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "LEADERBOARD_FILE", tmp_path / "lb.json")
    monkeypatch.setattr(store, "RESULTS_FILE", tmp_path / "res.json")


def _event():
    return {"event": "UFC X", "event_date": "2026-01-01", "predictor_picks": [
        {"name": "Qualia Model", "correct": True},
        {"name": "Qualia Model", "correct": False},
        {"name": "alice", "correct": True},
    ]}


def test_rescoring_same_event_does_not_double_count(isolated):
    store.record_scored_event(_event())
    m1, a1 = store.get_predictor("Qualia Model"), store.get_predictor("alice")
    assert (m1["total"], m1["correct"], m1["points"]) == (2, 1, 10)
    assert (a1["total"], a1["correct"], a1["points"]) == (1, 1, 10)

    store.record_scored_event(_event())  # exact same event again
    m2, a2 = store.get_predictor("Qualia Model"), store.get_predictor("alice")
    assert (m2["total"], m2["correct"], m2["points"]) == (2, 1, 10)  # unchanged
    assert (a2["total"], a2["correct"], a2["points"]) == (1, 1, 10)


def test_registered_user_survives_rebuild(isolated):
    store.ensure_predictor("bob")  # signed up, no picks yet
    store.record_scored_event(_event())
    bob = store.get_predictor("bob")
    assert bob is not None and bob["points"] == 0  # still on the board at 0


def test_distinct_events_accumulate_with_streak(isolated):
    store.record_scored_event({"event": "E1", "event_date": "2026-01-01",
                               "predictor_picks": [{"name": "alice", "correct": True}]})
    store.record_scored_event({"event": "E2", "event_date": "2026-01-08",
                               "predictor_picks": [{"name": "alice", "correct": True}]})
    a = store.get_predictor("alice")
    assert (a["total"], a["correct"], a["points"], a["streak"], a["best_streak"]) == (2, 2, 20, 2, 2)
