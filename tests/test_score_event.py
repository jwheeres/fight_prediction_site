"""score_event.build_payload: the model AND users get graded from a card + results."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import score_event  # noqa: E402

import app as app_module  # noqa: E402
from qualia import auth, picks, store  # noqa: E402

CARD = {
    "event": "UFC Test",
    "event_date": "2026-01-01",
    "fights": [
        {"fighter_a": "Islam Makhachev", "fighter_b": "Ian Garry", "predicted": "a"},   # model -> Makhachev
        {"fighter_a": "Sean O'Malley", "fighter_b": "Merab D", "predicted": "b"},        # model -> Merab
    ],
}
RESULTS = [
    {"fighter_a": "Islam Makhachev", "fighter_b": "Ian Garry", "winner": "Islam Makhachev"},  # model right
    {"fighter_a": "Sean O'Malley", "fighter_b": "Merab D", "winner": "Sean O'Malley"},         # model wrong
]


def test_build_payload_shapes_model_picks_and_results():
    payload = score_event.build_payload(CARD, RESULTS, "test")
    # results use the CARD's fight key + winner spelling (matches what users picked on)
    assert payload["results"] == [
        {"fight": "Islam Makhachev vs Ian Garry", "winner": "Islam Makhachev"},
        {"fight": "Sean O'Malley vs Merab D", "winner": "Sean O'Malley"},
    ]
    assert payload["summary"] == {"scored": 2, "correct": 1, "accuracy_pct": 50.0}
    assert {p["name"] for p in payload["predictor_picks"]} == {"Qualia Model"}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    monkeypatch.setattr(picks, "PICKS_FILE", tmp_path / "picks.json")
    monkeypatch.setattr(store, "LEADERBOARD_FILE", tmp_path / "leaderboard.json")
    monkeypatch.setattr(store, "RESULTS_FILE", tmp_path / "results.json")
    monkeypatch.setenv("SCORING_SECRET", "sekret")
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_scoring_grades_model_and_user_together(isolated):
    client = isolated
    client.post("/api/auth/register", json={"username": "buddy", "password": "strongpass1"})
    # buddy picks Makhachev on the same fight key the frontend uses
    client.post("/api/picks", json={"event": "UFC Test",
                                    "picks": {"Islam Makhachev vs Ian Garry": "Islam Makhachev"}})

    payload = score_event.build_payload(CARD, RESULTS, "test")
    r = client.post("/api/scheduled/score-results", headers={"X-Scoring-Secret": "sekret"}, json=payload)
    assert r.status_code == 201

    model = store.get_predictor("Qualia Model")
    assert (model["total"], model["correct"], model["points"]) == (2, 1, 10)

    buddy = store.get_predictor("buddy")
    assert (buddy["total"], buddy["correct"], buddy["points"]) == (1, 1, 10)  # graded, not ignored
