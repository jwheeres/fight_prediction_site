"""Scoring-endpoint lockdown, model-as-predictor, and the profile endpoint."""

import json
from pathlib import Path

import pytest

import app as app_module
from qualia import store


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Point the flat-file store at a temp dir so tests never touch real data."""
    monkeypatch.setattr(store, "LEADERBOARD_FILE", tmp_path / "leaderboard.json")
    monkeypatch.setattr(store, "RESULTS_FILE", tmp_path / "results.json")
    return tmp_path


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _score_payload():
    return {
        "event": "Test Event",
        "predictor_picks": [
            {"name": "Qualia Model", "correct": True},
            {"name": "Qualia Model", "correct": False},
        ],
    }


# --- scoring endpoint lockdown --------------------------------------------

def test_scoring_refused_when_secret_unset(client, isolated_store, monkeypatch):
    monkeypatch.delenv("SCORING_SECRET", raising=False)
    r = client.post("/api/scheduled/score-results", json=_score_payload())
    assert r.status_code == 503  # fail-safe, not open


def test_scoring_rejects_wrong_secret(client, isolated_store, monkeypatch):
    monkeypatch.setenv("SCORING_SECRET", "right-secret")
    r = client.post("/api/scheduled/score-results", json=_score_payload(),
                    headers={"X-Scoring-Secret": "wrong"})
    assert r.status_code == 401


def test_scoring_accepts_correct_secret_and_awards_model(client, isolated_store, monkeypatch):
    monkeypatch.setenv("SCORING_SECRET", "right-secret")
    r = client.post("/api/scheduled/score-results", json=_score_payload(),
                    headers={"X-Scoring-Secret": "right-secret"})
    assert r.status_code == 201
    model = store.get_predictor("Qualia Model")
    assert model is not None
    assert model["points"] == 10   # one correct pick * 10
    assert model["correct"] == 1
    assert model["total"] == 2


# --- predictor profile -----------------------------------------------------

def test_predictor_profile_returns_standing_and_record(client, isolated_store):
    store.save_leaderboard([
        {"name": "Qualia Model", "points": 30, "correct": 3, "total": 5,
         "streak": 1, "best_streak": 2, "is_model": True},
    ])
    r = client.get("/api/predictor/Qualia Model")
    assert r.status_code == 200
    body = r.get_json()
    assert body["predictor"]["name"] == "Qualia Model"
    assert body["rank"] == 1
    assert "track_record" in body  # model gets its honest track-record summary


def test_predictor_profile_unknown_404(client, isolated_store):
    store.save_leaderboard([])
    r = client.get("/api/predictor/Nobody")
    assert r.status_code == 404


# --- committed seed data ---------------------------------------------------

def test_seed_leaderboard_has_no_fictional_handles():
    board = json.loads((Path(__file__).resolve().parent.parent / "data" / "leaderboard.json").read_text())
    names = {e["name"] for e in board}
    fake = {"KO_Prophet", "OctagonOracle", "SubmissionSeer", "GroundGameGuru",
            "FightIQ", "TapologyTom", "CageWisdom"}
    assert not (names & fake), "fictional predictors must be gone from the seed board"
    assert "Qualia Model" in names, "the model should seed the real board"
