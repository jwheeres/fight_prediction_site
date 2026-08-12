"""Database-backed persistence (exercised via SQLite; prod uses Postgres/Neon).

Proves the same auth/picks/scoring flow works through a real database AND that
the data survives a simulated redeploy (dropping the connection and reopening
the same database file) — which is the whole point of moving off ephemeral
flat files.
"""

import pytest

import app as app_module
from qualia import auth, db, picks, store


@pytest.fixture
def db_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + str(tmp_path / "qualia.db"))
    monkeypatch.setenv("SCORING_SECRET", "sekret")
    db.reset()
    store.ensure_seed()  # seed the model into the fresh (empty) database
    yield
    db.reset()


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_flow_persists_through_the_database(db_mode, client):
    assert db.enabled()

    # Register, pick, score — all through the DB backend.
    assert client.post("/api/auth/register", json={"username": "neon_fan", "password": "strongpass1"}).status_code == 201
    client.post("/api/picks", json={"event": "UFC 1000", "picks": {"A vs B": "A"}})
    r = client.post("/api/scheduled/score-results",
                    headers={"X-Scoring-Secret": "sekret"},
                    json={"event": "UFC 1000", "results": [{"fight": "A vs B", "winner": "A"}]})
    assert r.status_code == 201
    assert store.get_predictor("neon_fan")["points"] == 10

    # Nothing was written to the flat files — it's all in the database.
    assert db.get("users", {}).get("neon_fan") is not None
    assert db.get("picks", {}).get("UFC 1000", {}).get("neon_fan") == {"A vs B": "A"}

    # Simulate a redeploy: drop the cached connection, reopen the same DB file.
    db.reset()
    assert auth.exists("neon_fan")                                  # account survived
    assert store.get_predictor("neon_fan")["points"] == 10         # points survived
    assert picks.get_picks("neon_fan", "UFC 1000") == {"A vs B": "A"}  # picks survived
    assert store.get_predictor("Qualia Model") is not None          # model seed survived


def test_falls_back_to_files_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db.reset()
    assert db.enabled() is False
