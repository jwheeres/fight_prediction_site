"""User auth (register/login/logout/me), pick submission, and grading picks
into the leaderboard when an event is scored."""

import pytest

import app as app_module
from qualia import auth, picks, store


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    monkeypatch.setattr(picks, "PICKS_FILE", tmp_path / "picks.json")
    monkeypatch.setattr(store, "LEADERBOARD_FILE", tmp_path / "leaderboard.json")
    monkeypatch.setattr(store, "RESULTS_FILE", tmp_path / "results.json")
    return tmp_path


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _register(client, u="fighter_fan", p="strongpass1"):
    return client.post("/api/auth/register", json={"username": u, "password": p})


# --- registration ----------------------------------------------------------

def test_register_success_logs_in_and_seeds_board(client, isolated):
    r = _register(client)
    assert r.status_code == 201
    assert r.get_json()["user"] == "fighter_fan"
    assert client.get("/api/auth/me").get_json()["user"] == "fighter_fan"
    assert store.get_predictor("fighter_fan") is not None  # on the board at 0


def test_register_rejects_duplicate_case_insensitive(client, isolated):
    _register(client, "Bob123", "strongpass1")
    r = _register(client, "bob123", "otherpass1")
    assert r.status_code == 400


def test_register_rejects_short_password_and_reserved_name(client, isolated):
    assert client.post("/api/auth/register", json={"username": "okname", "password": "short"}).status_code == 400
    assert client.post("/api/auth/register", json={"username": "Qualia Model", "password": "strongpass1"}).status_code == 400


# --- login / logout / me ---------------------------------------------------

def test_login_logout_cycle(client, isolated):
    _register(client, "carla", "strongpass1")
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").get_json()["user"] is None

    assert client.post("/api/auth/login", json={"username": "carla", "password": "wrong"}).status_code == 401
    ok = client.post("/api/auth/login", json={"username": "CARLA", "password": "strongpass1"})  # case-insensitive
    assert ok.status_code == 200 and ok.get_json()["user"] == "carla"
    assert client.get("/api/auth/me").get_json()["user"] == "carla"


def test_login_unknown_user_401(client, isolated):
    assert client.post("/api/auth/login", json={"username": "ghost", "password": "whatever12"}).status_code == 401


# --- picks -----------------------------------------------------------------

def test_picks_require_login(client, isolated):
    assert client.post("/api/picks", json={"event": "E1", "picks": {"A vs B": "A"}}).status_code == 401


def test_submit_and_read_picks(client, isolated):
    _register(client, "dana", "strongpass1")
    r = client.post("/api/picks", json={"event": "UFC 999", "picks": {"A vs B": "A", "C vs D": "D"}})
    assert r.status_code == 201
    got = client.get("/api/picks?event=UFC 999").get_json()["picks"]
    assert got == {"A vs B": "A", "C vs D": "D"}


# --- grading picks into the leaderboard on scoring -------------------------

def test_user_picks_graded_on_scoring(client, isolated, monkeypatch):
    _register(client, "erin", "strongpass1")
    client.post("/api/picks", json={"event": "UFC 999", "picks": {"A vs B": "A", "C vs D": "C"}})

    monkeypatch.setenv("SCORING_SECRET", "sekret")
    r = client.post(
        "/api/scheduled/score-results",
        headers={"X-Scoring-Secret": "sekret"},
        json={"event": "UFC 999", "results": [
            {"fight": "A vs B", "winner": "A"},   # erin right
            {"fight": "C vs D", "winner": "D"},   # erin wrong
        ]},
    )
    assert r.status_code == 201
    erin = store.get_predictor("erin")
    assert erin["total"] == 2
    assert erin["correct"] == 1
    assert erin["points"] == 10
