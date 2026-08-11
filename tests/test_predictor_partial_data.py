"""Tests for the fight predictor's partial-data handling.

Focus: an unknown fighter stat must NOT silently zero-fill into a
confident-looking prediction. Unknown-on-either-side is neutral; a genuine
full-unknown matchup is a coin flip; and the API surfaces when a result was
computed on partial data.
"""

import json
from pathlib import Path

import pytest

import app as app_module
from qualia import model as M

FIGHTERS = json.loads((Path(__file__).resolve().parent.parent / "data" / "fighters_stats.json").read_text())
_NAMES = list(FIGHTERS)
KNOWN = FIGHTERS[_NAMES[0]]


# --- pure diff/missing logic (model-independent, exact) --------------------

def test_missing_features_absent_and_none_only():
    stats = dict(KNOWN)
    stats["reach_in"] = None          # explicit None counts as missing
    del stats["age"]                  # absent key counts as missing
    stats["kd_per15"] = 0.0           # a real 0 is NOT missing
    missing = set(M.missing_features(stats, M.FEATURES))
    assert "reach_in" in missing
    assert "age" in missing
    assert "kd_per15" not in missing


def test_diff_neutralizes_unknown_side():
    # Same fighter minus one stat -> every diff is 0 (the dropped feature is
    # neutralized, the rest are identical), NOT a skew from zero-filling.
    partial = {k: v for k, v in KNOWN.items() if k != "reach_in"}
    diffs = M.diff_vector(KNOWN, partial)
    assert all(d == 0.0 for d in diffs)


# --- win_probability behavior ---------------------------------------------

def test_full_unknown_is_coin_flip():
    assert M.win_probability({}, {}) == pytest.approx(0.5, abs=0.02)


def test_partial_data_does_not_skew():
    # Regression guard: pre-fix this returned ~0.77 for a fighter vs itself
    # with one stat dropped. Neutralizing missing features must keep it ~0.5.
    partial = {k: v for k, v in KNOWN.items() if k != "reach_in"}
    assert M.win_probability(KNOWN, partial) == pytest.approx(0.5, abs=0.02)


def test_known_vs_known_is_symmetric():
    a, b = FIGHTERS[_NAMES[0]], FIGHTERS[_NAMES[10]]
    assert M.win_probability(a, b) + M.win_probability(b, a) == pytest.approx(1.0, abs=1e-9)


# --- API surface -----------------------------------------------------------

@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_predict_reports_real_model_kind(client):
    a, b = FIGHTERS[_NAMES[0]], FIGHTERS[_NAMES[10]]
    resp = client.post("/predict", json={"fighter_a": a, "fighter_b": b})
    assert resp.status_code == 200
    body = resp.get_json()
    # Must reflect the actual model, not a hardcoded "baseline".
    assert body["model_kind"] == M.MODEL_KIND
    # Full data -> no insufficient_data flag.
    assert "insufficient_data" not in body


def test_predict_flags_insufficient_data(client):
    a = FIGHTERS[_NAMES[0]]
    resp = client.post("/predict", json={"fighter_a": a, "fighter_b": {}})
    assert resp.status_code == 200
    body = resp.get_json()
    assert "insufficient_data" in body
    assert set(body["insufficient_data"]["fighter_b_missing"]) == set(M._feature_list())
    assert body["insufficient_data"]["fighter_a_missing"] == []
