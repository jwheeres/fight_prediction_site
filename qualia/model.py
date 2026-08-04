"""Fight win-probability model.

Two modes, resolved at import time:

  1. TRAINED model — if models/h2h_model.pkl exists (produced by
     qualia/train.py on a real dataset), it is loaded and used. This is a real
     scikit-learn estimator plus the feature order it was trained on.

  2. BASELINE — otherwise, a transparent, deterministic fallback so the app
     always runs. Clearly labeled as a baseline, not a trained model.

The repo's original rf_h2h_model_updated.pkl is NOT a model — it's the 16
feature *names* a model should expect. We still read it as the canonical
feature schema so training and inference agree on column order.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_FILE = ROOT / "rf_h2h_model_updated.pkl"
TRAINED_FILE = ROOT / "models" / "h2h_model.pkl"

# Per-fighter stat weights for the BASELINE. Positive = higher is better.
_WEIGHTS = {
    "striking_balance": 2.4, "finish_rate": 1.8, "strike_defense": 1.6,
    "takedown_success_rate": 1.1, "takedown_defense": 1.1, "knockdown_rate": 1.3,
    "ground_control": 0.9, "clinch_control": 0.7,
}


def load_feature_schema() -> list[str]:
    """The 16 feature names stored in the original pkl (canonical column order)."""
    try:
        obj = pickle.loads(SCHEMA_FILE.read_bytes())
        return [str(x) for x in list(obj)]
    except Exception:
        return []


FEATURE_SCHEMA = load_feature_schema()


def build_feature_row(stats_a: dict, stats_b: dict) -> dict:
    """Assemble the 16-feature row (schema-named) from two fighters' stats."""
    return {
        "f1_avg_fight_duration": stats_a.get("avg_fight_duration", 0.0),
        "f2_avg_fight_duration": stats_b.get("avg_fight_duration", 0.0),
        "f1_knockdown_rate": stats_a.get("knockdown_rate", 0.0),
        "f2_knockdown_rate": stats_b.get("knockdown_rate", 0.0),
        "f1_takedown_success_rate": stats_a.get("takedown_success_rate", 0.0),
        "f2_takedown_success_rate": stats_b.get("takedown_success_rate", 0.0),
        "striking_accuracy_diff": stats_a.get("striking_balance", 0.0) - stats_b.get("striking_balance", 0.0),
        "f1_strike_defense": stats_a.get("strike_defense", 0.0),
        "f2_strike_defense": stats_b.get("strike_defense", 0.0),
        "takedown_defense_diff": stats_a.get("takedown_defense", 0.0) - stats_b.get("takedown_defense", 0.0),
        "f1_striking_balance": stats_a.get("striking_balance", 0.0),
        "f2_striking_balance": stats_b.get("striking_balance", 0.0),
        "clinch_control_diff": stats_a.get("clinch_control", 0.0) - stats_b.get("clinch_control", 0.0),
        "ground_control_diff": stats_a.get("ground_control", 0.0) - stats_b.get("ground_control", 0.0),
        "f1_finish_rate": stats_a.get("finish_rate", 0.0),
        "f2_finish_rate": stats_b.get("finish_rate", 0.0),
    }


# ---- Trained-model loading (optional) --------------------------------------
_TRAINED = None
MODEL_KIND = "baseline"

def _try_load_trained():
    global _TRAINED, MODEL_KIND
    if not TRAINED_FILE.exists():
        return
    try:
        bundle = pickle.loads(TRAINED_FILE.read_bytes())
        # bundle = {"model": estimator, "features": [...], "kind": "...", ...}
        est = bundle["model"]
        feats = bundle.get("features") or FEATURE_SCHEMA
        classes = list(getattr(est, "classes_", [0, 1]))
        pos_index = classes.index(1) if 1 in classes else len(classes) - 1
        _TRAINED = {"est": est, "features": feats, "pos_index": pos_index}
        MODEL_KIND = bundle.get("kind", "trained")
    except Exception:
        _TRAINED = None
        MODEL_KIND = "baseline"

_try_load_trained()


def _baseline_prob(stats_a: dict, stats_b: dict) -> float:
    score = lambda s: sum(w * float(s.get(k, 0.0)) for k, w in _WEIGHTS.items())
    return 1.0 / (1.0 + math.exp(-(score(stats_a) - score(stats_b))))


def win_probability(stats_a: dict, stats_b: dict) -> float:
    """P(fighter A beats fighter B), in (0, 1).

    Uses the trained model if one is loaded, else the baseline.
    """
    if _TRAINED is not None:
        row = build_feature_row(stats_a, stats_b)
        vector = [[float(row.get(name, 0.0)) for name in _TRAINED["features"]]]
        proba = _TRAINED["est"].predict_proba(vector)[0]
        return float(proba[_TRAINED["pos_index"]])
    return _baseline_prob(stats_a, stats_b)
