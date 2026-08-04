"""Fight win-probability model.

IMPORTANT: the repo's `rf_h2h_model_updated.pkl` is NOT a trained model — it
deserializes to a NumPy array of 16 feature *names* (the schema a model would
expect). So this module does two things:

  1. Loads that schema from the pkl (so we're grounded in the real artifact and
     will notice if it changes).
  2. Provides a transparent, deterministic BASELINE predictor over those
     features. It is intentionally simple and explainable — not a black box —
     and clearly labeled as a baseline until a real trained model is supplied.

To plug in a real model later, replace `win_probability` with a call into your
trained estimator; keep the same signature and the rest of the app is unchanged.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

MODEL_FILE = Path(__file__).resolve().parent.parent / "rf_h2h_model_updated.pkl"

# Per-fighter stat weights for the baseline. Positive = higher is better.
# These are sensible priors, not fitted coefficients — hence "baseline".
_WEIGHTS = {
    "striking_balance": 2.4,       # striking accuracy / output balance
    "finish_rate": 1.8,            # ability to end fights
    "strike_defense": 1.6,         # not getting hit
    "takedown_success_rate": 1.1,  # grappling offense
    "takedown_defense": 1.1,       # grappling defense
    "knockdown_rate": 1.3,         # power
    "ground_control": 0.9,
    "clinch_control": 0.7,
}


def load_feature_schema() -> list[str]:
    """Return the 16 feature names stored in the pkl (the real artifact)."""
    try:
        obj = pickle.loads(MODEL_FILE.read_bytes())
        return [str(x) for x in list(obj)]
    except Exception:
        return []


FEATURE_SCHEMA = load_feature_schema()


def _fighter_score(stats: dict) -> float:
    return sum(weight * float(stats.get(stat, 0.0)) for stat, weight in _WEIGHTS.items())


def win_probability(stats_a: dict, stats_b: dict) -> float:
    """P(fighter A beats fighter B) from their stats, in (0, 1).

    Baseline: logistic of the weighted skill-score difference. Deterministic
    and monotonic in each stat, so results are explainable.
    """
    diff = _fighter_score(stats_a) - _fighter_score(stats_b)
    # Scale so a ~1.0 score edge maps to a meaningful (but not extreme) prob.
    return 1.0 / (1.0 + math.exp(-diff))


def build_feature_row(stats_a: dict, stats_b: dict) -> dict:
    """Assemble the 16-feature row matching the pkl schema (for inspection/debug)."""
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
