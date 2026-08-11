"""Fight win-probability model (real, trained on ufcstats data).

Feature vector for a fight = the per-fighter career-stat *differences*
(fighter A minus fighter B) over FEATURES. A trained scikit-learn model
(models/h2h_model.pkl, produced by qualia/train.py on data/training.csv) turns
that into P(A wins). If no trained model is present, a transparent baseline
keeps the app running.

Stats come from data/fighters_stats.json (built by scripts/build_fighter_data.py).
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINED_FILE = ROOT / "models" / "h2h_model.pkl"

# Per-fighter features, in the exact order the model expects the diffs.
# NOTE: career win_rate / finish_rate are deliberately EXCLUDED — computed over
# a fighter's whole career they encode the very outcomes we're predicting
# (look-ahead leakage) and would inflate accuracy. We keep style, grappling,
# and physical metrics, which describe *how* a fighter fights. Re-including an
# as-of (pre-fight) win_rate is the honest follow-up.
FEATURES = [
    "sig_str_acc", "sig_str_def", "slpm", "sapm",
    "td_acc", "td_def", "td_per15", "sub_per15", "kd_per15",
    "ctrl_pf", "reach_in", "height_in", "age", "ufc_fights",
]

# Baseline weights (used only if no trained model). Kept to 0-1 style features
# so the logistic is well-scaled; deliberately simple and explainable.
_BASELINE_W = {
    "win_rate": 3.0, "sig_str_acc": 1.5, "sig_str_def": 1.5,
    "td_def": 1.0, "finish_rate": 0.8,
}


def diff_vector(stats_a: dict, stats_b: dict) -> list[float]:
    return [float(stats_a.get(k, 0.0)) - float(stats_b.get(k, 0.0)) for k in FEATURES]


_TRAINED = None
MODEL_KIND = "baseline"
MODEL_METRICS: dict = {}


def _try_load_trained():
    global _TRAINED, MODEL_KIND, MODEL_METRICS
    if not TRAINED_FILE.exists():
        return
    try:
        bundle = pickle.loads(TRAINED_FILE.read_bytes())
        est = bundle["model"]
        feats = bundle.get("features", FEATURES)
        classes = list(getattr(est, "classes_", [0, 1]))
        pos = classes.index(1) if 1 in classes else len(classes) - 1
        _TRAINED = {"est": est, "features": feats, "pos": pos}
        MODEL_KIND = bundle.get("kind", "trained")
        MODEL_METRICS = bundle.get("metrics", {})
    except Exception:
        _TRAINED = None
        MODEL_KIND = "baseline"


_try_load_trained()


def _baseline_prob(stats_a: dict, stats_b: dict) -> float:
    s = sum(w * (float(stats_a.get(k, 0.0)) - float(stats_b.get(k, 0.0)))
            for k, w in _BASELINE_W.items())
    return 1.0 / (1.0 + math.exp(-s))


def win_probability(stats_a: dict, stats_b: dict) -> float:
    """P(fighter A beats fighter B), in (0, 1)."""
    if _TRAINED is not None:
        vec = [[float(stats_a.get(k, 0.0)) - float(stats_b.get(k, 0.0))
                for k in _TRAINED["features"]]]
        return float(_TRAINED["est"].predict_proba(vec)[0][_TRAINED["pos"]])
    return _baseline_prob(stats_a, stats_b)
