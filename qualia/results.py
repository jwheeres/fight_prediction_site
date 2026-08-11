"""Past results / model track record.

Reads scored historical fights from data/past_results.json and computes an
honest accuracy summary (overall + high-confidence). This is what powers the
"Track Record" tab — credibility comes from showing the misses too.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PAST_RESULTS_FILE = DATA_DIR / "past_results.json"


def _round(n: float, d: int = 1) -> float:
    return round(n, d)


def get_track_record() -> dict:
    if not PAST_RESULTS_FILE.exists():
        return {"events": [], "summary": {"scored": 0, "correct": 0, "accuracy": 0.0,
                                          "high_conf_scored": 0, "high_conf_correct": 0, "high_conf_accuracy": 0.0}}

    data = json.loads(PAST_RESULTS_FILE.read_text())
    events = data.get("events", [])

    scored = correct = hc_scored = hc_correct = 0
    for ev in events:
        for f in ev["fights"]:
            scored += 1
            if f.get("correct"):
                correct += 1
            if f.get("high_conf"):
                hc_scored += 1
                if f.get("correct"):
                    hc_correct += 1

    return {
        "events": events,
        "summary": {
            "scored": scored,
            "correct": correct,
            "accuracy": _round(correct / scored * 100) if scored else 0.0,
            "high_conf_scored": hc_scored,
            "high_conf_correct": hc_correct,
            "high_conf_accuracy": _round(hc_correct / hc_scored * 100) if hc_scored else 0.0,
        },
    }
