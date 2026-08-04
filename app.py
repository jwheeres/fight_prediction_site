"""Qualia Bet Market — backend API + static frontend.

Endpoints:
  GET  /                          -> the frontend (static/index.html)
  GET  /api/card                  -> fight card: model probs merged with live market odds
  GET  /api/leaderboard           -> community rankings
  POST /api/scheduled/score-results -> record a scored event, award points
  POST /predict                   -> single head-to-head prediction from raw stats

Notes vs. the previous version:
  - CORS is enabled so a browser frontend on another origin can call this.
  - Errors return real HTTP status codes (4xx/5xx), not 200-with-error-body,
    so the frontend can actually tell success from failure.
  - Odds are fetched server-side (key stays secret) and cached.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from qualia import card as card_service
from qualia import model as model_service
from qualia import store

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = Flask(__name__, static_folder=None)
CORS(app)  # allow cross-origin calls from the frontend


@app.route("/")
def home():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    # Serve other static assets (js/css/images) if added later.
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/card")
def api_card():
    force = request.args.get("refresh") == "1"
    try:
        return jsonify(card_service.build_card(force_refresh=force))
    except Exception as exc:  # pragma: no cover - defensive
        app.logger.exception("card build failed")
        return jsonify({"error": "Could not build the card.", "detail": str(exc)}), 500


@app.route("/api/leaderboard")
def api_leaderboard():
    board = store.get_leaderboard()
    return jsonify({"leaderboard": board, "count": len(board)})


@app.route("/api/scheduled/score-results", methods=["POST"])
def api_score_results():
    payload = request.get_json(silent=True)
    if not payload or "event" not in payload:
        return jsonify({"error": "Body must be JSON with at least an 'event' field."}), 400
    result = store.record_scored_event(payload)
    return jsonify({"status": "ok", **result}), 201


@app.route("/predict", methods=["POST"])
def predict():
    """Head-to-head prediction from two fighters' raw stats.

    Accepts either {"fighter_a": {...stats}, "fighter_b": {...stats}} or the
    legacy flat f1_/f2_ feature body for backward compatibility.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    try:
        if "fighter_a" in data and "fighter_b" in data:
            stats_a, stats_b = data["fighter_a"], data["fighter_b"]
        else:
            # Legacy flat body -> map onto stat dicts the baseline understands.
            stats_a = {
                "avg_fight_duration": data.get("f1_avg_fight_duration", 0),
                "knockdown_rate": data.get("f1_knockdown_rate", 0),
                "takedown_success_rate": data.get("f1_takedown_success_rate", 0),
                "strike_defense": data.get("f1_strike_defense", 0),
                "striking_balance": data.get("f1_striking_balance", 0),
                "finish_rate": data.get("f1_finish_rate", 0),
            }
            stats_b = {
                "avg_fight_duration": data.get("f2_avg_fight_duration", 0),
                "knockdown_rate": data.get("f2_knockdown_rate", 0),
                "takedown_success_rate": data.get("f2_takedown_success_rate", 0),
                "strike_defense": data.get("f2_strike_defense", 0),
                "striking_balance": data.get("f2_striking_balance", 0),
                "finish_rate": data.get("f2_finish_rate", 0),
            }

        prob_a = model_service.win_probability(stats_a, stats_b)
        winner = "fighter_a" if prob_a >= 0.5 else "fighter_b"
        return jsonify({
            "model_kind": "baseline",
            "prob_fighter_a": round(prob_a, 4),
            "prob_fighter_b": round(1 - prob_a, 4),
            "predicted_winner": winner,
        })
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": "Prediction failed.", "detail": str(exc)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "features": len(model_service.FEATURE_SCHEMA)})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
