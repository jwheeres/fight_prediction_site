"""Qualia Bet Market — backend API + static frontend.

Endpoints:
  GET  /                          -> the frontend (static/index.html)
  GET  /api/card                  -> fight card: model probs merged with live market odds
  GET  /api/leaderboard           -> community rankings
  GET  /api/results               -> model track record (past scored fights)
  POST /api/scheduled/score-results -> record a scored event, award points
  POST /predict                   -> single head-to-head prediction from raw stats

Notes vs. the previous version:
  - CORS is enabled so a browser frontend on another origin can call this.
  - Errors return real HTTP status codes (4xx/5xx), not 200-with-error-body,
    so the frontend can actually tell success from failure.
  - Odds are fetched server-side (key stays secret) and cached.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS

from qualia import auth as auth_service
from qualia import card as card_service
from qualia import db
from qualia import model as model_service
from qualia import odds as odds_service
from qualia import picks as picks_service
from qualia import results as results_service
from qualia import store

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = Flask(__name__, static_folder=None)
# Signs the session cookie that carries the logged-in user. MUST be set in
# production; the insecure fallback keeps local dev working but is logged.
app.secret_key = os.getenv("SECRET_KEY") or "dev-insecure-secret-change-me"
if app.secret_key == "dev-insecure-secret-change-me":
    app.logger.warning("SECRET_KEY not set — using an insecure dev key. Set SECRET_KEY in production.")
CORS(app, supports_credentials=True)  # allow the frontend to send the session cookie

# Seed the model onto the leaderboard (needed when DATABASE_URL points at a
# fresh, empty database). Best-effort: a DB hiccup shouldn't stop the app.
try:
    store.ensure_seed()
except Exception:  # pragma: no cover - defensive
    app.logger.exception("leaderboard seed failed")


@app.after_request
def no_cache_html(response):
    # The whole UI is one file (static/index.html). Browsers aggressively cache
    # it, so a fresh deploy wouldn't show up until a hard refresh — that's the
    # recurring "it didn't update" problem. Tell the browser to revalidate the
    # HTML shell every load. Static assets (js/css/images) can still be cached.
    if response.mimetype == "text/html":
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


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


@app.route("/api/results")
def api_results():
    return jsonify(results_service.get_track_record())


@app.route("/api/odds-status")
def api_odds_status():
    """Diagnostics for the live odds hookup (no secrets returned)."""
    return jsonify(odds_service.get_diagnostics())


def _scoring_authorized() -> tuple[bool, str, int]:
    """Gate the scoring endpoint behind a shared secret.

    Fail-safe: if SCORING_SECRET isn't configured, writes are refused entirely
    (503) rather than left open — a public, unauthenticated scoring endpoint
    lets anyone stuff the leaderboard. The scheduled scorer sends the secret in
    the X-Scoring-Secret header (see scripts/score_event.py).
    """
    secret = os.getenv("SCORING_SECRET", "").strip()
    if not secret:
        return False, "Scoring is not configured (set SCORING_SECRET).", 503
    provided = request.headers.get("X-Scoring-Secret", "")
    if not hmac.compare_digest(provided, secret):
        return False, "Unauthorized.", 401
    return True, "", 200


@app.route("/api/scheduled/score-results", methods=["POST"])
def api_score_results():
    ok, reason, code = _scoring_authorized()
    if not ok:
        return jsonify({"error": reason}), code
    payload = request.get_json(silent=True)
    if not payload or "event" not in payload:
        return jsonify({"error": "Body must be JSON with at least an 'event' field."}), 400
    # Grade stored user picks against the actual results and award them on the
    # same board as the model's picks. `results` is [{fight, winner}, ...].
    user_awards = picks_service.grade_users(payload["event"], payload.get("results", []))
    if user_awards:
        payload = {**payload, "predictor_picks": list(payload.get("predictor_picks", [])) + user_awards}
    result = store.record_scored_event(payload)
    return jsonify({"status": "ok", **result}), 201


@app.route("/api/predictor/<name>")
def api_predictor(name):
    """Public profile for one predictor: leaderboard standing + rank. For the
    model ('Qualia Model') it also includes the honest track-record summary."""
    entry = store.get_predictor(name)
    if entry is None:
        return jsonify({"error": f"No predictor named {name!r}."}), 404
    profile = {"predictor": entry, "rank": store.get_rank(name)}
    if entry.get("is_model") or name == "Qualia Model":
        profile["track_record"] = results_service.get_track_record()["summary"]
    return jsonify(profile)


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
        response = {
            "model_kind": model_service.MODEL_KIND,
            "prob_fighter_a": round(prob_a, 4),
            "prob_fighter_b": round(1 - prob_a, 4),
            "predicted_winner": winner,
        }

        # Flag predictions computed on partial data so a caller can tell a
        # confident result from a coin flip forced by missing stats.
        missing_a = model_service.missing_features(stats_a)
        missing_b = model_service.missing_features(stats_b)
        if missing_a or missing_b:
            response["insufficient_data"] = {
                "fighter_a_missing": missing_a,
                "fighter_b_missing": missing_b,
            }
        return jsonify(response)
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": "Prediction failed.", "detail": str(exc)}), 400


def current_user() -> str | None:
    return session.get("user")


@app.route("/api/auth/register", methods=["POST"])
def api_register():
    data = request.get_json(silent=True) or {}
    ok, err = auth_service.register(data.get("username", ""), data.get("password", ""))
    if not ok:
        return jsonify({"error": err}), 400
    username = auth_service.verify(data.get("username", ""), data.get("password", ""))
    store.ensure_predictor(username)  # appear on the board from signup
    session["user"] = username
    return jsonify({"user": username}), 201


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    username = auth_service.verify(data.get("username", ""), data.get("password", ""))
    if not username:
        return jsonify({"error": "Invalid username or password."}), 401
    session["user"] = username
    return jsonify({"user": username})


@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.pop("user", None)
    return jsonify({"status": "ok"})


@app.route("/api/auth/me")
def api_me():
    return jsonify({"user": current_user()})


@app.route("/api/picks", methods=["GET", "POST"])
def api_picks():
    user = current_user()
    if not user:
        return jsonify({"error": "Log in to make picks."}), 401
    if request.method == "GET":
        event = request.args.get("event", "")
        return jsonify({"event": event, "picks": picks_service.get_picks(user, event)})
    data = request.get_json(silent=True) or {}
    event = data.get("event")
    picks = data.get("picks")
    if not event or not isinstance(picks, dict) or not picks:
        return jsonify({"error": "Body needs 'event' and a non-empty 'picks' object."}), 400
    saved = picks_service.set_picks(user, event, picks)
    return jsonify({"event": event, "picks": saved}), 201


def _persistence_diag() -> dict:
    """Where users/picks are stored and whether it will survive a redeploy.

    'ephemeral-files' means DATABASE_URL is NOT set — data lives on the host's
    temporary disk and is wiped on every redeploy. 'database' means it persists.
    No secrets are exposed. This is what to check when picks 'disappear'.
    """
    if db.enabled():
        try:
            db.get("__ping__", None)
            persistence, database_ok = "database", True
        except Exception:
            persistence, database_ok = "database", False
    else:
        persistence, database_ok = "ephemeral-files", None
    try:
        users = auth_service.count()
    except Exception:
        users = None
    try:
        pick_events = picks_service.event_count()
    except Exception:
        pick_events = None
    return {
        "persistence": persistence,
        "database_ok": database_ok,
        "secret_key_set": bool(os.getenv("SECRET_KEY", "").strip()),
        "stored_users": users,
        "pick_events": pick_events,
    }


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": model_service.MODEL_KIND,
        "features": len(model_service.FEATURES),
        **_persistence_diag(),
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
