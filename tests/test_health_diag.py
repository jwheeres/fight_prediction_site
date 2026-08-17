"""The /health diagnostic reports where data is stored (to debug 'picks vanished')."""

import app as app_module
from qualia import db


def _client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_health_flags_ephemeral_storage_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db.reset()
    body = _client().get("/health").get_json()
    # This is the smoking gun for "picks disappear on redeploy".
    assert body["persistence"] == "ephemeral-files"
    assert "stored_users" in body and "pick_events" in body
    assert body["secret_key_set"] in (True, False)


def test_health_reports_database_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + str(tmp_path / "h.db"))
    db.reset()
    try:
        body = _client().get("/health").get_json()
        assert body["persistence"] == "database"
        assert body["database_ok"] is True
    finally:
        db.reset()
