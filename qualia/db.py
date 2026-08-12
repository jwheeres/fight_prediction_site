"""Optional database-backed persistence for the flat-file stores.

When DATABASE_URL is set, users / picks / leaderboard / results are kept in a
single key->JSON table (`qualia_kv`) instead of local files, so the data
survives redeploys on an ephemeral host like Render. Without DATABASE_URL,
everything stays in data/*.json — zero-setup local dev and tests.

One code path serves two drivers so the DB behaviour is actually testable
without a live Postgres:

  postgresql://…  /  postgres://…   -> psycopg  (production, e.g. Neon)
  sqlite:///path/to.db              -> sqlite3  (local / tests)

It's a whole-document key/value store (each module reads/writes its entire JSON
blob under one key), which keeps the existing module logic unchanged. Callers
already serialise read-modify-write with their own in-process lock; that's the
right grain for a single-instance hobby app. If you ever scale to multiple
instances, move the read-modify-write flows into real transactions.
"""

from __future__ import annotations

import json
import os
import threading

_LOCK = threading.Lock()
_conn = None
_kind = None  # "sqlite" | "postgres"


def url() -> str | None:
    u = os.getenv("DATABASE_URL", "").strip()
    return u or None


def enabled() -> bool:
    return url() is not None


def _sqlite_path(u: str) -> str:
    # sqlite:///relative.db  ->  relative.db ;  sqlite:////abs.db -> /abs.db
    return u[len("sqlite:///"):] if u.startswith("sqlite:///") else u[len("sqlite://"):]


def _connect():
    u = url()
    if u.startswith("sqlite"):
        import sqlite3
        conn = sqlite3.connect(_sqlite_path(u), check_same_thread=False)
        conn.execute("CREATE TABLE IF NOT EXISTS qualia_kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.commit()
        return conn, "sqlite"
    import psycopg  # imported lazily so sqlite/file modes need no driver
    conn = psycopg.connect(u, autocommit=True)
    with conn.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS qualia_kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    return conn, "psycopg"


def _get_conn():
    global _conn, _kind
    if _conn is None:
        _conn, _kind = _connect()
    return _conn, _kind


def _ph(kind: str) -> str:
    return "?" if kind == "sqlite" else "%s"


def _run(write: bool, sql_tmpl, params, fetch=False):
    """Execute a statement, reconnecting once if the connection went stale
    (Neon drops idle connections when it scales to zero)."""
    global _conn
    for attempt in (1, 2):
        try:
            conn, kind = _get_conn()
            sql = sql_tmpl.format(ph=_ph(kind))
            cur = conn.cursor()
            cur.execute(sql, params)
            row = cur.fetchone() if fetch else None
            if write and kind == "sqlite":
                conn.commit()
            return row
        except Exception:
            _conn = None  # force a fresh connection and retry once
            if attempt == 2:
                raise


def get(key: str, default):
    with _LOCK:
        row = _run(False, "SELECT value FROM qualia_kv WHERE key = {ph}", (key,), fetch=True)
    return json.loads(row[0]) if row else default


def set(key: str, value) -> None:  # noqa: A001 - mirrors dict-ish api on purpose
    payload = json.dumps(value)
    with _LOCK:
        _run(True,
             "INSERT INTO qualia_kv (key, value) VALUES ({ph}, {ph}) "
             "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
             (key, payload))


def reset() -> None:
    """Drop the cached connection (used by tests switching databases)."""
    global _conn, _kind
    try:
        if _conn is not None:
            _conn.close()
    except Exception:
        pass
    _conn = None
    _kind = None
