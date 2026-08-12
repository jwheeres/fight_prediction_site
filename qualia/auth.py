"""User accounts + password auth, stored flat-file in data/users.json.

Hobby-grade auth for a single-instance app, matching the rest of qualia's
flat-file style: passwords hashed with Werkzeug (scrypt by default), never
stored in the clear. The logged-in state itself lives in Flask's signed-cookie
session (see app.py) — this module only owns the user records. Swap the flat
file for a real DB when there are concurrent writers.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

from qualia import db

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
USERS_FILE = DATA_DIR / "users.json"

_LOCK = threading.Lock()

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\- ]{3,32}$")
MIN_PASSWORD = 8
# Names nobody may register — don't let a user impersonate the model.
RESERVED = {"qualia model", "admin", "the house"}

# A real hash to check against for unknown users, so login timing doesn't leak
# whether a username exists.
_DUMMY_HASH = generate_password_hash("not-a-real-password")


def _read() -> dict:
    if db.enabled():
        return db.get(USERS_FILE.stem, {})
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text())
    except Exception:
        return {}


def _write(users: dict) -> None:
    if db.enabled():
        db.set(USERS_FILE.stem, users)
        return
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = USERS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(users, indent=2))
    tmp.replace(USERS_FILE)  # atomic on POSIX


def validate_username(username: str) -> tuple[bool, str]:
    if not username or not USERNAME_RE.match(username):
        return False, "Username must be 3–32 chars: letters, numbers, space, _ . -"
    if username.strip().lower() in RESERVED:
        return False, "That name is reserved."
    return True, ""


def _find(users: dict, username: str) -> str | None:
    """Canonical stored name matching username case-insensitively, or None."""
    target = (username or "").strip().lower()
    for name in users:
        if name.lower() == target:
            return name
    return None


def register(username: str, password: str) -> tuple[bool, str]:
    """Create a user. Returns (ok, error_message)."""
    username = (username or "").strip()
    ok, err = validate_username(username)
    if not ok:
        return False, err
    if len(password or "") < MIN_PASSWORD:
        return False, f"Password must be at least {MIN_PASSWORD} characters."
    with _LOCK:
        users = _read()
        if _find(users, username) is not None:
            return False, "That username is taken."
        users[username] = {
            "password_hash": generate_password_hash(password),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write(users)
    return True, ""


def verify(username: str, password: str) -> str | None:
    """Return the canonical username on a correct password, else None."""
    users = _read()
    name = _find(users, username)
    if name is None:
        # Still run a hash check to keep timing ~constant against user probing.
        check_password_hash(_DUMMY_HASH, password or "")
        return None
    if check_password_hash(users[name]["password_hash"], password or ""):
        return name
    return None


def exists(username: str) -> bool:
    return _find(_read(), username) is not None
