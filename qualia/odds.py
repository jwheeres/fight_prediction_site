"""Live betting-odds service.

Fetches UFC/MMA head-to-head (moneyline) odds from The Odds API, aggregates
them across bookmakers into a de-vigged consensus probability, and exposes a
lookup keyed by fighter name. Falls back to a bundled sample file when no API
key is configured or the network call fails, so the app always renders.

The Odds API docs: https://the-odds-api.com/liveapi/guides/v4/
Set the key via the ODDS_API_KEY environment variable (never commit it).
"""

from __future__ import annotations

import json
import os
import time
import unicodedata
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_ODDS_FILE = DATA_DIR / "sample_odds.json"
# Last successful LIVE fetch, persisted so we keep showing real odds (not
# made-up sample data) across restarts and when credits run out.
ODDS_CACHE_FILE = DATA_DIR / "odds_cache.json"

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "mma_mixed_martial_arts"

# Simple in-process cache so we don't burn the API quota (free tier 500/mo).
_CACHE: dict[str, object] = {"fetched_at": 0.0, "events": None, "source": None,
                             "diag": None, "cooldown_until": 0.0}
_CACHE_TTL_SECONDS = 15 * 60
# After hitting zero credits, stop calling for a while so we don't hammer the
# API with doomed requests. force_refresh (the UI's Refresh button) bypasses it.
_OUT_OF_CREDITS_COOLDOWN = 30 * 60


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation so 'Uroš Medić' matches 'Uros Medic'."""
    stripped = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return "".join(c for c in stripped.lower() if c.isalnum() or c.isspace()).strip()


def american_to_implied(price: float) -> float:
    """American odds -> implied probability (includes the book's vig)."""
    if price < 0:
        return (-price) / ((-price) + 100.0)
    return 100.0 / (price + 100.0)


def implied_to_american(prob: float) -> int:
    """Probability -> representative American odds (for display)."""
    prob = min(max(prob, 1e-6), 1 - 1e-6)
    if prob >= 0.5:
        return -round(100.0 * prob / (1.0 - prob))
    return round(100.0 * (1.0 - prob) / prob)


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _parse_events(raw_events: list[dict]) -> list[dict]:
    """Turn The Odds API payload into consensus rows, one per matchup."""
    parsed = []
    for ev in raw_events:
        a = ev.get("home_team")
        b = ev.get("away_team")
        if not a or not b:
            continue

        implied = {a: [], b: []}
        prices = {a: [], b: []}
        books = 0
        last_update = None

        for book in ev.get("bookmakers", []):
            h2h = next((m for m in book.get("markets", []) if m.get("key") == "h2h"), None)
            if not h2h:
                continue
            outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", []) if "price" in o}
            if a in outcomes and b in outcomes:
                books += 1
                implied[a].append(american_to_implied(outcomes[a]))
                implied[b].append(american_to_implied(outcomes[b]))
                prices[a].append(outcomes[a])
                prices[b].append(outcomes[b])
                last_update = book.get("last_update") or last_update

        if books == 0:
            continue

        avg_a = sum(implied[a]) / books
        avg_b = sum(implied[b]) / books
        # Remove the vig so the two probabilities sum to 1.
        total = avg_a + avg_b
        devig_a = avg_a / total
        devig_b = avg_b / total

        parsed.append({
            "fighter_a": a,
            "fighter_b": b,
            "books": books,
            "market_prob_a": round(devig_a, 4),
            "market_prob_b": round(devig_b, 4),
            "odds_a": int(_median(prices[a])),
            "odds_b": int(_median(prices[b])),
            "commence_time": ev.get("commence_time"),
            "last_update": last_update,
        })
    return parsed


def _load_sample() -> list[dict]:
    if not SAMPLE_ODDS_FILE.exists():
        return []
    return json.loads(SAMPLE_ODDS_FILE.read_text())


def _save_disk_cache(raw_events: list[dict], fetched_at: float) -> None:
    try:
        ODDS_CACHE_FILE.write_text(json.dumps({"raw_events": raw_events, "fetched_at": fetched_at}))
    except Exception:
        pass  # best-effort; a read-only FS just means no cross-restart cache


def _load_disk_cache() -> tuple[list[dict] | None, float]:
    try:
        d = json.loads(ODDS_CACHE_FILE.read_text())
        return d.get("raw_events"), float(d.get("fetched_at", 0.0))
    except Exception:
        return None, 0.0


def get_odds(force_refresh: bool = False) -> dict:
    """Return {'source', 'fetched_at', 'events': [...]} with consensus odds.

    Credit-aware fallback ladder:
      1. fresh in-process cache
      2. a live API call (skipped while in a post-zero-credits cooldown)
      3. the last successful LIVE fetch persisted to disk (real odds, just
         older) — served as 'live' with an honest timestamp
      4. bundled sample data (only if we've never had a successful fetch)
    """
    now = time.time()
    if (
        not force_refresh
        and _CACHE["events"] is not None
        and now - float(_CACHE["fetched_at"]) < _CACHE_TTL_SECONDS
    ):
        return {"source": _CACHE["source"], "fetched_at": _CACHE["fetched_at"], "events": _CACHE["events"]}

    api_key = os.getenv("ODDS_API_KEY", "").strip()
    diag: dict = {"key_present": bool(api_key), "key_length": len(api_key)}
    in_cooldown = (not force_refresh) and now < float(_CACHE.get("cooldown_until", 0.0))

    if api_key and not in_cooldown:
        try:
            resp = requests.get(
                f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds",
                params={"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american"},
                timeout=12,
            )
            diag["http_status"] = resp.status_code
            diag["requests_remaining"] = resp.headers.get("x-requests-remaining")
            diag["requests_used"] = resp.headers.get("x-requests-used")
            if not resp.ok:
                diag["error"] = resp.text[:200]
                # Out of credits (or bad key): back off so we stop hammering.
                if resp.headers.get("x-requests-remaining") == "0" or resp.status_code in (401, 429):
                    _CACHE["cooldown_until"] = now + _OUT_OF_CREDITS_COOLDOWN
            resp.raise_for_status()
            raw_events = resp.json()
            diag["raw_events"] = len(raw_events)
            events = _parse_events(raw_events)
            diag["source"] = "live"
            diag["parsed_events"] = len(events)
            _save_disk_cache(raw_events, now)
            _CACHE.update({"fetched_at": now, "events": events, "source": "live",
                           "diag": diag, "cooldown_until": 0.0})
            return {"source": "live", "fetched_at": now, "events": events}
        except Exception as exc:
            diag.setdefault("error", str(exc)[:200])
    elif in_cooldown:
        diag["note"] = "credit cooldown active; serving cached odds"

    # Live call failed / skipped -> last real odds from disk, if we have them.
    disk_raw, disk_at = _load_disk_cache()
    if disk_raw:
        events = _parse_events(disk_raw)
        diag["source"] = "cached"
        diag["parsed_events"] = len(events)
        _CACHE.update({"fetched_at": disk_at or now, "events": events, "source": "live", "diag": diag})
        return {"source": "live", "fetched_at": disk_at or now, "events": events}

    # Never had a successful fetch -> honest sample data.
    events = _parse_events(_load_sample())
    diag["source"] = "sample"
    diag["parsed_events"] = len(events)
    _CACHE.update({"fetched_at": now, "events": events, "source": "sample", "diag": diag})
    return {"source": "sample", "fetched_at": now, "events": events}


def get_diagnostics() -> dict:
    """Non-secret snapshot of the last odds fetch, for troubleshooting.

    Reuses the cached result so checking status never costs an API credit; only
    triggers a fetch if nothing has been fetched yet this process. Never returns
    the key value (only whether one is present and its length).
    """
    if _CACHE.get("diag") is None:
        get_odds()
    return _CACHE.get("diag") or {"key_present": bool(os.getenv("ODDS_API_KEY", "").strip())}


def build_odds_index(force_refresh: bool = False) -> dict:
    """Map normalized fighter name -> consensus row, for merging with a card."""
    odds = get_odds(force_refresh=force_refresh)
    index = {}
    for row in odds["events"]:
        index[normalize_name(row["fighter_a"])] = {**row, "side": "a"}
        index[normalize_name(row["fighter_b"])] = {**row, "side": "b"}
    return {"source": odds["source"], "fetched_at": odds["fetched_at"], "index": index}
