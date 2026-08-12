#!/usr/bin/env python3
"""Build real fighter career stats + a training set from public UFC data.

Source: Greco1899/scrape_ufc_stats on GitHub (a maintained scrape of
ufcstats.com). We download the fight stats, results, tale-of-the-tape, and
event-details CSVs and emit:

  data/fighters_stats.json  -> name -> career feature dict (used LIVE for serving)
  data/training.csv         -> one row per decided fight: AS-OF featureA-featureB
                               + winner

Two different feature snapshots, on purpose:

* Serving (fighters_stats.json) uses a fighter's WHOLE career to date. For an
  upcoming fight that is correct, not leaky — the bout hasn't happened yet.
* Training (training.csv) uses AS-OF (pre-fight) stats: for each historical
  fight, each fighter's numbers are computed from ONLY their bouts strictly
  before that fight's date. This removes the look-ahead leakage that whole-career
  averages bake in (a career average already "knows" the outcome of the very
  fight we're trying to predict), and it lets win_rate be an honest feature.

Run this locally (raw.githubusercontent must be reachable), commit the two
outputs, then train with `python -m qualia.train`. Re-run to refresh.
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BASE = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main"
FILES = {
    "stats": "ufc_fight_stats.csv",
    "results": "ufc_fight_results.csv",
    "tott": "ufc_fighter_tott.csv",
    "events": "ufc_event_details.csv",
}

# Every per-fighter numeric feature we derive (order matters for the CSV).
# qualia.model.FEATURES selects which of these the model actually trains on.
FEATURES = [
    "sig_str_acc", "sig_str_def", "slpm", "sapm",
    "td_acc", "td_def", "td_per15", "sub_per15", "kd_per15",
    "ctrl_pf", "finish_rate", "win_rate", "reach_in", "height_in",
    "age", "ufc_fights",
]

# A fight only enters the training set if BOTH fighters have at least this many
# prior UFC bouts — otherwise their as-of averages are pure small-sample noise.
MIN_PRIOR_FIGHTS = 2


# --------------------------------------------------------------------------- #
# small parsers
# --------------------------------------------------------------------------- #

def _download(name: str) -> pd.DataFrame:
    cache = DATA_DIR / "_raw" / name
    if cache.exists():
        return pd.read_csv(cache)
    url = f"{BASE}/{name}"
    print(f"  downloading {name} …")
    r = requests.get(url, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(r.content)
    return pd.read_csv(io.StringIO(r.text))


def _landed_attempted(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse an 'X of Y' column into (landed, attempted) numeric series."""
    parts = series.fillna("0 of 0").astype(str).str.split(" of ", expand=True)
    landed = pd.to_numeric(parts[0], errors="coerce").fillna(0)
    attempted = pd.to_numeric(parts[1], errors="coerce").fillna(0)
    return landed, attempted


def _ctrl_seconds(series: pd.Series) -> pd.Series:
    def to_s(v):
        try:
            m, s = str(v).split(":")
            return int(m) * 60 + int(s)
        except Exception:
            return 0
    return series.apply(to_s)


def _inches_from_height(v: str) -> float:
    try:  # e.g. 5' 11"
        ft, inch = str(v).replace('"', "").split("' ")
        return int(ft) * 12 + int(inch)
    except Exception:
        return np.nan


def _inches_from_reach(v: str) -> float:
    try:
        return float(str(v).replace('"', "").strip())
    except Exception:
        return np.nan


def _parse_date(v: str):
    try:  # e.g. "August 08, 2026"
        return datetime.strptime(str(v).strip(), "%B %d, %Y")
    except Exception:
        return pd.NaT


def _parse_dob(v: str):
    try:  # e.g. "Jan 12, 1990"
        return datetime.strptime(str(v).strip(), "%b %d, %Y")
    except Exception:
        return pd.NaT


def _sdiv(a, b) -> np.ndarray:
    """Elementwise a/b, returning 0 where b == 0 (never NaN/inf)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


# --------------------------------------------------------------------------- #
# load + parse raw data into one per-fighter-per-bout table
# --------------------------------------------------------------------------- #

def load_parsed() -> dict:
    """Return the parsed building blocks shared by career + as-of paths."""
    stats = _download(FILES["stats"])
    results = _download(FILES["results"])
    tott = _download(FILES["tott"])
    events = _download(FILES["events"])

    stats.columns = [c.strip() for c in stats.columns]
    stats["FIGHTER"] = stats["FIGHTER"].str.strip()
    stats["BOUT"] = stats["BOUT"].str.strip()

    sig_l, sig_a = _landed_attempted(stats["SIG.STR."])
    td_l, td_a = _landed_attempted(stats["TD"])
    stats = stats.assign(
        sig_l=sig_l, sig_a=sig_a, td_l=td_l, td_a=td_a,
        kd=pd.to_numeric(stats["KD"], errors="coerce").fillna(0),
        sub=pd.to_numeric(stats["SUB.ATT"], errors="coerce").fillna(0),
        ctrl=_ctrl_seconds(stats["CTRL"]),
    )

    off = stats.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False).agg(
        sig_l=("sig_l", "sum"), sig_a=("sig_a", "sum"),
        td_l=("td_l", "sum"), td_a=("td_a", "sum"),
        kd=("kd", "sum"), sub=("sub", "sum"), ctrl=("ctrl", "sum"),
    )
    # Defense = what the OTHER fighter in the same bout did. Bouts are
    # 2-fighter, so each fighter row matches exactly one opponent row.
    merged = off.merge(off, on=["EVENT", "BOUT"], suffixes=("", "_opp"))
    merged = merged[merged["FIGHTER"] != merged["FIGHTER_opp"]].copy()

    results.columns = [c.strip() for c in results.columns]
    results["BOUT"] = results["BOUT"].str.strip()
    results["EVENT"] = results["EVENT"].str.strip()

    def dur_min(row):
        try:
            rnd = int(row["ROUND"])
            mm, ss = str(row["TIME"]).split(":")
            return (rnd - 1) * 5 + int(mm) + int(ss) / 60.0
        except Exception:
            return np.nan

    results["dur_min"] = results.apply(dur_min, axis=1)
    dur = results.set_index(["EVENT", "BOUT"])["dur_min"].to_dict()
    merged["dur_min"] = merged.apply(lambda r: dur.get((r["EVENT"], r["BOUT"]), np.nan), axis=1)

    def winner(row):
        b = str(row["BOUT"])
        if " vs. " not in b:
            return None
        a, c = b.split(" vs. ", 1)
        o = str(row["OUTCOME"]).strip()
        if o.startswith("W/"):
            return a.strip()
        if o.startswith("L/") or o.endswith("/W"):
            return c.strip()
        return None

    results["winner"] = results.apply(winner, axis=1)
    win_map = {(r["EVENT"], r["BOUT"]): r["winner"] for _, r in results.iterrows()}
    method_map = {(r["EVENT"], r["BOUT"]): str(r["METHOD"]).strip() for _, r in results.iterrows()}

    # Event dates (for chronological / as-of ordering).
    events.columns = [c.strip() for c in events.columns]
    events["EVENT"] = events["EVENT"].str.strip()
    event_date = {r["EVENT"]: _parse_date(r["DATE"]) for _, r in events.iterrows()}

    # Per-bout won/finish flags on the merged table.
    merged["won"] = merged.apply(
        lambda r: 1 if win_map.get((r["EVENT"], r["BOUT"])) == r["FIGHTER"] else 0, axis=1
    )
    merged["finish"] = merged.apply(
        lambda r: 1 if r["won"] and any(
            k in method_map.get((r["EVENT"], r["BOUT"]), "") for k in ("KO", "Submission")
        ) else 0,
        axis=1,
    )
    merged["date"] = merged["EVENT"].map(event_date)

    # Physical attributes / DOB from tale-of-the-tape.
    tott.columns = [c.strip() for c in tott.columns]
    tott["FIGHTER"] = tott["FIGHTER"].str.strip()
    height = {r["FIGHTER"]: _inches_from_height(r["HEIGHT"]) for _, r in tott.iterrows()}
    reach = {r["FIGHTER"]: _inches_from_reach(r["REACH"]) for _, r in tott.iterrows()}
    dob = {r["FIGHTER"]: _parse_dob(r["DOB"]) for _, r in tott.iterrows()}

    med_height = np.nanmedian([v for v in height.values() if not np.isnan(v)])
    med_reach = np.nanmedian([v for v in reach.values() if not np.isnan(v)])

    return {
        "merged": merged, "results": results, "win_map": win_map,
        "event_date": event_date, "height": height, "reach": reach, "dob": dob,
        "med_height": med_height, "med_reach": med_reach,
    }


# --------------------------------------------------------------------------- #
# career stats (whole career) -> served live
# --------------------------------------------------------------------------- #

def career_stats(P: dict) -> dict:
    merged, win_map = P["merged"], P["win_map"]
    rows = []
    for fighter, g in merged.groupby("FIGHTER"):
        total_min = g["dur_min"].sum()
        n = len(g)
        if total_min <= 0 or n == 0:
            continue
        sig_l_sum, sig_a_sum = g["sig_l"].sum(), g["sig_a"].sum()
        opp_sig_l, opp_sig_a = g["sig_l_opp"].sum(), g["sig_a_opp"].sum()
        td_l_sum, td_a_sum = g["td_l"].sum(), g["td_a"].sum()
        opp_td_l, opp_td_a = g["td_l_opp"].sum(), g["td_a_opp"].sum()
        wins = int(g["won"].sum())
        finishes = int(g["finish"].sum())
        med_age = 30.0  # sensible fallback if DOB missing
        dob = P["dob"].get(fighter, pd.NaT)
        age = (datetime.now() - dob).days / 365.25 if dob is not pd.NaT and pd.notna(dob) else med_age
        rows.append({
            "fighter": fighter,
            "sig_str_acc": _sdiv(sig_l_sum, sig_a_sum).item() if sig_a_sum else 0.0,
            "sig_str_def": (1 - opp_sig_l / opp_sig_a) if opp_sig_a else 0.0,
            "slpm": sig_l_sum / total_min,
            "sapm": opp_sig_l / total_min,
            "td_acc": (td_l_sum / td_a_sum) if td_a_sum else 0.0,
            "td_def": (1 - opp_td_l / opp_td_a) if opp_td_a else 0.0,
            "td_per15": td_l_sum / total_min * 15,
            "sub_per15": g["sub"].sum() / total_min * 15,
            "kd_per15": g["kd"].sum() / total_min * 15,
            "ctrl_pf": g["ctrl"].sum() / n,
            "finish_rate": (finishes / wins) if wins else 0.0,
            "win_rate": wins / n,
            "reach_in": P["reach"].get(fighter, np.nan),
            "height_in": P["height"].get(fighter, np.nan),
            "age": age,
            "ufc_fights": n,
        })
    df = pd.DataFrame(rows)
    df["reach_in"] = df["reach_in"].fillna(P["med_reach"])
    df["height_in"] = df["height_in"].fillna(P["med_height"])
    return {r["fighter"]: {k: round(float(r[k]), 4) for k in FEATURES} for _, r in df.iterrows()}


# --------------------------------------------------------------------------- #
# as-of stats (strictly pre-fight) -> training only
# --------------------------------------------------------------------------- #

def asof_table(P: dict) -> pd.DataFrame:
    """One row per (EVENT, BOUT, FIGHTER): that fighter's stats using ONLY their
    bouts before this fight's date. Rows with < MIN_PRIOR_FIGHTS priors are kept
    but flagged via n_prior so the caller can drop them.
    """
    df = P["merged"].dropna(subset=["date"]).sort_values(["FIGHTER", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    comp = ["sig_l", "sig_a", "sig_l_opp", "sig_a_opp", "td_l", "td_a",
            "td_l_opp", "td_a_opp", "kd", "sub", "ctrl", "dur_min", "won", "finish"]
    grp = df.groupby("FIGHTER")
    prior = grp[comp].cumsum() - df[comp]         # sum over strictly-earlier bouts
    n = grp.cumcount().to_numpy(dtype=float)      # number of prior bouts

    dur = prior["dur_min"].to_numpy()
    dob = pd.to_datetime(df["FIGHTER"].map(P["dob"]))
    age = (df["date"] - dob).dt.days / 365.25

    out = pd.DataFrame({
        "EVENT": df["EVENT"], "BOUT": df["BOUT"], "FIGHTER": df["FIGHTER"],
        "date": df["date"], "n_prior": n,
        "sig_str_acc": _sdiv(prior["sig_l"], prior["sig_a"]),
        "sig_str_def": np.where(prior["sig_a_opp"] > 0, 1 - _sdiv(prior["sig_l_opp"], prior["sig_a_opp"]), 0.0),
        "slpm": _sdiv(prior["sig_l"], dur),
        "sapm": _sdiv(prior["sig_l_opp"], dur),
        "td_acc": _sdiv(prior["td_l"], prior["td_a"]),
        "td_def": np.where(prior["td_a_opp"] > 0, 1 - _sdiv(prior["td_l_opp"], prior["td_a_opp"]), 0.0),
        "td_per15": _sdiv(prior["td_l"], dur) * 15,
        "sub_per15": _sdiv(prior["sub"], dur) * 15,
        "kd_per15": _sdiv(prior["kd"], dur) * 15,
        "ctrl_pf": _sdiv(prior["ctrl"], n),
        "finish_rate": _sdiv(prior["finish"], prior["won"]),
        "win_rate": _sdiv(prior["won"], n),
        "reach_in": df["FIGHTER"].map(P["reach"]).fillna(P["med_reach"]),
        "height_in": df["FIGHTER"].map(P["height"]).fillna(P["med_height"]),
        "age": age,
        "ufc_fights": n,
    })
    out["age"] = out["age"].fillna(out["age"].median())
    return out


def asof_training_rows(P: dict, features: list[str] = FEATURES, min_prior: int = MIN_PRIOR_FIGHTS) -> pd.DataFrame:
    tbl = asof_table(P)
    idx = {(r["EVENT"], r["BOUT"], r["FIGHTER"]): r for _, r in tbl.iterrows()}
    rows, seen = [], set()
    for _, r in P["results"].iterrows():
        b = str(r["BOUT"])
        if " vs. " not in b or not r["winner"]:
            continue
        a, c = [x.strip() for x in b.split(" vs. ", 1)]
        key = (r["EVENT"], b)
        if key in seen:
            continue
        ra, rc = idx.get((r["EVENT"], b, a)), idx.get((r["EVENT"], b, c))
        if ra is None or rc is None:
            continue
        if ra["n_prior"] < min_prior or rc["n_prior"] < min_prior:
            continue
        seen.add(key)
        row = {f"d_{k}": float(ra[k]) - float(rc[k]) for k in features}
        row["winner"] = 1 if r["winner"] == a else 0
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #

def build():
    print("Loading raw data…")
    P = load_parsed()

    stats_out = career_stats(P)
    (DATA_DIR / "fighters_stats.json").write_text(json.dumps(stats_out, indent=0))
    print(f"Wrote data/fighters_stats.json — {len(stats_out)} fighters (career, served live)")

    train = asof_training_rows(P)
    train.to_csv(DATA_DIR / "training.csv", index=False)
    print(f"Wrote data/training.csv — {len(train)} fights (as-of), "
          f"win-rate {train['winner'].mean():.3f}, min_prior={MIN_PRIOR_FIGHTS}")


if __name__ == "__main__":
    build()
