#!/usr/bin/env python3
"""Build real fighter career stats + a training set from public UFC data.

Source: Greco1899/scrape_ufc_stats on GitHub (a maintained scrape of
ufcstats.com). We download four CSVs, derive per-fighter career averages
(strike accuracy/defense, SLpM/SApM, takedown acc/def, control time, finish
rate, etc.), and emit:

  data/fighters_stats.json  -> name -> feature dict (used live + for training)
  data/training.csv         -> one row per decided fight: featureA-featureB + winner

Run this locally (raw.githubusercontent must be reachable), commit the two
outputs, then train with `python -m qualia.train`. Re-run to refresh.

Honest caveat: career averages include a fighter's whole history, so training
on them has mild look-ahead leakage; held-out accuracy is therefore a slight
over-estimate. Good enough for a v1 real model; as-of (pre-fight) features are
the follow-up.
"""

from __future__ import annotations

import io
import sys
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
}

# The per-fighter numeric features the model consumes (order matters).
FEATURES = [
    "sig_str_acc", "sig_str_def", "slpm", "sapm",
    "td_acc", "td_def", "td_per15", "sub_per15", "kd_per15",
    "ctrl_pf", "finish_rate", "win_rate", "reach_in", "height_in",
    "age", "ufc_fights",
]


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
    # e.g. 5' 11"
    try:
        ft, inch = str(v).replace('"', "").split("' ")
        return int(ft) * 12 + int(inch)
    except Exception:
        return np.nan


def _inches_from_reach(v: str) -> float:
    try:
        return float(str(v).replace('"', "").strip())
    except Exception:
        return np.nan


def _age(dob: str) -> float:
    try:
        d = datetime.strptime(str(dob), "%b %d, %Y")
        return (datetime.now() - d).days / 365.25
    except Exception:
        return np.nan


def build():
    print("Loading raw data…")
    stats = _download(FILES["stats"])
    results = _download(FILES["results"])
    tott = _download(FILES["tott"])

    stats.columns = [c.strip() for c in stats.columns]
    stats["FIGHTER"] = stats["FIGHTER"].str.strip()
    stats["BOUT"] = stats["BOUT"].str.strip()

    sig_l, sig_a = _landed_attempted(stats["SIG.STR."])
    td_l, td_a = _landed_attempted(stats["TD"])
    clinch_l, _ = _landed_attempted(stats["CLINCH"])
    ground_l, _ = _landed_attempted(stats["GROUND"])
    stats = stats.assign(
        sig_l=sig_l, sig_a=sig_a, td_l=td_l, td_a=td_a,
        kd=pd.to_numeric(stats["KD"], errors="coerce").fillna(0),
        sub=pd.to_numeric(stats["SUB.ATT"], errors="coerce").fillna(0),
        ctrl=_ctrl_seconds(stats["CTRL"]),
    )

    # Per fighter-per-bout offense totals.
    off = stats.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False).agg(
        sig_l=("sig_l", "sum"), sig_a=("sig_a", "sum"),
        td_l=("td_l", "sum"), td_a=("td_a", "sum"),
        kd=("kd", "sum"), sub=("sub", "sum"), ctrl=("ctrl", "sum"),
    )

    # Defense = what the OTHER fighter in the same bout did to them.
    opp = off.copy()
    merged = off.merge(opp, on=["EVENT", "BOUT"], suffixes=("", "_opp"))
    merged = merged[merged["FIGHTER"] != merged["FIGHTER_opp"]]
    # (bouts are 2-fighter, so exactly one opponent row per fighter row)

    # Fight duration per bout (from results), matched by EVENT+BOUT.
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

    # Winner + method per bout, from results OUTCOME ("W/L" => first name won).
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

    # Aggregate to career per-fighter numbers.
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

        wins = sum(1 for _, r in g.iterrows() if win_map.get((r["EVENT"], r["BOUT"])) == fighter)
        finishes = sum(
            1 for _, r in g.iterrows()
            if win_map.get((r["EVENT"], r["BOUT"])) == fighter
            and any(k in method_map.get((r["EVENT"], r["BOUT"]), "") for k in ("KO", "Submission"))
        )
        rows.append({
            "fighter": fighter,
            "sig_str_acc": sig_l_sum / sig_a_sum if sig_a_sum else 0.0,
            "sig_str_def": 1 - (opp_sig_l / opp_sig_a) if opp_sig_a else 0.0,
            "slpm": sig_l_sum / total_min,
            "sapm": opp_sig_l / total_min,
            "td_acc": td_l_sum / td_a_sum if td_a_sum else 0.0,
            "td_def": 1 - (opp_td_l / opp_td_a) if opp_td_a else 0.0,
            "td_per15": td_l_sum / total_min * 15,
            "sub_per15": g["sub"].sum() / total_min * 15,
            "kd_per15": g["kd"].sum() / total_min * 15,
            "ctrl_pf": g["ctrl"].sum() / n,
            "finish_rate": finishes / wins if wins else 0.0,
            "win_rate": wins / n,
            "ufc_fights": n,
        })

    fighters = pd.DataFrame(rows)

    # Physical attributes from tale-of-the-tape.
    tott.columns = [c.strip() for c in tott.columns]
    tott["FIGHTER"] = tott["FIGHTER"].str.strip()
    tott["height_in"] = tott["HEIGHT"].apply(_inches_from_height)
    tott["reach_in"] = tott["REACH"].apply(_inches_from_reach)
    tott["age"] = tott["DOB"].apply(_age)
    fighters = fighters.merge(
        tott[["FIGHTER", "height_in", "reach_in", "age"]],
        left_on="fighter", right_on="FIGHTER", how="left",
    ).drop(columns=["FIGHTER"])

    # Fill missing physicals with medians so the model always has a value.
    for col in ("height_in", "reach_in", "age"):
        fighters[col] = fighters[col].fillna(fighters[col].median())

    # Write the fighter stats table (name -> features).
    out = {}
    for _, r in fighters.iterrows():
        out[r["fighter"]] = {k: round(float(r[k]), 4) for k in FEATURES}
    import json
    (DATA_DIR / "fighters_stats.json").write_text(json.dumps(out, indent=0))
    print(f"Wrote data/fighters_stats.json — {len(out)} fighters")

    # Build the training set from decided fights where we have both fighters.
    stat_lookup = {f: v for f, v in out.items()}
    train_rows = []
    seen = set()
    for _, r in results.iterrows():
        b = str(r["BOUT"])
        if " vs. " not in b or not r["winner"]:
            continue
        a, c = [x.strip() for x in b.split(" vs. ", 1)]
        if a not in stat_lookup or c not in stat_lookup:
            continue
        key = (r["EVENT"], b)
        if key in seen:
            continue
        seen.add(key)
        fa, fc = stat_lookup[a], stat_lookup[c]
        row = {f"d_{k}": fa[k] - fc[k] for k in FEATURES}
        row["winner"] = 1 if r["winner"] == a else 0
        train_rows.append(row)

    train = pd.DataFrame(train_rows)
    train.to_csv(DATA_DIR / "training.csv", index=False)
    print(f"Wrote data/training.csv — {len(train)} fights, win-rate {train['winner'].mean():.3f}")


if __name__ == "__main__":
    build()
