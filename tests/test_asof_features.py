"""Leakage guard for the as-of (pre-fight) feature construction.

The whole point of scripts/build_fighter_data.asof_table is that a fighter's
stats for a given fight are computed from ONLY their earlier bouts. These tests
prove that with a tiny synthetic dataset — no network / raw download needed.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_fighter_data as B  # noqa: E402


def _P(merged: pd.DataFrame) -> dict:
    return {"merged": merged, "dob": {}, "reach": {}, "height": {},
            "med_reach": 72.0, "med_height": 70.0}


def _bout(event, fighter, date, won, finish=0):
    return dict(
        EVENT=event, BOUT=f"{fighter} vs. Opp", FIGHTER=fighter, FIGHTER_opp="Opp",
        sig_l=10, sig_a=20, sig_l_opp=8, sig_a_opp=20, td_l=1, td_a=2,
        td_l_opp=0, td_a_opp=2, kd=0, sub=0, ctrl=60, dur_min=15.0,
        won=won, finish=finish, date=pd.Timestamp(date),
    )


def test_asof_uses_only_prior_fights():
    merged = pd.DataFrame([
        _bout("E1", "A", "2020-01-01", won=1),
        _bout("E2", "A", "2020-06-01", won=0),
        _bout("E3", "A", "2021-01-01", won=1),
    ])
    tbl = B.asof_table(_P(merged)).sort_values("date").reset_index(drop=True)

    # Prior-fight count increments 0, 1, 2 — never counts the current bout.
    assert list(tbl["ufc_fights"]) == [0, 1, 2]
    assert list(tbl["n_prior"]) == [0, 1, 2]

    # win_rate is strictly pre-fight:
    #   E1: no priors -> 0.0   E2: won E1 -> 1/1   E3: 1 win of 2 priors -> 0.5
    assert tbl.loc[0, "win_rate"] == 0.0
    assert tbl.loc[1, "win_rate"] == 1.0
    assert tbl.loc[2, "win_rate"] == 0.5


def test_future_fights_do_not_change_past_asof():
    base = [
        _bout("E1", "A", "2020-01-01", won=1),
        _bout("E2", "A", "2020-06-01", won=0),
    ]
    t1 = B.asof_table(_P(pd.DataFrame(base))).sort_values("date").reset_index(drop=True)
    # Append a LATER fight; every earlier as-of row must be byte-for-byte the same.
    t2 = B.asof_table(
        _P(pd.DataFrame(base + [_bout("E3", "A", "2021-01-01", won=1)]))
    ).sort_values("date").reset_index(drop=True)

    for col in ["win_rate", "ufc_fights", "sig_str_acc", "slpm", "td_acc", "kd_per15"]:
        assert t1.loc[0, col] == t2.loc[0, col], col
        assert t1.loc[1, col] == t2.loc[1, col], col
