#!/usr/bin/env python3
"""Generate a SYNTHETIC fight dataset to validate the training pipeline.

This is NOT real data. It exists only to prove qualia/train.py runs end to end
and produces a working model artifact. For real predictions, replace it with a
real historical dataset in the same column format (see
data/fights_dataset.sample.csv).

Usage:
    python scripts/make_synth_dataset.py --n 4000 --out data/fights_synth.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qualia.model import FEATURE_SCHEMA  # noqa: E402

# A "true" latent weighting the generator uses to decide winners, so the model
# has a real (but noisy) signal to learn. Deliberately different from the app's
# baseline weights.
TRUE_W = {
    "f1_striking_balance": 2.0, "f2_striking_balance": -2.0,
    "f1_finish_rate": 1.5, "f2_finish_rate": -1.5,
    "f1_strike_defense": 1.2, "f2_strike_defense": -1.2,
    "f1_takedown_success_rate": 0.8, "f2_takedown_success_rate": -0.8,
    "striking_accuracy_diff": 1.0, "takedown_defense_diff": 0.6,
    "ground_control_diff": 0.5, "clinch_control_diff": 0.3,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=Path("data/fights_synth.csv"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.n
    cols = {}
    for name in FEATURE_SCHEMA:
        if name.endswith("_diff"):
            cols[name] = rng.normal(0, 0.15, n)
        elif "avg_fight_duration" in name:
            cols[name] = rng.uniform(6, 15, n)
        else:  # rates / accuracies / defenses in 0..1
            cols[name] = rng.uniform(0.2, 0.9, n)
    df = pd.DataFrame(cols)

    logit = sum(w * df[name] for name, w in TRUE_W.items())
    logit = logit - logit.mean()
    prob = 1.0 / (1.0 + np.exp(-logit))
    df["winner"] = (rng.uniform(0, 1, n) < prob).astype(int)

    df = df[FEATURE_SCHEMA + ["winner"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} synthetic rows -> {args.out} (win rate {df['winner'].mean():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
