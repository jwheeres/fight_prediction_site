#!/usr/bin/env python3
"""Train a real head-to-head win model and save it for the app to use.

The app (qualia/model.py) automatically loads models/h2h_model.pkl if it
exists, so training here is all that's needed to replace the baseline.

Dataset format — a CSV with one row per historical fight:
  * the 16 feature columns named in the schema (see qualia.model.FEATURE_SCHEMA),
  * a `winner` column = 1 if fighter A (the f1_* fighter) won, else 0.

`data/fights_dataset.sample.csv` documents the exact columns.

Usage:
    python -m qualia.train --data data/fights_dataset.csv
    python -m qualia.train --data data/fights.csv --model logistic --test-size 0.2

The pipeline reports honest held-out metrics (accuracy, log-loss, Brier, AUC)
so you can judge whether the model actually beats a coin flip / the market —
per the project's "expectancy, not vibes" ethos.
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from qualia.model import FEATURE_SCHEMA

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "models" / "h2h_model.pkl"


def build_estimator(kind: str):
    if kind == "logistic":
        return LogisticRegression(max_iter=1000)
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
    raise ValueError(f"unknown model kind: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Training CSV")
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "logistic"])
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibrate", action="store_true", help="Wrap in probability calibration")
    args = parser.parse_args()

    if not FEATURE_SCHEMA:
        print("Could not read feature schema from rf_h2h_model_updated.pkl.")
        return 1

    df = pd.read_csv(args.data)
    missing = [c for c in FEATURE_SCHEMA + ["winner"] if c not in df.columns]
    if missing:
        print(f"Dataset is missing required columns: {missing}")
        return 1

    X = df[FEATURE_SCHEMA].to_numpy(dtype=float)
    y = df["winner"].to_numpy(dtype=int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y if len(set(y)) > 1 else None
    )

    est = build_estimator(args.model)
    if args.calibrate:
        est = CalibratedClassifierCV(est, cv=3)
    est.fit(X_tr, y_tr)

    proba = est.predict_proba(X_te)[:, list(est.classes_).index(1)]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "accuracy": round(float(accuracy_score(y_te, preds)), 4),
        "log_loss": round(float(log_loss(y_te, proba, labels=[0, 1])), 4),
        "brier": round(float(brier_score_loss(y_te, proba)), 4),
        "auc": round(float(roc_auc_score(y_te, proba)), 4) if len(set(y_te)) > 1 else None,
    }

    print("=== Held-out metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("  (baseline to beat: accuracy 0.5, log_loss 0.693, brier 0.25)")

    if args.model == "random_forest" and not args.calibrate:
        importances = sorted(zip(FEATURE_SCHEMA, est.feature_importances_), key=lambda t: -t[1])
        print("=== Top features ===")
        for name, imp in importances[:6]:
            print(f"  {name}: {imp:.3f}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": est,
        "features": FEATURE_SCHEMA,
        "kind": args.model,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
    }
    OUT_FILE.write_bytes(pickle.dumps(bundle))
    print(f"\nSaved trained model -> {OUT_FILE.relative_to(ROOT)}")
    print("The app will use it automatically on next start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
