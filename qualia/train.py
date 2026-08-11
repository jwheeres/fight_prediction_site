#!/usr/bin/env python3
"""Train the head-to-head model on real UFC data.

Reads data/training.csv (built by scripts/build_fighter_data.py), where each
row is the career-stat difference (fighter A minus B) plus winner (1 = A won).

Key detail: we SYMMETRIZE — every fight is added both ways (A-B, win) and
(B-A, loss) — so the model can't exploit corner/ordering bias and must learn
from real skill differences. Fights are split into train/test *before*
symmetrizing so no fight leaks across the split.

    python -m qualia.train                # random forest (default)
    python -m qualia.train --model logistic

Honest note: features are whole-career averages, so there's mild look-ahead
leakage; held-out numbers slightly over-state real forecasting skill.
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qualia.model import FEATURES

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "training.csv"
OUT_FILE = ROOT / "models" / "h2h_model.pkl"


def build_estimator(kind: str):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=400, max_depth=9, min_samples_leaf=12,
            max_features="sqrt", random_state=42, n_jobs=-1,
        )
    raise ValueError(kind)


def _symmetrize(X: np.ndarray, y: np.ndarray):
    Xs = np.vstack([X, -X])
    ys = np.concatenate([y, 1 - y])
    return Xs, ys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="random_forest", choices=["random_forest", "logistic"])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(TRAIN_FILE)
    cols = [f"d_{k}" for k in FEATURES]
    X = df[cols].to_numpy(dtype=float)
    y = df["winner"].to_numpy(dtype=int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    X_tr, y_tr = _symmetrize(X_tr, y_tr)
    X_te, y_te = _symmetrize(X_te, y_te)

    est = build_estimator(args.model)
    est.fit(X_tr, y_tr)

    proba = est.predict_proba(X_te)[:, list(est.classes_).index(1)]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "n_train_fights": int(len(X_tr) // 2),
        "n_test_fights": int(len(X_te) // 2),
        "accuracy": round(float(accuracy_score(y_te, preds)), 4),
        "log_loss": round(float(log_loss(y_te, proba, labels=[0, 1])), 4),
        "brier": round(float(brier_score_loss(y_te, proba)), 4),
        "auc": round(float(roc_auc_score(y_te, proba)), 4),
    }
    print("=== Held-out metrics (symmetrized, so 0.5 = coin flip) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.model == "random_forest":
        imp = sorted(zip(FEATURES, est.feature_importances_), key=lambda t: -t[1])
        print("=== Top features ===")
        for name, i in imp[:8]:
            print(f"  {name}: {i:.3f}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_bytes(pickle.dumps({
        "model": est, "features": FEATURES, "kind": args.model,
        "metrics": metrics, "trained_at": datetime.now(timezone.utc).isoformat(),
    }))
    print(f"\nSaved {OUT_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
