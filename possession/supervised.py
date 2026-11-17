"""Frame classifiers trained on real labels: degraded DFL matches, controlled-possession targets.

The self-trained model (possession/model.py) learns from rule-based pseudo-labels on the match
itself. Those rules need the ball to name a team, so inside ball gaps -- 41% of graded frames --
nearly every pseudo-label is D, and the "blind" model has never seen a live frame without a ball.
Here every frame of seven degraded Bundesliga matches has a real label, gaps included.

Same two-model split as model.py, for the same reason (no visibility shortcut):

    seen    ball known (seen or interpolated): every feature except visibility bookkeeping
    blind   ball unknown: only features that exist without a ball (team shape, speeds, boxes)

Smart Tagging columns do not exist on DFL, so these models never use them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from possession.model import ball_known, feature_columns

SUBSAMPLE = 5           # neighbouring frames are near-duplicates
# One thread: with several, OpenMP sums gradients in a varying order and the probabilities differ
# run to run in the last bits (measured: ~1e-15). The submission must be reproduced bit for bit.
THREADS = 1
SEED = 0
CODES = {"H": 0, "A": 1, "D": 2}


def target_codes(labels: np.ndarray) -> np.ndarray:
    """H/A/D -> 0/1/2, anything else (X) -> -1."""
    out = np.full(len(labels), -1, dtype=np.int8)
    for name, code in CODES.items():
        out[labels == name] = code
    return out


def _classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, class_weight="balanced", early_stopping=False, random_state=SEED)


def stack(matches: list[tuple[pd.DataFrame, np.ndarray]], encode=target_codes, subsample: int = SUBSAMPLE):
    """Training rows from [(features, labels)]: graded, subsampled frames of every match.

    `encode` turns one match's labels into integer targets, -1 for frames left out of training;
    `subsample` keeps every n-th frame (1 when the rows were subsampled already).
    """
    frames, targets = [], []
    for features, labels in matches:
        codes = encode(labels)
        keep = (codes >= 0) & (np.arange(len(codes)) % subsample == 0)
        frames.append(features.loc[keep])
        targets.append(codes[keep])
    return pd.concat(frames, ignore_index=True), np.concatenate(targets)


def train(matches: list[tuple[pd.DataFrame, np.ndarray]], encode=target_codes, subsample: int = SUBSAMPLE) -> dict:
    """{"seen": model, "blind": model} trained on [(features, labels)] of several matches."""
    features, targets = stack(matches, encode, subsample)
    known = ball_known(features)
    models = {}
    for kind, mask in (("seen", known), ("blind", ~known)):
        columns = [c for c in feature_columns(features, kind) if not c.startswith("st_")]
        model = _classifier()
        with threadpool_limits(THREADS):
            model.fit(features.loc[mask, columns].values, targets[mask])
        model.feature_columns_ = columns
        models[kind] = model
    return models


def predict_proba(models: dict, features: pd.DataFrame) -> np.ndarray:
    """(N, classes) in class-code order (H, A, D for the possession models): the seen model where
    the ball is known, the blind one elsewhere."""
    def proba(model, frame):
        with threadpool_limits(THREADS):
            out = model.predict_proba(frame[model.feature_columns_].values)
        return out[:, np.argsort(model.classes_)]

    known = ball_known(features)
    out = proba(models["blind"], features)
    if known.any():
        out[known] = proba(models["seen"], features.loc[known])
    return out
