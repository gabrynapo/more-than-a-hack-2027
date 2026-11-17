"""A turnover detector, and the switching cost it sets in the HMM.

The HMM charges the same price for every change of team (decode.py: a possession lasts ~15 s, so it
is left with probability 1 / 375 per frame). That price is right on average and wrong exactly where
spellF1 is decided: a real one-second possession between two of the other team needs a burst of
evidence strong enough to pay two switches, and Viterbi prefers the merge (A-H-A decoded as A),
which costs up to three matched possessions.

So a second classifier answers a different question from the possession classifiers: "does the
team in possession change within WINDOW_FRAMES of this frame?" -- trained on the same degraded DFL
matches and labels, with the past/future context features (context.py) that localise a change.
Its odds then scale the H <-> A transition probability frame by frame:

    P(H -> A at t) = base * odds(c_t) ** gamma          (clipped to a valid probability)

gamma = 0 is the fixed-cost HMM. Where the detector is sure a turnover happens, switching gets
cheaper; where it is sure none does, dearer. Dead-ball transitions are left alone. This is a hidden
Markov model with input-dependent transitions (an "input-output HMM", Bengio & Frasconi 1995), with
the input being the detector's output.
"""

from __future__ import annotations

import numpy as np

from possession import supervised

WINDOW_FRAMES = 12          # +/-0.5 s: a frame is "at a turnover" this close to an H <-> A change
MIN_SCALE, MAX_SCALE = 1e-4, 1e4


def turnover_targets(labels: np.ndarray, window: int = WINDOW_FRAMES) -> np.ndarray:
    """1 within `window` frames of a change between H and A, 0 on other graded frames, -1 on X."""
    labels = np.asarray(labels)
    out = np.where(labels == "X", -1, 0).astype(np.int8)
    change = np.where((labels[1:] != labels[:-1]) & np.isin(labels[1:], ("H", "A"))
                      & np.isin(labels[:-1], ("H", "A")))[0] + 1
    near = np.zeros(len(labels) + 1, dtype=np.int32)
    np.add.at(near, np.maximum(change - window, 0), 1)
    np.add.at(near, np.minimum(change + window, len(labels)), -1)
    out[(np.cumsum(near)[:-1] > 0) & (out == 0)] = 1
    return out


def training_rows(features, labels) -> tuple:
    """(context features, targets) of one match on the frames the classifiers train on.

    The context columns are added here and only the subsampled graded rows are kept, so that
    twenty-one training copies fit in memory together.
    """
    from possession.context import add_context
    targets = turnover_targets(labels)
    keep = (targets >= 0) & (np.arange(len(targets)) % supervised.SUBSAMPLE == 0)
    return add_context(features).loc[keep].reset_index(drop=True), targets[keep]


def train(rows) -> dict:
    """Seen / blind binary classifiers on [training_rows(...)] of several matches."""
    return supervised.train(rows, encode=lambda targets: targets, subsample=1)


def predict(models: dict, features) -> np.ndarray:
    """(N,) P(turnover near this frame)."""
    return supervised.predict_proba(models, features)[:, 1]


def switch_scale(p: np.ndarray, gamma: float) -> np.ndarray:
    """Per-frame multiplier of the H <-> A transition probability: odds(p) ** gamma."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.clip((p / (1 - p)) ** gamma, MIN_SCALE, MAX_SCALE)
