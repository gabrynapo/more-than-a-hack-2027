"""Symmetry copies of a tracking dict: the same match seen with the teams swapped or the pitch mirrored.

Seven DFL matches are few, and the temporal networks go through them about fifty times. Football
has two exact symmetries the labels obey, so each gives new training sequences for free:

    swap     the home side becomes the away side and back: every H label becomes A and every A, H.
             The features are built in a "home attacks +x" frame, so the swapped match is also
             played in the other direction.
    mirror   the pitch is reflected across its long axis (y -> width - y): left wing and right wing
             trade places, labels unchanged.

The transform is applied to the tracking (clean, before `noise.degrade`, or noisy), never to the
features, so nothing has to know which feature columns are signed. Labels and the half index live
in the same dict (`dfl.load`) and are transformed with it.
"""

from __future__ import annotations

import numpy as np

from possession.tracking import AWAY, HOME

SWAPPED_LABEL = {"H": "A", "A": "H"}


def swap_labels(labels: np.ndarray) -> np.ndarray:
    """H <-> A, everything else unchanged."""
    labels = np.asarray(labels)
    out = labels.copy()
    out[labels == "H"] = "A"
    out[labels == "A"] = "H"
    return out


def swap_proba(proba: np.ndarray) -> np.ndarray:
    """(N, 3) H, A, D probabilities of the swapped match, read back in the original team order."""
    return np.asarray(proba)[:, [1, 0, 2]]


def transform(tracking: dict, swap: bool = False, mirror: bool = False) -> dict:
    """A copy of `tracking` with the teams swapped and / or the pitch mirrored across its long axis."""
    out = {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in tracking.items()}
    if swap:
        team = np.asarray(tracking["team"])
        out["team"] = np.where(team == HOME, AWAY, HOME).astype(team.dtype)
        if "labels" in tracking:
            out["labels"] = swap_labels(tracking["labels"])
    if mirror:
        width = float(tracking["pitch"][1])
        out["pos"][..., 1] = width - out["pos"][..., 1]
        out["ball"][..., 1] = width - out["ball"][..., 1]
    return out
