"""possession.split on synthetic probabilities."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession.decode import decode  # noqa: E402
from possession.split import SplitConfig, insert_turnovers, runs  # noqa: E402

PRIOR = np.full(3, 1.0 / 3)


def proba_of(text, confidence):
    """Per-frame probabilities leaning `confidence` to each letter of `text`."""
    index = {"H": 0, "A": 1, "D": 2}
    out = np.full((len(text), 3), (1 - confidence) / 2)
    out[np.arange(len(text)), [index[c] for c in text]] = confidence
    return out


def test_runs():
    assert runs(np.array(list("AAHHHD"))) == [(0, 2, "A"), (2, 5, "H"), (5, 6, "D")]


def test_a_short_turnover_merged_by_the_coarse_decoder_is_reopened():
    truth = "A" * 400 + "H" * 30 + "A" * 400
    proba = proba_of(truth, 0.62)
    half = np.ones(len(truth), dtype=int)
    coarse = decode(proba, half, PRIOR, evidence_weight=0.35)
    assert (coarse == "A").all()                       # the HMM merges the short H away
    out = insert_turnovers(coarse, proba, half, PRIOR, SplitConfig(fine_weight=1.0, threshold=0.55))
    assert "".join(out) == truth


def test_nothing_is_inserted_when_the_classifier_does_not_lean():
    truth = "A" * 400 + "H" * 30 + "A" * 400
    proba = proba_of(truth, 0.62)
    half = np.ones(len(truth), dtype=int)
    coarse = np.array(list("A" * len(truth)))
    strict = insert_turnovers(coarse, proba, half, PRIOR, SplitConfig(threshold=0.99))
    assert (strict == "A").all()


def test_runs_at_a_possession_edge_or_across_dead_ball_are_left_alone():
    # H then D: a fine H run touching the stoppage is not "inside" an A possession
    truth = "A" * 400 + "H" * 30 + "D" * 200
    proba = proba_of(truth, 0.62)
    half = np.ones(len(truth), dtype=int)
    coarse = np.array(list("A" * 430 + "D" * 200))
    out = insert_turnovers(coarse, proba, half, PRIOR, SplitConfig(threshold=0.55))
    assert "".join(out) == "".join(coarse)


def test_too_short_runs_are_ignored():
    truth = "A" * 400 + "H" * 8 + "A" * 400
    proba = proba_of(truth, 0.9)
    half = np.ones(len(truth), dtype=int)
    coarse = np.array(list("A" * len(truth)))
    out = insert_turnovers(coarse, proba, half, PRIOR, SplitConfig(min_frames=12, threshold=0.55))
    assert (out == "A").all()
