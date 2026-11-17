"""possession.turnover and the frame-dependent switching cost of the decoder, on synthetic data."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession.decode import decode, scaled_transitions, transition_matrix, viterbi  # noqa: E402
from possession.turnover import switch_scale, turnover_targets  # noqa: E402

PRIOR = np.full(3, 1.0 / 3)


def lab(text):
    return np.array(list(text), dtype="<U1")


def proba_from(text, confidence=0.8):
    index = {"H": 0, "A": 1, "D": 2}
    proba = np.full((len(text), 3), (1 - confidence) / 2)
    for t, c in enumerate(text):
        proba[t, index[c]] = confidence
    return proba


def test_targets_mark_team_changes_only():
    labels = lab("X" * 3 + "H" * 10 + "A" * 10 + "D" * 10 + "H" * 5 + "X" * 2)
    targets = turnover_targets(labels, window=2)
    assert (targets[:3] == -1).all() and (targets[-2:] == -1).all()
    assert np.where(targets == 1)[0].tolist() == [11, 12, 13, 14]      # around the H -> A at 13
    assert (targets[3:-2] >= 0).all()                                  # A -> D and D -> H are not turnovers


def test_gamma_zero_and_unit_scale_keep_the_fixed_hmm():
    assert (switch_scale(np.array([0.01, 0.5, 0.99]), 0.0) == 1).all()
    assert np.allclose(scaled_transitions(np.ones(4))[2], transition_matrix())
    text = "H" * 200 + "A" * 40 + "H" * 150 + "D" * 100 + "A" * 200
    proba = proba_from(text, 0.6)
    assert (viterbi(proba, PRIOR, 0.35, np.ones(len(text))) == viterbi(proba, PRIOR, 0.35)).all()


def test_cheap_switch_at_the_detected_turnover_keeps_a_short_possession():
    text = "A" * 200 + "H" * 25 + "A" * 200                            # one second of H
    proba = proba_from(text, 0.6)
    merged = viterbi(proba, PRIOR, 0.35)
    assert (merged == 1).all()                                         # fixed cost: merged into A
    detector = np.full(len(text), 0.05)
    detector[195:205] = detector[220:230] = 0.99                      # sure of a change at both ends
    kept = viterbi(proba, PRIOR, 0.35, switch_scale(detector, 1.0))
    assert (kept[205:220] == 0).all() and (kept[:190] == 1).all() and (kept[235:] == 1).all()


def test_decode_passes_the_scale_per_half():
    half = np.array([0] * 5 + [1] * 200 + [2] * 200)
    proba = proba_from("H" * 405)
    out = decode(proba, half, PRIOR, 0.35, switch_scale=np.ones(405))
    assert "".join(out[:5]) == "XXXXX" and (out[5:] == "H").all()
