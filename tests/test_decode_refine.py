"""Synthetic-data tests for the temporal decoder and boundary clean-up."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession.decode import decode, viterbi  # noqa: E402
from possession.refine import drop_fragments  # noqa: E402

PRIOR = np.full(3, 1.0 / 3)


def proba_from(text, confidence=0.8):
    index = {"H": 0, "A": 1, "D": 2}
    proba = np.full((len(text), 3), (1 - confidence) / 2)
    for t, c in enumerate(text):
        proba[t, index[c]] = confidence
    return proba


def test_viterbi_ignores_isolated_flicker():
    text = "H" * 200 + "A" * 3 + "H" * 200
    path = viterbi(proba_from(text), PRIOR)
    assert (path == 0).all()


def test_viterbi_follows_a_sustained_change():
    text = "H" * 300 + "A" * 300
    path = viterbi(proba_from(text), PRIOR)
    assert path[0] == 0 and path[-1] == 1
    assert abs(int(np.argmax(path == 1)) - 300) < 25


def test_decode_writes_x_outside_halves():
    half = np.array([0] * 5 + [1] * 50 + [0] * 5)
    labels = decode(proba_from("H" * 60), half, PRIOR)
    assert (labels[:5] == "X").all() and (labels[-5:] == "X").all()
    assert (labels[5:55] == "H").all()


def test_drop_fragments_merges_slivers_into_supported_neighbour():
    labels = np.array(list("H" * 30 + "A" * 3 + "H" * 30), dtype="<U1")
    proba = proba_from("H" * 63)
    half = np.ones(63, dtype=int)
    out = drop_fragments(labels, proba, half, min_frames=12)
    assert (out == "H").all()


def test_posterior_is_normalised_and_agrees_on_a_clear_signal():
    from possession.decode import posterior
    text = "H" * 300 + "A" * 300
    post = posterior(proba_from(text), PRIOR)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-9)
    assert (np.argmax(post, axis=1) == viterbi(proba_from(text), PRIOR)).mean() > 0.97


def test_decode_rejects_unknown_method():
    import pytest
    with pytest.raises(ValueError):
        decode(proba_from("HH"), np.array([1, 1]), PRIOR, method="nope")


def test_force_pre_kickoff_dead_only_touches_that_half_before_kickoff():
    from possession.pipeline import force_pre_kickoff_dead
    labels = np.array(list("HHHHAAAA"), dtype="<U1")
    half = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    out = force_pre_kickoff_dead(labels, half, {2: 6})
    assert "".join(out) == "HHHHDDAA"
    assert "".join(force_pre_kickoff_dead(labels, half, {})) == "HHHHAAAA"


def test_hybrid_takes_dead_from_one_model_and_team_split_from_the_other():
    from possession.pipeline import hybrid_proba
    alive_dead = np.array([[0.1, 0.1, 0.8], [0.45, 0.45, 0.1]])
    team = np.array([[0.6, 0.2, 0.2], [0.1, 0.3, 0.6]])
    out = hybrid_proba(alive_dead, team)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    np.testing.assert_allclose(out[:, 2], [0.8, 0.1])
    np.testing.assert_allclose(out[0, :2], [0.2 * 0.75, 0.2 * 0.25])
    np.testing.assert_allclose(out[1, :2], [0.9 * 0.25, 0.9 * 0.75])
