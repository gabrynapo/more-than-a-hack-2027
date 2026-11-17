"""Symmetry copies (possession/augment.py) on a synthetic tracking dict."""

import numpy as np

from possession import augment
from possession.tracking import AWAY, HOME


def tracking():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 60, size=(50, 4, 2)).astype(np.float32)
    pos[5, 1] = np.nan
    ball = rng.uniform(0, 60, size=(50, 2)).astype(np.float32)
    ball[10:15] = np.nan
    return {"pos": pos, "speed": np.ones((50, 4), dtype=np.float32), "ball": ball,
            "team": np.array([HOME, HOME, AWAY, AWAY], dtype=np.int8),
            "pitch": np.array([105.0, 68.0], dtype=np.float32),
            "labels": np.array(["X"] * 5 + ["H"] * 15 + ["D"] * 10 + ["A"] * 20),
            "half": np.array([0] * 5 + [1] * 45)}


def test_swap_exchanges_teams_and_labels_only():
    source = tracking()
    out = augment.transform(source, swap=True)
    assert list(out["team"]) == [AWAY, AWAY, HOME, HOME]
    assert list(out["labels"][5:20]) == ["A"] * 15 and list(out["labels"][30:]) == ["H"] * 20
    assert list(out["labels"][:5]) == ["X"] * 5 and list(out["labels"][20:30]) == ["D"] * 10
    np.testing.assert_array_equal(out["pos"], source["pos"])
    np.testing.assert_array_equal(out["ball"], source["ball"])


def test_mirror_reflects_y_and_keeps_gaps():
    source = tracking()
    out = augment.transform(source, mirror=True)
    np.testing.assert_allclose(out["pos"][..., 1], 68.0 - source["pos"][..., 1], rtol=1e-6)
    np.testing.assert_array_equal(out["pos"][..., 0], source["pos"][..., 0])
    assert np.isnan(out["ball"][10:15]).all() and np.isnan(out["pos"][5, 1]).all()
    assert list(out["labels"]) == list(source["labels"])


def test_transforms_are_involutions_and_leave_the_source_untouched():
    source = tracking()
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in source.items()}
    twice = augment.transform(augment.transform(source, swap=True, mirror=True), swap=True, mirror=True)
    for key in ("pos", "ball"):
        np.testing.assert_allclose(twice[key], before[key], rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(source[key], before[key])
    assert list(twice["team"]) == list(before["team"]) and list(twice["labels"]) == list(before["labels"])


def test_swap_proba_exchanges_home_and_away_columns():
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    np.testing.assert_array_equal(augment.swap_proba(proba), [[0.2, 0.7, 0.1], [0.3, 0.1, 0.6]])
    np.testing.assert_array_equal(augment.swap_labels(np.array(["H", "A", "D", "X"])), ["A", "H", "D", "X"])
