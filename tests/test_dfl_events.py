"""possession.dfl_events on synthetic timelines: synchronisation and event-backed labels."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import dfl  # noqa: E402
from possession.dfl_events import control_frames, event_labels, synchronise  # noqa: E402


def lab(text):
    return np.array(list(text), dtype="<U1")


def test_short_spell_with_control_evidence_is_kept():
    labels = lab("A" * 100 + "H" * 20 + "A" * 100)          # 20 frames = 0.8 s
    kept = event_labels(labels, {"H": np.array([110]), "A": np.array([], dtype=np.int64)}, 1.5)
    assert "".join(kept) == "".join(labels)


def test_short_spell_without_evidence_is_folded_like_controlled_labels():
    labels = lab("A" * 100 + "H" * 20 + "A" * 100)
    none = {"H": np.array([], dtype=np.int64), "A": np.array([50, 150])}
    assert "".join(event_labels(labels, none, 1.5)) == "".join(dfl.controlled_labels(labels, 1.5)) == "A" * 220


def test_evidence_of_the_other_team_or_far_away_does_not_count():
    labels = lab("A" * 100 + "H" * 20 + "A" * 100)
    wrong_team = {"H": np.array([], dtype=np.int64), "A": np.array([110])}
    far = {"H": np.array([60, 160]), "A": np.array([], dtype=np.int64)}
    assert (event_labels(labels, wrong_team, 1.5) == "A").all()
    assert (event_labels(labels, far, 1.5) == "A").all()


def test_tolerance_just_outside_the_spell():
    labels = lab("A" * 100 + "H" * 20 + "A" * 100)
    near = {"H": np.array([97]), "A": np.array([], dtype=np.int64)}
    assert (event_labels(labels, near, 1.5, tolerance=5) == labels).all()
    assert (event_labels(labels, near, 1.5, tolerance=2) == "A").all()


def test_restart_spells_and_long_spells_are_untouched():
    labels = lab("D" * 50 + "H" * 10 + "A" * 100)          # short H after a stoppage = restart
    none = {"H": np.array([], dtype=np.int64), "A": np.array([], dtype=np.int64)}
    assert (event_labels(labels, none, 1.5) == labels).all()


def test_control_frames_keeps_synchronised_control_events_only():
    rows = [{"team": "H", "frame": 30, "control": True}, {"team": "H", "frame": -1, "control": True},
            {"team": "A", "frame": 10, "control": False}, {"team": "A", "frame": 20, "control": True}]
    frames = control_frames(rows)
    assert frames["H"].tolist() == [30] and frames["A"].tolist() == [20]


def test_synchronise_moves_the_event_to_the_players_touch():
    n = 600
    half = np.zeros(n, dtype=np.int8)
    half[50:550] = 1
    half_starts_at = 50
    ball = np.tile(np.array([[50.0, 30.0]], dtype=np.float32), (n, 1))
    pos = np.full((n, 2, 2), 80.0, dtype=np.float32)
    pos[300, 1] = [50.2, 30.0]                               # player slot 1 touches the ball at frame 300
    tracking = {"half": half, "ball": ball, "pos": pos}
    t0 = 1000.0
    meta = {"players": [("T1", "P0"), ("T2", "P1")],
            "clock": {"firstHalf": {"t0": t0, "t1": t0 + 20}, "secondHalf": {"t0": t0 + 9999, "t1": t0 + 99999}}}
    stated = t0 + (300 - half_starts_at) / 25 + 1.5           # typed 1.5 s late
    row = synchronise([{"player": "P1", "time": stated}], tracking, meta)[0]
    assert row["frame"] == 300 and abs(row["offset_s"] + 1.5) <= 1 / 25      # within one frame of rounding
    lost = synchronise([{"player": "P0", "time": stated}], tracking, meta)[0]
    assert lost["frame"] == -1                               # P0 never near the ball
