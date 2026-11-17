"""possession.metrica_sample: labels from Metrica-style events, on synthetic event lists."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession.metrica_sample import _row, event_labels, player_speed  # noqa: E402


def ev(team, kind, start, end=None, subtypes=()):
    return _row(team, kind, subtypes, 1, start, start if end is None else end)


def text(labels):
    return "".join(labels)


def test_possession_follows_the_latest_possession_event():
    events = [ev("H", "SET PIECE", 0, subtypes=("KICK OFF",)), ev("H", "PASS", 0, 10),
              ev("H", "BALL LOST", 20, 25, ("INTERCEPTION",)), ev("A", "RECOVERY", 30),
              ev("A", "CHALLENGE", 35, subtypes=("GROUND", "LOST")), ev("A", "PASS", 40, 45)]
    labels = event_labels(events, 1, 0, 49)
    assert text(labels) == "H" * 30 + "A" * 20                   # A only once it recovers the ball


def test_ball_out_is_dead_until_the_restart():
    events = [ev("H", "PASS", 0, 5), ev("A", "BALL OUT", 10, 15, ("CLEARANCE",)),
              ev("H", "SET PIECE", 30, subtypes=("THROW IN",)), ev("H", "PASS", 30, 35)]
    labels = event_labels(events, 1, 0, 39)
    assert text(labels) == "H" * 15 + "D" * 15 + "H" * 10       # a clearance out is no possession of A


def test_a_foul_stops_play_but_advantage_does_not():
    foul = [ev("H", "PASS", 0, 5), ev("A", "CHALLENGE", 10, subtypes=("TACKLE", "FAULT", "LOST")),
            ev("H", "FAULT RECEIVED", 10), ev("H", "SET PIECE", 25, subtypes=("FREE KICK",))]
    assert text(event_labels(foul, 1, 0, 29)) == "H" * 10 + "D" * 15 + "H" * 5
    advantage = [ev("H", "PASS", 0, 5), ev("A", "CHALLENGE", 10, subtypes=("GROUND", "ADVANTAGE", "LOST")),
                 ev("H", "PASS", 15, 20)]
    assert text(event_labels(advantage, 1, 0, 24)) == "H" * 25


def test_goal_kick_and_shot_out():
    events = [ev("A", "SHOT", 0, 10, ("OFF TARGET", "OUT")), ev("H", "PASS", 40, 50, ("GOAL KICK",))]
    assert text(event_labels(events, 1, 0, 59)) == "A" * 10 + "D" * 30 + "H" * 20


def test_player_speed_within_halves():
    pos = np.zeros((20, 1, 2), dtype=np.float32)
    pos[:, 0, 0] = np.arange(20) * 0.2                          # 0.2 m per frame = 5 m/s
    half = np.array([0] * 2 + [1] * 16 + [0] * 2, dtype=np.int8)
    speed = player_speed(pos, half)
    assert np.allclose(speed[4:16, 0], 5.0)
    assert np.isnan(speed[:4, 0]).all() and np.isnan(speed[16:, 0]).all()
