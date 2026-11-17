"""possession.sequence on synthetic sequences: pooling, a learnable toy task, determinism."""

import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import sequence  # noqa: E402
from possession.sequence import SequenceConfig, pool, predict_proba, step_labels, train  # noqa: E402

TINY = SequenceConfig(channels=8, layers=3, chunk=64, batch=4, iterations=60, learning_rate=1e-2, dropout=0.0)


def toy_match(seed, n=3000):
    """A signal that says who has the ball, with noise; the labels follow it; X around the halves."""
    rng = np.random.default_rng(seed)
    state = np.repeat(rng.integers(0, 3, size=n // 100), 100)
    signal = np.eye(3)[state] + rng.normal(0, 0.8, size=(n, 3))
    half = np.zeros(n, dtype=np.int8)
    half[100:1450], half[1550:2900] = 1, 2
    features = pd.DataFrame({"tm_a": signal[:, 0], "tm_b": signal[:, 1], "tm_c": signal[:, 2],
                             "half": half, "ball_seen": 1.0, "ball_interp": 0.0})
    labels = np.array(list("HAD"))[state]
    labels[half == 0] = "X"
    return features, labels


def test_pool_is_a_nan_aware_block_mean():
    values = np.array([[1.0], [3.0], [np.nan], [np.nan], [5.0]], dtype=np.float32)
    assert np.allclose(pool(values, step=2)[:, 0], [2.0, np.nan, 5.0], equal_nan=True)


def test_step_labels_take_the_block_centre():
    labels = np.array(list("HHHAAXX"))
    assert step_labels(labels, step=3).tolist() == [0, 1, -1]  # centres 1, 4; the last block is frame 6


def test_network_learns_a_toy_task_and_is_deterministic():
    sequence.THREADS = 1
    matches = [toy_match(seed) for seed in range(3)]
    model = train(matches, TINY)
    test_features, test_labels = toy_match(99)
    proba = predict_proba(model, test_features)
    graded = test_labels != "X"
    accuracy = (np.array(list("HAD"))[proba.argmax(axis=1)][graded] == test_labels[graded]).mean()
    assert accuracy > 0.8
    assert np.allclose(proba[test_features["half"].values == 0], 1 / 3)
    again = predict_proba(train(matches, TINY), test_features)
    assert np.array_equal(proba, again)                       # bit for bit


def test_multi_stage_network_learns_and_is_deterministic():
    sequence.THREADS = 1
    config = SequenceConfig(channels=8, layers=3, chunk=64, batch=4, iterations=60, learning_rate=1e-2,
                            dropout=0.0, stages=2, refine_channels=4, refine_layers=3, smooth_weight=0.15)
    matches = [toy_match(seed) for seed in range(3)]
    test_features, test_labels = toy_match(99)
    proba = predict_proba(train(matches, config), test_features)
    graded = test_labels != "X"
    accuracy = (np.array(list("HAD"))[proba.argmax(axis=1)][graded] == test_labels[graded]).mean()
    assert accuracy > 0.7
    assert np.array_equal(proba, predict_proba(train(matches, config), test_features))


def test_ensemble_is_the_mean_of_its_networks():
    sequence.THREADS = 1
    matches = [toy_match(seed) for seed in range(2)]
    test_features, _ = toy_match(99)
    other = SequenceConfig(channels=8, layers=3, chunk=64, batch=4, iterations=60, learning_rate=1e-2,
                           dropout=0.0, seed=1)
    single = predict_proba(train(matches, TINY), test_features)
    assert np.array_equal(sequence.ensemble_proba(matches, test_features, (TINY,)), single)
    both = sequence.ensemble_proba(matches, test_features, (TINY, other))
    assert np.allclose(both, (single + predict_proba(train(matches, other), test_features)) / 2)
    assert set(sequence.ENSEMBLES) == {"base", "all4", "light_symsw", "all4_symsw"}
    assert sequence.ENSEMBLES["base"] == (SequenceConfig(),)
    recipe = sequence.ENSEMBLES["all4_symsw"]
    assert [c.iterations for c in recipe] == [2 * c.iterations for c in sequence.ENSEMBLES["all4"]]
    assert all(c.short_weight == 4.0 for c in recipe) and sequence.ENSEMBLES["light_symsw"] == recipe[:2]


def test_boundary_targets_mark_steps_near_team_changes():
    labels = np.array(list("H" * 30 + "A" * 30 + "D" * 30 + "XXXXX"))
    targets = sequence.boundary_targets(labels, step=5)
    assert targets[5] == 1 and targets[0] == 0       # step 5 = frames 25-29, next to the change at 30
    assert targets[12] == 0                          # the A -> D change is not a turnover
    assert targets[-1] == -1


def test_boundary_network_finds_changes_and_is_deterministic():
    sequence.THREADS = 1
    matches = [toy_match(seed) for seed in range(3)]
    test_features, test_labels = toy_match(99)
    p = sequence.predict_boundary(sequence.train_boundary(matches, TINY), test_features)
    from possession.turnover import turnover_targets
    near = turnover_targets(test_labels)
    assert p[near == 1].mean() > p[near == 0].mean() + 0.2
    assert np.all(p[test_features["half"].values == 0] == 0.5)
    assert np.array_equal(p, sequence.predict_boundary(sequence.train_boundary(matches, TINY), test_features))


def test_spell_weights_raise_short_possessions_only():
    labels = np.array(["X"] * 10 + ["H"] * 25 + ["A"] * 250 + ["D"] * 20 + ["A"] * 50)
    weights = sequence.spell_weights(labels, max_weight=4.0, ref_s=5.0)
    assert np.all(weights[:10] == 1) and np.all(weights[285:305] == 1)       # ungraded and dead ball
    assert np.all(weights[10:35] == 4.0)                                      # 1 s: 5 / 1, capped at 4
    assert np.all(weights[35:285] == 1.0)                                     # 10 s: no extra weight
    np.testing.assert_allclose(weights[305:], 2.5)                            # 2 s: 5 / 2


def test_weighted_training_runs_and_default_is_unchanged():
    sequence.THREADS = 1
    matches = [toy_match(seed) for seed in (0, 1)]
    config = replace(TINY, iterations=5)
    plain = sequence.predict_proba(sequence.train(matches, config), matches[0][0])
    again = sequence.predict_proba(sequence.train(matches, replace(config, short_weight=1.0)), matches[0][0])
    weighted = sequence.predict_proba(sequence.train(matches, replace(config, short_weight=4.0)), matches[0][0])
    np.testing.assert_array_equal(plain, again)
    assert weighted.shape == plain.shape and np.isfinite(weighted).all()
    assert not np.array_equal(weighted, plain)


def test_swapped_team_averaging_is_the_mean_of_the_two_readings():
    features, labels = toy_match(0)
    model = train([(features, labels)], TINY)
    other = features.rename(columns={"tm_a": "tm_b", "tm_b": "tm_a"})[features.columns]
    plain, swapped = predict_proba(model, features), predict_proba(model, other)
    assert np.array_equal(sequence.predict_proba_tta(model, features, None), plain)
    both = sequence.predict_proba_tta(model, features, other)
    assert np.allclose(both, (plain + swapped[:, [1, 0, 2]]) / 2)
    assert np.allclose(both.sum(axis=1), 1.0, atol=1e-5)
