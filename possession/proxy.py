"""The validation proxy: DFL matches, degraded to look like the competition ATD, scored officially.

Seventeen public possessions cannot tell two versions of the pipeline apart. Seven DFL matches carry
~2,000 frame-labelled possessions of the same kind, so here a difference can be measured. The
degradation (possession/noise.py) is what makes it relevant to the competition match.

Pooled scoring: all matches are concatenated with ungraded ("X") separators and scored once by the
official `possession_metrics.evaluate`, so spells never straddle two matches and the result is the
F1 over every possession of every match, not an average of per-match F1s.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from possession_metrics import evaluate, label_codes  # noqa: E402

from possession import augment, dfl  # noqa: E402
from possession.features import build_from_tracking  # noqa: E402
from possession.noise import NoiseConfig, degrade  # noqa: E402

NOISE_VERSION = "n2"        # bump whenever NoiseConfig defaults change, to invalidate caches
# Proxy label definitions (dfl.controlled_labels): 1.5 s reproduces the public window's possession
# rate and duration profile; 1.0 and 2.0 bracket it, and a choice has to hold on all three.
LABEL_TAUS = (1.0, 1.5, 2.0)
PRIMARY_TAU = 1.5
BASE_SEED = 2027


def seed_for(match_id: str, replica: int = 0) -> int:
    return BASE_SEED + 100 * dfl.MATCH_IDS.index(match_id) + replica


# Noise presets. "n2" is what dfl_sup_v1 was trained on and must stay byte-identical so that v1 can
# be regenerated. "n3" is calibrated against label-free statistics of the competition match
# (scripts/calibrate_noise.py, candidate H): low-frequency wander instead of frame-by-frame jitter,
# a ball rarely seen near the lines, gaps concentrated at stoppages and slow play.
N3 = NoiseConfig(ball_jitter_m=0.55, ball_wander_frames=5, player_wander_frames=6, player_missing=0.03,
                 line_margin_m=5, line_visible_min=0.3, missing_alive=0.05, missing_dead=0.62,
                 gap_speed_beta=5.0)
NOISE_PRESETS = {"n2": NoiseConfig(), "n3": N3}


def prepare(match_id: str, replica: int = 0, config: NoiseConfig | None = None,
            use_cache: bool = True, noise: str = "n2"):
    """(noisy tracking, features, gt labels) for one DFL match degraded with preset `noise`
    (or an explicit `config`, never cached). Features of presets are cached."""
    preset = config is None
    config = NOISE_PRESETS[noise] if preset else config
    clean = dfl.load(match_id)
    labels, half = clean["labels"], clean["half"]
    noisy = degrade(clean, labels, half, seed_for(match_id, replica), config)
    cache = os.path.join(dfl.CACHE_DIR, "{}_{}_r{}_features.parquet".format(match_id, noise, replica))
    if use_cache and preset and os.path.isfile(cache):
        features = pd.read_parquet(cache)
    else:
        features = build_from_tracking(noisy, half, dfl.half_starts(half))
        if use_cache and preset:
            os.makedirs(dfl.CACHE_DIR, exist_ok=True)
            features.to_parquet(cache)
    return noisy, features, labels


# Domain randomisation for the final supervised models: each DFL match is used three times, under
# the preset and two other settings, so the classifiers do not depend on the exact rates we
# assumed. Validated by scripts/proxy_robustness.py ("rand" rows): never worse than default-only
# training under any test noise, +0.017 spellF1 under the harshest. (name, noise, seed replica)
TRAINING_NOISE = {
    "n2": (
        ("default", NoiseConfig(), 0),
        ("rand_a", NoiseConfig(missing_alive=0.36, missing_dead=0.62, gap_median_s=12.0, ball_smooth_frames=7.0,
                               ball_jitter_m=0.4, player_missing=0.10, player_jitter_m=0.25), 10),
        ("rand_b", NoiseConfig(missing_alive=0.25, missing_dead=0.50, gap_median_s=8.0, gap_sigma=1.2,
                               ball_smooth_frames=3.0, player_missing=0.05), 20),
    ),
    "n3": (
        ("default", N3, 0),
        ("rand_a", replace(N3, ball_jitter_m=0.65, line_margin_m=6, line_visible_min=0.25, missing_alive=0.08,
                           missing_dead=0.70, gap_median_s=12.0, ball_smooth_frames=7.0, player_missing=0.05), 10),
        ("rand_b", replace(N3, ball_jitter_m=0.45, line_margin_m=4, line_visible_min=0.4, missing_alive=0.03,
                           missing_dead=0.55, gap_median_s=8.0, gap_sigma=1.2, ball_smooth_frames=3.0,
                           gap_speed_beta=3.0), 20),
    ),
}


def extras_cache(match_id: str, noise: str, name: str) -> str:
    """Cache of the extra columns (possession/extra_features.py) of one copy of a DFL match."""
    return os.path.join(dfl.CACHE_DIR, "{}_{}_{}_extra.parquet".format(match_id, noise, name))


def with_extras(match_id: str, noise: str, name: str, features, noisy, half):
    """`features` with the extra columns appended, cached. `noisy` is the degraded tracking the
    features were built from: a callable returning it, only called on a cache miss."""
    from possession.extra_features import extra_columns
    path = extras_cache(match_id, noise, name)
    if os.path.isfile(path):
        extra = pd.read_parquet(path)
    else:
        extra = extra_columns(noisy(), half, features)
        os.makedirs(dfl.CACHE_DIR, exist_ok=True)
        extra.to_parquet(path)
    return pd.concat([features.reset_index(drop=True), extra], axis=1)


def training_copies(match_id: str, noise: str = "n2", extras: bool = False):
    """[(features, raw labels)] of one DFL match under every TRAINING_NOISE[noise] setting, cached.
    `extras` appends the columns of possession/extra_features.py to every copy."""
    if extras:
        clean = dfl.load(match_id)
        return [(with_extras(match_id, noise, name, features, lambda c=config, r=replica: degrade(
                     clean, clean["labels"], clean["half"], seed_for(match_id, r), c), clean["half"]), labels)
                for (name, config, replica), (features, labels) in zip(TRAINING_NOISE[noise],
                                                                       training_copies(match_id, noise))]
    out = []
    for name, config, replica in TRAINING_NOISE[noise]:
        if name == "default":
            _, features, labels = prepare(match_id, replica, noise=noise)
            out.append((features, labels))
            continue
        cache = os.path.join(dfl.CACHE_DIR, "{}_{}_{}_r{}_features.parquet".format(match_id, noise, name, replica))
        clean = dfl.load(match_id)
        if os.path.isfile(cache):
            features = pd.read_parquet(cache)
        else:
            noisy = degrade(clean, clean["labels"], clean["half"], seed_for(match_id, replica), config)
            features = build_from_tracking(noisy, clean["half"], dfl.half_starts(clean["half"]))
            os.makedirs(dfl.CACHE_DIR, exist_ok=True)
            features.to_parquet(cache)
        out.append((features, clean["labels"]))
    return out


# Symmetry copies for the temporal networks (possession/augment.py): each DFL match once more with
# the teams swapped, once mirrored and once both, each under one of the TRAINING_NOISE settings and
# a noise seed of its own. (name, swap, mirror, TRAINING_NOISE setting, seed replica)
SYMMETRY_COPIES = (
    ("swap", True, False, "default", 30),
    ("mirror", False, True, "rand_a", 40),
    ("swapmirror", True, True, "rand_b", 50),
)


def symmetry_copies(match_id: str, noise: str = "n3", extras: bool = False):
    """[(features, raw labels, swapped)] of one DFL match under every SYMMETRY_COPIES entry, cached.

    The labels are the match's own raw labels; where `swapped` is True the features describe the
    match with the teams exchanged, so the caller swaps H and A in whatever labels it derives.
    `extras` appends the columns of possession/extra_features.py to every copy.
    """
    if extras:
        clean = dfl.load(match_id)
        configs = {name: config for name, config, _ in TRAINING_NOISE[noise]}

        def noisy(swap, mirror, setting, replica):
            copy = augment.transform(clean, swap=swap, mirror=mirror)
            return degrade(copy, copy["labels"], copy["half"], seed_for(match_id, replica), configs[setting])

        return [(with_extras(match_id, noise, "sym_" + name, features,
                             lambda s=swap, m=mirror, t=setting, r=replica: noisy(s, m, t, r), clean["half"]),
                 labels, swapped)
                for (name, swap, mirror, setting, replica), (features, labels, swapped)
                in zip(SYMMETRY_COPIES, symmetry_copies(match_id, noise))]
    configs = {name: config for name, config, _ in TRAINING_NOISE[noise]}
    out = []
    clean = None
    for name, swap, mirror, setting, replica in SYMMETRY_COPIES:
        cache = os.path.join(dfl.CACHE_DIR, "{}_{}_sym_{}_r{}_features.parquet".format(match_id, noise, name, replica))
        clean = dfl.load(match_id) if clean is None else clean
        if os.path.isfile(cache):
            features = pd.read_parquet(cache)
        else:
            copy = augment.transform(clean, swap=swap, mirror=mirror)
            noisy = degrade(copy, copy["labels"], copy["half"], seed_for(match_id, replica), configs[setting])
            features = build_from_tracking(noisy, copy["half"], dfl.half_starts(copy["half"]))
            os.makedirs(dfl.CACHE_DIR, exist_ok=True)
            features.to_parquet(cache)
        out.append((features, clean["labels"], swap))
    return out


def swapped_test_features(match_id: str, noise: str = "n3"):
    """Features of the held-out view of a match (prepare(match_id, noise=noise)) with the teams
    swapped after the noise: the same degraded tracking, read from the other side. Cached."""
    cache = os.path.join(dfl.CACHE_DIR, "{}_{}_r0_swapped_features.parquet".format(match_id, noise))
    if os.path.isfile(cache):
        return pd.read_parquet(cache)
    clean = dfl.load(match_id)
    noisy = degrade(clean, clean["labels"], clean["half"], seed_for(match_id, 0), NOISE_PRESETS[noise])
    features = build_from_tracking(augment.transform(noisy, swap=True), clean["half"], dfl.half_starts(clean["half"]))
    features.to_parquet(cache)
    return features


def pooled_score(pairs) -> dict:
    """Official metrics over [(gt_labels, pred_labels), ...], matches separated by X frames."""
    gap = np.array(["X"])
    gt = np.concatenate([np.concatenate((g, gap)) for g, _ in pairs])
    pred = np.concatenate([np.concatenate((p, gap)) for _, p in pairs])
    return evaluate(label_codes(gt), label_codes(pred))


def summary(result: dict, pairs) -> dict:
    """The few numbers that matter, plus the possession rate per minute of graded play."""
    graded_frames = sum(int(np.sum(g != "X")) for g, _ in pairs)
    minutes = graded_frames / 25.0 / 60.0
    pred = np.concatenate([p[g != "X"] for g, p in pairs])
    return {
        "spellF1": result["spell_f1_mean"],
        "macroF1": result["macro_f1"],
        "n_gt": result["n_gt_spells"],
        "n_pred": result["n_pred_spells"],
        "gt/min": result["n_gt_spells"] / minutes,
        "pred/min": result["n_pred_spells"] / minutes,
        "D%": float(np.mean(pred == "D")),
        "alive/dead F1": result["alive_dead_f1"],
        "team acc": result["team_acc_both_alive"],
    }
