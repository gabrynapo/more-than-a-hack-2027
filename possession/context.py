"""Directional temporal context: what happened just before a frame vs just after it.

The frame classifiers see each frame through centred rolling means (features.py, +/-0.5 s and
+/-1.5 s). A centred window is blind to *where* inside it a change happens, and the proxy error
analysis shows exactly that cost: possessions under 5 s are 44% of all and mostly missed, turnovers
are found far less often than restarts, and matched boundaries are off by more than 2.8 s in 10% of
cases.

Change-point detection compares a past window with a future window: their difference peaks where
the regime changes (e.g. arXiv 2009.11829). This module adds, for a few base signals x and windows
w, the mean of x over the past w frames, over the next w frames, and their difference -- so the
classifier can tell "home before, away after" from "home throughout".

The same tools build the second-stage inputs (`stage2_inputs`): the first-stage probabilities read
at several past and future scales -- multi-stage refinement in the spirit of MS-TCN / ASRF, with
gradient boosting instead of convolutions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTEXT_BASE = ("bp_d_diff", "bp_d_h_min", "bp_d_a_min", "bp_nearest_home", "bp_n_h_5", "bp_n_a_5",
                "ball_speed", "ball_out_dist", "tm_speed_all", "tm_frac_slow_all", "tm_h_width",
                "tm_a_width", "tm_vx_att_diff", "tm_cx_diff", "ev_kick")
CONTEXT_WINDOWS = (25, 75)                       # 1 s and 3 s
STAGE2_WINDOWS = (12, 25, 50, 100, 200)          # 0.5 s ... 8 s


def one_sided_means(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """(past, future): NaN-ignoring means over [t-window, t) and (t, t+window]. NaN if empty."""
    valid = ~np.isnan(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(x))

    def mean(low, high):
        low, high = np.clip(low, 0, len(x)), np.clip(high, 0, len(x))
        n = count[high] - count[low]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)

    return mean(index - window, index), mean(index + 1, index + window + 1)


def add_context(features: pd.DataFrame, base=CONTEXT_BASE, windows=CONTEXT_WINDOWS) -> pd.DataFrame:
    """A copy of `features` with cx_<name>_p<w> / _f<w> / _d<w> columns (past, future, future-past).

    Halves are not separated here: the ungraded frames between them are NaN-free tracking, and a
    window reaching across halftime only touches frames that are never scored.
    """
    columns = {}
    for name in base:
        if name not in features.columns:
            continue
        x = features[name].values.astype(np.float64)
        for w in windows:
            past, future = one_sided_means(x, w)
            columns["cx_{}_p{}".format(name, w)] = past
            columns["cx_{}_f{}".format(name, w)] = future
            columns["cx_{}_d{}".format(name, w)] = future - past
    extra = pd.DataFrame({k: v.astype(np.float32) for k, v in columns.items()}, index=features.index)
    return pd.concat([features, extra], axis=1)


def stage2_inputs(proba: np.ndarray, features: pd.DataFrame, windows=STAGE2_WINDOWS) -> pd.DataFrame:
    """Second-stage table: first-stage P(H), P(A), P(D) and their past / future means at each scale,
    plus the visibility flag (the stage-1 posterior means something different without a ball)."""
    columns = {"s1_h": proba[:, 0], "s1_a": proba[:, 1], "s1_d": proba[:, 2],
               "ball_known": ((features["ball_seen"].values > 0)
                              | (features["ball_interp"].values > 0)).astype(np.float64)}
    for k, name in enumerate(("h", "a", "d")):
        for w in windows:
            past, future = one_sided_means(proba[:, k].astype(np.float64), w)
            columns["s1_{}_p{}".format(name, w)] = past
            columns["s1_{}_f{}".format(name, w)] = future
            columns["s1_{}_d{}".format(name, w)] = future - past
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in columns.items()},
                        index=features.index)
