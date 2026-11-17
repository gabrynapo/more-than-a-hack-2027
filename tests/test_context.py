"""Synthetic tests for directional temporal context."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession.context import add_context, one_sided_means, stage2_inputs  # noqa: E402


def test_one_sided_means_exclude_the_frame_itself():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    past, future = one_sided_means(x, 2)
    assert np.isnan(past[0]) and past[1] == 1.0 and past[2] == 1.5 and past[4] == 3.5
    assert future[0] == 2.5 and future[3] == 5.0 and np.isnan(future[4])


def test_one_sided_means_ignore_nan():
    x = np.array([1.0, np.nan, 3.0, np.nan])
    past, future = one_sided_means(x, 3)
    assert past[3] == 2.0 and future[0] == 3.0


def test_difference_peaks_at_a_step():
    x = np.concatenate((np.zeros(50), np.ones(50)))
    frame = pd.DataFrame({"bp_d_diff": x})
    out = add_context(frame, base=("bp_d_diff",), windows=(10,))
    d = out["cx_bp_d_diff_d10"].values
    assert np.nanargmax(d) in (49, 50) and d[20] == 0.0
    assert list(frame.columns) == ["bp_d_diff"]          # input untouched


def test_stage2_inputs_shape():
    proba = np.full((30, 3), 1 / 3)
    frame = pd.DataFrame({"ball_seen": np.ones(30), "ball_interp": np.zeros(30)})
    table = stage2_inputs(proba, frame, windows=(5,))
    assert len(table) == 30 and "s1_h_d5" in table and table["ball_known"].eq(1).all()
