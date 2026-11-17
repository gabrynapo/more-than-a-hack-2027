"""Frame classifiers trained on the rule-based pseudo-labels: P(H), P(A), P(D) for every frame.

The rules only fire where the answer is obvious. The classifiers see the same frames through the
full feature table and so carry what they learned into the frames the rules leave open.

Two models, because of where the labels come from. The rules that name a team need the ball, so
nearly every pseudo-label without a ball is a D. A single model learns the shortcut "no ball ->
dead" (it did: 99% of ball-less frames came out D, against 56% on the public window). So:

    seen    all features except ball visibility itself; trained and applied where the ball is seen
            or bridged by interpolation
    blind   only features that exist without a ball -- team shape, speeds, Smart Tagging -- trained
            on every labelled frame and applied in the gaps. It cannot tell whether the ball is
            visible, so it cannot learn the shortcut

The public 5-minute window (plus a margin, because neighbouring frames are near-duplicates) is never
trained on, so scoring against it stays an honest check.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from possession.labeller import UNLABELLED

# Params
EXCLUDED_FEATURES = ("half", "graded", "t_half",                         # bookkeeping
                     "ball_seen", "ball_interp", "ball_since_s",          # visibility shortcuts
                     "ball_until_s", "ball_gap_s", "ball_from_broadcast")
BLIND_PREFIXES = ("tm_", "st_", "ev_speed_", "geo_box_max")
PUBLIC_WINDOW = (11185, 18684)
PUBLIC_MARGIN = 250                                # 10 s either side
SUBSAMPLE = 5                                      # neighbouring frames are near-duplicates
SEED = 0


def ball_known(features):
    return (features["ball_seen"].values > 0) | (features["ball_interp"].values > 0)


def feature_columns(features, kind):
    columns = [name for name in features.columns if name not in EXCLUDED_FEATURES]
    if kind == "blind":
        columns = [name for name in columns if name.startswith(BLIND_PREFIXES)]
    return columns


def training_mask(features, labels, exclude_window=PUBLIC_WINDOW):
    """Labelled, subsampled frames outside `exclude_window` (the public window by default; None to
    exclude nothing, as on a match with no held-out window)."""
    index = np.arange(len(features))
    held_out = np.zeros(len(features), dtype=bool)
    if exclude_window is not None:
        held_out = ((index >= exclude_window[0] - PUBLIC_MARGIN)
                    & (index <= exclude_window[1] + PUBLIC_MARGIN))
    return (labels != UNLABELLED) & ~held_out & (index % SUBSAMPLE == 0)


def _fit(features, labels, mask, columns):
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, class_weight="balanced", early_stopping=False, random_state=SEED)
    with threadpool_limits(1):     # bit-for-bit reproducible, see supervised.THREADS
        model.fit(features.loc[mask, columns].values, labels[mask])
    model.feature_columns_ = columns
    return model


def train(features, labels, exclude_window=PUBLIC_WINDOW):
    """{"seen": model, "blind": model}."""
    mask = training_mask(features, labels, exclude_window)
    return {
        "seen": _fit(features, labels, mask & ball_known(features),
                     feature_columns(features, "seen")),
        "blind": _fit(features, labels, mask, feature_columns(features, "blind")),
    }


def _proba(model, features):
    with threadpool_limits(1):
        proba = model.predict_proba(features[model.feature_columns_].values)
    return proba[:, np.argsort(model.classes_)]


def predict_proba(models, features):
    """(N, 3) probabilities in H, A, D order: the seen model where the ball is known, else blind."""
    known = ball_known(features)
    proba = _proba(models["blind"], features)
    proba[known] = _proba(models["seen"], features.loc[known])
    return proba
