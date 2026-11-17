"""Build the per-frame feature table and sanity-check it against the public 5 minutes.

    python scripts/build_features.py            # build, cache, report
    python scripts/build_features.py --top 40   # longer report

The table is written to data/cache/ (gitignored -- it is derived tracking and must not be
published). The report scores each feature on its own against the public ground truth:

    AUC H/A   how well the feature separates home from away possession (alive frames only)
    AUC D     how well it separates dead ball from live play
    cover     share of graded public frames where the feature is not NaN

AUC is folded to >= 0.5, so it says how informative a feature is, not in which direction. It is a
check that the features mean what they should -- 17 possessions are far too few to select on.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.features import build, cache_path  # noqa: E402

GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def auc(values, positive):
    """Mann-Whitney AUC of `values` for the boolean `positive`, NaN rows dropped, folded >= 0.5."""
    keep = ~np.isnan(values)
    values, positive = values[keep], positive[keep]
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0 or np.ptp(values) == 0:
        return np.nan
    ranks = rankdata(values)
    score = (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return max(score, 1 - score)


def report(features, top):
    labels = pd.read_csv(GT_PUBLIC)["label"].values
    graded = labels != "X"
    alive = np.isin(labels, ["H", "A"])
    rows = []
    for name in features.columns:
        values = features[name].values.astype(np.float64)
        rows.append({
            "feature": name,
            "AUC H/A": auc(values[alive], labels[alive] == "H"),
            "AUC D": auc(values[graded], labels[graded] == "D"),
            "cover": np.mean(~np.isnan(values[graded])),
        })
    table = pd.DataFrame(rows).set_index("feature")
    pd.set_option("display.width", 120)
    for key in ("AUC H/A", "AUC D"):
        print("\ntop {} features by {} (public 5 min, {} frames)".format(top, key, graded.sum()))
        print(table.sort_values(key, ascending=False).head(top).round(3).to_string())

    # Does the Smart Tagging team tag agree with the ground truth? Attacking phases should name
    # the team in possession, defending phases the other one.
    print("\nSmart Tagging vs public ground truth, alive frames where the phase is active")
    for name in [c for c in features.columns if c.startswith("st_") and c + "_on" in features.columns]:
        tag = features[name].values
        mask = alive & (tag != 0)
        if mask.sum() == 0:
            continue
        agree = np.mean((tag[mask] > 0) == (labels[mask] == "H"))
        print("  {:<38} frames {:>5}   tag team == possessor {:.0%}".format(name, mask.sum(), agree))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    start = time.time()
    features = build()
    path = cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    features.to_parquet(path)
    print("built {} frames x {} features in {:.0f}s -> {}".format(
        len(features), features.shape[1], time.time() - start, os.path.relpath(path, ROOT)))
    print("home attacks +x in half: {}".format(
        {h: "yes" if s > 0 else "no (rotated)" for h, s in features.attrs["home_attack_signs"].items()}))
    report(features, args.top)
