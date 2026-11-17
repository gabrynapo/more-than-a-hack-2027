"""Re-open short turnovers that the HMM merged away.

spellF1 punishes a merge far more than a spurious cut. If the decoder misses a one-second H between
two A possessions, the three true possessions become one long predicted A; it overlaps none of them
by IoU >= 0.5, so three matches are lost and one false possession appears. A wrongly inserted cut
usually leaves the larger piece of the long possession still matched.

The HMM pays the same price for every switch and so prefers the merge. This module decodes the
same probabilities a second time, at a higher evidence weight (finer: it keeps short runs), and
copies back the short opposite-team runs that fall strictly inside one coarse possession -- the A-H-A
pattern -- when the classifier really leans to the other team there.

    insert_turnovers(labels, proba, half, prior, fine_weight, threshold, min_frames)

All three parameters are chosen on the DFL proxy (scripts/proxy_short_spells.py), never on the
competition match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possession.decode import decode

TEAM_INDEX = {"H": 0, "A": 1}
OTHER = {"H": "A", "A": "H"}


@dataclass(frozen=True)
class SplitConfig:
    fine_weight: float = 1.0      # evidence weight of the fine decoding (the coarse one is 0.35)
    threshold: float = 0.6        # minimum mean P(team) / (P(H) + P(A)) over the inserted run
    min_frames: int = 12          # shorter fine runs are ignored (0.5 s at 25 fps)
    max_frames: int = 250         # longer ones are not "short turnovers" (10 s)


def runs(labels):
    """[(start, stop, label)] maximal constant runs."""
    if not len(labels):
        return []
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(labels)]))
    return [(int(a), int(b), labels[a]) for a, b in zip(bounds[:-1], bounds[1:])]


def insert_turnovers(labels, proba, half, prior, config: SplitConfig = SplitConfig()):
    """`labels` with the fine decoder's short opposite-team runs copied in where they split a possession."""
    fine = decode(proba, half, prior, evidence_weight=config.fine_weight)
    out = labels.copy()
    for start, stop, team in runs(fine):
        if team not in TEAM_INDEX or not config.min_frames <= stop - start <= config.max_frames:
            continue
        host = OTHER[team]
        # strictly inside one coarse possession of the other team, same half on both sides
        if start == 0 or stop >= len(labels) or half[start - 1] != half[stop]:
            continue
        if labels[start - 1] != host or labels[stop] != host or (labels[start:stop] != host).any():
            continue
        live = proba[start:stop, 0] + proba[start:stop, 1]
        share = proba[start:stop, TEAM_INDEX[team]] / np.maximum(live, 1e-12)
        if share.mean() >= config.threshold:
            out[start:stop] = team
    return out
