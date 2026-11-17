"""Align the broadcast recording to the tactical one, frame by frame.

The broadcast camera follows the ball up close, so its tracking sees the ball where the wide tactical
camera loses it -- in crowded boxes, tackles and aerial duels. But it is on its own timeline, with
cuts, replays and zooms, and no mapping is provided.

Both trackings are in the same pitch coordinates, so a broadcast frame can be located in the
tactical recording by its players: the tactical frame whose players sit where the broadcast's
visible players sit, team by team, is the same moment. The cost of a candidate is a one-sided
chamfer distance -- for every visible broadcast player, the distance to the nearest tactical player
of the same team -- so a zoomed broadcast frame showing eight players still matches the tactical
frame showing all twenty-two.

The live broadcast turns out to be continuous within each half: one offset per half, drifting by a
few frames. So the alignment finds those live offsets once, then checks every broadcast frame
against them. Replays and slow motion sit on other offsets and are simply left unmapped.

Measured: where both cameras see the ball, they agree to 0.75 m (median). But the broadcast tends
to lose the ball when the tactical camera does, so it fills only a small share of the gaps.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.tracking import AWAY, HOME, load as load_tracking  # noqa: E402

# Params
MIN_PLAYERS = 6          # broadcast frames with fewer visible players are not fingerprinted
CHUNK = 20000            # tactical candidates per vectorised block
COARSE_STEP = 5          # global search resolution
REFINE = 12              # +/- frames of drift searched around each live offset
ACCEPT_COST = 1.5        # m: a probe match this good is trusted
MIN_MARGIN = 1.0         # m: ...if the best match elsewhere in the match is this much worse
FRAME_COST = 2.0         # m: a frame keeps an offset only if its own players agree
BASE_PROBE_STEP = 250    # broadcast frames between global probes (10 s)
MIN_OFFSET_SUPPORT = 5   # probes needed before an offset counts as live footage
FILL_FRAMES = 50         # close-ups up to 2 s long inherit the offset around them


def chamfer_costs(query, query_team, candidates, team):
    """Mean distance from each query player to the nearest same-team candidate player.

    query (k, 2), query_team (k,), candidates (M, slots, 2), team (slots,) -> (M,) costs in metres.
    """
    total = np.zeros(len(candidates))
    for side in (HOME, AWAY):
        points = query[query_team == side]
        if len(points) == 0:
            continue
        pool = candidates[:, team == side]                                   # (M, s, 2)
        distance = np.linalg.norm(pool[:, None, :, :] - points[None, :, None, :], axis=3)
        nearest = np.nanmin(np.where(np.isnan(distance), np.inf, distance), axis=2)
        total += nearest.sum(axis=1)
    return total / len(query)


def locate(query, query_team, tactical_pos, team, frames):
    """(best_frame, best_cost, second_best_cost_far_away) over the candidate `frames`.

    The runner-up is taken at least 5 s from the best, so a sharp, unique minimum shows as a large
    gap between the two.
    """
    costs = np.empty(len(frames))
    for start in range(0, len(frames), CHUNK):
        block = frames[start:start + CHUNK]
        costs[start:start + CHUNK] = chamfer_costs(query, query_team, tactical_pos[block], team)
    best = int(np.argmin(costs))
    far = np.abs(frames - frames[best]) > 125
    runner_up = costs[far].min() if far.any() else np.inf
    return int(frames[best]), float(costs[best]), float(runner_up)


def fingerprint(broadcast, j):
    """Visible players of broadcast frame j: (positions (k, 2), teams (k,))."""
    seen = ~np.isnan(broadcast["pos"][j, :, 0])
    return broadcast["pos"][j, seen].astype(np.float64), broadcast["team"][seen]


def base_offsets(broadcast, tac_pos, team):
    """The live offsets of the recording: one dominant value per stretch of live footage.

    The live broadcast turns out to be continuous within each half (it runs at the tactical rate,
    drifting by a few frames), so a sparse global search finds a handful of offsets that cover it.
    Offsets supported by fewer than MIN_OFFSET_SUPPORT probes are replays or false matches.
    """
    counts = (~np.isnan(broadcast["pos"][..., 0])).sum(axis=1)
    coarse = np.arange(0, len(tac_pos), COARSE_STEP)
    found = []
    for j in np.arange(0, len(counts), BASE_PROBE_STEP):
        if counts[j] < 10:
            continue
        query, query_team = fingerprint(broadcast, j)
        frame, cost, runner_up = locate(query, query_team, tac_pos, team, coarse)
        if cost < ACCEPT_COST and runner_up > cost + MIN_MARGIN:
            found.append(frame - j)
    found = np.round(np.array(found) / 10.0) * 10        # pool probes that agree within ~5 frames
    values, support = np.unique(found, return_counts=True)
    return [int(v) for v in values[support >= MIN_OFFSET_SUPPORT]]


def align(use_cache=True):
    """(tactical_of_broadcast, cost): the tactical frame each broadcast frame shows (-1: none).

    Every broadcast frame tries each live offset, corrected by up to +/-REFINE frames for drift,
    and keeps the best if its own players agree (cost below FRAME_COST). Close-ups with too few
    players to judge inherit the offset of their neighbours when the frames either side agree --
    exactly the zoomed shots of a duel where the broadcast sees the ball and the tactical does not.
    """
    from possession.tracking import CACHE_DIR
    cache = os.path.join(CACHE_DIR, "BAR-ATH_broadcast_alignment.npz")
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as data:
            return data["tactical_of_broadcast"], data["cost"]

    tactical, broadcast = load_tracking("tactical"), load_tracking("broadcast")
    tac_pos = tactical["pos"].astype(np.float64)
    team = tactical["team"]
    n_tac, n_bc = len(tac_pos), len(broadcast["ball"])
    offsets = base_offsets(broadcast, tac_pos, team)
    print("live offsets:", offsets)

    mapping = np.full(n_bc, -1, dtype=np.int64)
    cost = np.full(n_bc, np.inf)
    shifts = np.arange(-REFINE, REFINE + 1)
    for j in range(n_bc):
        query, query_team = fingerprint(broadcast, j)
        if len(query) < MIN_PLAYERS:
            continue
        candidates = np.unique(np.concatenate([j + o + shifts for o in offsets]))
        candidates = candidates[(candidates >= 0) & (candidates < n_tac)]
        if len(candidates) == 0:
            continue
        costs = chamfer_costs(query, query_team, tac_pos[candidates], team)
        k = int(np.argmin(costs))
        if costs[k] < FRAME_COST:
            mapping[j], cost[j] = candidates[k], costs[k]

    # Continuity: an unmapped frame between two mapped frames with the same offset, close by.
    index = np.arange(n_bc)
    mapped = mapping >= 0
    prev = np.maximum.accumulate(np.where(mapped, index, -1))
    nxt = np.minimum.accumulate(np.where(mapped, index, n_bc)[::-1])[::-1]
    fill = ~mapped & (prev >= 0) & (nxt < n_bc) & (nxt - prev <= FILL_FRAMES)
    same = np.zeros(n_bc, dtype=bool)
    same[fill] = (mapping[prev[fill]] - prev[fill]) == (mapping[nxt[fill]] - nxt[fill])
    fill &= same
    mapping[fill] = index[fill] + (mapping[prev[fill]] - prev[fill])
    cost[fill] = FRAME_COST                                  # trusted less than a direct match

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, tactical_of_broadcast=mapping, cost=cost)
    return mapping, cost


def broadcast_ball_on_tactical(n_tactical, use_cache=True):
    """(n_tactical, 2) ball position seen by the broadcast at each tactical frame, NaN elsewhere.

    Where several broadcast frames land on one tactical frame (a replay of a live moment), the
    best-matched one wins.
    """
    mapping, cost = align(use_cache)
    ball = load_tracking("broadcast")["ball"].astype(np.float64)
    out = np.full((n_tactical, 2), np.nan)
    best = np.full(n_tactical, np.inf)
    has = (mapping >= 0) & ~np.isnan(ball[:, 0])
    for j in np.where(has)[0][np.argsort(-cost[has])]:   # worst first, best overwrites
        out[mapping[j]] = ball[j]
        best[mapping[j]] = cost[j]
    return out


if __name__ == "__main__":
    import time
    start = time.time()
    mapping, cost = align(use_cache=False)
    mapped = mapping >= 0
    print("aligned {:.1%} of broadcast frames in {:.0f}s; median frame cost {:.2f} m".format(
        mapped.mean(), time.time() - start, np.median(cost[mapped])))
    offsets = mapping[mapped] - np.where(mapped)[0]
    values, counts = np.unique(offsets, return_counts=True)
    top = np.argsort(-counts)[:8]
    print("most common offsets (frames: broadcast frames):",
          ", ".join("{:+d}: {}".format(values[k], counts[k]) for k in top))

    tactical = load_tracking("tactical")
    tac_ball = tactical["ball"].astype(np.float64)
    from_broadcast = broadcast_ball_on_tactical(len(tac_ball), use_cache=True)
    from games import half_video_blocks
    graded = np.zeros(len(tac_ball), dtype=bool)
    for _, first, last in half_video_blocks("BAR-ATH"):
        graded[first:last + 1] = True
    tac_seen = ~np.isnan(tac_ball[:, 0])
    bc_seen = ~np.isnan(from_broadcast[:, 0])
    both = graded & tac_seen & bc_seen
    gap = np.linalg.norm(tac_ball[both] - from_broadcast[both], axis=1)
    print("where both cameras see the ball: {} frames, distance median {:.2f} m, p90 {:.2f} m".format(
        both.sum(), np.median(gap), np.percentile(gap, 90)))
    print("graded frames without a tactical ball: {}; recovered from broadcast: {} ({:.1%})".format(
        (graded & ~tac_seen).sum(), (graded & ~tac_seen & bc_seen).sum(),
        (graded & ~tac_seen & bc_seen).sum() / (graded & ~tac_seen).sum()))
