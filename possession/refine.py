"""Snap decoded boundaries onto the touch that caused them.

The HMM finds the right possessions but places their edges loosely: its evidence is deliberately
damped, so a switch lands wherever enough frames have accumulated, up to a few seconds off. The
ground truth is sharper than that. On the public window nearly every boundary sits within a few
frames of a touch -- a kick leaving a foot, or a player of the new side reaching the ball -- for
changes of side, for the whistle and for the restart alike.

So each boundary moves to a touch near it:

    H <-> A       a touch by the new side: its kick, or its player reaching the ball
    D -> H / A    a kick: the restart (by the new side, or by nobody the tracking can place)
    H / A -> D    a kick: the ball played out, or the last action before the whistle

Arrivals at the ball do not count around a stoppage -- during one, players fetch and place the
ball, and that is not play.

Among the candidates in the window, the one that best splits the classifier's own frame
probabilities wins -- a change-point restricted to the frames where a change can physically happen.
With no candidate in the window (typically a ball gap) the boundary stays where the HMM put it.
"""

import numpy as np

# Params
WINDOW = 75             # frames either side of the decoded boundary (3 s)
TOUCH_DIST = 2.0        # m: a player this close to the ball can have touched it
MIN_SEGMENT_FRAMES = 12  # 0.5 s
FOUL_WINDOW_FRAMES = 100 # 4 s: tackle, whistle and the pitch braking all happen within this
FPS = 25
EPS = 1e-6
STATE_INDEX = {"H": 0, "A": 1, "D": 2}


def _rising(mask):
    return mask & ~np.concatenate(([False], mask[:-1]))


def touch_events(features):
    """(frames, sides, is_kick): the start of every kick, and every arrival of a player at the ball.

    The side is that of the nearest player within TOUCH_DIST, one frame before the event (the ball
    has already left the foot when a kick is detected); "" when nobody is close enough.
    """
    d_h = features["bp_d_h_min"].values
    d_a = features["bp_d_a_min"].values
    seen = features["ball_seen"].values > 0
    with np.errstate(invalid="ignore"):
        home_near = seen & (d_h < TOUCH_DIST) & (d_h <= d_a)
        away_near = seen & (d_a < TOUCH_DIST) & (d_a < d_h)
    kick_start = _rising(features["ev_kick"].values > 0)
    events = kick_start | _rising(home_near) | _rising(away_near)

    frames = np.where(events)[0]
    before = np.maximum(frames - 1, 0)
    sides = np.where(home_near[before] | home_near[frames], "H",
                     np.where(away_near[before] | away_near[frames], "A", ""))
    return frames, sides, kick_start[frames]


def refine(labels, proba, half, events):
    """Labels with each boundary moved onto its best supporting touch. X frames stay X."""
    frames, sides, is_kick = events
    out = labels.copy()
    log_proba = np.log(proba + EPS)
    for number in (1, 2):
        index = np.where(half == number)[0]
        if len(index) == 0:
            continue
        first, last = index[0], index[-1] + 1
        segment = out[first:last]
        cuts = list(np.where(segment[1:] != segment[:-1])[0] + 1 + first)
        for k, cut in enumerate(cuts):
            before, after = out[cut - 1], out[cut]
            low = max(cuts[k - 1] + 1 if k > 0 else first + 1, cut - WINDOW)
            high = min(cuts[k + 1] - 1 if k + 1 < len(cuts) else last - 1, cut + WINDOW)
            wanted = (frames >= low) & (frames <= high)
            if "D" in (before, after):
                # Play stops and restarts with a kick; arrivals during a stoppage are players
                # fetching and placing the ball.
                wanted &= is_kick
            if after != "D":
                wanted &= (sides == after) | (is_kick & (sides == ""))
            candidates = frames[wanted]
            if len(candidates) == 0:
                continue
            # Score of a cut at c: evidence for `before` on [low, c) plus for `after` on [c, high].
            ratio = log_proba[low:high + 1, STATE_INDEX[before]] - log_proba[low:high + 1, STATE_INDEX[after]]
            gain = np.concatenate(([0.0], np.cumsum(ratio)))
            best = candidates[np.argmax(gain[candidates - low])]
            if best < cut:
                out[best:cut] = after
            elif best > cut:
                out[cut:best] = before
            cuts[k] = best
    return out


def _segments(labels, first, last):
    """[(state, start, end)] of the runs in labels[first:last], end exclusive."""
    segment = labels[first:last]
    starts = np.concatenate(([0], np.where(segment[1:] != segment[:-1])[0] + 1)) + first
    ends = np.concatenate((starts[1:], [last]))
    return [(labels[s], s, e) for s, e in zip(starts, ends)]


def undo_foul_regains(labels, whistle, half, window=FOUL_WINDOW_FRAMES):
    """A foul does not change possession: give the tackle before a whistle back to the fouled side.

    Before a whistle the tracking typically shows the tackler reaching the ball, so the decoder
    reports a brief "regain" -- often a sliver of each side -- right before the stoppage. Under the
    rules of the game the fouled side keeps the ball and restarts with a free kick.

    Only stoppages that open with a whistle (the pitch braking, `whistle`) qualify. Before an exit
    the last side on the ball genuinely had it, short or not.
    """
    out = labels.copy()
    for number in (1, 2):
        index = np.where(half == number)[0]
        if len(index) == 0:
            continue
        runs = _segments(out, index[0], index[-1] + 1)
        for k, (state, start, _) in enumerate(runs):
            if state != "D" or not whistle[max(0, start - FPS):start + FPS].any():
                continue
            j = k - 1
            while j >= 0 and runs[j][0] != "D" and start - runs[j][1] <= window:
                j -= 1
            # runs[j + 1:k] all start within `window` of the stoppage; runs[j] is what came before.
            if j < 0 or runs[j][0] == "D" or j + 1 >= k:
                continue
            fouled = runs[j][0]
            chain = runs[j + 1:k]
            if any(s != fouled for s, _, _ in chain):
                out[chain[0][1]:start] = fouled
    return out


def drop_fragments(labels, proba, half, min_frames=MIN_SEGMENT_FRAMES):
    """Merge segments shorter than `min_frames` into a neighbour, shortest first.

    Snapping two nearby boundaries onto touches can squeeze the segment between them to a few
    frames, and each such sliver is a false possession. The shortest real one on the public
    window lasts 0.8 s. A sliver joins whichever neighbour its frames' probabilities favour.
    """
    out = labels.copy()
    for number in (1, 2):
        index = np.where(half == number)[0]
        if len(index) == 0:
            continue
        first, last = index[0], index[-1] + 1
        while True:
            segment = out[first:last]
            starts = np.concatenate(([0], np.where(segment[1:] != segment[:-1])[0] + 1)) + first
            ends = np.concatenate((starts[1:], [last]))
            lengths = ends - starts
            short = np.where(lengths < min_frames)[0]
            if len(short) == 0 or len(starts) == 1:
                break
            k = short[np.argmin(lengths[short])]
            s, e = starts[k], ends[k]
            neighbours = [out[s - 1]] if k > 0 else []
            neighbours += [out[e]] if k + 1 < len(starts) else []
            support = {state: proba[s:e, STATE_INDEX[state]].mean() for state in neighbours}
            out[s:e] = max(support, key=support.get)
    return out
