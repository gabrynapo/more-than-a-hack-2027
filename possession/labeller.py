"""Rule-based pseudo-labels: H / A / D where the tracking makes the answer obvious, -1 elsewhere.

These are the training targets for the frame classifier. They cover the whole match, so the model
learns from ~150k frames instead of the 7,500 public ones -- which stay out of training entirely and
remain an honest check.

The rules aim for precision, not coverage. Whatever is ambiguous (passes in flight, contested balls,
long ball gaps) is left unlabelled and is the classifier's job.

    control   ball seen, one side's nearest player within CONTROL_DIST and clearly closer than the
              other side's, for at least MIN_CONTROL_FRAMES in a row -> that side
    carry     between two control runs of the same side, less than MAX_CARRY_S apart, with no
              dead-ball evidence and no ball gap over 1 s in between -> that side (possession
              persists through loose phases)
    dead      ball clearly outside the lines; or out and not yet kicked back into play; or a gap that
              opens with the ball over a line; or no kick for 3 s either side (ball observed) with
              everyone walking; or the whole pitch braking hard and then walking (a whistle); or
              the whole pitch walking for a sustained stretch with no control anywhere near; or
              graded frames before a half's kickoff
"""

import numpy as np

from possession.features import rolling_nanmean

# Params
FPS = 25
CONTROL_DIST = 1.5          # m: within playing distance of the ball
CONTROL_MARGIN = 1.5        # m: the other side's nearest must be at least this much further away
MIN_CONTROL_FRAMES = 8      # 0.32 s
MAX_CARRY_S = 8.0
MAX_CARRY_GAP_FRAMES = 25   # a carry never bridges more than 1 s without the ball: a stoppage
                            # can start and end unseen inside a longer gap
OUT_DIST = 1.0              # m beyond the lines
SLOW_FRAC = 0.85            # share of players walking
SLOW_MIN_FRAMES = 75        # 3 s
NO_CONTROL_PAD = 50         # frames around a control run in which "everyone walking" is not trusted
KICKOFF_FRAMES = {2: 113742}  # published in FAQ.md: graded from 110,808, kickoff at 113,742
MAX_UNRESOLVED_S = 60.0     # an exit with no kick after it stays a stoppage at most this long
QUIET_S = 3                 # seconds without a kick, either side
QUIET_SPEED = 1.5           # m/s, mean player speed over +/-1.5 s
MIN_OBSERVED = 0.6          # share of the +/-3 s window with the ball seen
GAP_EXIT_M = 0.5            # ball last seen this far over a line before a gap
WHISTLE_DROP = 1.5          # m/s: mean player speed falls this much within 2 s
WHISTLE_WALK_FRAMES = 50    # ...and the pitch then walks for at least 2 s
WHISTLE_LEAD_FRAMES = 75    # the braking starts at most 3 s before the walking

UNLABELLED, H, A, D = -1, 0, 1, 2


def _runs(mask):
    """[(start, end)] of True runs, end exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def _run_lengths(mask):
    """Per frame: length of the True run it belongs to, 0 outside runs."""
    out = np.zeros(len(mask), dtype=np.int64)
    for start, end in _runs(mask):
        out[start:end] = end - start
    return out


def _dilate(mask, pad):
    out = mask.copy()
    for start, end in _runs(mask):
        out[max(0, start - pad):end + pad] = True
    return out


def control_runs(features):
    """[(start, end, side)] of sustained control, in frame order."""
    d_h = features["bp_d_h_min"].values
    d_a = features["bp_d_a_min"].values
    seen = features["ball_seen"].values > 0
    with np.errstate(invalid="ignore"):
        home = seen & (d_h < CONTROL_DIST) & (d_a > d_h + CONTROL_MARGIN)
        away = seen & (d_a < CONTROL_DIST) & (d_h > d_a + CONTROL_MARGIN)
    runs = [(s, e, H) for s, e in _runs(home) if e - s >= MIN_CONTROL_FRAMES]
    runs += [(s, e, A) for s, e in _runs(away) if e - s >= MIN_CONTROL_FRAMES]
    return sorted(runs)


def dead_evidence(features, controls):
    n = len(features)
    seen = features["ball_seen"].values > 0
    with np.errstate(invalid="ignore"):
        out = features["ball_out_dist"].values > OUT_DIST
        out &= seen
        slow = features["tm_frac_slow_all_r3"].values > SLOW_FRAC

        # The ball went out and nobody has kicked it since: it is being fetched, carried back and
        # placed, whoever is standing over it.
        unresolved = ((features["ev_out_unresolved"].values > 0)
                      & (features["ev_since_out_s"].values < MAX_UNRESOLVED_S))

        # Nobody kicks the ball for a while either side and everyone walks. Absence of kicks only
        # counts where the ball could have been seen kicking, or every long gap would qualify.
        observed = rolling_nanmean(seen.astype(float), QUIET_S * FPS) >= MIN_OBSERVED
        quiet = ((features["ev_since_kick_s"].values >= QUIET_S)
                 & (features["ev_until_kick_s"].values >= QUIET_S)
                 & (features["tm_speed_all_r3"].values < QUIET_SPEED)
                 & observed)

        # A gap that opens with the ball over a line is a stoppage for as long as it lasts.
        gap_after_exit = ~seen & (features["ball_prev_out"].values > GAP_EXIT_M)
    slow_runs = np.zeros(n, dtype=bool)
    for start, end in _runs(slow):
        if end - start >= SLOW_MIN_FRAMES:
            slow_runs[start:end] = True
    controlled = np.zeros(n, dtype=bool)
    for start, end, _ in controls:
        controlled[start:end] = True
    slow_runs &= ~_dilate(controlled, NO_CONTROL_PAD)
    return out | slow_runs | unresolved | quiet | gap_after_exit | whistle_stops(features)


def whistle_stops(features):
    """The whole pitch brakes hard, then walks: a whistle, typically a foul, often unseen.

    Dead from the onset of the braking until the next kick (the free kick), and never beyond the
    walking stretch. This is the stoppage the other rules miss when the ball is lost in the
    challenge that caused it.
    """
    n = len(features)
    with np.errstate(invalid="ignore"):
        walking = features["tm_speed_all_r3"].values < QUIET_SPEED
        braking = features["ev_speed_drop"].values > WHISTLE_DROP
    until_kick = features["ev_until_kick_s"].values
    out = np.zeros(n, dtype=bool)
    for start, end in _runs(walking):
        if end - start < WHISTLE_WALK_FRAMES:
            continue
        lead = max(0, start - WHISTLE_LEAD_FRAMES)
        hits = np.where(braking[lead:start + 1])[0]
        if len(hits) == 0:
            continue
        onset = lead + hits[0]
        stop = min(end, onset + int(until_kick[onset] * FPS))
        out[onset:stop] = True
    return out


def pseudo_labels(features, kickoff_frames=KICKOFF_FRAMES):
    """int8 array: H / A / D / UNLABELLED per frame. Only graded frames are ever labelled.

    `kickoff_frames` {half: frame} marks graded frames before a kickoff as dead; the default is the
    competition match's. Pass {} for a match whose halves start at the kickoff.
    """
    n = len(features)
    labels = np.full(n, UNLABELLED, dtype=np.int8)
    controls = control_runs(features)
    dead = dead_evidence(features, controls)

    unseen_run = _run_lengths(features["ball_seen"].values == 0)
    for (s0, e0, side0), (s1, e1, side1) in zip(controls, controls[1:]):
        if (side0 == side1 and (s1 - e0) <= MAX_CARRY_S * FPS and not dead[e0:s1].any()
                and unseen_run[e0:s1].max(initial=0) <= MAX_CARRY_GAP_FRAMES):
            labels[e0:s1] = side0
    for start, end, side in controls:
        labels[start:end] = side
    labels[dead] = D

    half = features["half"].values
    for number, kickoff in kickoff_frames.items():
        before = (half == number) & (np.arange(n) < kickoff)
        labels[before] = D
    labels[half == 0] = UNLABELLED
    return labels
