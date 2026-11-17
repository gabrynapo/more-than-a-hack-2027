"""Per-frame features from the tracking and the Smart Tagging. One row per video frame.

Everything is expressed in a "home attacks +x" frame: in each half the pitch is rotated so that the
home side attacks towards x = L. A feature then means the same thing in both halves, and an `_h` /
`_a` pair is directly comparable.

Feature groups, by prefix:

    (none)  context            half, graded, t_half
    ball_   ball state         seen, position, speed, out-of-pitch distance, gaps in the track
    bp_     ball-to-players    nearest home / away distances, crowding, who moves with the ball
    geo_    restart geometry   distance to centre spot, corners, goal areas; players in the boxes
    ev_     events             kicks and exits, time since / until them, collective slow-down
    tm_     team shape         centroid, spread, length/width, speeds, defensive lines, marking
    st_     Smart Tagging      active phases with their team, set-piece types, a possession vote
    *_r1 / *_r3                centred rolling means over +/-0.5 s and +/-1.5 s

The ball is missing on 41% of graded frames, mostly in long stretches (median ~10 s). Short gaps
(<= MAX_INTERP_FRAMES) are bridged by linear interpolation and flagged; across long gaps the ball
features are NaN, and the team-shape and tagging features are what is left to go on.

Where the tactical camera loses the ball, the broadcast camera often still has it: its ball is
moved onto the tactical timeline by possession/align.py and fills the gaps (`ball_from_broadcast`
flags those frames). Every ball feature below then treats it as seen.

The undocumented third ball field is deliberately unused: it is filled on ~10% of frames, mostly
when the ball itself is not seen, and does not correlate with ball speed.
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from possession.tracking import AWAY, HOME, load as load_tracking  # noqa: E402

# Params
FPS = 25
MAX_INTERP_FRAMES = 50          # bridge ball gaps up to 2 s
MAX_STEP_M = 1.0                # a slot moving >1 m in one frame (25 m/s) is an identity switch
SLOW_SPEED = 1.5                # m/s: walking. Dead balls are mostly walking
NEAR_RADII = (2.0, 5.0, 10.0)   # metres, for crowding around the ball
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16
GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH = 5.5, 9.16
GOAL_HALF_WIDTH = 3.66
STILL_DISP_M = 1.0              # ball moves less than this over a centred second: placed, not played
KICK_SPEED = 8.0                # m/s: faster than a carried or dribbled ball
MAX_BALL_SPEED = 40.0           # m/s: above this the "ball" is a detection jump, not a kick
OUT_EVENT_M = 1.0               # m beyond a line before an exit is believed
EVENT_CAP_S = 60.0
USE_BROADCAST = True            # fill tactical ball gaps from the aligned broadcast (align.py)
TEAM_TAGS = {"barcelona": 1, "atletic club": -1}   # Smart Tagging club names -> +1 home, -1 away
DEFENDING_CODES = ("defending in", "defensive transition")  # Team tag names the defending side
# Their Team tag agrees with the possessor only at chance level on the public window (44-54%), so
# they stay as raw columns but do not vote.
NON_VOTING_CODES = ("ball in final third", "ball in the box", "players in the box")
ROLL_WINDOWS = {"r1": 12, "r3": 37}                 # half-widths in frames
ROLLED = ("bp_d_diff", "bp_d_h_min", "bp_d_a_min", "bp_n_h_5", "bp_n_a_5", "ball_speed",
          "ball_out_dist", "tm_speed_all", "tm_frac_slow_all", "tm_cx_all", "tm_vx_att_diff",
          "tm_mark_h", "tm_mark_a", "st_vote")


# -- helpers ----------------------------------------------------------------------------------

def rolling_nanmean(x, half_width):
    """Centred mean over +/- half_width frames, ignoring NaN. NaN where the window is empty."""
    valid = ~np.isnan(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(x))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(x), index + half_width + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def _nan_sorted(values):
    """Sort along the last axis with NaN last."""
    return np.sort(np.where(np.isnan(values), np.inf, values), axis=-1)


def _inf_to_nan(values):
    return np.where(np.isinf(values), np.nan, values)


def _quiet(func, *args, **kwargs):
    """numpy's nan-reductions warn on all-NaN rows; those rows are expected here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return func(*args, **kwargs)


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


# -- frame of reference -----------------------------------------------------------------------

def half_index(n_frames, game=GAME):
    """Per frame: 1 or 2 inside a graded half, 0 elsewhere."""
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in half_video_blocks(game):
        half[first:last + 1] = number
    return half


def home_attack_signs(pos, team, half, length):
    """{half: +1 if home attacks towards x = L, else -1}.

    Decided by who is the leftmost player on the pitch: that is almost always the goalkeeper of
    the side defending the x = 0 goal. A team's average position is not used, because a dominant
    side spends most of a half in the opponent's half.
    """
    x = np.where(np.isnan(pos[..., 0]), np.inf, pos[..., 0])
    leftmost_is_home = team[np.argmin(x, axis=1)] == HOME
    seen = np.isfinite(x).any(axis=1)
    signs = {}
    for number in (1, 2):
        mask = (half == number) & seen
        signs[number] = 1 if leftmost_is_home[mask].mean() > 0.5 else -1
    return signs


def to_home_attack_frame(xy, sign_per_frame, pitch):
    """Rotate 180 degrees where home attacks towards x = 0. Works on (N, 2) and (N, k, 2)."""
    out = xy.copy()
    flip = sign_per_frame < 0
    out[flip, ..., 0] = pitch[0] - xy[flip, ..., 0]
    out[flip, ..., 1] = pitch[1] - xy[flip, ..., 1]
    return out


# -- ball -------------------------------------------------------------------------------------

def ball_track(ball):
    """Interpolated ball position and the gap bookkeeping around it.

    Returns (position, interpolated_flag, frames_since_seen, frames_until_seen, gap_length,
    last_seen_index, next_seen_index); the indices are -1 / N where there is none.
    """
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, index, -1))
    following = np.minimum.accumulate(np.where(seen, index, n)[::-1])[::-1]
    since = np.where(last >= 0, index - last, np.nan).astype(np.float64)
    until = np.where(following < n, following - index, np.nan).astype(np.float64)
    gap = np.where(seen, 0.0, since + until - 1)

    position = ball.astype(np.float64).copy()
    bridge = (~seen) & (gap <= MAX_INTERP_FRAMES)
    if bridge.any():
        lo, hi = last[bridge], following[bridge]
        weight = ((index[bridge] - lo) / (hi - lo))[:, None]
        position[bridge] = ball[lo] * (1 - weight) + ball[hi] * weight
    return position, bridge, since, until, gap, last, following


def central_velocity(xy):
    """(x[t+1] - x[t-1]) / 2 frames, in units per second. Works on (N, 2) and (N, k, 2)."""
    velocity = np.full(xy.shape, np.nan)
    velocity[1:-1] = (xy[2:] - xy[:-2]) * (FPS / 2.0)
    return velocity


# -- feature groups ---------------------------------------------------------------------------

def ball_features(ball, interpolated, since, until, gap, pitch):
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    velocity = central_velocity(ball)
    speed = np.linalg.norm(velocity, axis=1)
    outside = np.max(np.stack([-x, x - length, -y, y - width]), axis=0)
    return {
        "ball_seen": (since == 0).astype(float),
        "ball_interp": interpolated.astype(float),
        "ball_x": x, "ball_y": y,
        "ball_vx": velocity[:, 0],                 # > 0: towards the goal home attacks
        "ball_speed": speed,
        "ball_speed_med5": pd.Series(speed).rolling(5, center=True, min_periods=3).median().values,
        "ball_out_dist": outside,                  # > 0 outside the lines, < 0 distance inside
        "ball_since_s": since / FPS,
        "ball_until_s": until / FPS,
        "ball_gap_s": gap / FPS,
    }


def restart_features(ball, raw_ball, last, following, pitch):
    """What a stoppage looks like from the ball: it sits still, or it vanishes off the pitch.

    `ball_disp_1s` is how far the ball travels over a centred second; a ball being placed for a
    free kick or a throw-in barely moves, which per-frame speed (jittery) shows less cleanly.
    `ball_still_s` is the length of the stationary run the frame belongs to.

    The `ball_prev_*` / `ball_next_*` columns describe the ball at the last frame before and the
    first frame after the current gap in the raw track (the frame itself when the ball is seen):
    a gap that opens with the ball over a line and closes with it still at a corner, a touchline
    or the centre spot is a stoppage, however long it lasts.
    """
    length, width = pitch
    n = len(ball)
    displacement = np.full(n, np.nan)
    displacement[12:-12] = np.linalg.norm(ball[24:] - ball[:-24], axis=1)
    with np.errstate(invalid="ignore"):
        still = displacement < STILL_DISP_M
    run_length = np.zeros(n)
    edges = np.diff(np.concatenate(([0], still.astype(np.int8), [0])))
    for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        run_length[start:end] = (end - start) / FPS

    def at(index, values):
        valid = (index >= 0) & (index < n)
        out = np.full(n, np.nan)
        out[valid] = values[index[valid]]
        return out

    x, y = raw_ball[:, 0], raw_ball[:, 1]
    outside = np.max(np.stack([-x, x - length, -y, y - width]), axis=0)
    in_goal = ((x < 0) | (x > length)) & (np.abs(y - width / 2) < GOAL_HALF_WIDTH)
    corners = np.array([[0, 0], [0, width], [length, 0], [length, width]])
    to_corner = np.linalg.norm(raw_ball[:, None, :] - corners[None], axis=2).min(axis=1)
    return {
        "ball_disp_1s": displacement,
        "ball_still_s": run_length,
        "ball_prev_out": at(last, outside),
        "ball_prev_in_goal": at(last, in_goal.astype(float)),
        "ball_next_out": at(following, outside),
        "ball_next_still_s": at(following, run_length),
        "ball_next_d_touchline": at(following, np.minimum(y, width - y)),
        "ball_next_d_corner": at(following, to_corner),
        "ball_next_d_centre": at(following, np.hypot(x - length / 2, y - width / 2)),
        "ball_next_d_goalline": at(following, np.minimum(x, length - x)),
    }


def _since_until(events, cap_s=EVENT_CAP_S):
    """Seconds since the last and until the next True frame, capped. The frame itself counts 0."""
    n = len(events)
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(events, index, -10 ** 9))
    following = np.minimum.accumulate(np.where(events, index, 10 ** 9)[::-1])[::-1]
    return np.minimum((index - last) / FPS, cap_s), np.minimum((following - index) / FPS, cap_s)


def event_features(ball_speed_med5, raw_ball, speed_all, pitch):
    """Kicks and exits: the events that end and restart play.

    A kick is the ball, seen and inside the lines, travelling faster than a player can carry it.
    Open play has one every few seconds even when everybody walks; a stoppage has none until the
    restart. An exit is the ball seen clearly over a line. `ball_out_unresolved` is 1 while the
    most recent exit has not yet been followed by a kick -- the ball is out and nobody has
    restarted.
    """
    length, width = pitch
    x, y = raw_ball[:, 0], raw_ball[:, 1]
    with np.errstate(invalid="ignore"):
        outside = np.max(np.stack([-x, x - length, -y, y - width]), axis=0)
        kick = (outside < 0) & (ball_speed_med5 > KICK_SPEED) & (ball_speed_med5 < MAX_BALL_SPEED)
        exit_ = outside > OUT_EVENT_M
    since_kick, until_kick = _since_until(kick)
    since_out, _ = _since_until(exit_)

    smooth = rolling_nanmean(speed_all, 12)
    lag = 2 * FPS
    drop = np.full(len(smooth), np.nan)
    rise = np.full(len(smooth), np.nan)
    drop[lag:] = smooth[:-lag] - smooth[lag:]          # > 0: slowed down over the last 2 s
    rise[:-lag] = smooth[lag:] - smooth[:-lag]         # > 0: speeds up over the next 2 s
    return {
        "ev_kick": kick.astype(float),
        "ev_since_kick_s": since_kick,
        "ev_until_kick_s": until_kick,
        "ev_since_out_s": since_out,
        "ev_out_unresolved": (since_out < since_kick).astype(float),
        "ev_speed_drop": drop,
        "ev_speed_rise": rise,
    }


def ball_player_features(ball, pos, team, pitch):
    """Who is near the ball, and who moves with it. NaN wherever the ball position is unknown."""
    distance = np.linalg.norm(pos - ball[:, None, :], axis=2)            # (N, slots)
    home, away = team == HOME, team == AWAY
    d_home, d_away = _nan_sorted(distance[:, home]), _nan_sorted(distance[:, away])
    no_ball = np.isnan(ball[:, 0])

    out = {
        "bp_d_h_min": _inf_to_nan(d_home[:, 0]), "bp_d_h_2nd": _inf_to_nan(d_home[:, 1]),
        "bp_d_a_min": _inf_to_nan(d_away[:, 0]), "bp_d_a_2nd": _inf_to_nan(d_away[:, 1]),
    }
    out["bp_d_diff"] = out["bp_d_h_min"] - out["bp_d_a_min"]           # < 0: home is closer
    for radius in NEAR_RADII:
        tag = "{:g}".format(radius)
        out["bp_n_h_" + tag] = np.where(no_ball, np.nan, (d_home <= radius).sum(axis=1))
        out["bp_n_a_" + tag] = np.where(no_ball, np.nan, (d_away <= radius).sum(axis=1))

    # The nearest player overall, and whether the ball travels with them: a carried or dribbled
    # ball has the carrier's velocity, a passed one does not.
    safe = np.where(np.isnan(distance), np.inf, distance)
    nearest = np.argmin(safe, axis=1)
    rows = np.arange(len(ball))
    has_nearest = np.isfinite(safe[rows, nearest])

    player_velocity = central_velocity(pos)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=2)
    jump = np.zeros(pos.shape[:2], dtype=bool)
    jump[1:] |= step > MAX_STEP_M
    jump[:-1] |= step > MAX_STEP_M
    player_velocity[jump] = np.nan

    ball_velocity = central_velocity(ball)
    relative = np.linalg.norm(ball_velocity - player_velocity[rows, nearest], axis=1)
    out["bp_nearest_home"] = np.where(has_nearest, team[nearest] == HOME, np.nan).astype(float)
    out["bp_nearest_rel_speed"] = np.where(has_nearest, relative, np.nan)
    return out


def geometry_features(ball, pos, pitch):
    """Where restarts happen: centre spot, corners, goal areas, touchlines -- and crowded boxes."""
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    corners = np.array([[0, 0], [0, width], [length, 0], [length, width]])
    to_corner = np.linalg.norm(ball[:, None, :] - corners[None], axis=2).min(axis=1)
    centred_y = np.abs(y - width / 2)
    in_goal_area = (((x < GOAL_AREA_DEPTH) | (x > length - GOAL_AREA_DEPTH))
                    & (centred_y < GOAL_AREA_HALF_WIDTH))

    px, py = pos[..., 0], pos[..., 1]
    in_box_width = np.abs(py - width / 2) < BOX_HALF_WIDTH
    box_low = ((px < BOX_DEPTH) & in_box_width).sum(axis=1)
    box_high = ((px > length - BOX_DEPTH) & in_box_width).sum(axis=1)
    return {
        "geo_d_centre": np.hypot(x - length / 2, y - width / 2),
        "geo_d_corner": to_corner,
        "geo_d_touchline": np.minimum(y, width - y),
        "geo_d_goalline": np.minimum(x, length - x),
        "geo_in_goal_area": np.where(np.isnan(x), np.nan, in_goal_area),
        "geo_box_max": np.maximum(box_low, box_high).astype(float),
    }


def _team_shape(pos, speed, prefix):
    x, y = pos[..., 0], pos[..., 1]
    count = (~np.isnan(x)).sum(axis=1).astype(float)
    p10x, p90x = _quiet(np.nanpercentile, x, [10, 90], axis=1)
    p10y, p90y = _quiet(np.nanpercentile, y, [10, 90], axis=1)
    return {
        prefix + "n": count,
        prefix + "cx": _quiet(np.nanmean, x, axis=1),
        prefix + "cy": _quiet(np.nanmean, y, axis=1),
        prefix + "sx": _quiet(np.nanstd, x, axis=1),
        prefix + "sy": _quiet(np.nanstd, y, axis=1),
        prefix + "length": p90x - p10x,
        prefix + "width": p90y - p10y,
        prefix + "speed": _quiet(np.nanmean, speed, axis=1),
        prefix + "speed_max": _quiet(np.nanmax, speed, axis=1),
        prefix + "frac_slow": np.where(count > 0, (speed < SLOW_SPEED).sum(axis=1) / np.maximum(count, 1), np.nan),
    }


def _marking(attackers, defenders, chunk=20000):
    """Mean over `attackers` of the distance to the nearest of `defenders`. Chunked for memory."""
    out = np.full(len(attackers), np.nan)
    for start in range(0, len(attackers), chunk):
        a, d = attackers[start:start + chunk], defenders[start:start + chunk]
        pair = np.linalg.norm(a[:, :, None, :] - d[:, None, :, :], axis=3)
        nearest = _quiet(np.nanmin, pair, axis=2)
        out[start:start + chunk] = _quiet(np.nanmean, nearest, axis=1)
    return out


def team_features(pos, speed, team, pitch):
    length = pitch[0]
    home, away = team == HOME, team == AWAY
    out = {}
    out.update(_team_shape(pos[:, home], speed[:, home], "tm_h_"))
    out.update(_team_shape(pos[:, away], speed[:, away], "tm_a_"))

    all_count = (~np.isnan(speed)).sum(axis=1)
    out["tm_cx_all"] = _quiet(np.nanmean, pos[..., 0], axis=1)
    out["tm_speed_all"] = _quiet(np.nanmean, speed, axis=1)
    out["tm_frac_slow_all"] = np.where(all_count > 0, (speed < SLOW_SPEED).sum(axis=1) / np.maximum(all_count, 1), np.nan)
    out["tm_cx_diff"] = out["tm_h_cx"] - out["tm_a_cx"]

    # Centroid velocity towards the team's own attacking goal, from centroids smoothed over 1 s so
    # that slots blinking in and out do not read as movement.
    for side, sign in (("h", 1.0), ("a", -1.0)):
        smooth = rolling_nanmean(out["tm_{}_cx".format(side)], 12)
        velocity = np.full(len(smooth), np.nan)
        velocity[12:-12] = (smooth[24:] - smooth[:-24]) * (FPS / 24.0)
        out["tm_{}_vx_att".format(side)] = sign * velocity
    out["tm_vx_att_diff"] = out["tm_h_vx_att"] - out["tm_a_vx_att"]

    # Defensive lines: the second-deepest player of each side (the deepest is usually the keeper),
    # as a distance from the goal that side defends.
    home_x, away_x = _nan_sorted(pos[:, home, 0]), _nan_sorted(-pos[:, away, 0])
    out["tm_h_def_line"] = _inf_to_nan(home_x[:, 1])
    out["tm_a_def_line"] = _inf_to_nan(length + away_x[:, 1])

    # How tightly each side is marked: mean distance from its players to the nearest opponent.
    out["tm_mark_h"] = _marking(pos[:, home], pos[:, away])
    out["tm_mark_a"] = _marking(pos[:, away], pos[:, home])
    return out


def tagging_features(n_frames, game=GAME):
    """Smart Tagging phases painted onto frames.

    For every code, `st_<code>` is +1 / -1 for a home / away tag and 0 otherwise (including N/A),
    and `st_<code>_on` is 1 whenever the phase is active at all. `st_vote` sums the possession each
    active phase implies: an attacking phase votes for its team, a defending one for the opponent
    (checked on the public window: defending tags name the possessor 0-27% of the time).
    """
    from load_smart_tagging import load as load_tags
    events = load_tags(game)
    out = {}
    vote = np.zeros(n_frames)
    set_piece_starts = []

    for event in events:
        code = _slug(event["code"])
        first, end = max(0, event["start_frame"]), min(n_frames, event["end_frame"])
        team_name = (event["tags"].get("Team") or "").strip().lower()
        sign = TEAM_TAGS.get(team_name, 0)
        out.setdefault("st_" + code, np.zeros(n_frames))[first:end] = sign
        out.setdefault("st_" + code + "_on", np.zeros(n_frames))[first:end] = 1
        raw_code = event["code"].strip().lower()
        if raw_code not in NON_VOTING_CODES:
            vote[first:end] += -sign if raw_code.startswith(DEFENDING_CODES) else sign
        if code == "set_pieces":
            set_piece_starts.append(first)
            kind = _slug(event["tags"].get("Type") or "unknown")
            out.setdefault("st_sp_" + kind, np.zeros(n_frames))[first:end] = 1

    out["st_vote"] = vote
    starts = np.array(sorted(set_piece_starts))
    index = np.arange(n_frames)
    position = np.searchsorted(starts, index, side="right")
    previous = np.where(position > 0, starts[np.maximum(position - 1, 0)], -10 ** 9)
    following = np.where(position < len(starts), starts[np.minimum(position, len(starts) - 1)], 10 ** 9)
    out["st_since_set_piece_s"] = np.minimum((index - previous) / FPS, 600)
    out["st_until_set_piece_s"] = np.minimum((following - index) / FPS, 600)
    return out


# -- assembly ---------------------------------------------------------------------------------

def merged_ball(tracking, use_broadcast):
    """The tactical ball, with its gaps filled from the aligned broadcast where it saw the ball.

    Returns (ball (N, 2) in raw pitch metres, from_broadcast (N,) bool).
    """
    ball = tracking["ball"].astype(np.float64)
    from_broadcast = np.zeros(len(ball), dtype=bool)
    if use_broadcast:
        from possession.align import broadcast_ball_on_tactical
        other = broadcast_ball_on_tactical(len(ball))
        from_broadcast = np.isnan(ball[:, 0]) & ~np.isnan(other[:, 0])
        ball[from_broadcast] = other[from_broadcast]
    return ball, from_broadcast


def build(game=GAME, source="tactical", use_broadcast=USE_BROADCAST, swap=False):
    """The full feature table of the competition match: one row per video frame, float32 columns.

    swap: the same match with the two teams swapped (possession/augment.py), without the Smart
    Tagging columns -- what the temporal networks read for their swapped-team averaging.
    """
    tracking = load_tracking(source, game)
    n_frames = len(tracking["ball"])
    ball_raw, from_broadcast = merged_ball(tracking, use_broadcast)
    starts = {number: first for number, first, _ in half_video_blocks(game)}
    if swap:
        from possession.augment import transform
        return build_from_tracking(transform(tracking, swap=True), half_index(n_frames, game), starts,
                                   ball_raw, from_broadcast)
    return build_from_tracking(tracking, half_index(n_frames, game), starts, ball_raw,
                               from_broadcast, tagging_features(n_frames, game))


def build_from_tracking(tracking, half, starts, ball_raw=None, from_broadcast=None, tagging=None):
    """The feature table for any tracking dict shaped like `possession.tracking.load()`.

    half      per frame: 1 / 2 inside a graded half, 0 elsewhere
    starts    {half number: first frame of that half}
    ball_raw  ball track to use, raw pitch metres (default: tracking["ball"])
    tagging   Smart Tagging columns (`tagging_features`), or None for a match without tagging --
              the st_ columns are then absent, so a model trained on such matches never sees them
    """
    pitch = tracking["pitch"].astype(np.float64)
    team = tracking["team"]
    n_frames = len(tracking["ball"])
    if ball_raw is None:
        ball_raw = tracking["ball"].astype(np.float64)
    if from_broadcast is None:
        from_broadcast = np.zeros(n_frames, dtype=bool)

    signs = home_attack_signs(tracking["pos"], team, half, pitch[0])
    sign_per_frame = np.where(half == 2, signs[2], signs[1])
    pos = to_home_attack_frame(tracking["pos"].astype(np.float64), sign_per_frame, pitch)
    raw_ball = to_home_attack_frame(ball_raw, sign_per_frame, pitch)
    speed = tracking["speed"].astype(np.float64)

    ball, interpolated, since, until, gap, last, following = ball_track(raw_ball)

    t_half = np.where(half > 0, (np.arange(n_frames) - np.vectorize(starts.get)(np.maximum(half, 1))) / FPS, np.nan)
    columns = {"half": half.astype(float), "graded": (half > 0).astype(float), "t_half": t_half,
               "ball_from_broadcast": from_broadcast.astype(float)}
    columns.update(ball_features(ball, interpolated, since, until, gap, pitch))
    columns.update(restart_features(ball, raw_ball, last, following, pitch))
    columns.update(ball_player_features(ball, pos, team, pitch))
    columns.update(geometry_features(ball, pos, pitch))
    columns.update(team_features(pos, speed, team, pitch))
    columns.update(event_features(columns["ball_speed_med5"], raw_ball, columns["tm_speed_all"], pitch))
    if tagging is not None:
        columns.update(tagging)

    for name in ROLLED:
        if name not in columns:
            continue
        for suffix, half_width in ROLL_WINDOWS.items():
            columns["{}_{}".format(name, suffix)] = rolling_nanmean(columns[name], half_width)

    frame = pd.DataFrame({key: np.asarray(value, dtype=np.float32) for key, value in columns.items()})
    frame.index.name = "frame"
    frame.attrs["home_attack_signs"] = signs
    return frame


def cache_path(game=GAME, source="tactical", swap=False):
    from possession.tracking import CACHE_DIR
    return os.path.join(CACHE_DIR, "{}_{}{}_features.parquet".format(game, source, "_swapped" if swap else ""))
