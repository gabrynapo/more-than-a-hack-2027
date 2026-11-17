"""Four extra families of per-frame features (docs/MIGLIORAMENTI_SPELLF1.md, section 34).

Experimental: off by default; scripts/proxy_extra_features.py tests them, and
`make_submission.py --extra-features` builds a submission with them (dfl_sup_v10_xf), for comparison.

    ball_prev_d_*   where the ball was last seen before the current gap: the "before" half of the
                    existing ball_next_d_* pair (a ball lost at the touchline is often out of play)
    gk_*            each side's deepest player (almost always the goalkeeper): distance to the
                    ball, in the box, nearest opponent and whether opponents are backing off --
                    a keeper holding the ball is possession, not a stoppage
    rs_*            restart taker: while the ball sits still, the side of the nearest player and
                    how isolated they are; the last taker's side and the time since, carried on
    conv_*          where each side is running to: the least-squares meeting point of its players'
                    velocity lines, a ball estimate when the ball is lost (pressers converge on it)

All positions are in the "home attacks +x" frame of possession/features.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from possession.features import home_attack_signs, to_home_attack_frame
from possession.tracking import AWAY, HOME

FPS = 25
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16
TREND_FRAMES = 25
VELOCITY_FRAMES = 2
MIN_SPEED = 2.0
STILL_S = 1.0


def interpolate(ball):
    """Linear interpolation of the ball across gaps; NaN before the first and after the last sighting."""
    out = np.full_like(ball, np.nan, dtype=float)
    seen = np.flatnonzero(~np.isnan(ball[:, 0]))
    if len(seen) == 0:
        return out
    index = np.arange(len(ball))
    inside = (index >= seen[0]) & (index <= seen[-1])
    for k in range(2):
        out[inside, k] = np.interp(index[inside], seen, ball[seen, k])
    return out


def last_seen(ball):
    """Per frame: index of the last frame at or before it with a ball, -1 if none."""
    index = np.where(~np.isnan(ball[:, 0]), np.arange(len(ball)), -1)
    return np.maximum.accumulate(index)


def shifted(values, k):
    """values[t + k] (NaN outside)."""
    out = np.full_like(values, np.nan, dtype=float)
    if k >= 0:
        out[:len(values) - k] = values[k:]
    else:
        out[-k:] = values[:k]
    return out


def meeting_point(pos, velocity):
    """(N, 2) least-squares point closest to the lines through `pos` along `velocity` (N, k, 2),
    weighted by speed, over players faster than MIN_SPEED; NaN with fewer than three of them."""
    speed = np.linalg.norm(velocity, axis=2)
    use = (speed > MIN_SPEED) & ~np.isnan(pos[..., 0])
    w = np.where(use, speed, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dx, dy = velocity[..., 0] / speed, velocity[..., 1] / speed
    dx, dy = np.where(use, dx, 0.0), np.where(use, dy, 0.0)
    px, py = np.where(use, pos[..., 0], 0.0), np.where(use, pos[..., 1], 0.0)
    a11, a12, a22 = (w * (1 - dx * dx)).sum(1), (w * -dx * dy).sum(1), (w * (1 - dy * dy)).sum(1)
    b1 = (w * ((1 - dx * dx) * px - dx * dy * py)).sum(1)
    b2 = (w * (-dx * dy * px + (1 - dy * dy) * py)).sum(1)
    det = a11 * a22 - a12 * a12
    ok = (use.sum(1) >= 3) & (det > 1e-3 * w.sum(1) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        x, y = (a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det
    return np.where(ok[:, None], np.stack([x, y], axis=1), np.nan)


def extra_columns(tracking: dict, half: np.ndarray, features: pd.DataFrame) -> pd.DataFrame:
    pitch = tracking["pitch"].astype(float)
    length, width = pitch
    team = np.asarray(tracking["team"])
    signs = home_attack_signs(tracking["pos"], team, half, length)
    sign = np.where(half == 2, signs[2], signs[1])
    pos = to_home_attack_frame(tracking["pos"].astype(float), sign, pitch)
    raw = to_home_attack_frame(tracking["ball"].astype(float), sign, pitch)
    ball = interpolate(raw)
    n = len(raw)
    out = {}

    last = last_seen(raw)
    at_last = np.where(last[:, None] >= 0, raw[np.maximum(last, 0)], np.nan)
    out["ball_prev_d_touchline"] = np.minimum(at_last[:, 1], width - at_last[:, 1])
    out["ball_prev_d_goalline"] = np.minimum(at_last[:, 0], length - at_last[:, 0])

    home, away = team == HOME, team == AWAY
    with np.errstate(invalid="ignore"):
        distance = np.linalg.norm(pos - ball[:, None, :], axis=2)
    for side, own, other, deepest in (("h", home, away, np.nanargmin), ("a", away, home, np.nanargmax)):
        x = pos[:, own, 0]
        valid = ~np.isnan(x).all(axis=1)
        pick = np.zeros(n, dtype=int)
        pick[valid] = deepest(x[valid], axis=1)
        keeper = np.where(valid[:, None], pos[:, own][np.arange(n), pick], np.nan)
        goal_x = 0.0 if side == "h" else length
        # A frame with no tracked opponent gives NaN on purpose: silence nanmin's all-NaN warning.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["gk_{}_d_ball".format(side)] = np.linalg.norm(keeper - ball, axis=1)
            out["gk_{}_in_box".format(side)] = ((np.abs(keeper[:, 0] - goal_x) < BOX_DEPTH)
                                                & (np.abs(keeper[:, 1] - width / 2) < BOX_HALF_WIDTH)).astype(float)
            opp = np.nanmin(np.linalg.norm(pos[:, other] - keeper[:, None, :], axis=2), axis=1) \
                if other.any() else np.full(n, np.nan)
        out["gk_{}_opp_min".format(side)] = opp
        out["gk_{}_opp_trend".format(side)] = shifted(opp, TREND_FRAMES) - shifted(opp, -TREND_FRAMES)

    safe = np.where(np.isnan(distance), np.inf, distance)
    nearest = np.argmin(safe, axis=1)
    found = np.isfinite(safe[np.arange(n), nearest])
    near_sign = np.where(found, np.where(team[nearest] == HOME, 1.0, -1.0), 0.0)
    d_home = np.min(np.where(home[None, :], safe, np.inf), axis=1)
    d_away = np.min(np.where(away[None, :], safe, np.inf), axis=1)
    still = np.nan_to_num(features["ball_still_s"].values.astype(float)) >= STILL_S
    taker = np.where(still & found, near_sign, 0.0)
    with np.errstate(invalid="ignore"):  # inf - inf on frames without players: zeroed just below
        isolation = np.where(taker > 0, d_away - d_home, np.where(taker < 0, d_home - d_away, 0.0))
    out["rs_taker"] = taker
    out["rs_isolation"] = np.where(np.isfinite(isolation), np.clip(isolation, -30, 30), 0.0)
    index = np.arange(n)
    last_still = np.maximum.accumulate(np.where(taker != 0, index, -1))
    out["rs_last_taker"] = np.where(last_still >= 0, taker[np.maximum(last_still, 0)], 0.0)
    out["rs_since_s"] = np.where(last_still >= 0, np.minimum((index - last_still) / FPS, 600), 600)

    velocity = (np.roll(pos, -VELOCITY_FRAMES, axis=0) - np.roll(pos, VELOCITY_FRAMES, axis=0)) \
        * FPS / (2 * VELOCITY_FRAMES)
    points = {}
    for side, own in (("h", home), ("a", away)):
        point = meeting_point(pos[:, own], velocity[:, own])
        point = np.where((np.abs(point - pitch / 2) <= pitch / 2 + 10).all(axis=1)[:, None], point, np.nan)
        points[side] = point
        with np.errstate(invalid="ignore"):
            out["conv_{}_d_ball".format(side)] = np.linalg.norm(point - ball, axis=1)
            to_point = np.linalg.norm(pos - point[:, None, :], axis=2)
        near = np.argmin(np.where(np.isnan(to_point), np.inf, to_point), axis=1)
        out["conv_{}_near_sign".format(side)] = np.where(~np.isnan(point[:, 0]),
                                                         np.where(team[near] == HOME, 1.0, -1.0), 0.0)
    with np.errstate(invalid="ignore"):
        out["conv_gap"] = np.linalg.norm(points["h"] - points["a"], axis=1)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()})
