"""Degrade clean, corrected tracking until it looks like the competition's automatic ATD.

The DFL open data is corrected optical tracking: the ball is there on every frame, every player on
every frame. The competition tracking is automatic and uncorrected. A model trained or a decoder
tuned on the clean data would meet a different world on the real match, so the clean data is
corrupted with the same defects, at the rates the competition documents and our own aggregate
measurements report (DATA.md; notebooks/01 summary):

    ball gaps        missing on ~41% of graded frames, in long gaps (median ~10 s), more often
                     while the ball is dead (~56% missing) than alive (~30%), and more often while
                     the players are slow (gap_speed_beta)
    ball near lines  the tactical camera rarely sees the ball near or beyond the lines
                     (line_margin_m, line_visible_min)
    ball smoothing   the ATD ball is strongly smoothed: a kick ramps from ~2 to ~20 m/s in ~0.8 s
    player dropout   ~20.5 of 22 players tracked on average, missing in short stretches
    wander           low-frequency position error on players and ball: white noise smoothed over
                     `*_wander_frames`, scaled to `*_jitter_m`. (Version n2 added the noise frame by
                     frame, after smoothing: it inflated the ball speed from ~3 to ~8 m/s and made the
                     kick detector fire on almost every frame -- see docs/MIGLIORAMENTI_SPELLF1.md.)

The defaults are calibrated so that label-free aggregate distributions of the features (medians,
percentiles over the whole competition match, from scripts/domain_shift_check.py) match the
proxy's. No label -- public or private -- and no frame of the competition data enters here. Every
random draw comes from the generator passed in, so a seed fixes the result.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

FPS = 25


@dataclass(frozen=True)
class NoiseConfig:
    missing_alive: float = 0.30         # share of alive frames with no ball (gap process only)
    missing_dead: float = 0.56          # share of dead frames with no ball (gap process only)
    gap_median_s: float = 10.0          # ball gaps: lognormal median ...
    gap_sigma: float = 1.0              # ... and log-standard deviation
    gap_speed_beta: float = 0.0         # gap starts x exp(-beta * (mean player speed - 1.4 m/s))
    ball_smooth_frames: float = 5.0     # Gaussian sigma of the ball track
    ball_jitter_m: float = 0.3          # amplitude of the ball's low-frequency wander
    ball_wander_frames: float = 0.0     # its correlation (Gaussian sigma, frames); 0 = white (n2)
    line_margin_m: float = 0.0          # ball fully visible only this far inside the lines; 0 = off
    line_visible_min: float = 1.0       # visibility at and beyond the lines
    player_missing: float = 0.07        # share of on-pitch player frames lost
    player_gap_median_s: float = 1.5
    player_gap_sigma: float = 1.0
    player_jitter_m: float = 0.15
    player_wander_frames: float = 0.0


def _gap_hazard(missing_share: float, mean_gap_frames: float) -> float:
    """Per-frame probability of a gap opening, for a target share of missing frames.

    A renewal process that opens a gap with probability p per seen frame and keeps it for G frames
    on average is missing on p*G / (1 + p*G) of the time.
    """
    ratio = missing_share / (1.0 - missing_share)
    return ratio / mean_gap_frames


def _lognormal_frames(rng: np.random.Generator, median_s: float, sigma: float) -> int:
    return max(1, int(round(rng.lognormal(np.log(median_s * FPS), sigma))))


def ball_gaps(dead: np.ndarray, graded: np.ndarray, rng: np.random.Generator,
              config: NoiseConfig, player_speed: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of frames where the ball goes missing. Gap starts depend on the frame's state
    and, with gap_speed_beta > 0, on how slowly the players move (normalised per state, so the
    target missing shares hold)."""
    mean_gap = config.gap_median_s * FPS * np.exp(config.gap_sigma ** 2 / 2)
    hazard = np.where(dead, _gap_hazard(config.missing_dead, mean_gap),
                      _gap_hazard(config.missing_alive, mean_gap))
    if config.gap_speed_beta > 0 and player_speed is not None:
        speed = np.where(np.isnan(player_speed), 1.4, player_speed)
        factor = np.clip(np.exp(-config.gap_speed_beta * (speed - 1.4)), 0.1, 10.0)
        for state in (dead, ~dead):
            mask = state & graded
            if mask.any():
                factor[mask] /= factor[mask].mean()
        hazard = hazard * factor
    hazard = np.where(graded, hazard, 0.0)
    draws = rng.random(len(dead))
    missing = np.zeros(len(dead), dtype=bool)
    t = 0
    while t < len(dead):
        if draws[t] < hazard[t]:
            length = _lognormal_frames(rng, config.gap_median_s, config.gap_sigma)
            missing[t:t + length] = True
            t += length
        else:
            t += 1
    return missing


def line_dropout(ball: np.ndarray, pitch: np.ndarray, rng: np.random.Generator,
                 config: NoiseConfig) -> np.ndarray:
    """Boolean mask of ball frames lost near the lines. Visibility rises linearly from
    line_visible_min at the lines (and beyond) to 1 at line_margin_m inside; the draws are
    correlated over ~1 s so that losses come in stretches, not as flicker."""
    if config.line_margin_m <= 0:
        return np.zeros(len(ball), dtype=bool)
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    inside = np.min(np.stack([x, length - x, y, width - y]), axis=0)
    visible = config.line_visible_min + (1 - config.line_visible_min) * np.clip(
        inside / config.line_margin_m, 0.0, 1.0)
    z = gaussian_filter1d(rng.normal(size=len(ball)), 25.0)
    u = norm.cdf(z / z.std())
    return ~np.isnan(x) & (u > visible)


def _wander(rng: np.random.Generator, shape: tuple, amplitude: float, frames: float) -> np.ndarray:
    """Position error with standard deviation `amplitude`, correlated over `frames` along axis 0
    (white when frames == 0, the n2 behaviour)."""
    noise = rng.normal(0.0, 1.0, shape)
    if frames > 0:
        noise = gaussian_filter1d(noise, frames, axis=0)
        noise /= noise.std()
    return amplitude * noise


def player_dropout(present: np.ndarray, rng: np.random.Generator, config: NoiseConfig) -> np.ndarray:
    """Boolean mask (frames, slots) of player frames to delete, in short stretches."""
    mean_gap = config.player_gap_median_s * FPS * np.exp(config.player_gap_sigma ** 2 / 2)
    hazard = _gap_hazard(config.player_missing, mean_gap)
    drop = np.zeros(present.shape, dtype=bool)
    n_frames, n_slots = present.shape
    for slot in range(n_slots):
        starts = np.where(rng.random(n_frames) < hazard)[0]
        for start in starts:
            length = _lognormal_frames(rng, config.player_gap_median_s, config.player_gap_sigma)
            drop[start:start + length, slot] = True
    return drop & present


def _smooth_runs(track: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of an (N, 2) track, run by run so that NaN gaps never smear."""
    out = track.copy()
    seen = ~np.isnan(track[:, 0])
    edges = np.diff(np.concatenate(([0], seen.astype(np.int8), [0])))
    for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        out[start:end] = gaussian_filter1d(track[start:end], sigma, axis=0, mode="nearest")
    return out


def degrade(tracking: dict, labels: np.ndarray, half: np.ndarray, seed: int,
            config: NoiseConfig = NoiseConfig()) -> dict:
    """A noisy copy of a tracking dict. `labels` / `half` only steer where gaps fall."""
    rng = np.random.default_rng(seed)
    out = {key: (value.copy() if isinstance(value, np.ndarray) else value)
           for key, value in tracking.items()}

    ball = tracking["ball"].astype(np.float64)
    if config.ball_smooth_frames > 0:
        ball = _smooth_runs(ball, config.ball_smooth_frames)
    ball += _wander(rng, ball.shape, config.ball_jitter_m, config.ball_wander_frames)
    speed = tracking["speed"].astype(np.float64).copy()
    with warnings.catch_warnings():          # frames without any player: expected, filled below
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_speed = gaussian_filter1d(np.nan_to_num(np.nanmean(speed, axis=1), nan=1.4), 37.0)
    lost = ball_gaps(labels == "D", half > 0, rng, config, mean_speed)
    lost |= line_dropout(ball, tracking["pitch"].astype(np.float64), rng, config)
    ball[lost] = np.nan
    out["ball"] = ball.astype(np.float32)

    pos = tracking["pos"].astype(np.float64)
    present = ~np.isnan(pos[..., 0])
    pos += _wander(rng, pos.shape, config.player_jitter_m, config.player_wander_frames)
    drop = player_dropout(present, rng, config)
    pos[drop] = np.nan
    speed[drop] = np.nan
    out["pos"], out["speed"] = pos.astype(np.float32), speed.astype(np.float32)
    return out
