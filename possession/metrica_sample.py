"""Metrica Sports' public sample matches as possession validation and training material.

Three anonymised matches (github.com/metrica-sports/sample-data, fetched by
scripts/fetch_metrica_sample.sh), tracked and annotated by the provider of the competition data:

    game 1, 2   CSV tracking (one file per team) and CSV events (older annotation: no CARRY)
    game 3      FIFA EPTS tracking -- the competition's own format -- and JSON events

25 fps, coordinates normalised to a 105 x 68 m pitch. The players are present on every frame; the
ball is missing on about a third of them, much like the competition ATD.

There is no possession flag in this tracking, so the labels come from the human-annotated events,
read with Metrica's own definitions (documentation/events-definitions.pdf): "a player is regarded as
being in possession as soon as they touch the ball [...] until another player takes over
possession [...] or the ball goes out of play".

    live team   the team of the latest possession event started at or before the frame:
                SET PIECE, RECOVERY, PASS, CARRY, SHOT, BALL LOST (the losing team still had it),
                FAULT RECEIVED. A CHALLENGE alone changes nothing; a won one is followed by the
                winner's RECOVERY or action.
    dead ball   from a stoppage to the first play event after it: the end of a BALL OUT, of a SHOT
                that goes OUT or is a GOAL, of a BALL LOST to OFFSIDE / HANDBALL / FORCED /
                END HALF; the start of a CHALLENGE with FAULT (not ADVANTAGE) or a FAULT RECEIVED.
                The next SET PIECE, goal kick or RECOVERY (dropped ball) restarts play.
    graded      from each half's kick-off to its last tracked frame; the rest is X.

load(game) returns exactly what possession.dfl.load returns, so the pipeline runs on it unchanged;
features(game) is its feature table and training_matches() the (features, labels) pairs that
`--extra-metrica` adds to training (each match REPEATS times, so it weighs what a DFL match with its
three noise copies weighs).
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

from possession import dfl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "external", "metrica_sample")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
GAMES = ("1", "2", "3")
FPS = 25
PITCH = (105.0, 68.0)
SPEED_HALF_WINDOW = 2          # player speed from the displacement over +/-2 frames
MAX_STOPPAGE_FRAMES = 180 * FPS
REPEATS = 3

POSSESSION_TYPES = {"SET PIECE", "RECOVERY", "PASS", "CARRY", "SHOT", "BALL LOST", "FAULT RECEIVED"}
STOP_AT_END = {"BALL OUT"}
STOPPING_LOSSES = {"OFFSIDE", "HANDBALL", "HAND BALL", "FORCED", "END HALF"}
NOT_PLAY = {"CHALLENGE", "CARD", "BALL OUT", "FAULT RECEIVED"}


# -- events -------------------------------------------------------------------------------------

def _row(team, kind, subtypes, period, start, end):
    return {"team": team, "type": kind, "subtypes": frozenset(s for s in subtypes if s),
            "period": int(period), "start": int(start), "end": int(end)}


def read_events(game: str) -> list[dict]:
    """[{team H/A, type, subtypes, period, start, end}] in file order (frame numbers of the tracking)."""
    if game == "3":
        with open(os.path.join(DATA_DIR, "Sample_Game_3_events.json"), encoding="utf-8") as handle:
            data = json.load(handle)["data"]
        out = []
        for e in data:
            subtypes = e.get("subtypes") or []
            subtypes = [subtypes] if isinstance(subtypes, dict) else subtypes
            team = "H" if e["team"]["id"] == "FIFATMA" else "A"
            out.append(_row(team, e["type"]["name"], [s["name"] for s in subtypes], e["period"],
                            e["start"]["frame"], e["end"]["frame"]))
        return out
    path = os.path.join(DATA_DIR, "Sample_Game_{}_RawEventsData.csv".format(game))
    with open(path, newline="", encoding="utf-8") as handle:
        return [_row("H" if r["Team"] == "Home" else "A", r["Type"], r["Subtype"].split("-"), r["Period"],
                     r["Start Frame"], r["End Frame"]) for r in csv.DictReader(handle)]


def stoppage_frame(event: dict) -> int | None:
    """Frame where play stops because of this event, or None."""
    kind, sub = event["type"], event["subtypes"]
    if kind in STOP_AT_END:
        return event["end"]
    if kind == "SHOT" and sub & {"OUT", "GOAL"}:
        return event["end"]
    if kind == "BALL LOST" and sub & STOPPING_LOSSES:
        return event["end"]
    if kind == "CHALLENGE" and "FAULT" in sub and "ADVANTAGE" not in sub:
        return event["start"]
    if kind == "FAULT RECEIVED":
        return event["start"]
    return None


def event_labels(events: list[dict], period: int, first: int, last: int) -> np.ndarray:
    """H / A / D for frames first..last (inclusive) of one period, from its events."""
    own = sorted((e for e in events if e["period"] == period), key=lambda e: (e["start"], e["end"]))
    n = last - first + 1
    labels = np.full(n, "D", dtype="<U1")            # before the first possession event: kick-off
    team = None
    starts = [(e["start"], e["team"]) for e in own if e["type"] in POSSESSION_TYPES]
    k = 0
    for t in range(n):
        frame = first + t
        while k < len(starts) and starts[k][0] <= frame:
            team = starts[k][1]
            k += 1
        if team is not None:
            labels[t] = team
    plays = [e["start"] for e in own if e["type"] not in NOT_PLAY and stoppage_frame(e) is None]
    for e in own:
        stop = stoppage_frame(e)
        if stop is None:
            continue
        later = [p for p in plays if p > stop]
        restart = min(later[0] if later else last + 1, stop + MAX_STOPPAGE_FRAMES)
        lo, hi = max(stop, first) - first, min(restart, last + 1) - first
        if hi > lo:
            labels[lo:hi] = "D"
    return labels


# -- tracking -----------------------------------------------------------------------------------

def _dataset(game: str):
    from kloppy import metrica
    if game == "3":
        return metrica.load_tracking_epts(meta_data=os.path.join(DATA_DIR, "Sample_Game_3_metadata.xml"),
                                          raw_data=os.path.join(DATA_DIR, "Sample_Game_3_tracking.txt"))
    return metrica.load_tracking_csv(
        home_data=os.path.join(DATA_DIR, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(game)),
        away_data=os.path.join(DATA_DIR, "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(game)))


def player_speed(pos: np.ndarray, half: np.ndarray, window: int = SPEED_HALF_WINDOW) -> np.ndarray:
    """(n, slots) m/s from the displacement over +/-window frames, within each half."""
    speed = np.full(pos.shape[:2], np.nan, dtype=np.float32)
    for number in (1, 2):
        index = np.where(half == number)[0]
        if len(index) <= 2 * window:
            continue
        p = pos[index]
        moved = np.linalg.norm(p[2 * window:] - p[:-2 * window], axis=2) * FPS / (2 * window)
        speed[index[window:-window]] = moved
    return speed


def assemble(frames, columns: dict, team_of: dict, events: list[dict]) -> dict:
    """Dense arrays on dfl's padded timeline: each half runs from its kick-off to its last frame."""
    period, frame_id = frames["period_id"].values.astype(int), frames["frame_id"].values.astype(int)
    kickoff = {p: min(e["start"] for e in events if e["period"] == p and e["type"] == "SET PIECE"
                      and "KICK OFF" in e["subtypes"]) for p in (1, 2)}
    spans, offsets, offset = {}, {}, dfl.PAD
    for p in (1, 2):
        mine = frame_id[period == p]
        spans[p] = (max(kickoff[p], int(mine.min())), int(mine.max()))
        offsets[p] = offset
        offset += spans[p][1] - spans[p][0] + 1 + dfl.PAD
    n = offset
    players = sorted(columns)
    pos = np.full((n, len(players), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    labels = np.full(n, "X", dtype="<U1")
    half = np.zeros(n, dtype=np.int8)
    for p in (1, 2):
        first, last = spans[p]
        keep = (period == p) & (frame_id >= first) & (frame_id <= last)
        rows = offsets[p] + frame_id[keep] - first
        half[offsets[p]:offsets[p] + last - first + 1] = p
        labels[offsets[p]:offsets[p] + last - first + 1] = event_labels(events, p, first, last)
        ball[rows, 0] = frames["ball_x"].values[keep] * PITCH[0]
        ball[rows, 1] = frames["ball_y"].values[keep] * PITCH[1]
        for slot, player in enumerate(players):
            x_col, y_col = columns[player]
            pos[rows, slot, 0] = frames[x_col].values[keep] * PITCH[0]
            pos[rows, slot, 1] = frames[y_col].values[keep] * PITCH[1]
    team = np.array([dfl.HOME if team_of[p] == "H" else dfl.AWAY for p in players], dtype=np.int8)
    return {"pos": pos, "speed": player_speed(pos, half), "team": team, "ball": ball,
            "ball_extra": np.full(n, np.nan, dtype=np.float32),
            "pitch": np.array(PITCH, dtype=np.float32), "labels": labels, "half": half}


def load(game: str, use_cache: bool = True) -> dict:
    """One sample match as a tracking dict (see possession.dfl.assemble), cached as .npz."""
    cache = os.path.join(CACHE_DIR, "game{}.npz".format(game))
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as data:
            return {key: data[key] for key in data.files}
    dataset = _dataset(game)
    frames = dataset.to_df()
    columns, team_of = {}, {}
    for team in dataset.metadata.teams:
        for player in team.players:
            x_col, y_col = "{}_x".format(player.player_id), "{}_y".format(player.player_id)
            if x_col in frames.columns:
                columns[player.player_id] = (x_col, y_col)
                team_of[player.player_id] = "H" if team is dataset.metadata.teams[0] else "A"
    out = assemble(frames, columns, team_of, read_events(game))
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, **out)
    return out


def features(game: str, use_cache: bool = True):
    """The pipeline's feature table of one sample match (possession.features), cached as parquet."""
    import pandas as pd
    from possession.features import build_from_tracking
    path = os.path.join(CACHE_DIR, "game{}_features.parquet".format(game))
    if use_cache and os.path.isfile(path):
        return pd.read_parquet(path)
    tracking = load(game, use_cache)
    table = build_from_tracking(tracking, tracking["half"], dfl.half_starts(tracking["half"]))
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        table.to_parquet(path)
    return table


def training_matches(games=GAMES) -> list:
    """[(features, event labels)] of the sample matches, each REPEATS times (the same objects)."""
    out = []
    for game in games:
        pair = (features(game), load(game)["labels"])
        out += [pair] * REPEATS
    return out
