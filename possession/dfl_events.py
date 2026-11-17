"""DFL human-annotated events, synchronised to the tracking, as evidence of *controlled* possession.

Why: the competition counts a change of possession only when the other team gains **controlled**
possession; a touch, a deflection or a won header without control do not count (CHALLENGE.md).
DFL's tracking flag `BallPossession` flips on every touch, and `dfl.controlled_labels` cleans it with
a duration rule -- every spell under tau seconds is folded away. That rule cannot tell a real
one-second possession (a player controls the ball and is tackled) from a rebound, so the models
never see the former. The DFL event feed can: annotators record passes, shots, ball claims and duels,
and duels say whether possession changed and whether the loser had the ball under control.

    events(match_id)            parsed and synchronised events, cached as JSON
    control_frames(match_id)    {"H": frames, "A": frames} where a team demonstrably had control
    event_labels(labels, control, min_control_s)
                                controlled_labels, except that a short spell with control evidence
                                of its own team is kept as a real possession

Synchronisation. Event times are typed live and drift by up to a few seconds from the tracking.
Each event names its player, and the tracking carries every player's id, so the event is moved to
the frame, within a window around its stated time, where that player is closest to the ball; it is
dropped if the player never comes within MAX_DISTANCE_M of the ball. This is the usual event/tracking
alignment idea (e.g. ETSY, Van Roy et al. 2023), reduced to what control evidence needs.

Same data, same licence as dfl.py (Bassek et al., Scientific Data 2025, CC-BY 4.0): public, not
about the competition match.
"""

from __future__ import annotations

import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

from possession import dfl

FPS = 25
WINDOW_BEFORE_S, WINDOW_AFTER_S = 5.0, 5.0   # search around the stated event time
MAX_DISTANCE_M = 2.0                          # player-ball distance that counts as "on the ball"
TIME_PENALTY_M_PER_S = 0.1                    # tie-break: prefer frames near the stated time
TOLERANCE_FRAMES = 5                          # an event may sit this close outside its spell
SET_PIECES = {"KickOff", "ThrowIn", "FreeKick", "CornerKick", "GoalKick", "Penalty"}
CONTROL_TACKLE_RESULTS = {"ballClaimed", "ballControlRetained"}


def _epoch(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


def _paths(match_id: str) -> tuple[str, str]:
    found = glob.glob(os.path.join(dfl.DFL_DIR, "DFL_03_02_events_raw_*{}.xml".format(match_id)))
    if not found:
        raise FileNotFoundError("DFL events of {} not found -- run scripts/fetch_dfl.sh".format(match_id))
    return found[0], dfl._paths(match_id)[1]


_FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)"[^>]*PersonId="([^"]+)"')
_FIRST = re.compile(r'<Frame N="(\d+)" T="([^"]+)"')


def tracking_meta(positions_path: str) -> dict:
    """Player ids in tracking slot order, and each half's ball clock (first and last frame N, T)."""
    players, clock, current, last = set(), {}, None, None
    with open(positions_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<FrameSet "):
                section, team_id, person_id = _FRAMESET.match(line).groups()
                current = (section, team_id)
                if team_id not in ("BALL", "referee"):
                    players.add((team_id, person_id))
                last = None
            elif current is not None and current[1] == "BALL" and line.startswith("<Frame "):
                n, t = _FIRST.match(line).groups()
                entry = clock.setdefault(current[0], {"n0": int(n), "t0": _epoch(t)})
                last = (int(n), t)
                entry["n1"], entry["t1"] = last[0], _epoch(last[1])
    return {"players": sorted(players), "clock": clock}


def _team_letter(team_id, home_id):
    return "H" if team_id == home_id else "A"


def parse_events(events_path: str, home_id: str) -> list[dict]:
    """Ball events with a player: kind, team (H/A), player id, stated time, control flag."""
    out = []
    for event in ET.parse(events_path).getroot().iter("Event"):
        time = _epoch(event.get("EventTime"))
        set_piece = any(child.tag in SET_PIECES for child in event)
        for node in event.iter():
            row = None
            if node.tag == "Play" and node.get("Player"):
                row = {"kind": "play", "team": node.get("Team"), "player": node.get("Player"), "control": True,
                       "detail": node.get("Evaluation", "")}
            elif node.tag == "ShotAtGoal" and node.get("Player"):
                row = {"kind": "shot", "team": node.get("Team"), "player": node.get("Player"), "control": True,
                       "detail": ""}
            elif node.tag == "BallClaiming" and node.get("Player"):
                row = {"kind": "claim", "team": node.get("Team"), "player": node.get("Player"), "control": True,
                       "detail": node.get("Type", "")}
            elif node.tag == "OtherBallAction" and node.get("Player"):
                row = {"kind": "other", "team": node.get("Team"), "player": node.get("Player"), "control": False,
                       "detail": ""}
            elif node.tag == "TacklingGame" and node.get("Winner"):
                result = node.get("WinnerResult", "")
                won = node.get("PossessionChange") == "true" or result in CONTROL_TACKLE_RESULTS
                row = {"kind": "tackle", "team": node.get("WinnerTeam"), "player": node.get("Winner"),
                       "control": won, "detail": result}
                if node.get("LoserRole") == "withBallControl" and node.get("Loser"):
                    # the loser had the ball under control when the duel started
                    out.append({"kind": "tackle_loser", "team": _team_letter(node.get("LoserTeam"), home_id),
                                "player": node.get("Loser"), "control": True, "detail": result,
                                "time": time, "set_piece": set_piece})
            if row is not None:
                row.update(team=_team_letter(row["team"], home_id), time=time, set_piece=set_piece)
                out.append(row)
    return out


def synchronise(rows: list[dict], tracking: dict, meta: dict) -> list[dict]:
    """Adds frame (timeline index, -1 if not found), offset_s and distance_m to every row."""
    slot_of = {person: k for k, (_, person) in enumerate(meta["players"])}
    half, ball, pos = tracking["half"], tracking["ball"], tracking["pos"]
    starts = dfl.half_starts(half)
    out = []
    for row in rows:
        row = dict(row, frame=-1, offset_s=float("nan"), distance_m=float("nan"))
        for number, section in enumerate(dfl.SECTIONS, start=1):
            clock = meta["clock"][section]
            if not clock["t0"] - 60 <= row["time"] <= clock["t1"] + 60:
                continue
            stated = starts[number] + int(round((row["time"] - clock["t0"]) * FPS))
            lo = max(stated - int(WINDOW_BEFORE_S * FPS), 0)
            hi = min(stated + int(WINDOW_AFTER_S * FPS) + 1, len(half))
            slot = slot_of.get(row["player"])
            if slot is None or hi <= lo:
                break
            distance = np.linalg.norm(pos[lo:hi, slot] - ball[lo:hi], axis=1)
            distance[half[lo:hi] != number] = np.nan
            if np.isnan(distance).all():
                break
            score = distance + TIME_PENALTY_M_PER_S * np.abs(np.arange(lo, hi) - stated) / FPS
            best = int(np.nanargmin(score))
            if distance[best] <= MAX_DISTANCE_M:
                row.update(frame=lo + best, offset_s=(lo + best - stated) / FPS, distance_m=float(distance[best]))
            break
        out.append(row)
    return out


def events(match_id: str, use_cache: bool = True) -> list[dict]:
    """Parsed and synchronised events of one DFL match (cached as JSON next to the tracking cache)."""
    cache = os.path.join(dfl.CACHE_DIR, "{}_events.json".format(match_id))
    if use_cache and os.path.isfile(cache):
        with open(cache, encoding="utf-8") as handle:
            return json.load(handle)
    events_path, positions_path = _paths(match_id)
    info_path = dfl._paths(match_id)[0]
    rows = parse_events(events_path, dfl.home_team_id(info_path))
    out = synchronise(rows, dfl.load(match_id), tracking_meta(positions_path))
    if use_cache:
        os.makedirs(dfl.CACHE_DIR, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as handle:
            json.dump(out, handle)
    return out


def control_frames(rows: list[dict]) -> dict:
    """{"H": sorted frames, "A": sorted frames} of synchronised control events."""
    return {team: np.array(sorted(r["frame"] for r in rows if r["control"] and r["frame"] >= 0 and r["team"] == team),
                           dtype=np.int64) for team in ("H", "A")}


def event_labels(labels: np.ndarray, control: dict, min_control_s: float, fps: int = FPS,
                 tolerance: int = TOLERANCE_FRAMES) -> np.ndarray:
    """dfl.controlled_labels, except that a short spell is kept when its own team shows control in it.

    A live spell shorter than `min_control_s` that follows live play is folded into the possession
    before it (an uncontrolled touch) -- unless a synchronised control event of the spell's team
    (a pass, a shot, a ball claim, a duel won with the ball, a duel lost while in control) falls
    inside it, give or take `tolerance` frames. Then it is a real, short possession and stays.
    """
    out = labels.copy()
    min_frames = int(round(min_control_s * fps))
    change = np.concatenate(([True], out[1:] != out[:-1]))
    starts = np.where(change)[0]
    ends = np.concatenate((starts[1:], [len(out)]))
    previous = None
    for start, end in zip(starts, ends):
        state = out[start]
        if state in ("H", "A") and previous in ("H", "A") and end - start < min_frames:
            frames = control[state]
            first = np.searchsorted(frames, start - tolerance)
            if not (first < len(frames) and frames[first] < end + tolerance):
                out[start:end] = previous
                continue
        previous = state if state != "X" else None
    return out
