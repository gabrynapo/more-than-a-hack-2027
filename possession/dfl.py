"""DFL / Sportec open data (Bassek et al., Scientific Data 2025, CC-BY 4.0) as possession training
and validation material.

Seven full Bundesliga matches, 25 fps, corrected optical tracking. Every ball frame carries the
provider's own `BallStatus` (1 alive, 0 dead) and `BallPossession` (1 home, 2 guest) -- the same kind
of label the competition ground truth is: dead where the corrected tracking marks the ball dead,
otherwise the team the tracking names as owning it.

This module turns one match into exactly what `possession.tracking.load()` returns for the
competition match (dense arrays, metres, origin at a pitch corner), plus a frame label array and a
half index, so the whole pipeline can run on it unchanged:

    timeline   [PAD ungraded] [first half] [PAD ungraded] [second half] [PAD ungraded]
    labels     "X" on the pads, "H" / "A" / "D" inside the halves
    half       0 on the pads, 1 / 2 inside

The pads keep rolling windows from bleeding across halftime, exactly like the real halftime does.

Download the data with scripts/fetch_dfl.sh. It is public, so unlike the competition match it may
be inspected freely.
"""

from __future__ import annotations

import glob
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

from possession.tracking import AWAY, HOME

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFL_DIR = os.path.join(ROOT, "external", "dfl")
CACHE_DIR = os.path.join(DFL_DIR, "cache")
MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
PAD = 250                       # ungraded frames around each half (10 s)
KMH_TO_MS = 1 / 3.6             # DFL `S` is km/h (3.60x the displacement speed); the ATD is m/s
SECTIONS = ("firstHalf", "secondHalf")

_FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)"[^>]*PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"(?: Z="[-\d.]+")?'
                    r'(?: D="[-\d.]+")? S="([-\d.]+)"')
_BALL_FLAGS = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def _paths(match_id: str) -> tuple[str, str]:
    info = glob.glob(os.path.join(DFL_DIR, "DFL_02_01_matchinformation_*{}.xml".format(match_id)))
    positions = glob.glob(os.path.join(DFL_DIR, "DFL_04_03_positions_raw_observed_*{}.xml".format(match_id)))
    if not info or not positions:
        raise FileNotFoundError("DFL match {} not found in {} -- run scripts/fetch_dfl.sh".format(
            match_id, DFL_DIR))
    return info[0], positions[0]


def home_team_id(info_path: str) -> str:
    general = ET.parse(info_path).getroot().find(".//General")
    return general.get("HomeTeamId")


def parse_positions(path: str) -> dict:
    """{(section, team_id, person_id): (frames, x, y, speed[, possession, status])} as numpy arrays.

    A line-oriented regex pass: the file is one <Frame/> per line and ~400 MB, which a DOM parser
    would hold in memory several times over.
    """
    sets: dict = {}
    key = None
    rows: list = []
    flags: list = []

    def close() -> None:
        if key is None or not rows:
            return
        table = np.array(rows, dtype=np.float64)
        entry = [table[:, 0].astype(np.int64), table[:, 1], table[:, 2], table[:, 3]]
        if flags:
            ball = np.array(flags, dtype=np.int8)
            entry += [ball[:, 0], ball[:, 1]]
        sets[key] = tuple(entry)

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                match = _FRAME.match(line)
                if match is None:
                    raise ValueError("unparsed frame line in {}: {}".format(path, line[:120]))
                rows.append(match.groups())
                if key[1] == "BALL":
                    ball = _BALL_FLAGS.search(line)
                    flags.append(ball.groups() if ball else (0, 0))
            elif line.startswith("<FrameSet "):
                close()
                key, rows, flags = _FRAMESET.match(line).groups(), [], []
            elif line.startswith("<PitchSize "):
                size = re.search(r'X="([\d.]+)" Y="([\d.]+)"', line)
                sets["pitch"] = (float(size.group(1)), float(size.group(2)))
    close()
    return sets


def assemble(sets: dict, home_id: str) -> dict:
    """Dense arrays on one padded timeline. See the module docstring for the layout."""
    length, width = sets["pitch"]
    ball_sets = {key[0]: value for key, value in sets.items() if key != "pitch" and key[1] == "BALL"}

    # Each half spans exactly the ball's frames, which run kickoff to final whistle.
    spans, offset, offsets = {}, PAD, {}
    for section in SECTIONS:
        frames = ball_sets[section][0]
        spans[section] = (int(frames.min()), int(frames.max()))
        offsets[section] = offset
        offset += spans[section][1] - spans[section][0] + 1 + PAD
    n_frames = offset

    players = sorted({(key[1], key[2]) for key in sets
                      if key != "pitch" and key[1] not in ("BALL", "referee")})
    slot_of = {player: k for k, player in enumerate(players)}
    team = np.array([HOME if team_id == home_id else AWAY for team_id, _ in players], dtype=np.int8)

    pos = np.full((n_frames, len(players), 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, len(players)), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    labels = np.full(n_frames, "X", dtype="<U1")
    half = np.zeros(n_frames, dtype=np.int8)

    def rows_of(section: str, frames: np.ndarray) -> np.ndarray:
        return offsets[section] + frames - spans[section][0]

    for number, section in enumerate(SECTIONS, start=1):
        first = offsets[section]
        last = first + spans[section][1] - spans[section][0]
        half[first:last + 1] = number
        labels[first:last + 1] = "D"            # frames the ball record skips stay dead

        frames, x, y, _, possession, status = ball_sets[section]
        index = rows_of(section, frames)
        ball[index, 0] = x + length / 2
        ball[index, 1] = y + width / 2
        alive = status == 1
        labels[index[alive & (possession == 1)]] = "H"
        labels[index[alive & (possession == 2)]] = "A"

    for key, value in sets.items():
        if key == "pitch" or key[1] in ("BALL", "referee"):
            continue
        section, team_id, person_id = key
        frames, x, y, s = value[:4]
        index = rows_of(section, frames)
        inside = (index >= 0) & (index < n_frames)
        slot = slot_of[(team_id, person_id)]
        pos[index[inside], slot, 0] = x[inside] + length / 2
        pos[index[inside], slot, 1] = y[inside] + width / 2
        speed[index[inside], slot] = s[inside] * KMH_TO_MS

    return {
        "pos": pos, "speed": speed, "team": team, "ball": ball,
        "ball_extra": np.full(n_frames, np.nan, dtype=np.float32),
        "pitch": np.array([length, width], dtype=np.float32),
        "labels": labels, "half": half,
    }


def load(match_id: str, use_cache: bool = True) -> dict:
    """One DFL match as a tracking dict (see `assemble`), cached as .npz under external/dfl/cache."""
    cache = os.path.join(CACHE_DIR, "{}.npz".format(match_id))
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as data:
            return {key: data[key] for key in data.files}
    info_path, positions_path = _paths(match_id)
    out = assemble(parse_positions(positions_path), home_team_id(info_path))
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, **out)
    return out


def controlled_labels(labels: np.ndarray, min_control_s: float, fps: int = 25) -> np.ndarray:
    """Labels with uncontrolled touches folded back into the possession they interrupted.

    The DFL `BallPossession` flag follows every touch: deflections, duels and blocked passes all
    flip it, so its possessions are short (median ~3 s, 17% under a second). The competition counts
    a change only on *controlled* possession -- "a simple touch of the ball without control", "a
    deflection of a pass that does not intercept it" do not count (CHALLENGE.md). So a live spell
    shorter than `min_control_s` that follows live play is given to the team before it: possession
    stays with the last team that had it. Spells after a stoppage are restarts and stay as they
    are. One left-to-right pass, so a chain of short flips collapses into the possession it broke.
    """
    out = labels.copy()
    min_frames = int(round(min_control_s * fps))
    if min_frames <= 0:
        return out
    change = np.concatenate(([True], out[1:] != out[:-1]))
    starts = np.where(change)[0]
    ends = np.concatenate((starts[1:], [len(out)]))
    previous = None                      # label of the last output spell, None at a boundary
    for start, end in zip(starts, ends):
        state = out[start]
        if state in ("H", "A") and previous in ("H", "A") and end - start < min_frames:
            out[start:end] = previous
            continue
        previous = state if state != "X" else None
    return out


def half_starts(half: np.ndarray) -> dict:
    """{half number: first frame}, the counterpart of games.half_video_blocks for a DFL timeline."""
    return {number: int(np.argmax(half == number)) for number in (1, 2) if (half == number).any()}


if __name__ == "__main__":
    import time
    for match_id in MATCH_IDS:
        start = time.time()
        data = load(match_id)
        labels = data["labels"][data["half"] > 0]
        alive = labels != "D"
        changes = np.sum(labels[1:] != labels[:-1])
        print("{}  {:>6} graded frames  D {:.1%}  H share of alive {:.1%}  label changes {}  "
              "slots {}  ({:.0f}s)".format(match_id, len(labels), np.mean(~alive),
                                         np.mean(labels[alive] == "H"), changes,
                                         len(data["team"]), time.time() - start))
