"""Read the raw ATD into dense numpy arrays, indexed by video frame, in metres.

One pass over the 196 MB text file takes a while, so the result is cached as an .npz under
data/cache/. That folder sits inside data/, which is gitignored: frame-level tracking is exactly the
kind of derived data the terms forbid publishing, so it must never leave it.

Layout of a raw line (see DataFormatSpecifications in the .xml):

    <frame_id>:<x,y,speed>;<x,y,speed>;... (46 slots) :<ball_x,ball_y,ball_extra>

frame_id counts from 1, video frames from 0, so row i of every array here is video frame i.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, require, source_frames  # noqa: E402

CACHE_DIR = os.path.join(DATA_DIR, "cache")
HOME, AWAY = 0, 1


def slot_teams(meta_path):
    """Array over track slots: HOME or AWAY. The first <Team> in the file is the home side."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def _parse_raw(raw_path, n_slots, n_frames):
    """All numbers of the file in one vectorised parse.

    Every line has the same number of fields once ':' and ';' are treated like ',', so the whole
    file is one flat float array that reshapes to (frames, 1 + 3 * n_slots + 3).
    """
    with open(raw_path) as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",").replace("\n", ",").rstrip(",")
    flat = np.array(text.split(","), dtype=np.float64)
    width = 1 + 3 * n_slots + 3
    if flat.size != width * n_frames:
        raise ValueError("{}: expected {} x {} numbers, got {}".format(
            raw_path, n_frames, width, flat.size))
    return flat.reshape(n_frames, width)


def load(source="tactical", game=GAME, use_cache=True):
    """dict of arrays for one source:

        pos     (N, slots, 2) float32  player position, metres, origin at a pitch corner
        speed   (N, slots)    float32  m/s as given by the tracker
        team    (slots,)      int8     HOME / AWAY per slot
        ball    (N, 2)        float32  ball position, metres; NaN where unseen
        ball_extra (N,)       float32  the undocumented third ball field
        pitch   (2,)          length, width in metres

    NaN marks every missing observation.
    """
    cache = os.path.join(CACHE_DIR, "{}_{}_tracking.npz".format(game, source))
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as data:
            return {key: data[key] for key in data.files}

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, source=source)
    team = slot_teams(meta_path)
    n_slots, n_frames = len(team), source_frames(source, game)
    table = _parse_raw(raw_path, n_slots, n_frames)

    frame_ids = table[:, 0].astype(np.int64)
    if not np.array_equal(frame_ids, np.arange(1, n_frames + 1)):
        raise ValueError("{}: frame ids are not 1..{}".format(raw_path, n_frames))

    length, width = GAMES[game]["pitch"]
    players = table[:, 1:1 + 3 * n_slots].reshape(n_frames, n_slots, 3)
    pos = np.empty((n_frames, n_slots, 2), dtype=np.float32)
    pos[..., 0] = players[..., 0] * length
    pos[..., 1] = players[..., 1] * width
    ball = np.stack([table[:, -3] * length, table[:, -2] * width], axis=1).astype(np.float32)

    out = {
        "pos": pos,
        "speed": players[..., 2].astype(np.float32),
        "team": team,
        "ball": ball,
        "ball_extra": table[:, -1].astype(np.float32),
        "pitch": np.array([length, width], dtype=np.float32),
    }
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, **out)
    return out


if __name__ == "__main__":
    import time
    start = time.time()
    data = load(use_cache=False)
    print("parsed in {:.1f}s".format(time.time() - start))
    for key, value in data.items():
        print("  {:<11} {} {}".format(key, value.shape, value.dtype))
    extra = data["ball_extra"]
    seen = ~np.isnan(data["ball"][:, 0])
    filled = ~np.isnan(extra)
    print("ball seen {:.1%}; ball_extra filled {} frames ({} while ball seen)".format(
        seen.mean(), filled.sum(), (filled & seen).sum()))
    if filled.any():
        print("  ball_extra percentiles 1/50/99:", np.percentile(extra[filled], [1, 50, 99]))
