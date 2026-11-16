"""Read the raw ATD once and keep it as arrays, so nothing else has to wait for the text file.

    python src/parse.py                 # builds cache/BAR-ATH_tactical.npz

Positions are in metres, using the pitch in games.py. Arrays are indexed by video frame, so row i
is row i of a submission. Missing observations stay NaN.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))   # the shared repository, for games.py
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "starter"))
from baseline import FIELD_SEP, GROUP_SEP, MISSING, team_of_each_track  # noqa: E402
from games import DEFAULT_SOURCE, GAME, GAMES, require, source_frames  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")


def cache_path(game=GAME, source=DEFAULT_SOURCE):
    return os.path.join(CACHE_DIR, "{}_{}.npz".format(game, source))


def read_atd(raw_path, n_slots, n_frames, pitch):
    """(ball_xy, ball_speed, player_xy, player_speed) from one pass over the raw file.

    frame_id counts from 1 and arrays from 0, so a record lands on row frame_id - 1.
    """
    length, width = pitch
    ball_xy = np.full((n_frames, 2), np.nan)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    player_xy = np.full((n_frames, n_slots, 2), np.nan)
    player_speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip().split(FIELD_SEP)
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1

            groups = parts[1].split(GROUP_SEP)
            if len(groups) != n_slots:
                raise ValueError("frame_id {}: {} player groups, expected {}".format(
                    parts[0], len(groups), n_slots))
            for slot, group in enumerate(groups):
                if group[0] == MISSING:
                    continue
                fields = group.split(",")
                player_xy[row, slot, 0] = float(fields[0]) * length
                player_xy[row, slot, 1] = float(fields[1]) * width
                player_speed[row, slot] = float(fields[2])

            if parts[2][0] != MISSING:
                fields = parts[2].split(",")
                ball_xy[row, 0] = float(fields[0]) * length
                ball_xy[row, 1] = float(fields[1]) * width
                ball_speed[row] = float(fields[2])

    return ball_xy, ball_speed, player_xy, player_speed


def build(game=GAME, source=DEFAULT_SOURCE):
    """Parse the raw ATD and write the cache. Takes a minute or two."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, source=source)
    team_of_slot = team_of_each_track(meta_path)
    arrays = read_atd(raw_path, len(team_of_slot), source_frames(source, game),
                      GAMES[game]["pitch"])

    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    path = cache_path(game, source)
    np.savez(path, ball_xy=arrays[0], ball_speed=arrays[1], player_xy=arrays[2],
             player_speed=arrays[3], team_of_slot=team_of_slot)
    return path


def load(game=GAME, source=DEFAULT_SOURCE, rebuild=False):
    """The cached arrays as a dict, building them the first time.

    ball_xy       (frames, 2)         metres
    ball_speed    (frames,)           as in the ATD
    player_xy     (frames, slots, 2)  metres
    player_speed  (frames, slots)     as in the ATD
    team_of_slot  (slots,)            0 home, 1 away
    """
    path = cache_path(game, source)
    if rebuild or not os.path.isfile(path):
        build(game, source)
    with np.load(path) as cached:
        return {name: cached[name] for name in cached.files}


if __name__ == '__main__':
    data = load(rebuild=True)
    print("wrote {}".format(os.path.relpath(cache_path(), ROOT)))
    print("{} frames, {} track slots".format(*data["player_xy"].shape[:2]))
