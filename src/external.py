"""Turn the open IDSSE matches into the same arrays as src/parse.py, plus their labels.

These are seven Bundesliga matches from the open DFL dataset (Bassek et al. 2025, CC-BY 4.0). They
are other games, so their human-made labels are allowed for training and for choosing between
versions, which the provided match cannot offer. The files live in external/idsse and are not
committed:

    https://huggingface.co/datasets/pysport/idsse-data

    DFL_02_01_matchinformation_*.xml      home and away team ids
    DFL_04_03_positions_raw_observed_*    one FrameSet per person per half, 25 fps

Row layout: first half, one separator row, second half. The separator is NaN in the arrays and X
in the labels, which is what keeps the scorer from joining the two halves.

The labels follow the challenge: D where BallStatus is 0, otherwise the team in BallPossession.
Their definition of possession is not the challenge's, so a score here is a guide, not a forecast.
"""

import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from team import AWAY, HOME  # noqa: E402

# Params
SOURCE_DIR = os.path.join(ROOT, "external", "idsse")
CACHE_DIR = os.path.join(ROOT, "cache", "external")
HALF_LENGTH, HALF_WIDTH = 52.5, 34.0  # DFL coordinates are centred, the ATD's start at a corner
KMH_TO_MS = 1 / 3.6  # DFL speeds are km/h, the ATD's are m/s
HOME_POSSESSION, DEAD_STATUS = 1, 0
SECTIONS = ("firstHalf", "secondHalf")

MATCH_ID = re.compile(r"DFL-MAT-(\w+)\.xml$")
FRAMESET = re.compile(r'GameSection="(\w+)".*TeamId="([^"]+)" PersonId="([^"]+)"')
FRAME = re.compile(r'N="(\d+)".* X="([-\d.]+)" Y="([-\d.]+)"(?: Z="[-\d.]+")? D="[-\d.]+" '
                   r'S="([-\d.]+)"(?:.*BallPossession="(\d)" BallStatus="(\d)")?')


def match_ids():
    return sorted(MATCH_ID.search(path).group(1) for path in glob.glob(
        os.path.join(SOURCE_DIR, "DFL_04_03_positions_raw_observed_*.xml")))


def source_file(kind, match_id):
    return glob.glob(os.path.join(SOURCE_DIR, "DFL_{}_*{}.xml".format(kind, match_id)))[0]


def home_team_id(match_id):
    root = ET.parse(source_file("02_01_matchinformation", match_id)).getroot()
    return next(team.get("TeamId") for team in root.iter("Team") if team.get("Role") == "home")


def read_framesets(path):
    """{(section, team, person): (frames, x, y, speed, possession, status)} from one pass."""
    framesets, current = {}, None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                current.append(FRAME.search(line).groups())
            elif line.startswith("<FrameSet "):
                current = []
                framesets[FRAMESET.search(line).groups()] = current

    out = {}
    for key, rows in framesets.items():
        columns = list(zip(*rows))
        is_ball = columns[4][0] is not None
        out[key] = (np.array(columns[0], dtype=np.int64),
                    np.array(columns[1], dtype=np.float64),
                    np.array(columns[2], dtype=np.float64),
                    np.array(columns[3], dtype=np.float32),
                    np.array(columns[4], dtype=np.int8) if is_ball else None,
                    np.array(columns[5], dtype=np.int8) if is_ball else None)
    return out


def convert(match_id):
    """The same arrays as parse.load, plus labels and half blocks, saved to the cache."""
    framesets = read_framesets(source_file("04_03_positions_raw_observed", match_id))
    home_id = home_team_id(match_id)

    ball = {key[0]: value for key, value in framesets.items() if key[1] == "BALL"}
    people = sorted({key[1:] for key in framesets if key[1] not in ("BALL", "referee")})
    slot_of = {person: slot for slot, person in enumerate(people)}

    first_frame, row_of_start, blocks, row = {}, {}, [], 0
    for half, section in enumerate(SECTIONS, start=1):
        frames = ball[section][0]
        first_frame[section], row_of_start[section] = frames[0], row
        blocks.append((half, row, row + len(frames) - 1))
        row += len(frames) + 1
    n_rows = row - 1

    ball_xy = np.full((n_rows, 2), np.nan)
    ball_speed = np.full(n_rows, np.nan, dtype=np.float32)
    player_xy = np.full((n_rows, len(people), 2), np.nan)
    player_speed = np.full((n_rows, len(people)), np.nan, dtype=np.float32)
    labels = np.full(n_rows, "X", dtype="<U1")

    for section in SECTIONS:
        frames, x, y, speed, possession, status = ball[section]
        rows = frames - first_frame[section] + row_of_start[section]
        ball_xy[rows, 0], ball_xy[rows, 1] = x + HALF_LENGTH, y + HALF_WIDTH
        ball_speed[rows] = speed * KMH_TO_MS
        labels[rows] = np.where(status == DEAD_STATUS, "D",
                                np.where(possession == HOME_POSSESSION, "H", "A"))

    for (section, team_id, person), value in framesets.items():
        if (team_id, person) not in slot_of:
            continue
        frames, x, y, speed = value[:4]
        rows = frames - first_frame[section] + row_of_start[section]
        keep = (rows >= row_of_start[section]) & (rows < n_rows)
        slot = slot_of[(team_id, person)]
        player_xy[rows[keep], slot, 0] = x[keep] + HALF_LENGTH
        player_xy[rows[keep], slot, 1] = y[keep] + HALF_WIDTH
        player_speed[rows[keep], slot] = speed[keep] * KMH_TO_MS

    team_of_slot = np.array([HOME if team_id == home_id else AWAY for team_id, _ in people],
                            dtype=np.int8)
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    path = os.path.join(CACHE_DIR, match_id + ".npz")
    np.savez(path, ball_xy=ball_xy, ball_speed=ball_speed, player_xy=player_xy,
             player_speed=player_speed, team_of_slot=team_of_slot, labels=labels,
             blocks=np.array(blocks, dtype=np.int64))
    return path


def load(match_id):
    """One match as a dict, converting it the first time. blocks are (half, first, last) rows."""
    path = os.path.join(CACHE_DIR, match_id + ".npz")
    if not os.path.isfile(path):
        convert(match_id)
    with np.load(path) as cached:
        return {name: cached[name] for name in cached.files}


if __name__ == '__main__':
    for match_id in match_ids():
        data = load(match_id)
        counts = {label: int(np.sum(data["labels"] == label)) for label in "HADX"}
        print(match_id, data["player_xy"].shape[:2], counts)
