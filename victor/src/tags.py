"""Smart Tagging as an independent check on the provided match.

The tags are automatic, not human-made, so comparing against them is allowed where the public
ground truth is not. They are phases, not possessions, so full agreement is neither possible nor
the goal: this shows whether a version behaves sensibly on the real tracking.

    Attacking phases carry the team that has the ball, and defending phases the other one.
    A set piece is a restart, so the moments before its end should often be dead.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "starter"))
import load_smart_tagging  # noqa: E402

# Params
TEAM_LABEL = {"barcelona": "H", "atletic club": "A"}
OTHER_TEAM = {"H": "A", "A": "H"}
ATTACKING = {"build up", "progression", "creating chances", "attacking transition"}
DEFENDING = {"defending in defensive third", "defending in middle third",
             "defending in attacking third", "defensive transition"}
SET_PIECE = "set pieces"


def expected_team(event):
    """The side the tag says has the ball, or None if it does not say."""
    side = TEAM_LABEL.get((event["tags"].get("Team") or "").strip().lower())
    code = (event["code"] or "").strip().lower()
    if side is None:
        return None
    if code in ATTACKING:
        return side
    if code in DEFENDING:
        return OTHER_TEAM[side]
    return None


def agreement(labels):
    """How a set of frame labels sits with the tags. Three shares, computed over frames."""
    events = load_smart_tagging.load()
    agree = total = alive_total = dead_in_sets = set_frames = 0
    for event in events:
        window = labels[event["start_frame"]:event["end_frame"]]
        side = expected_team(event)
        if side is not None:
            total += len(window)
            alive = np.isin(window, ("H", "A"))
            alive_total += int(alive.sum())
            agree += int(np.sum(window == side))
        elif (event["code"] or "").strip().lower() == SET_PIECE:
            set_frames += len(window)
            dead_in_sets += int(np.sum(window == "D"))
    return {"team_agreement": agree / float(max(alive_total, 1)),
            "phase_frames_alive": alive_total / float(max(total, 1)),
            "set_piece_frames_dead": dead_in_sets / float(max(set_frames, 1))}
