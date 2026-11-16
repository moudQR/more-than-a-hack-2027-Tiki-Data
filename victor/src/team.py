"""Decide which team has the ball on each frame."""

import numpy as np

# Params
HOME, AWAY, NOBODY = 0, 1, -1
CHUNK_FRAMES = 20000  # squared distances are built in chunks to keep memory flat


def nearest_team(data):
    """Team of the tracked player closest to the ball, NOBODY where there is no answer.

    The same rule as starter/baseline.py, on the cached arrays.
    """
    ball_xy, player_xy = data["ball_xy"], data["player_xy"]
    team_of_slot = data["team_of_slot"]
    out = np.full(len(ball_xy), NOBODY, dtype=np.int8)

    for start in range(0, len(ball_xy), CHUNK_FRAMES):
        stop = start + CHUNK_FRAMES
        offsets = player_xy[start:stop] - ball_xy[start:stop, np.newaxis, :]
        distances = np.sum(offsets * offsets, axis=2)
        known = ~np.isnan(distances).all(axis=1)
        nearest = np.argmin(np.where(np.isnan(distances), np.inf, distances), axis=1)
        out[start:stop][known] = team_of_slot[nearest[known]]
    return out
