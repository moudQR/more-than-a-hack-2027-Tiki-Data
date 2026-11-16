"""Decide whether the ball is in play, from how the tracking behaves.

Three things go wrong with play at a stoppage: everyone slows down, the ball leaves the pitch or
stops moving, and the tracking tends to lose it. None is reliable alone, so a frame is called dead
when the signals agree, and very short stretches of either kind are then removed.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

# Params
PITCH = (105.0, 68.0)
OUTSIDE_MARGIN_M = 1.0      # a ball this far beyond a line has left the pitch
SPEED_WINDOW = 51           # frames averaged for the team speed, 2 s
STILL_WINDOW = 25           # frames the ball must stay put, 1 s
STILL_DISTANCE_M = 1.0

SLOW_SPEED = 1.2            # m/s, mean over all tracked players: slow enough on its own
WALKING_SPEED = 2.2         # m/s: slow enough when the ball is also unseen, out or still
MIN_GAP_FRAMES = 50         # the ball unseen this long counts as a signal
MIN_DEAD_FRAMES = 75        # dead stretches shorter than this are ignored, 3 s
MIN_ALIVE_FRAMES = 75       # live stretches shorter than this are ignored, 3 s


def team_speed(player_speed):
    """Mean speed of all tracked players, smoothed. Frames with no one tracked count as zero."""
    tracked = ~np.isnan(player_speed)
    total = np.where(tracked, player_speed, 0).sum(axis=1)
    return uniform_filter1d(total / np.maximum(tracked.sum(axis=1), 1), SPEED_WINDOW,
                            mode="nearest")


def unseen_length(ball_xy):
    """For each frame, how long the gap it sits in is. 0 where the ball is seen."""
    missing = np.isnan(ball_xy[:, 0]).astype(np.int8)
    edges = np.diff(np.concatenate(([0], missing, [0])))
    out = np.zeros(len(ball_xy), dtype=np.int32)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[start:stop] = stop - start
    return out


def ball_outside(ball_xy, pitch=PITCH, margin=OUTSIDE_MARGIN_M):
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > pitch[0] + margin) | (y < -margin) | (y > pitch[1] + margin)


def ball_still(ball_xy, window=STILL_WINDOW, distance=STILL_DISTANCE_M):
    """Frames where the ball is seen and has barely moved over the window."""
    moved = np.full(len(ball_xy), np.nan)
    moved[window:] = np.linalg.norm(ball_xy[window:] - ball_xy[:-window], axis=1)
    with np.errstate(invalid="ignore"):
        return moved < distance


def remove_short(flags, min_length, value):
    """Flip stretches of `value` shorter than min_length."""
    out = flags.copy()
    marked = np.concatenate(([0], (flags == value).astype(np.int8), [0]))
    edges = np.diff(marked)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start < min_length:
            out[start:stop] = not value
    return out


def detect(data, ball_xy, slow=SLOW_SPEED, walking=WALKING_SPEED, min_gap=MIN_GAP_FRAMES,
           min_dead=MIN_DEAD_FRAMES, min_alive=MIN_ALIVE_FRAMES):
    """Boolean per frame, True where the ball is dead.

    Uses the raw ball for 'unseen', because filling short gaps would hide part of the signal, and
    the cleaned ball for position.
    """
    speed = team_speed(data["player_speed"])
    lost = unseen_length(data["ball_xy"]) >= min_gap
    dead = (speed < slow) | ((speed < walking)
                             & (lost | ball_outside(ball_xy) | ball_still(ball_xy)))
    dead = remove_short(dead, min_dead, True)
    return remove_short(dead, min_alive, False)
