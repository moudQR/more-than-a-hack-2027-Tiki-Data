"""Clean the ball path: drop impossible points, then fill short gaps."""

import numpy as np
from scipy.ndimage import median_filter

# Params
MEDIAN_POINTS = 5       # seen points in the local median each point is compared with
MAX_JUMP_M = 5.0        # further than this from the local median is not the ball
MAX_FILL_FRAMES = 25    # gaps up to 1 s are filled with a straight line


def drop_jumps(ball_xy, median_points=MEDIAN_POINTS, max_jump=MAX_JUMP_M):
    """A copy with points far from their seen neighbours set to NaN."""
    out = ball_xy.copy()
    seen = np.flatnonzero(~np.isnan(ball_xy[:, 0]))
    if len(seen) < median_points:
        return out
    points = ball_xy[seen]
    local = median_filter(points, size=(median_points, 1), mode="nearest")
    out[seen[np.linalg.norm(points - local, axis=1) > max_jump]] = np.nan
    return out


def fill_short_gaps(ball_xy, max_fill=MAX_FILL_FRAMES):
    """A copy with gaps of up to max_fill frames bridged by a straight line."""
    out = ball_xy.copy()
    seen = np.flatnonzero(~np.isnan(ball_xy[:, 0]))
    if len(seen) < 2:
        return out
    for start, stop in zip(seen[:-1], seen[1:]):
        gap = stop - start - 1
        if 0 < gap <= max_fill:
            steps = np.arange(1, gap + 1)[:, np.newaxis] / float(gap + 1)
            out[start + 1:stop] = ball_xy[start] + steps * (ball_xy[stop] - ball_xy[start])
    return out


def clean(ball_xy, max_fill=MAX_FILL_FRAMES):
    return fill_short_gaps(drop_jumps(ball_xy), max_fill)
