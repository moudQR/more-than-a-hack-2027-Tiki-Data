"""Find touches: moments the ball changes speed or direction with a player next to it.

Control is what separates a change of possession from a deflection, and the challenge suggests
consecutive passes as a proxy for it. So the clues here say who touched the ball last, who touches
it next and how many touches in a row one team has made.

Nothing here is a final answer, because touches are missed wherever the ball is unseen. The clues
are handed to the model alongside everything else.
"""

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

# Params
FPS = 25
SMOOTH_FRAMES = 5           # the path is averaged first, or position noise looks like touches
STEP_FRAMES = 5             # velocity before and after a frame is measured over 0.2 s
TURN_QUANTILE = 0.85        # the sharpest changes of each match count, because the size of a
                            # change depends on how smooth that tracking system is
PEAK_FRAMES = 11            # one touch per this many frames
MAX_TOUCH_DISTANCE = 3.0    # metres from the nearest player
MAX_AGE_FRAMES = 250        # clues stop counting 10 s away from a touch
NO_TEAM = -1


def smooth_path(ball_xy):
    """Moving average of the ball path, ignoring gaps. NaN stays NaN."""
    seen = ~np.isnan(ball_xy[:, 0])
    weight = uniform_filter1d(seen.astype(np.float64), SMOOTH_FRAMES, mode="constant")
    total = uniform_filter1d(np.nan_to_num(ball_xy), SMOOTH_FRAMES, axis=0, mode="constant")
    out = np.full_like(ball_xy, np.nan)
    out[seen] = total[seen] / weight[seen, np.newaxis]
    return out


def velocity_change(ball_xy):
    """How much the ball's velocity changes at each frame, m/s. NaN where the ball is unknown."""
    smooth = smooth_path(ball_xy)
    step = STEP_FRAMES
    before = np.full_like(smooth, np.nan)
    after = np.full_like(smooth, np.nan)
    before[step:] = smooth[step:] - smooth[:-step]
    after[:-step] = smooth[step:] - smooth[:-step]
    return np.linalg.norm(after - before, axis=1) * FPS / step


def detect(ball_xy, near_home, near_away, blocks):
    """(frames, teams) of the touches found. teams is 0 home, 1 away, nearest player's side.

    The threshold comes from the two halves alone, because the ball track outside them is erratic
    and would push it too high.
    """
    turn = velocity_change(ball_xy)
    filled = np.nan_to_num(turn)
    in_play = np.zeros(len(turn), dtype=bool)
    for _, first, last in blocks:
        in_play[first:last + 1] = True
    threshold = np.quantile(filled[in_play & ~np.isnan(turn)], TURN_QUANTILE)
    peak = (filled == maximum_filter1d(filled, PEAK_FRAMES)) & (filled >= threshold) & in_play

    nearest = np.fmin(near_home, near_away)
    with np.errstate(invalid="ignore"):
        close = nearest <= MAX_TOUCH_DISTANCE
    frames = np.flatnonzero(peak & close)
    teams = (near_away[frames] < near_home[frames]).astype(np.int8)
    return frames, teams


def distance_to_touch(frames, n_frames, forward):
    """Frames until the next touch (forward) or since the last one, capped at MAX_AGE_FRAMES."""
    out = np.full(n_frames, float(MAX_AGE_FRAMES), dtype=np.float32)
    if not len(frames):
        return out
    index = np.arange(n_frames)
    if forward:
        slots = np.searchsorted(frames, index, side="left")
        known = slots < len(frames)
        out[known] = frames[slots[known]] - index[known]
    else:
        slots = np.searchsorted(frames, index, side="right") - 1
        known = slots >= 0
        out[known] = index[known] - frames[slots[known]]
    return np.minimum(out, MAX_AGE_FRAMES)


def run_of_touches(teams):
    """For each touch, how many touches in a row its team has made, counting this one."""
    runs = np.ones(len(teams), dtype=np.int32)
    for i in range(1, len(teams)):
        if teams[i] == teams[i - 1]:
            runs[i] = runs[i - 1] + 1
    return runs


def clues(frames, teams, n_frames):
    """Paired h_ and a_ columns, as a dict of arrays over frames."""
    out = {}
    for side, code in (("h", 0), ("a", 1)):
        mine = frames[teams == code]
        out[side + "_since_touch"] = distance_to_touch(mine, n_frames, False)
        out[side + "_until_touch"] = distance_to_touch(mine, n_frames, True)

        marks = np.zeros(n_frames, dtype=np.float32)
        marks[mine] = 1.0
        for half_width in (50, 125):
            out["{}_touches_r{}".format(side, half_width)] = uniform_filter1d(
                marks, 2 * half_width + 1, mode="constant") * (2 * half_width + 1)

    runs = run_of_touches(teams)
    for side, code in (("h", 0), ("a", 1)):
        behind = np.zeros(n_frames, dtype=np.float32)
        ahead = np.zeros(n_frames, dtype=np.float32)
        if len(frames):
            index = np.arange(n_frames)
            last = np.searchsorted(frames, index, side="right") - 1
            has_last = last >= 0
            mine = has_last.copy()
            mine[has_last] = teams[last[has_last]] == code
            behind[mine] = runs[last[mine]]

            upcoming = last + 1
            has_next = upcoming < len(frames)
            mine = has_next.copy()
            mine[has_next] = teams[upcoming[has_next]] == code
            ahead[mine] = runs[upcoming[mine]]
        out[side + "_run_behind"] = np.minimum(behind, 10)
        out[side + "_run_ahead"] = np.minimum(ahead, 10)
    return out
