"""Turn per-frame answers into possessions."""

import numpy as np
from scipy.ndimage import uniform_filter1d

# Params
DEAD = "D"
NO_ANSWER = "X"
STATES = np.array(["H", "A", "D"])

FPS = 25
WAIT_LIVE_SPEED = 2.0           # m/s, players' mean speed (2 s smoothed) at match pace
WAIT_LIVE_FRAMES = 3 * FPS      # match pace must hold this long
WAIT_STILL_SPEED = 1.2          # the kickoff is the last frame slower than this before match pace
WAIT_MIN_FRAMES = 30 * FPS      # anything shorter is an ordinary kickoff, not a wait
WAIT_SEARCH_FRAMES = 10 * 60 * FPS


def kickoff_wait(player_speed, blocks):
    """True on the frames at the head of a half before its kickoff, where the ball is dead.

    A half can be graded from well before its kickoff: the second half of the provided match is,
    by about two minutes. Players stand around until the kickoff, then reach match pace within
    seconds, so the kickoff is the last slow frame before the first sustained stretch of match
    pace. Waits under 30 s are left alone, so an ordinary kickoff is never touched.
    """
    tracked = ~np.isnan(player_speed)
    total = np.where(tracked, player_speed, 0).sum(axis=1)
    speed = uniform_filter1d(total / np.maximum(tracked.sum(axis=1), 1), 2 * FPS + 1,
                             mode="nearest")
    wait = np.zeros(len(speed), dtype=bool)
    for _, first, last in blocks:
        window = speed[first:min(last + 1, first + WAIT_SEARCH_FRAMES)]
        fast = (window > WAIT_LIVE_SPEED).astype(np.int32)
        held = np.flatnonzero(np.convolve(fast, np.ones(WAIT_LIVE_FRAMES, dtype=np.int32),
                                          "valid") == WAIT_LIVE_FRAMES)
        if not len(held):
            continue
        still = np.flatnonzero(window[:held[0]] < WAIT_STILL_SPEED)
        kickoff = still[-1] + 1 if len(still) else 0
        if kickoff >= WAIT_MIN_FRAMES:
            wait[first:first + kickoff] = True
    return wait


def best_path(log_scores, switch, stop, restart):
    """The H / A / D sequence with the highest total score, paying for every change (Viterbi).

    log_scores is (frames, 3). switch is the cost of H <-> A, stop of going to D, restart of leaving
    D. Plain Python floats, because the three-state step is faster that way than through numpy.
    """
    n_frames = len(log_scores)
    back = np.zeros((n_frames, 3), dtype=np.int8)
    rows = log_scores.tolist()
    h, a, d = rows[0]
    for t in range(1, n_frames):
        to_h, to_a, to_d = rows[t]
        from_a, from_d = a - switch, d - restart
        if h >= from_a and h >= from_d:
            new_h, back_h = h, 0
        elif from_a >= from_d:
            new_h, back_h = from_a, 1
        else:
            new_h, back_h = from_d, 2

        from_h = h - switch
        if a >= from_h and a >= from_d:
            new_a, back_a = a, 1
        elif from_h >= from_d:
            new_a, back_a = from_h, 0
        else:
            new_a, back_a = from_d, 2

        stop_h, stop_a = h - stop, a - stop
        if d >= stop_h and d >= stop_a:
            new_d, back_d = d, 2
        elif stop_h >= stop_a:
            new_d, back_d = stop_h, 0
        else:
            new_d, back_d = stop_a, 1

        back[t] = back_h, back_a, back_d
        h, a, d = new_h + to_h, new_a + to_a, new_d + to_d

    path = np.zeros(n_frames, dtype=np.int8)
    path[-1] = int(np.argmax((h, a, d)))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def best_path_min(log_scores, switch, stop, restart, min_live, min_dead):
    """Like best_path, but every H or A spell lasts at least min_live frames and every D spell at
    least min_dead, except the first spell of the half.

    full[k][t] is the best score of being in state k at frame t with its minimum already met.
    entry[(j, k)][s] is the score of leaving a full j at s - 1 into k at s, minus the cumulative
    score of k up to s, so the spell first counts as full at t = s + n - 1 with entry plus the
    cumulative score to t. O(1) per frame. Staying beats entering on ties, so it is deterministic.
    """
    n_frames = len(log_scores)
    cum = np.vstack([np.zeros((1, 3)), np.cumsum(log_scores, axis=0)]).T.tolist()
    rows = log_scores.tolist()
    cost = {(0, 1): switch, (1, 0): switch, (2, 0): restart, (2, 1): restart,
            (0, 2): stop, (1, 2): stop}
    need = {k: (min_dead if k == 2 else min_live) for k in range(3)}
    pairs = list(cost)
    into = {k: [j for j, kk in pairs if kk == k] for k in range(3)}
    neg = -1e18
    full = [[neg] * n_frames for _ in range(3)]
    back = [[-1] * n_frames for _ in range(3)]
    entry = {pair: [neg] * n_frames for pair in pairs}
    for k in range(3):
        full[k][0] = rows[0][k]
    for t in range(1, n_frames):
        for j, k in pairs:
            entry[(j, k)][t] = full[j][t - 1] - cost[(j, k)] - cum[k][t]
        for k in range(3):
            best, arg = full[k][t - 1] + rows[t][k], -1
            s = t - need[k] + 1
            if s >= 1:
                for j in into[k]:
                    value = entry[(j, k)][s] + cum[k][t + 1]
                    if value > best:
                        best, arg = value, j
            full[k][t] = best
            back[k][t] = arg

    path = np.zeros(n_frames, dtype=np.int8)
    k = int(np.argmax([full[0][-1], full[1][-1], full[2][-1]]))
    t = n_frames - 1
    while t >= 0:
        j = back[k][t]
        if j == -1:
            path[t] = k
            t -= 1
        else:
            s = t - need[k] + 1
            path[s:t + 1] = k
            t, k = s - 1, j
    return path


def decode_min(scores, blocks, switch, stop, restart, min_live, min_dead):
    """decode, with minimum spell lengths (see best_path_min)."""
    labels = np.full(len(scores), NO_ANSWER, dtype="<U1")
    log_scores = np.log(np.clip(scores, 1e-6, 1.0))
    for _, first, last in blocks:
        path = best_path_min(log_scores[first:last + 1], switch, stop, restart,
                             min_live, min_dead)
        labels[first:last + 1] = STATES[path]
    return labels


def decode(scores, blocks, switch, stop, restart):
    """Labels from (frames, 3) H / A / D probabilities, one best path per half."""
    labels = np.full(len(scores), NO_ANSWER, dtype="<U1")
    log_scores = np.log(np.clip(scores, 1e-6, 1.0))
    for _, first, last in blocks:
        path = best_path(log_scores[first:last + 1], switch, stop, restart)
        labels[first:last + 1] = STATES[path]
    return labels


def apply_dead_ball(labels, dead, blocks):
    """Make every dead frame D and give every live frame a team.

    Live frames the smoothing left without a team take the last team seen in that half, or the
    next one at the head of a live stretch, because in play some team always has the ball.
    """
    out = labels.copy()
    for _, first, last in blocks:
        window = out[first:last + 1]
        has_team = window != DEAD
        if has_team.any():
            known = np.flatnonzero(has_team)
            carry = np.maximum.accumulate(np.where(has_team, np.arange(len(window)), -1))
            carry[carry < 0] = known[0]
            window[:] = window[carry]
        window[dead[first:last + 1]] = DEAD
    return out


def abstain_outside(labels, blocks):
    """X on every frame outside the two halves, where nothing is graded."""
    out = np.full(len(labels), NO_ANSWER, dtype=labels.dtype)
    for _, first, last in blocks:
        out[first:last + 1] = labels[first:last + 1]
    return out
