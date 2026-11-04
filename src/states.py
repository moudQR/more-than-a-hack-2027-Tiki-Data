"""Turn per-frame answers into possessions."""

import numpy as np

# Params
DEAD = "D"
NO_ANSWER = "X"
STATES = np.array(["H", "A", "D"])


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
