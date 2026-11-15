"""Per-frame clues about who has the ball and whether it is in play.

Paired clues start with h_ and a_, one for each team, so that flip() can swap the teams. Training on
both versions stops the model learning anything about home sides or a particular club.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

import ball
import dead_ball
import touches

# Params
PITCH_LENGTH = 105.0
NEAR_RADIUS_M, WIDE_RADIUS_M = 3.0, 8.0
TREND_FRAMES = 25                   # forward speed of a team is measured over 1 s
BALL_SPEED_FRAMES = 5
ROLL_WINDOWS = (12, 25, 50, 125)    # half-widths, +/- 0.5 s to +/- 5 s
CHUNK_FRAMES = 20000
PRESS_RADII_M = (2.0, 5.0)          # opponents this close to the team's player nearest the ball
SHAPE_WINDOWS = (25, 125)           # rolling half-widths for the team-shape clues


def rolling_mean(values, half_width):
    """Centred mean ignoring NaN. NaN where the whole window is empty."""
    seen = ~np.isnan(values)
    totals = np.concatenate(([0.0], np.cumsum(np.where(seen, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(seen)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    amount = counts[high] - counts[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(amount > 0, (totals[high] - totals[low]) / amount, np.nan)


def nan_mean(values, weights):
    """Mean over the slots marked in weights, ignoring NaN, axis 1."""
    seen = ~np.isnan(values) & weights
    amount = seen.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(amount > 0, np.where(seen, values, 0).sum(axis=1) / amount, np.nan)


def attack_direction(home_x, blocks):
    """+1 where the home side attacks towards larger x, -1 otherwise, per half."""
    out = np.ones(len(home_x), dtype=np.float32)
    for _, first, last in blocks:
        out[first:last + 1] = 1.0 if np.nanmean(home_x[first:last + 1]) < PITCH_LENGTH / 2 else -1.0
    return out


def ball_distances(ball_xy, player_xy, is_home):
    """Nearest distance and nearby head-counts per team, NaN where the ball is unknown."""
    n_frames = len(ball_xy)
    near = {side: np.full(n_frames, np.nan) for side in "ha"}
    close = {side: np.zeros(n_frames, dtype=np.float32) for side in "ha"}
    wide = {side: np.zeros(n_frames, dtype=np.float32) for side in "ha"}

    for start in range(0, n_frames, CHUNK_FRAMES):
        stop = start + CHUNK_FRAMES
        offsets = player_xy[start:stop] - ball_xy[start:stop, np.newaxis, :]
        distances = np.sqrt(np.sum(offsets * offsets, axis=2))
        for side, slots in (("h", is_home), ("a", ~is_home)):
            part = distances[:, slots]
            filled = np.where(np.isnan(part), np.inf, part)
            best = filled.min(axis=1)
            near[side][start:stop] = np.where(np.isfinite(best), best, np.nan)
            close[side][start:stop] = (filled < NEAR_RADIUS_M).sum(axis=1)
            wide[side][start:stop] = (filled < WIDE_RADIUS_M).sum(axis=1)

    known = ~np.isnan(ball_xy[:, 0])
    for side in "ha":
        close[side] = np.where(known, close[side], np.nan)
        wide[side] = np.where(known, wide[side], np.nan)
    return near, close, wide


def sorted_rows(values):
    """(rows sorted ascending with NaN last, count of values per row)."""
    return np.sort(values, axis=1), (~np.isnan(values)).sum(axis=1)


def kth(ordered, count, k, from_top=False):
    """k-th smallest (or largest) per row of sorted_rows, NaN where fewer than 2k+2 values.

    Quantiles rather than min or max, so a duplicated or wrong-team track barely moves them.
    """
    index = count - 1 - k if from_top else np.full(len(count), k)
    picked = np.take_along_axis(ordered, np.clip(index, 0, ordered.shape[1] - 1)[:, None], 1)[:, 0]
    return np.where(count >= 2 * k + 2, picked, np.nan)


def median(ordered, count):
    """Row median of sorted_rows, NaN where empty."""
    last = ordered.shape[1] - 1
    low = np.take_along_axis(ordered, np.clip((count - 1) // 2, 0, last)[:, None], 1)[:, 0]
    high = np.take_along_axis(ordered, np.clip(count // 2, 0, last)[:, None], 1)[:, 0]
    return np.where(count > 0, 0.5 * (low + high), np.nan)


def pressure(ball_xy, player_xy, is_home):
    """Around each team's player nearest the ball: opponents within PRESS_RADII_M, nearest one."""
    n_frames = len(ball_xy)
    out = {}
    for side in "ha":
        out[side + "_press_near"] = np.full(n_frames, np.nan)
        for radius in PRESS_RADII_M:
            out["{}_press{:g}".format(side, radius)] = np.full(n_frames, np.nan)
    known = ~np.isnan(ball_xy[:, 0])
    for start in range(0, n_frames, CHUNK_FRAMES):
        stop = min(n_frames, start + CHUNK_FRAMES)
        xy = player_xy[start:stop]
        offsets = xy - ball_xy[start:stop, np.newaxis, :]
        distances = np.sqrt(np.sum(offsets * offsets, axis=2))
        rows = np.arange(stop - start)
        for side, own, other in (("h", is_home, ~is_home), ("a", ~is_home, is_home)):
            own_d = np.where(np.isnan(distances[:, own]), np.inf, distances[:, own])
            carrier = xy[:, own][rows, own_d.argmin(axis=1)]
            gap = xy[:, other] - carrier[:, np.newaxis, :]
            gap = np.sqrt(np.sum(gap * gap, axis=2))
            ok = known[start:stop] & np.isfinite(own_d.min(axis=1))
            filled = np.where(np.isnan(gap), np.inf, gap)
            best = filled.min(axis=1)
            out[side + "_press_near"][start:stop] = np.where(ok & np.isfinite(best), best, np.nan)
            for radius in PRESS_RADII_M:
                out["{}_press{:g}".format(side, radius)][start:stop] = np.where(
                    ok, (filled < radius).sum(axis=1), np.nan)
    return out


def shape(player_xy, is_home, direction):
    """Defensive-line height, length, width, compactness per team, along its attacking direction.

    These need no ball, so they keep working while it is unseen.
    """
    centre = PITCH_LENGTH / 2
    out = {}
    for side, slots, sign in (("h", is_home, direction), ("a", ~is_home, -direction)):
        forward = sign[:, np.newaxis] * (player_xy[:, slots, 0] - centre)
        across = player_xy[:, slots, 1]
        f_sorted, count = sorted_rows(forward)
        y_sorted, _ = sorted_rows(across)
        back = kth(f_sorted, count, 1)
        out[side + "_line"] = 0.5 * (back + kth(f_sorted, count, 2))   # 2nd/3rd deepest
        out[side + "_length"] = kth(f_sorted, count, 1, from_top=True) - back
        out[side + "_width"] = kth(y_sorted, count, 1, from_top=True) - kth(y_sorted, count, 1)
        reach = np.hypot(forward - median(f_sorted, count)[:, None],
                         across - median(y_sorted, count)[:, None])
        out[side + "_compact"] = median(*sorted_rows(reach))
    out["h_line_gap"] = out["h_line"] - out["a_line"]
    out["a_line_gap"] = -out["h_line_gap"]
    return out


def build(data, blocks):
    """(names, X): one row per frame, float32, NaN where a clue does not exist."""
    player_xy, player_speed = data["player_xy"], data["player_speed"]
    is_home = data["team_of_slot"] == 0
    tracked_home, tracked_away = np.tile(is_home, (len(player_xy), 1)), None
    tracked_away = ~tracked_home
    ball_xy = ball.clean(data["ball_xy"])

    cols = {}
    home_x = nan_mean(player_xy[:, :, 0], tracked_home)
    away_x = nan_mean(player_xy[:, :, 0], tracked_away)
    direction = attack_direction(home_x, blocks)
    centre = PITCH_LENGTH / 2

    near, close, wide = ball_distances(ball_xy, player_xy, is_home)
    pairs = {"near": near, "close": close, "wide": wide}
    for name, values in pairs.items():
        for side in "ha":
            cols["{}_{}".format(side, name)] = values[side]
            for half_width in ROLL_WINDOWS:
                cols["{}_{}_r{}".format(side, name, half_width)] = rolling_mean(values[side],
                                                                                half_width)

    frames, teams = touches.detect(ball_xy, near["h"], near["a"], blocks)
    cols.update(touches.clues(frames, teams, len(ball_xy)))
    cols["ball_turn"] = touches.velocity_change(ball_xy)
    cols["ball_turn_r25"] = rolling_mean(cols["ball_turn"], 25)

    closer_home = np.where(np.isnan(near["h"]) | np.isnan(near["a"]), np.nan,
                           (near["h"] < near["a"]).astype(np.float32))
    closer_away = 1.0 - closer_home
    for half_width in ROLL_WINDOWS:
        cols["h_closer_r{}".format(half_width)] = rolling_mean(closer_home, half_width)
        cols["a_closer_r{}".format(half_width)] = rolling_mean(closer_away, half_width)

    for side, team_x, tracked, sign in (("h", home_x, tracked_home, direction),
                                        ("a", away_x, tracked_away, -direction)):
        forward = sign * (team_x - centre)
        cols[side + "_forward"] = forward
        cols[side + "_push"] = np.concatenate((np.zeros(TREND_FRAMES),
                                               forward[TREND_FRAMES:] - forward[:-TREND_FRAMES]))
        cols[side + "_speed"] = uniform_filter1d(
            np.nan_to_num(nan_mean(player_speed, tracked)), 25, mode="nearest")
        spread_x = player_xy[:, :, 0] - team_x[:, np.newaxis]
        spread_y = player_xy[:, :, 1] - nan_mean(player_xy[:, :, 1], tracked)[:, np.newaxis]
        cols[side + "_spread"] = np.sqrt(nan_mean(spread_x ** 2 + spread_y ** 2, tracked))
        cols[side + "_ball_forward"] = sign * (ball_xy[:, 0] - centre)

    for name, values in pressure(ball_xy, player_xy, is_home).items():
        cols[name] = values
        for half_width in ROLL_WINDOWS:
            cols["{}_r{}".format(name, half_width)] = rolling_mean(values, half_width)
    for name, values in shape(player_xy, is_home, direction).items():
        cols[name] = values
        for half_width in SHAPE_WINDOWS:
            cols["{}_r{}".format(name, half_width)] = rolling_mean(values, half_width)

    ball_step = np.full(len(ball_xy), np.nan)
    ball_step[BALL_SPEED_FRAMES:] = np.linalg.norm(
        ball_xy[BALL_SPEED_FRAMES:] - ball_xy[:-BALL_SPEED_FRAMES], axis=1)
    cols["ball_speed"] = ball_step * 25.0 / BALL_SPEED_FRAMES
    cols["ball_y_edge"] = np.minimum(ball_xy[:, 1], 68.0 - ball_xy[:, 1])
    cols["ball_seen_raw"] = (~np.isnan(data["ball_xy"][:, 0])).astype(np.float32)
    cols["ball_seen_r50"] = rolling_mean(cols["ball_seen_raw"], 50)
    cols["ball_seen_r125"] = rolling_mean(cols["ball_seen_raw"], 125)
    cols["unseen_length"] = dead_ball.unseen_length(data["ball_xy"]).astype(np.float32)
    cols["ball_outside"] = dead_ball.ball_outside(ball_xy).astype(np.float32)
    cols["ball_still"] = dead_ball.ball_still(ball_xy).astype(np.float32)
    cols["team_speed"] = dead_ball.team_speed(player_speed)
    cols["team_speed_r125"] = rolling_mean(cols["team_speed"], 125)
    cols["rule_dead"] = dead_ball.detect(data, ball_xy).astype(np.float32)

    names = sorted(cols)
    return names, np.column_stack([cols[name] for name in names]).astype(np.float32)


def flip(names, X):
    """The same frames with the two teams swapped."""
    position = {name: i for i, name in enumerate(names)}
    order = [position[("a_" + name[2:]) if name.startswith("h_") else
                      ("h_" + name[2:]) if name.startswith("a_") else name]
             for name in names]
    return X[:, order]
