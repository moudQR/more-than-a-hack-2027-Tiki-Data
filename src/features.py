"""Per-frame clues about who has the ball and whether it is in play.

Paired clues start with h_ and a_, one for each team, so that flip() can swap the teams. Training on
both versions stops the model learning anything about home sides or a particular club.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

import ball
import dead_ball

# Params
PITCH_LENGTH = 105.0
NEAR_RADIUS_M, WIDE_RADIUS_M = 3.0, 8.0
TREND_FRAMES = 25                   # forward speed of a team is measured over 1 s
BALL_SPEED_FRAMES = 5
ROLL_WINDOWS = (12, 25, 50, 125)    # half-widths, +/- 0.5 s to +/- 5 s
CHUNK_FRAMES = 20000


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
