"""Make clean tracking look like the provided automatic tracking.

The open matches come from a much cleaner system, so a score on them as they are says little. This
removes the ball and the players the way the real ATD loses them, using the lengths of the seen and
unseen stretches measured on the provided match, softens the ball path as much as the real
one, then adds a little position noise.

Only those lengths and the size of its velocity changes are taken from the provided match: no
labels and no frame numbers. The lengths stay in cache/, because they are derived from the match
data.

MAX_ALIVE_GAP and MAX_DEAD_SEEN were set so the four baselines land near their published full-match
scores with the ball visible about as often as DATA.md says, so no ground truth was involved.

What this does not copy: in play the real ball goes missing more often in crowds and in the air,
whereas here gaps fall at random, and tracks here never swap team.
"""

import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402

# Params
PROFILE_PATH = os.path.join(ROOT, "cache", "tracking_profile.npz")
BALL_NOISE_M = 0.05             # the real ball path is smooth, not noisy
BALL_SMOOTH_FRAMES = 15         # matches the size of velocity changes in the real ATD
PLAYER_NOISE_M = 0.3
PLAYER_COVERAGE = 20.5 / 22     # DATA.md: 20.5 tracked objects on average, of 22 players
MAX_ALIVE_GAP = 200             # longest ball gap in play, frames
MAX_DEAD_SEEN = 250             # longest visible ball stretch in dead time, frames
LONG_GAP_SHARE = 0.1            # longer gaps in play, relative to the short ones
BOUNDARY_JITTER = 75            # dead-time damage starts and ends up to 3 s off the true edge
PLAYER_SPEED_NOISE = 0.3        # m/s


def run_lengths(seen):
    """(lengths of True runs, lengths of False runs) in a boolean array."""
    edges = np.flatnonzero(np.diff(seen.astype(np.int8))) + 1
    bounds = np.concatenate(([0], edges, [len(seen)]))
    lengths = np.diff(bounds)
    first_is_seen = bool(seen[0])
    return (lengths[0 if first_is_seen else 1::2], lengths[1 if first_is_seen else 0::2])


def measure_profile(data, blocks=None):
    """Seen and unseen stretch lengths of the ball and of tracked players, over the two halves."""
    blocks = half_video_blocks(GAME) if blocks is None else blocks
    ball_seen, ball_gap, player_seen, player_gap = [], [], [], []
    for _, first, last in blocks:
        window = data["ball_xy"][first:last + 1, 0]
        seen, gap = run_lengths(~np.isnan(window))
        ball_seen.append(seen)
        ball_gap.append(gap)
        for slot in range(data["player_xy"].shape[1]):
            track = ~np.isnan(data["player_xy"][first:last + 1, slot, 0])
            if track.any():
                seen, gap = run_lengths(track)
                player_seen.append(seen)
                player_gap.append(gap)
    return {"ball_seen": np.concatenate(ball_seen), "ball_gap": np.concatenate(ball_gap),
            "player_seen": np.concatenate(player_seen), "player_gap": np.concatenate(player_gap)}


def load_profile():
    """The measured stretch lengths, measured from the cached tracking the first time."""
    if not os.path.isfile(PROFILE_PATH):
        import parse
        profile = measure_profile(parse.load())
        np.savez(PROFILE_PATH, **profile)
    with np.load(PROFILE_PATH) as cached:
        return {name: cached[name] for name in cached.files}


def visible_mask(n_frames, seen_lengths, gap_lengths, rng):
    """A boolean mask built by drawing seen and unseen stretches until it covers n_frames."""
    mask = np.zeros(n_frames, dtype=bool)
    position, seen = 0, bool(rng.integers(2))
    while position < n_frames:
        length = int(rng.choice(seen_lengths if seen else gap_lengths))
        mask[position:position + length] = seen
        position += length
        seen = not seen
    return mask


def blurred_dead_zone(labels, rng, jitter=BOUNDARY_JITTER):
    """The dead stretches, each edge moved at random.

    A model trained on damage that switches exactly at the true boundary would learn that fact
    instead of anything about football, and real tracking offers no such clue.
    """
    zone = np.zeros(len(labels), dtype=bool)
    marked = np.concatenate(([0], (labels == "D").astype(np.int8), [0]))
    edges = np.diff(marked)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        shift_start, shift_stop = rng.integers(-jitter, jitter + 1, 2)
        zone[max(0, start + shift_start):max(0, stop + shift_stop)] = True
    return zone


def ball_mask(labels, profile, rng, max_alive_gap=None, max_dead_seen=None):
    """Where the ball is visible. It gets lost far more during dead balls than in play.

    The baseline can only answer D where the ball is unseen, so its published macroF1 shows the two
    go together. Long gaps are made rare in play, not impossible, so that a long gap never proves
    a dead ball, and seen stretches are capped in dead time.
    """
    max_alive_gap = MAX_ALIVE_GAP if max_alive_gap is None else max_alive_gap
    max_dead_seen = MAX_DEAD_SEEN if max_dead_seen is None else max_dead_seen
    gaps, seen = profile["ball_gap"], profile["ball_seen"]
    short, long_gaps = gaps[gaps <= max_alive_gap], gaps[gaps > max_alive_gap]
    n_long = int(np.ceil(len(short) * LONG_GAP_SHARE))
    alive_gaps = np.concatenate((short, rng.choice(long_gaps, n_long)))
    alive = visible_mask(len(labels), seen, alive_gaps, rng)
    dead = visible_mask(len(labels), np.minimum(seen, max_dead_seen), gaps, rng)
    return np.where(blurred_dead_zone(labels, rng), dead, alive)


def smooth_ball(ball_xy, blocks):
    """The ball path with sharp changes softened, half by half."""
    out = ball_xy.copy()
    for _, first, last in blocks:
        out[first:last + 1] = uniform_filter1d(ball_xy[first:last + 1], BALL_SMOOTH_FRAMES, axis=0,
                                               mode="nearest")
    return out


def player_gap_lengths(profile, rng):
    """Short gaps that leave PLAYER_COVERAGE of each player tracked.

    A lost player soon returns under another slot, so the long empty stretches of a slot say
    nothing about how long a person goes unseen.
    """
    mean_gap = profile["player_seen"].mean() * (1 - PLAYER_COVERAGE) / PLAYER_COVERAGE
    return 1 + rng.exponential(mean_gap, 10000).astype(np.int64)


def degrade(data, profile, seed, **ball_options):
    """A copy of one match with the tracking damaged. The same seed gives the same copy."""
    rng = np.random.default_rng(seed)
    n_frames, n_slots = data["player_xy"].shape[:2]
    out = dict(data)

    ball_xy = smooth_ball(data["ball_xy"], data["blocks"])
    ball_xy = ball_xy + rng.normal(0, BALL_NOISE_M, ball_xy.shape)
    ball_mask_ = ball_mask(data["labels"], profile, rng, **ball_options)
    ball_xy[~ball_mask_] = np.nan
    ball_speed = data["ball_speed"].copy()
    ball_speed[~ball_mask_] = np.nan

    player_xy = data["player_xy"] + rng.normal(0, PLAYER_NOISE_M, data["player_xy"].shape)
    player_speed = np.abs(data["player_speed"] + rng.normal(
        0, PLAYER_SPEED_NOISE, data["player_speed"].shape)).astype(np.float32)
    player_gaps = player_gap_lengths(profile, rng)
    for slot in range(n_slots):
        mask = visible_mask(n_frames, profile["player_seen"], player_gaps, rng)
        player_xy[~mask, slot] = np.nan
        player_speed[~mask, slot] = np.nan

    out.update(ball_xy=ball_xy, ball_speed=ball_speed, player_xy=player_xy,
               player_speed=player_speed)
    return out
