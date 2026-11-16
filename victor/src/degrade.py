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
sys.path.insert(0, os.path.dirname(ROOT))   # the shared repository, for games.py
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
EXTRA_SLOTS = 3                 # spare slots per team for tracks that follow the wrong person
WRONG_TEAM_GAP_MEAN = 4500      # frames a spare slot rests between wrong-team stretches
DUPLICATE_GAP_MEAN = 6000       # frames a spare slot rests between duplicate stretches
FLAW_MEDIAN_FRAMES = 100        # real stretches with a team over 11: median 72-85, mean 137-166
FLAW_SIGMA = 1.0                # log-normal spread of those lengths
FLAW_FRAMES = (10, 1500)
DUPLICATE_OFFSET_M = (1.0, 3.0)


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
    """The ball path with sharp changes softened, half by half. Gaps stay gaps.

    The filter is a running sum, so a single NaN would spread through the rest of the half: gaps
    are bridged for the filter only and put back afterwards.
    """
    out = ball_xy.copy()
    for _, first, last in blocks:
        part = ball_xy[first:last + 1]
        seen = np.flatnonzero(~np.isnan(part[:, 0]))
        if not len(seen):
            continue
        frames = np.arange(len(part))
        filled = np.column_stack([np.interp(frames, seen, part[seen, axis]) for axis in (0, 1)])
        smooth = uniform_filter1d(filled, BALL_SMOOTH_FRAMES, axis=0, mode="nearest")
        smooth[np.isnan(part[:, 0])] = np.nan
        out[first:last + 1] = smooth
    return out


def player_gap_lengths(profile, rng):
    """Short gaps that leave PLAYER_COVERAGE of each player tracked.

    A lost player soon returns under another slot, so the long empty stretches of a slot say
    nothing about how long a person goes unseen.
    """
    mean_gap = profile["player_seen"].mean() * (1 - PLAYER_COVERAGE) / PLAYER_COVERAGE
    return 1 + rng.exponential(mean_gap, 10000).astype(np.int64)


def flaw_lengths(n, rng):
    lengths = np.exp(rng.normal(np.log(FLAW_MEDIAN_FRAMES), FLAW_SIGMA, n))
    return np.clip(lengths, *FLAW_FRAMES).astype(np.int64)


def add_track_flaws(player_xy, player_speed, team_of_slot, rng):
    """Spare slots per team that now and then follow the wrong person.

    Measured on the provided match, a team shows more than 11 tracks on 13% of graded frames and
    in 84% of those the other team shows 10 or fewer: mostly one player tracked under a slot of
    the other team, for a few seconds. So most stretches move a player into a spare slot of the
    other team and blank their own slot meanwhile. A few copy a player into a spare slot of their
    own team, a little apart, as the duplicates behind the 2% of frames with more than 22 tracks.
    """
    n_frames = len(player_xy)
    teams = [np.flatnonzero(team_of_slot == side) for side in (0, 1)]
    new_xy = [player_xy]
    new_speed = [player_speed]
    new_team = [team_of_slot]
    for spare_team in (0, 1):
        for _ in range(EXTRA_SLOTS):
            xy = np.full((n_frames, 1, 2), np.nan)
            speed = np.full((n_frames, 1), np.nan, dtype=player_speed.dtype)
            for wrong_team, gap_mean in ((True, WRONG_TEAM_GAP_MEAN), (False, DUPLICATE_GAP_MEAN)):
                position = int(rng.exponential(gap_mean))
                while position < n_frames:
                    length = int(flaw_lengths(1, rng)[0])
                    stop = min(n_frames, position + length)
                    pool = teams[1 - spare_team] if wrong_team else teams[spare_team]
                    seen = pool[~np.isnan(player_xy[position, pool, 0])]
                    free = np.isnan(xy[position:stop, 0, 0]).all()
                    if len(seen) and free:
                        source = rng.choice(seen)
                        xy[position:stop, 0] = player_xy[position:stop, source]
                        speed[position:stop, 0] = player_speed[position:stop, source]
                        if wrong_team:
                            player_xy[position:stop, source] = np.nan
                            player_speed[position:stop, source] = np.nan
                        else:
                            angle = rng.uniform(0, 2 * np.pi)
                            offset = rng.uniform(*DUPLICATE_OFFSET_M)
                            drift = rng.normal(0, 0.2, (stop - position, 2)).cumsum(axis=0)
                            drift *= 0.5 / max(1.0, np.abs(drift).max())
                            xy[position:stop, 0] += offset * np.array([np.cos(angle),
                                                                       np.sin(angle)]) + drift
                    position = stop + int(rng.exponential(gap_mean))
            new_xy.append(xy)
            new_speed.append(speed)
            new_team.append(np.array([spare_team], dtype=team_of_slot.dtype))
    return (np.concatenate(new_xy, axis=1), np.concatenate(new_speed, axis=1),
            np.concatenate(new_team))


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

    # own generator, so the rest of the damage stays what it was for this seed
    player_xy, player_speed, team_of_slot = add_track_flaws(
        player_xy, player_speed, data["team_of_slot"], np.random.default_rng([seed, 1]))
    out.update(ball_xy=ball_xy, ball_speed=ball_speed, player_xy=player_xy,
               player_speed=player_speed, team_of_slot=team_of_slot)
    return out
