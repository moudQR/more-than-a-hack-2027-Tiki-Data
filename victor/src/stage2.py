"""Second stage: a model that reads the first model's H / A / D probabilities in context.

Trained on out-of-fold stage-1 probabilities only (each match scored by the model that left it
out), so it learns how the first model errs on matches it has not seen. Paired h_ / a_ inputs and
team-swapped copies keep it free of any home or away bias.
"""

import os
import sys

import lightgbm as lgb
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import degrade  # noqa: E402
import external  # noqa: E402
import model  # noqa: E402

# Params
MODEL_PATH = os.path.join(ROOT, "models", "stage2.txt")
WINDOWS = (12, 25, 50, 125, 250)    # half-widths, +/- 0.5 s to +/- 10 s
FRAME_STRIDE = 5
PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 31,
          "min_data_in_leaf": 400, "feature_fraction": 0.7, "bagging_fraction": 0.7,
          "bagging_freq": 1, "lambda_l2": 1.0, "seed": 0, "deterministic": True,
          "force_row_wise": True, "num_threads": 6, "verbosity": -1}
N_TREES = 200
CLASSES = {"H": 0, "A": 1, "D": 2}


def _since(mask):
    """Frames since mask was last true (capped), and frames until it is next true."""
    n = len(mask)
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(mask, index, -10 ** 6))
    nxt = np.minimum.accumulate(np.where(mask, index, 10 ** 7)[::-1])[::-1]
    cap = 1500.0
    return np.minimum(index - last, cap), np.minimum(nxt - index, cap)


def _block_features(p):
    cols = {}
    series = {"h": p[:, 0], "a": p[:, 1], "d": p[:, 2]}
    with np.errstate(divide="ignore"):
        lp = np.log(np.clip(p, 1e-4, 1))
    series["h_margin"] = lp[:, 0] - lp[:, 1]          # h minus a; a_margin is its negative
    series["a_margin"] = -series["h_margin"]
    series["h_live"] = lp[:, 0] - lp[:, 2]
    series["a_live"] = lp[:, 1] - lp[:, 2]
    for name, x in series.items():
        prefix, base = (name[:2], name[2:]) if name[:2] in ("h_", "a_") else ("", name)
        if prefix == "" and name in ("h", "a"):
            prefix, base = name + "_", "p"
        key = prefix + base
        cols[key] = x
        for w in WINDOWS:
            size = 2 * w + 1
            cols["{}_mean{}".format(key, w)] = uniform_filter1d(x, size, mode="nearest")
            cols["{}_min{}".format(key, w)] = minimum_filter1d(x, size, mode="nearest")
            cols["{}_max{}".format(key, w)] = maximum_filter1d(x, size, mode="nearest")
            # mean ahead minus mean behind
            ahead = uniform_filter1d(x, w + 1, mode="nearest", origin=-(w // 2))
            behind = uniform_filter1d(x, w + 1, mode="nearest", origin=w // 2)
            cols["{}_slope{}".format(key, w)] = ahead - behind
    top = np.argmax(p, axis=1)
    for side, k in (("h_", 0), ("a_", 1), ("d_", 2)):
        back, ahead = _since(top == k)
        cols[side + "since"] = back
        cols[side + "until"] = ahead
    return cols


def build(p, blocks):
    """(names, X) for every frame; rows outside the blocks are NaN."""
    names, X = None, None
    for _, first, last in blocks:
        cols = _block_features(p[first:last + 1].astype(np.float64))
        if names is None:
            names = sorted(cols)
            X = np.full((len(p), len(names)), np.nan, dtype=np.float32)
        X[first:last + 1] = np.column_stack([cols[n] for n in names])
    return names, X


def _swap_name(name):
    for a, b in (("h_", "a_"), ("a_", "h_")):
        if name.startswith(a):
            return b + name[2:]
    return name


def flip(names, X):
    position = {n: i for i, n in enumerate(names)}
    return X[:, [position[_swap_name(n)] for n in names]]


def training_rows(runs):
    """runs: iterable of (p, blocks, labels). Every 5th labelled frame, teams swapped too."""
    rows, targets, names = [], [], None
    for p, blocks, labels in runs:
        names, X = build(p, blocks)
        keep = np.flatnonzero(labels != "X")[::FRAME_STRIDE]
        y = np.array([CLASSES[l] for l in labels[keep]], dtype=np.int8)
        rows.extend((X[keep], flip(names, X[keep])))
        targets.extend((y, np.where(y == 2, 2, 1 - y)))
    return names, np.vstack(rows), np.concatenate(targets)


def train(runs):
    names, X, y = training_rows(runs)
    data = lgb.Dataset(X, label=y, feature_name=names, free_raw_data=True)
    return lgb.train(PARAMS, data, num_boost_round=N_TREES)


def out_of_fold_runs(match_ids):
    """(p, blocks, labels) per match and training seed, each match scored by the stage-1 model
    that left it out (cache/models, from train_model.py --cv)."""
    profile = degrade.load_profile()
    for match_id in match_ids:
        data = external.load(match_id)
        blocks = [tuple(block) for block in data["blocks"].tolist()]
        booster = model.load(model.cv_path(match_id))
        for seed in model.TRAIN_SEEDS:
            p = model.scores(booster, degrade.degrade(data, profile, seed), blocks)
            yield p, blocks, data["labels"]


def load(path=MODEL_PATH):
    return lgb.Booster(model_file=path)


def cv_path(match_id):
    return os.path.join(model.CV_DIR, "stage2_" + match_id + ".txt")


def scores(booster, p, blocks):
    """(frames, 3) H / A / D probabilities; frames outside the blocks keep stage 1's."""
    names, X = build(p, blocks)
    if names != booster.feature_name():
        raise ValueError("stage-2 inputs do not match the trained model; train it again")
    out = p.astype(np.float64).copy()
    inside = ~np.isnan(X[:, 0])
    # symmetric: average with the team-swapped prediction
    a = booster.predict(X[inside])
    b = booster.predict(flip(names, X[inside]))[:, [1, 0, 2]]
    out[inside] = (a + b) / 2
    return out
