"""Learn per-frame H / A / D scores from the labelled open matches.

Trained on damaged copies only, so the model sees what the real tracking looks like. Nothing
from the provided match is used: the damage is built from DATA.md figures.
"""

import os
import sys

import lightgbm as lgb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import degrade  # noqa: E402
import external  # noqa: E402
import features  # noqa: E402

# Params
MODEL_PATH = os.path.join(ROOT, "models", "possession.txt")
CV_DIR = os.path.join(ROOT, "cache", "models")
TRAIN_SEEDS = (10, 11)      # different from the seeds score_external.py tests on
FRAME_STRIDE = 5            # neighbouring frames are nearly identical
CLASSES = {"H": 0, "A": 1, "D": 2}
PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.07, "num_leaves": 63,
          "min_data_in_leaf": 200, "feature_fraction": 0.7, "seed": 0, "deterministic": True,
          "force_row_wise": True, "num_threads": 8, "verbosity": -1}
N_TREES = 150


def training_rows(match_ids):
    """(names, X, y) from every match and seed, each frame also with the teams swapped."""
    profile = degrade.load_profile()
    rows, targets, names = [], [], None
    for match_id in match_ids:
        data = external.load(match_id)
        blocks = [tuple(block) for block in data["blocks"].tolist()]
        scored = np.flatnonzero(data["labels"] != "X")[::FRAME_STRIDE]
        y = np.array([CLASSES[label] for label in data["labels"][scored]], dtype=np.int8)
        swapped = np.where(y == 2, 2, 1 - y)
        for seed in TRAIN_SEEDS:
            names, X = features.build(degrade.degrade(data, profile, seed), blocks)
            X = X[scored]
            rows.extend((X, features.flip(names, X)))
            targets.extend((y, swapped))
    return names, np.vstack(rows), np.concatenate(targets)


def train(match_ids):
    names, X, y = training_rows(match_ids)
    dataset = lgb.Dataset(X, label=y, feature_name=names, free_raw_data=True)
    return lgb.train(PARAMS, dataset, num_boost_round=N_TREES)


def load(path=MODEL_PATH):
    return lgb.Booster(model_file=path)


def cv_path(match_id):
    return os.path.join(CV_DIR, match_id + ".txt")


def scores(booster, data, blocks):
    """(frames, 3) array of H, A, D probabilities."""
    names, X = features.build(data, blocks)
    if names != booster.feature_name():
        raise ValueError("feature list does not match the trained model; train it again")
    return booster.predict(X)
