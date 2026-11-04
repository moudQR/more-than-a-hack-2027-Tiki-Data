"""Numbers and plots for the notebooks. Nothing in run.py depends on this file."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Params
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT, "experiments_log.csv")
COLOURS = {"H": "#a50044", "A": "#ee2523", "D": "#c8c8c8", "X": "#ffffff"}
FPS = 25


def possessions(labels, blocks):
    """DataFrame of possessions: half, team, first_frame, frames, seconds."""
    rows = []
    for half, first, last in blocks:
        window = labels[first:last + 1]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if window[start] in ("H", "A"):
                rows.append({"half": half, "team": str(window[start]),
                             "first_frame": first + int(start), "frames": int(stop - start),
                             "seconds": (stop - start) / float(FPS)})
    return pd.DataFrame(rows)


def overview(labels, blocks):
    """What a set of labels says, in the terms an analyst would use."""
    graded = np.concatenate([labels[first:last + 1] for _, first, last in blocks])
    spells = possessions(labels, blocks)
    alive = np.isin(graded, ("H", "A"))
    return {"possessions": len(spells),
            "mean_possession_s": round(float(spells["seconds"].mean()), 1),
            "median_possession_s": round(float(spells["seconds"].median()), 1),
            "dead_share": round(float(np.mean(graded == "D")), 3),
            "home_share_of_live": round(float(np.sum(graded == "H")) / max(int(alive.sum()), 1), 3)}


def plot_timeline(labels, blocks, width=1500):
    """One strip per half: claret home, red away, grey dead."""
    fig, axes = plt.subplots(len(blocks), 1, figsize=(14, 1.6 * len(blocks)), squeeze=False)
    for axis, (half, first, last) in zip(axes[:, 0], blocks):
        window = labels[first:last + 1]
        picks = np.linspace(0, len(window) - 1, min(width, len(window))).astype(int)
        axis.bar(np.arange(len(picks)), 1, width=1, color=[COLOURS[window[i]] for i in picks])
        axis.set_xlim(0, len(picks))
        axis.set_yticks([])
        axis.set_xticks([])
        axis.set_ylabel("half {}".format(half))
    fig.suptitle("Possession timeline (claret home, red away, grey dead)")
    return fig


def log_result(name, note, table, path=LOG_PATH):
    """Add one test-bed result to the running log. Open matches only, so safe to commit."""
    row = table[table["version"] == name]
    entry = pd.DataFrame([{"version": name, "note": note,
                           "spellF1": round(float(row["spellF1"].mean()), 4),
                           "macroF1": round(float(row["macroF1"].mean()), 4),
                           "poss": round(float(row["poss"].mean()), 1)}])
    entry.to_csv(path, mode="a", header=not os.path.isfile(path), index=False)
    return pd.read_csv(path)
