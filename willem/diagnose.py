"""Sanity checks on a submission beyond the format validator."""
import sys
from collections import Counter

import numpy as np
import pandas as pd

import config

PUBLIC_GT_PCTS = {"H": 46.0, "A": 34.4, "D": 19.6}


def spell_stats(labels: np.ndarray):
    """Return list of (label, start, end, length_frames) per maximal run."""
    spells = []
    i = 0
    while i < len(labels):
        j = i + 1
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        spells.append((labels[i], i, j, j - i))
        i = j
    return spells


def main(path):
    df = pd.read_csv(path)
    labels = df["label"].values

    graded = np.zeros(len(labels), dtype=bool)
    graded[config.HALF1_START:config.HALF1_END] = True
    graded[config.HALF2_START:config.HALF2_END] = True
    g = labels[graded]

    print("=" * 60)
    print(f"{path}")
    print("=" * 60)

    print("\nGraded-frame distribution vs public ground truth:")
    print(f"{'label':>6}  {'count':>8}  {'pct':>6}  {'target':>7}  {'delta':>7}")
    for lab in "HAD":
        n = int((g == lab).sum())
        pct = 100 * n / len(g)
        target = PUBLIC_GT_PCTS[lab]
        print(f"{lab:>6}  {n:>8}  {pct:>5.1f}%  {target:>6.1f}%  {pct - target:>+6.1f}")

    print("\nSpells on the whole video (including ungraded X runs):")
    spells = spell_stats(labels)
    by_label = Counter(s[0] for s in spells)
    print(f"  total spells: {len(spells)}")
    for lab in "HADX":
        print(f"  {lab}: {by_label.get(lab, 0)}")

    print("\nPer-label duration (seconds, at 25 fps):")
    for lab in "HAD":
        durs = [s[3] / config.FPS for s in spells if s[0] == lab]
        if durs:
            print(f"  {lab}: n={len(durs):>4}  "
                  f"mean={np.mean(durs):>6.1f}s  median={np.median(durs):>5.1f}s  "
                  f"max={np.max(durs):>6.1f}s")

    # How many possessions does a coach see? ~15-25 per half.
    n_possessions = sum(1 for lab, _, _, d in spells
                        if lab in "HA" and d >= 5 * config.FPS)
    print(f"\nPossessions (HA spells ≥ 5s): {n_possessions}")
    print(f"  ground truth estimate: ~30-50 for the full match")

    # Longest D runs — a D that spans minutes is a bug, not a stoppage.
    long_d = [s for s in spells if s[0] == "D" and s[3] > 30 * config.FPS]
    print(f"\nD spells longer than 30s: {len(long_d)}")
    for lab, start, end, length in long_d[:5]:
        print(f"  frames {start:>6}..{end:>6}  ({length/25:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation/results/solution.csv")