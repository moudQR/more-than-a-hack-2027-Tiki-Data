"""End-to-end: features → trained classifier + Viterbi → smart-tagging → CSV."""
from pathlib import Path

import numpy as np
import pandas as pd

import config
from features import extract_frame_features, load_team_slots
from model import predict
from smart_tagging import load_set_piece_mask

def _majority_vote_within(labels: np.ndarray, start: int, end: int,
                          window: int = 75) -> np.ndarray:
    """Majority vote within a graded segment. Cumulative-sum, O(n)."""
    seg = labels[start:end]
    n = len(seg)
    h = np.concatenate(([0], np.cumsum(seg == "H")))
    a = np.concatenate(([0], np.cumsum(seg == "A")))
    d = np.concatenate(([0], np.cumsum(seg == "D")))

    idx = np.arange(n)
    lo = np.maximum(0, idx - window)
    hi = np.minimum(n, idx + window + 1)
    hc = h[hi] - h[lo]
    ac = a[hi] - a[lo]
    dc = d[hi] - d[lo]

    out = np.where((hc > ac) & (hc > dc), "H",
               np.where((ac > hc) & (ac > dc), "A",
                        np.where((dc > hc) & (dc > ac), "D", seg)))
    labels[start:end] = out
    return labels


def smooth_per_half(labels: np.ndarray, window: int = 75) -> np.ndarray:
    """Apply majority vote inside each graded half. X regions are untouched."""
    labels = labels.copy()
    for start, end in ((config.HALF1_START, config.HALF1_END + 1), (config.HALF2_START, config.HALF2_END + 1)):
        labels = _majority_vote_within(labels, start, end, window)
    return labels

def build_submission(out_path: Path | None = None) -> Path:
    out_path = out_path or (config.RESULTS_DIR / "solution.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = config.TACTICAL_DIR / "BAR-ATH_tactical_atd.xml"
    raw = config.TACTICAL_DIR / "BAR-ATH_tactical_atd.txt"
    smart = config.TACTICAL_DIR / "BAR-ATH_tactical_smart_tagging.xml"

    print("Extracting per-frame features from ATD …")
    slots = load_team_slots(meta)
    X = extract_frame_features(raw, slots, config.PITCH, config.N_FRAMES)

    print("Running trained classifier + Viterbi over 199,319 frames …")
    labels = predict(X)

    print("Smoothing within each half …")
    labels = smooth_per_half(labels, window=125)

    if smart.exists():
        print("Applying Smart Tagging: forcing D on set-piece intervals …")
        sp_mask = load_set_piece_mask(smart, config.FPS, config.N_FRAMES)
        labels[sp_mask] = "D"
        print(f"  {int(sp_mask.sum())} frames forced to D")
    else:
        print(f"  {smart.name} not found — skipping set-piece override")

    graded = np.zeros(config.N_FRAMES, dtype=bool)
    graded[config.HALF1_START:config.HALF1_END + 1] = True
    graded[config.HALF2_START:config.HALF2_END + 1] = True
    labels[~graded] = "X"

    df = pd.DataFrame({"frame": np.arange(config.N_FRAMES), "label": labels})
    df.to_csv(out_path, index=False)

    print(f"\nWrote {out_path}")
    print(df["label"].value_counts().to_string())
    return out_path


if __name__ == "__main__":
    build_submission()