"""Per-frame features. One core function, two adapters (ATD, Metrica CSV).

The features deliberately exclude anything that requires player identity — the
challenge ATD has none, so the training data must be reduced to the same
information regime.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"
DIST_THRESHOLDS = (3.0, 5.0, 10.0, 15.0)


def frame_features(home_xy, away_xy, ball_xy, pitch):
    f = np.full(18, np.nan)      # was np.full(16, np.nan)
    f[0] = 0.0
    if ball_xy is None or np.isnan(ball_xy).any():
        return f.tolist()

    bx, by = ball_xy
    f[0] = 1.0
    length, width = pitch
    f[14] = bx / length
    f[15] = by / width

    def _dist(arr):
        if arr.size == 0:
            return None
        return np.hypot(arr[:, 0] - bx, arr[:, 1] - by)

    dh = _dist(home_xy)
    da = _dist(away_xy)

    if dh is not None:
        f[1] = dh.min()
        for k, thr in enumerate(DIST_THRESHOLDS):
            f[4 + k] = (dh < thr).sum()
    if da is not None:
        f[2] = da.min()
        for k, thr in enumerate(DIST_THRESHOLDS):
            f[8 + k] = (da < thr).sum()

    # --- scale-invariant features (new) ---
    # (dh - da) is signed: positive means home is closer, negative away.
    # The tree classifier can use the SIGN of this feature — which is the same
    # on Metrica (ball at possessing player) and on the challenge (ball near
    # the possessing player). That sign is what the classifier actually needs
    # to distinguish H from A. Without it, the absolute distances learn
    # thresholds that only match Metrica's ball-at-centroid representation.
    if dh is not None and da is not None:
        eps = 1e-2
        dh_min = dh.min() + eps
        da_min = da.min() + eps
        f[3] = float(np.log(da_min / dh_min))              # (unchanged, still informative)
        f[16] = float(da_min - dh_min)                     # signed difference
        f[17] = float(dh_min / (dh_min + da_min))          # relative, 0-1

    if dh is not None:
        cx, cy = home_xy.mean(axis=0)
        f[12] = np.hypot(cx - bx, cy - by)
    if da is not None:
        cx, cy = away_xy.mean(axis=0)
        f[13] = np.hypot(cx - bx, cy - by)

    return f.tolist()


# ---------- ATD adapter ----------

def load_team_slots(meta_path: Path) -> np.ndarray:
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home_id = team_ids[0]
    return np.array(
        [0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
        dtype=np.int8,
    )


def extract_frame_features(raw_path: Path, team_of_slot: np.ndarray,
                           pitch: tuple[float, float], n_frames: int) -> np.ndarray:
    """Return (n_frames, 18) feature matrix from the challenge ATD."""
    length, width = pitch
    X = np.full((n_frames, 18), np.nan)               # was 16
    with open(raw_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            fi = int(parts[0]) - 1
            if fi >= n_frames:
                break

            home, away = [], []
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if not group or group[0] == MISSING:
                    continue
                xs, ys = group.split(",")[:2]
                pt = (float(xs) * length, float(ys) * width)
                (home if team_of_slot[slot] == 0 else away).append(pt)

            ball = parts[2].strip()
            ball_xy = None
            if ball and ball[0] != MISSING:
                bx, by = ball.split(",")[:2]
                ball_xy = np.array([float(bx) * length, float(by) * width])

            X[fi] = frame_features(np.array(home).reshape(-1, 2),
                                   np.array(away).reshape(-1, 2),
                                   ball_xy, pitch)
    return X


# ---------- Metrica sample-data adapter ----------

def _load_metrica_csv(path: Path) -> "pd.DataFrame":
    """Read a Metrica tracking CSV robustly, positionally.

    Column names differ between releases (Player1_x vs Home vs Home.1 ...) and
    y-coordinates are frequently unnamed, so we cannot select by name. Instead
    we find the first row whose first three cells parse as numbers, drop
    everything above it, and coerce the rest to float.
    """
    import pandas as pd
    raw = pd.read_csv(path, header=None, low_memory=False)

    start = None
    for i in range(min(10, len(raw))):
        try:
            float(raw.iat[i, 0]); float(raw.iat[i, 1]); float(raw.iat[i, 2])
            start = i
            break
        except (ValueError, TypeError, IndexError):
            continue
    if start is None:
        raise ValueError(f"No numeric data rows found in {path}")

    df = raw.iloc[start:].reset_index(drop=True)
    return df.apply(pd.to_numeric, errors="coerce")


def _row_objects(row, start_col: int = 3) -> list[tuple[float, float]]:
    """Return (x, y) pairs from a row, starting at `start_col`."""
    out = []
    n = len(row)
    for k in range(start_col, n - 1, 2):
        out.append((row.iat[k], row.iat[k + 1]))
    return out


def extract_frame_features_metrica(home_csv: Path, away_csv: Path,
                                   pitch: tuple[float, float]) -> np.ndarray:
    """Per-frame 18-feature matrix from a Metrica sample match."""
    home = _load_metrica_csv(home_csv)
    away = _load_metrica_csv(away_csv)
    n = min(len(home), len(away))
    length, width = pitch

    X = np.full((n, 18), np.nan)                      # was 16
    for i in range(n):
        ho = _row_objects(home.iloc[i])
        ao = _row_objects(away.iloc[i])

        # Both Metrica CSVs carry 14 players + the ball as the last object.
        # Dropping the last from only one of them makes the ball count as an
        # away player, so nearest_away_dist is 0 on every visible frame and the
        # classifier learns to ignore the away side entirely.
        h_pts = [[x * length, y * width] for x, y in ho[:-1]
                 if not (np.isnan(x) or np.isnan(y))]
        a_pts = [[x * length, y * width] for x, y in ao[:-1]
                 if not (np.isnan(x) or np.isnan(y))]

        ball = None
        if ho:
            bx, by = ho[-1]
            if not (np.isnan(bx) or np.isnan(by)):
                ball = np.array([bx * length, by * width])

        X[i] = frame_features(np.array(h_pts, dtype=float).reshape(-1, 2),
                              np.array(a_pts, dtype=float).reshape(-1, 2),
                              ball, pitch)
    return X