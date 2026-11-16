"""Train a possession classifier from Metrica sample matches."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from config import LABELS, TRAINING_DIR, MODELS_DIR, PITCH
from features import extract_frame_features_metrica

POSSESSION_EVENT_TYPES = {"PASS", "BALL RECOVERY", "CARRY", "SHOT", "SET PIECE"}
STOPPAGE_EVENT_TYPES = {"BALL OUT", "FOUL", "CARD", "CORNER", "OFFSIDE",
                        "GOAL KICK", "THROW IN", "FREE KICK"}


def _frames_from_events(events, n_frames: int, home_team: str = "home") -> np.ndarray:
    labels = np.full(n_frames, "", dtype="<U1")
    for _, ev in events.iterrows():
        etype = str(ev.get("Type", "")).strip().upper()
        team = str(ev.get("Team", "")).strip().lower()
        try:
            start = int(ev["Start Frame"]); end = int(ev["End Frame"])
        except (KeyError, ValueError, TypeError):
            continue
        start, end = max(0, start), min(n_frames, end + 1)
        if etype in STOPPAGE_EVENT_TYPES:
            labels[start:end] = "D"
        elif team == home_team:
            labels[start:end] = "H"
        elif team:
            labels[start:end] = "A"
    prev = "D"
    for i in range(n_frames):
        if labels[i] == "":
            labels[i] = prev
        else:
            prev = labels[i]
    return labels


def _learn_transition_matrix(labels_list):
    idx = {s: i for i, s in enumerate(LABELS)}
    counts = np.zeros((len(LABELS), len(LABELS)))
    for labels in labels_list:
        prev = labels[0]
        for cur in labels[1:]:
            counts[idx[prev], idx[cur]] += 1
            prev = cur
    counts += 1e-2
    return counts / counts.sum(axis=1, keepdims=True)


def _active_columns(X: np.ndarray, min_distinct: int = 2) -> np.ndarray:
    """Column indices with >= min_distinct non-NaN unique values.

    HistGradientBoostingClassifier bins thresholds from distinct values and
    crashes on 0- or 1-valued columns. Dropping them is also the right model
    choice: a constant column carries no information. NaN-valued columns are
    also dropped here if they are entirely NaN in this slice.
    """
    active = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col[~np.isnan(col)]
        if len(np.unique(col)) >= min_distinct:
            active.append(j)
    return np.array(active, dtype=int)


def _load_match(game_dir: Path):
    paths = {k: game_dir / f"{k}.csv" for k in ("events", "home", "away")}
    if not all(p.exists() for p in paths.values()):
        print(f"  skipping {game_dir.name} (missing files)")
        return None
    events = pd.read_csv(paths["events"])
    X = extract_frame_features_metrica(paths["home"], paths["away"], PITCH)
    n = len(X)
    y = _frames_from_events(events, n, home_team="home")

    # TEST: flip H↔A. If the Metrica sample data's "Home" (filename) is not the
    # same team as "Home" in the events file, the model learned the opposite
    # sign. Flipping reveals whether the team signal is inverted.
    y = np.where(y == "H", "TMP", y)
    y = np.where(y == "A", "H", y)
    y = np.where(y == "TMP", "A", y).astype("<U1")

    print(f"  {game_dir.name}: {n} frames, "
          f"labels={dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def train() -> None:
    game_dirs = sorted(TRAINING_DIR.glob("Sample_Game_*"))
    if not game_dirs:
        raise SystemExit("No training matches. Run `python solution/run.py download` first.")

    datasets = []
    for gd in game_dirs:
        result = _load_match(gd)
        if result is not None:
            datasets.append((gd.name, *result))
    if not datasets:
        raise SystemExit("No usable training matches found.")

    X_all = np.vstack([d[1] for d in datasets])
    y_all = np.concatenate([d[2] for d in datasets])
    groups = np.concatenate([[d[0]] * len(d[1]) for d in datasets])

    keep = ~np.isnan(X_all[:, 1:]).all(axis=1)
    X_all, y_all, groups = X_all[keep], y_all[keep], groups[keep]
    print(f"\nTraining rows after drop: {len(X_all)}")

    if len(datasets) < 2:
        raise SystemExit("Need at least 2 training matches for hold-out validation.")

    folds = list(GroupKFold(n_splits=len(datasets)).split(X_all, y_all, groups))
    fold_scores = []
    for fold_idx, (tr, te) in enumerate(folds, 1):
        active_fold = _active_columns(X_all[tr])
        dropped = sorted(set(range(X_all.shape[1])) - set(active_fold.tolist()))
        if dropped:
            print(f"  fold {fold_idx}: dropping constant cols {dropped}")
        X_tr = X_all[tr][:, active_fold]
        X_te = X_all[te][:, active_fold]

        w_tr = compute_sample_weight("balanced", y_all[tr])
        clf = HistGradientBoostingClassifier(
            max_iter=200, max_depth=3, learning_rate=0.05, random_state=0,
        )
        clf.fit(X_tr, y_all[tr], sample_weight=w_tr)
        preds = clf.predict(X_te)
        f1 = f1_score(y_all[te], preds, average="macro", labels=list(LABELS))
        fold_scores.append(f1)
        print(f"  hold-out fold {fold_idx}: macro-F1 = {f1:.3f}")
    print(f"  mean hold-out macro-F1 = {np.mean(fold_scores):.3f}")

    active_all = _active_columns(X_all)
    dropped_all = sorted(set(range(X_all.shape[1])) - set(active_all.tolist()))
    if dropped_all:
        print(f"  final fit: dropping constant cols {dropped_all}")
    X_active = X_all[:, active_all]

    w_all = compute_sample_weight("balanced", y_all)
    clf = HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.05, random_state=0,
    )
    clf.fit(X_active, y_all, sample_weight=w_all)

    transition = _learn_transition_matrix([d[2] for d in datasets])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(clf, MODELS_DIR / "classifier.joblib")
    np.save(MODELS_DIR / "transition.npy", transition)
    np.save(MODELS_DIR / "active_columns.npy", active_all)
    (MODELS_DIR / "meta.json").write_text(json.dumps({
        "n_training_frames": int(len(X_all)),
        "holdout_macro_f1_mean": float(np.mean(fold_scores)),
        "active_columns": active_all.tolist(),
        "label_counts": {k: int(v) for k, v in zip(*np.unique(y_all, return_counts=True))},
    }, indent=2), encoding="utf-8")

    print(f"\nSaved classifier + transition matrix to {MODELS_DIR}")
    print("Transition matrix (H, A, D):")
    print(np.round(transition, 3))


if __name__ == "__main__":
    train()