"""Sanity checks on a submission file (does not use the public GT)."""
from pathlib import Path

import pandas as pd

import config


def validate(path: Path) -> None:
    df = pd.read_csv(path)
    assert list(df.columns) == ["frame", "label"], f"bad columns: {list(df.columns)}"
    assert len(df) == config.N_FRAMES, f"expected {config.N_FRAMES} rows, got {len(df)}"
    assert (df["frame"].values == range(config.N_FRAMES)).all(), "frames must be 0..n-1"
    assert set(df["label"]) <= set("HADX"), f"bad labels: {set(df['label']) - set('HADX')}"

    # Graded frames must not be X
    graded_mask = pd.Series(False, index=df.index)
    graded_mask.iloc[config.HALF1_START:config.HALF1_END] = True
    graded_mask.iloc[config.HALF2_START:config.HALF2_END] = True
    n_x_graded = (df.loc[graded_mask, "label"] == "X").sum()
    assert n_x_graded == 0, f"{n_x_graded} graded frames are labelled X"

    print(f"OK — {path} is a valid submission")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    import sys
    validate(Path(sys.argv[1]) if len(sys.argv) > 1 else config.RESULTS_DIR / "solution.csv")