"""Compare versions of the pipeline on the open matches. This is how to choose between them.

    python score_external.py                # every version in VERSIONS, damaged tracking
    python score_external.py --clean        # the same on the undamaged tracking

Scored with the official possession_metrics, so the numbers mean the same as on the board. The
baseline rows are also the check on the test bed itself: the closer they sit to the published
full-match baseline scores, the more this damaged data behaves like the real thing.

Published full match, for comparison:
    baseline_raw        macroF1 0.6532   spellF1 0.0486
    baseline_smooth25   macroF1 0.6752   spellF1 0.1996
    baseline_smooth50   macroF1 0.6724   spellF1 0.2433
    baseline_smooth125  macroF1 0.6356   spellF1 0.2242
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "starter"))
sys.path.insert(0, os.path.join(HERE, "evaluation"))
import degrade  # noqa: E402
import external  # noqa: E402
import model  # noqa: E402
import possession_metrics as pm  # noqa: E402
import states  # noqa: E402
import team  # noqa: E402
from baseline import smooth  # noqa: E402
from run import predict, predict_rules  # noqa: E402

# Params
SEEDS = (0, 1, 2)   # each match is damaged this many ways, so one lucky draw cannot decide


def baseline(window):
    def version(data, blocks, match_id):
        return states.abstain_outside(smooth(team.nearest_team(data), window), blocks)
    return version


def rules(data, blocks, match_id):
    return predict_rules(data, blocks)


def current(data, blocks, match_id):
    """Scored by the model that never saw this match. See train_model.py --cv."""
    return predict(data, blocks, model.load(model.cv_path(match_id)))


# name -> function(data, blocks, match_id) returning labels. Add each new idea here.
VERSIONS = {
    "baseline_raw": baseline(0),
    "baseline_smooth25": baseline(25),
    "baseline_smooth50": baseline(50),
    "baseline_smooth125": baseline(125),
    "rules": rules,
    "current": current,
}


def evaluate(gt, labels):
    result = pm.evaluate(gt, pm.label_codes(labels))
    return (result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
            result["n_gt_spells"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", action="store_true", help="skip the damage")
    args = parser.parse_args()

    profile = None if args.clean else degrade.load_profile()
    scores = {name: [] for name in VERSIONS}
    for match_id in external.match_ids():
        data = external.load(match_id)
        gt = pm.label_codes(data["labels"])
        blocks = [tuple(block) for block in data["blocks"].tolist()]
        for seed in ((0,) if args.clean else SEEDS):
            damaged = data if args.clean else degrade.degrade(data, profile, seed)
            for name, version in VERSIONS.items():
                scores[name].append(evaluate(gt, version(damaged, blocks, match_id)))

    print("{:<20} {:>8} {:>8} {:>6} {:>6}  spellF1 by match range".format(
        "version", "spellF1", "macroF1", "poss", "true"))
    for name, rows in sorted(scores.items(), key=lambda item: -np.mean(item[1], axis=0)[0]):
        rows = np.array(rows)
        print("{:<20} {:>8.4f} {:>8.4f} {:>6.0f} {:>6.0f}  {:.3f} .. {:.3f}".format(
            name, rows[:, 0].mean(), rows[:, 1].mean(), rows[:, 2].mean(), rows[:, 3].mean(),
            rows[:, 0].min(), rows[:, 0].max()))
    print("\nGaps below about 0.04 are noise on one match; these are means over many runs.")
