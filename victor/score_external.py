"""Compare versions of the pipeline on the open matches. This is how to choose between them.

    python victor/score_external.py         # every version in VERSIONS, damaged tracking
    python victor/score_external.py --clean # the same on the undamaged tracking

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
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)   # the shared repository; this folder is victor/
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(REPO, "starter"))
sys.path.insert(0, os.path.join(REPO, "evaluation"))
sys.path.insert(0, REPO)
import degrade  # noqa: E402
import external  # noqa: E402
import model  # noqa: E402
import stage2  # noqa: E402
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
    return predict(data, blocks, model.load(model.cv_path(match_id)),
                   stage2.load(stage2.cv_path(match_id)))


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
    """The numbers the comparison looks at, for one match and one version."""
    result = pm.evaluate(gt, pm.label_codes(labels))
    return {"spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
            "poss": result["n_pred_spells"], "true_poss": result["n_gt_spells"],
            "alive_dead_f1": result["alive_dead_f1"],
            "team_acc": result["team_acc_both_alive"]}


def matches(seeds=SEEDS, clean=False):
    """Yield (match_id, seed, damaged data, blocks, gt) for every run of the test bed."""
    profile = None if clean else degrade.load_profile()
    for match_id in external.match_ids():
        data = external.load(match_id)
        gt = pm.label_codes(data["labels"])
        blocks = [tuple(block) for block in data["blocks"].tolist()]
        for seed in ((0,) if clean else seeds):
            damaged = data if clean else degrade.degrade(data, profile, seed)
            yield match_id, seed, damaged, blocks, gt


def score_versions(versions=None, seeds=SEEDS, clean=False):
    """One row per version, match and seed."""
    versions = VERSIONS if versions is None else versions
    rows = []
    for match_id, seed, damaged, blocks, gt in matches(seeds, clean):
        for name, version in versions.items():
            row = evaluate(gt, version(damaged, blocks, match_id))
            row.update(version=name, match=match_id, seed=seed)
            rows.append(row)
    return pd.DataFrame(rows)


def what_if(version, seeds=SEEDS):
    """What a version would score with perfect dead-ball calls, or with perfect team calls.

    Only possible here, where every frame has a label. The larger of the two gains shows which
    half of the problem is costing more.
    """
    rows = []
    for match_id, seed, damaged, blocks, gt in matches(seeds):
        labels = version(damaged, blocks, match_id)
        truth = np.array(pm.LABELS, dtype="<U1")[gt]

        perfect_dead = states.apply_dead_ball(np.where(labels == "X", "D", labels), truth == "D",
                                              blocks)
        perfect_team = labels.copy()
        both_alive = np.isin(labels, ("H", "A")) & np.isin(truth, ("H", "A"))
        perfect_team[both_alive] = truth[both_alive]

        rows.append({"match": match_id, "seed": seed,
                     "as_it_is": evaluate(gt, labels)["spellF1"],
                     "perfect_dead_ball": evaluate(gt, perfect_dead)["spellF1"],
                     "perfect_team": evaluate(gt, perfect_team)["spellF1"]})
    return pd.DataFrame(rows)


def scorecard(versions=None, seeds=SEEDS):
    """Every number the official scoreboard reports, per version, pooled over the test bed.

    Returns (card, confusion): card is a DataFrame with one row per version; confusion maps each
    version to its 4x4 frame counts, rows the truth and columns the prediction, in H A D X order.
    Per-class figures are recomputed from the pooled counts rather than averaged.
    """
    versions = VERSIONS if versions is None else versions
    pooled = {name: np.zeros((pm.N_LABELS, pm.N_LABELS), dtype=np.int64) for name in versions}
    rows = {name: [] for name in versions}
    for match_id, seed, damaged, blocks, gt in matches(seeds):
        for name, version in versions.items():
            result = pm.evaluate(gt, pm.label_codes(version(damaged, blocks, match_id)))
            pooled[name] += np.array(result["confusion"])
            loose, tight = result["transitions"][pm.TAUS[-1]], result["transitions"][pm.TAUS[0]]
            row = {"spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
                   "kappa": result["kappa"], "accuracy": result["accuracy"],
                   "alive_dead_f1": result["alive_dead_f1"],
                   "team_acc": result["team_acc_both_alive"],
                   "trans_f1_0.5s": tight["f1"], "trans_f1_1s": loose["f1"],
                   "mean_latency_frames": loose["mean_latency"],
                   "poss": result["n_pred_spells"], "true_poss": result["n_gt_spells"],
                   "mean_poss_s": result["mean_spell_duration_s"],
                   "true_mean_poss_s": result["gt_mean_spell_duration_s"],
                   "fragmentation": result["fragmentation"], "home_share_err": result["share_err"]}
            for iou in pm.SPELL_IOUS:
                row["spellF1@{:.1f}".format(iou)] = result["spell_f1"][iou]["f1"]
            rows[name].append(row)

    card = pd.DataFrame({name: pd.DataFrame(values).mean() for name, values in rows.items()}).T
    for name, counts in pooled.items():
        scored = pm.scored_confusion(counts)
        for k, label in enumerate(pm.SCORED):
            precision, recall, f1 = pm.per_class_prf(scored, k)
            card.loc[name, "precision_" + label] = precision
            card.loc[name, "recall_" + label] = recall
            card.loc[name, "f1_" + label] = f1
    return card.sort_values("spellF1", ascending=False), pooled


def summary(table):
    """One row per version, best first."""
    grouped = table.groupby("version")
    out = grouped[["spellF1", "macroF1", "poss", "true_poss", "alive_dead_f1", "team_acc"]].mean()
    out["worst_match"] = grouped["spellF1"].min()
    out["best_match"] = grouped["spellF1"].max()
    return out.sort_values("spellF1", ascending=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", action="store_true", help="skip the damage")
    parser.add_argument("--detail", action="store_true",
                        help="every scoreboard number and the confusion matrices")
    args = parser.parse_args()

    if args.detail:
        card, confusion = scorecard()
        print(card.round(4).T.to_string())
        for name, counts in confusion.items():
            print("\n{}  rows = truth, columns = prediction".format(name))
            print(pd.DataFrame(counts, index=pm.LABELS, columns=pm.LABELS).iloc[:3].to_string())
    else:
        print(summary(score_versions(clean=args.clean)).round(4).to_string())
    print("\nGaps below about 0.04 are noise on one match; these are means over many runs.")
