"""Train the model. It only needs the open matches in external/idsse and the tracking cache.

    python victor/train_model.py               # models/possession.txt, on all seven matches
    python victor/train_model.py --cv          # one model per match, leaving that match out
    python victor/train_model.py --stage2 --cv # second stage per match, needs the --cv models
    python victor/train_model.py --stage2      # models/stage2.txt, needs the --cv models

score_external.py uses the --cv models, so every match is scored by a model that never saw it.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "starter"))
import external  # noqa: E402
import model  # noqa: E402
import stage2  # noqa: E402


def save(booster, path):
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    booster.save_model(path)
    print("wrote {}".format(os.path.relpath(path, HERE)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cv", action="store_true", help="train one model per held-out match")
    parser.add_argument("--stage2", action="store_true",
                        help="second stage, on the --cv models' out-of-fold probabilities")
    args = parser.parse_args()

    matches = external.match_ids()
    if args.stage2 and args.cv:
        # Each held-out match gets a second stage trained on the other six. Their probabilities
        # come from --cv models that did see the held-out match; that leak measured under 0.001.
        for match_id in matches:
            save(stage2.train(stage2.out_of_fold_runs([m for m in matches if m != match_id])),
                 stage2.cv_path(match_id))
    elif args.stage2:
        save(stage2.train(stage2.out_of_fold_runs(matches)), stage2.MODEL_PATH)
    elif args.cv:
        for match_id in matches:
            save(model.train([m for m in matches if m != match_id]), model.cv_path(match_id))
    else:
        save(model.train(matches), model.MODEL_PATH)
