"""Train the model. It only needs the open matches in external/idsse and the tracking cache.

    python train_model.py               # models/possession.txt, on all seven matches
    python train_model.py --cv          # one model per match, leaving that match out

score_external.py uses the --cv models, so every match is scored by a model that never saw it.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "starter"))
import external  # noqa: E402
import model  # noqa: E402


def save(booster, path):
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    booster.save_model(path)
    print("wrote {}".format(os.path.relpath(path, HERE)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cv", action="store_true", help="train one model per held-out match")
    args = parser.parse_args()

    matches = external.match_ids()
    if args.cv:
        for match_id in matches:
            save(model.train([m for m in matches if m != match_id]), model.cv_path(match_id))
    else:
        save(model.train(matches), model.MODEL_PATH)
