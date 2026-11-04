"""Data in, submission out. This one command regenerates the file.

    python run.py                       # writes evaluation/results/tiki_data.csv
    python run.py --name my_test
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "starter"))
from baseline import smooth, write_submission  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "src"))
import ball  # noqa: E402
import dead_ball  # noqa: E402
import model  # noqa: E402
import parse  # noqa: E402
import states  # noqa: E402
import team  # noqa: E402

# Params
RESULTS_DIR = os.path.join(HERE, "evaluation", "results")
SMOOTH_WINDOW = 50      # +/- 2 s majority vote, the published floor's window
CHANGE_COST = 10        # what the best path pays for each H, A or D change, chosen on the test bed
DEFAULT_NAME = "tiki_data"


def predict(data, blocks, booster, cost=CHANGE_COST):
    """Labels for every frame. blocks is [(half, first_frame, last_frame)]."""
    labels = states.decode(model.scores(booster, data, blocks), blocks, cost, cost, cost)
    return states.abstain_outside(labels, blocks)


def predict_rules(data, blocks, window=SMOOTH_WINDOW):
    """The version without a model: nearest player, smoothing and the dead-ball rules."""
    ball_xy = ball.clean(data["ball_xy"])
    labels = smooth(team.nearest_team(dict(data, ball_xy=ball_xy)), window)
    labels = states.apply_dead_ball(labels, dead_ball.detect(data, ball_xy), blocks)
    return states.abstain_outside(labels, blocks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=DEFAULT_NAME, help="file name, without .csv")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="parse the raw tracking again")
    args = parser.parse_args()

    path = write_submission(os.path.join(args.out, args.name + ".csv"),
                            predict(parse.load(rebuild=args.rebuild_cache),
                                    half_video_blocks(GAME), model.load()))
    print("wrote {}".format(os.path.relpath(path, HERE)))
