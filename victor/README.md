# Victor's pipeline (team Tiki-Data)

Everything in this folder is Victor's approach, kept separate so the team can compare methods and
merge the best parts. The shared files stay at the repository root: the match data in `data/`, the
organisers' `starter/`, `evaluation/` and `games.py`.

From the repository root:

```bash
pip install -r victor/requirements-run.txt     # Python 3.11, any OS
python victor/run.py                           # -> evaluation/results/victor.csv
python evaluation/validate_submission.py evaluation/results/victor.csv
```

Unpack the match data into `data/` first, as described in [DATA.md](../DATA.md). How it works, which
data it uses, how it was tested and how to retrain it: **[TECHNICAL.md](TECHNICAL.md)**.

| Path | What it is |
|---|---|
| `run.py` | Labels the match with the shipped models and writes the submission file |
| `train_model.py` | Retrains both models on the open matches |
| `score_external.py` | Scores every version on the open-match test bed |
| `src/` | The pipeline: parsing, ball cleaning, clues, models, decoding, test-bed damage |
| `models/` | The two trained LightGBM models used by `run.py` |
| `notebooks/` | 04 walkthrough, 05 experiments, 06 scorecard, 07 replay |
| `cache/`, `external/` | Local only, never committed: arrays built from the match and the open matches |

Notebook outputs from 04 and 07 contain match data. `nbstripout` strips them on commit; keep them out
of cloud AI tools.
