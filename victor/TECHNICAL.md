# Technical document: Tiki-Data

More Than a Hack 2027, Phase 1: frame-by-frame possession for FC Barcelona vs Athletic Club.

> Upload this file as `<team_name>.md` (or export it as `<team_name>.pdf`), named after the
> registered team, together with `<team_name>.csv`.

> **Where things are.** This is Victor's folder in the team repository. Paths to the shared files
> (`data/`, `starter/`, `evaluation/`, `games.py`) are from the repository root; every other path
> (`src/`, `models/`, `notebooks/`, `cache/`, `external/`) is inside `victor/`. Commands run from
> the repository root.

## 1. Summary

We label every frame of the tactical video as `H`, `A`, `D` or `X` in four steps:

1. **Clues.** From the automatic tracking (ATD) alone we compute 133 numbers per frame: who is near
   the ball, who touched it last and who touches it next, whether the ball is out, still or unseen,
   how fast the players move, which way each team pushes, how much pressure the player on the ball
   is under, and each team's shape (defensive line, length, width, compactness).
2. **Per-frame model.** A LightGBM classifier turns those clues into a probability of `H`, `A` and
   `D` for each frame. It is trained only on seven open Bundesliga matches whose tracking we
   damaged to look like the provided ATD.
3. **Second stage.** A second LightGBM model reads those probabilities in context (their rolling
   means, minimums, maximums and slopes over up to ±10 s) and gives corrected probabilities. It is
   trained on the first model's out-of-fold probabilities only.
4. **Best path.** A three-state Viterbi decoder picks, per half, the `H` / `A` / `D` sequence with
   the highest total log-probability, paying a fixed cost of 10 for every change of state, with
   possessions at least 1.4 s long. This turns flickering frame guesses into whole possessions.

Frames outside the two graded halves are `X`. Inside them every frame gets `H`, `A` or `D`.

**Held-out check of the submitted models** on Metrica Sports' public sample game 3, a match no
model of ours was trained on, laid out like the competition bundle by `victor/testbed_metrica3.py`
and scored with the official `possession_metrics.py`. "Damaged" uses only DATA.md figures (41% of
the ball missing, about 20.5 of 22 players tracked, position noise):

| Metrica sample game 3 | spellF1 clean | spellF1 damaged | macroF1 damaged |
|---|---|---|---|
| Baseline: nearest player, ±50-frame majority vote | 0.368 | 0.276 | 0.717 |
| Rule-based alternative the team built and dropped | 0.518 | 0.474 | 0.753 |
| **Our pipeline (submitted)** | **0.613** | **0.589** | **0.794** |

These are guides, not a forecast of our real score, which we expect to be lower (see section 7).
The test-bed figures in section 4.4 were measured during development, before the final retraining
described in section 4.2.

## 2. Approach, and why this rather than something else

**The problem we had to solve first was measurement.** The provided match has no labels apart from
the public 5 minutes, which the rules allow only as a pipeline check. There is no leaderboard. So
any choice made by looking at the provided match would be a guess. We therefore built a test bed on
other matches with official per-frame labels (section 4), damaged to resemble our tracking, and
made every design decision there, scored with the official `possession_metrics.py`.

**Why a learned per-frame model.** Hand-written rules (nearest player, smoothing, dead-ball rules)
were our first version and are still in the code (`run.py: predict_rules`). They cannot weigh many
weak signals against each other, and they collapse when the ball is unseen, which happens on about
41% of graded frames (`DATA.md`). Gradient-boosted trees combine dozens of clues, cope with missing
values natively, train in minutes on a laptop and are deterministic with a fixed seed. Team-level
clues (which side is pushing forward, how fast each team runs, how spread out it is) keep working
when the ball is missing. On the test bed, moving from rules to the model raised team accuracy from
about 78% to 88%.

**Why Viterbi on top.** spellF1 counts whole possessions, so a single false switch splits one
possession into two and loses both. Frame-wise argmax of the model flickers badly (on the test bed
it scores far lower on its own). The Viterbi decoder finds the globally best sequence for
a half given a cost per change, which removes flicker without the lag that a majority-vote window
adds at real changes. It has one parameter, the change cost, set to 10 on the test bed. It is not
learned.

**What we did not build.** A sequence neural net (LSTM/TCN) needs much more labelled data than
seven matches and is harder to make byte-reproducible. A learned HMM would add little over
LightGBM + Viterbi. Our own ball detector on the video, or fusing the broadcast tracking, would
attack the missing ball directly, but needs a GPU pipeline and timeline alignment we did not have
time for.

## 3. Assets: used, checked against, dropped

| Asset | Status | How |
|---|---|---|
| Tactical ATD (`BAR-ATH_tactical_atd.txt` + `.xml`) | **Model input** | Parsed once into arrays in metres (`src/parse.py`), cached in `cache/`. Ball and player positions and speeds, team of each track slot (reusing `starter/baseline.py`'s `team_of_each_track`). |
| `games.py` constants | Used | Pitch size and the graded frame blocks of each half. Frames outside them are `X`. |
| Tactical ATD, for training | **Not used** | The organisers allow the provided match for validation only. Nothing measured on it is used in training: the damage applied to the open matches is built from figures published in `DATA.md` (section 4.2). |
| Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`) | **Check only, not model input** | `src/tags.py` measures how often our team labels agree with the team implied by attacking and defending phase tags. It is automatic data, so this is allowed, and it is our only independent check on the real match: team agreement 0.85 for the final pipeline vs 0.73 for the rule-only version. It was never used to train or tune anything. |
| Tactical video | Not used as input | |
| Homography (tactical and broadcast) | Dropped | Not needed: the ATD is already in pitch coordinates. |
| Broadcast video and broadcast ATD | Dropped | Not synchronised with the tactical timeline, has cuts and replays; aligning it was out of scope. |
| Public 5-minute ground truth | Pipeline check only | Used to confirm the file lines up frame for frame. Never used to train, tune or choose between versions. |

## 4. Training data

### 4.1 The only labelled dataset: IDSSE (open DFL data)

| | |
|---|---|
| Dataset | Bassek, M. et al. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data. |
| Licence | CC-BY 4.0 |
| Source | https://huggingface.co/datasets/pysport/idsse-data |
| Matches | 7 Bundesliga matches (ids J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9) |
| Files used | `DFL_02_01_matchinformation_*` (home/away team ids) and `DFL_04_03_positions_raw_observed_*` (25 fps positions and speeds of players and ball) |
| Labels | DFL's official per-frame `BallStatus` (in play / dead) and `BallPossession` (home / away), shipped with the position data |
| Who labelled it | DFL / Sportec Solutions, as part of their official data. **Not labelled by us.** No hand labels of any kind were made by the team. |
| How much | All frames of both halves of all 7 matches. Training uses every 5th labelled frame. |

No labelled data from the provided game is used anywhere.

**Converting the labels** (`src/external.py`): a frame is `D` where `BallStatus` is 0, otherwise `H`
or `A` from `BallPossession`. DFL gives possession to whoever touches the ball, so about 30% of its
possessions last under 2 s, whereas the challenge counts only controlled possession. We therefore
fold every possession shorter than 2 s (50 frames) into the previous team's possession in that half,
or into the next team's if it directly follows a dead ball. This takes the open data from about 437
to 279 possessions per match (mean length 7.2 s to 11.3 s), changes about 2% of frames and no dead
frames. Retraining on these labels gained +0.010 spellF1 on the test bed, better on 5 of 7 matches.

### 4.2 Damaging the open tracking (`src/degrade.py`)

The DFL tracking is far cleaner than the provided automatic tracking, so both training and scoring
on it as-is would mislead. Each copy is damaged with a fixed seed:

| Real ATD property | What we do to the open data |
|---|---|
| Ball missing on about 41% of graded frames (DATA.md) | Remove the ball by drawing seen/unseen stretch lengths from log-normal distributions whose means give about 59% visible after damage, as DATA.md reports |
| Ball lost more often at stoppages | Long gaps made rare in play; dead time uses the full gap distribution with seen stretches capped; the edges of the dead-time zone are jittered by up to 3 s so the model cannot learn the exact boundary from the damage |
| About 20.5 of 22 players tracked | Remove player stretches to about 20.4 tracked |
| Smooth ball path | Smooth the ball lightly (5-frame mean, a judgement value) |
| Noise | 0.05 m ball noise, 0.3 m player position noise, 0.3 m/s player speed noise |
| The tracker occasionally duplicates someone, so more than 22 objects are tracked (DATA.md) | Three spare slots per team that now and then copy a teammate 1 to 3 m away, at a judgement rate |

**Nothing is measured on the provided match.** Every damage setting is either a figure published in
`DATA.md` or a judgement value, so training needs no competition data at all: the final models were
retrained on a machine that did not have the competition data.

### 4.3 Training set and model

| Setting | Value |
|---|---|
| Model | LightGBM 4.7, `multiclass` (H, A, D), 150 trees, learning rate 0.07, 63 leaves, min 200 rows per leaf, feature fraction 0.7, seed 0, `deterministic=True` |
| Rows | Every 5th labelled frame of each match, from 2 differently damaged copies (seeds 10 and 11), each frame also included with the two teams swapped |
| Why swap teams | Every paired clue has an `h_` and `a_` version; swapping them stops the model learning anything about home sides or particular clubs |
| Features | 133 clues (section 5) |
| Final model | Trained on all 7 matches, saved as `models/possession.txt` |
| Test-bed models | One per match, trained on the other 6 (`train_model.py --cv`), so every test-bed score comes from a model that never saw that match. Test damage uses seeds 0, 1, 2, different from training. |
| Second stage (`src/stage2.py`) | LightGBM, 200 trees, learning rate 0.05, 31 leaves, `deterministic=True`. About 250 inputs per frame built only from the first model's H/A/D probabilities: their log-ratios and their rolling mean, minimum, maximum and slope over ±0.5, 1, 2, 5 and 10 s, plus frames since and until the first model last favoured each state. Trained on out-of-fold probabilities (each match scored by the first-stage model that left it out), every 5th frame of seeds 10 and 11, with teams swapped; predictions are averaged with their team-swapped mirror. Saved as `models/stage2.txt`. A fully nested version (no first-stage model ever saw the held-out match) scored within 0.001, so the small leak in the held-out second-stage models does not inflate the test-bed score. |

### 4.4 How far to trust the test bed

The baselines land close to their published full-match numbers (±50-frame baseline: 0.2431 on the
test bed vs 0.2433 published), which suggests the damage is about right for spellF1.

**A second, independent check.** We also scored the final model on Metrica Sports' three public
sample games (https://github.com/metrica-sports/sample-data), never used for training, with
answers derived from Metrica's own event data and folded the same way. It scores spellF1 0.784
clean and 0.691 damaged, against 0.439 and 0.289 for the ±50 baseline. This is another tracking provider and an independent possession
definition, so it suggests the model is not tied to the DFL data. Adding these games to training
did not change the test-bed score, so they are kept as a check only. 
**A third check.** SkillCorner's open data (https://github.com/SkillCorner/opendata, MIT,
"SkillCorner open data, in collaboration with PySport") gives 20 A-League matches of broadcast
tracking with SkillCorner's own automatic per-frame possession; dead ball is taken as the gaps
between its phases of play. Never used for training. The pipeline scores 0.502 clean and 0.493
damaged against 0.280 and 0.199 for the baseline, better on all 20 matches. The absolute level is
lower because broadcast tracking sees only about 9 players per frame and the labels come from
another automatic model, but the gap to the baseline (about 2.5 times) is the same as on the other
two sets, which is what we would not see if the model were fitted to the DFL data.

The Metrica and SkillCorner converters were one-off scripts and are not part of the pipeline.
All test sets differ from the real task in league, tracking system or possession definition, and
our damage drops the ball at random rather than in crowds and in the air. Gaps below about
0.04 spellF1 between two versions on one match are within noise, so every change was kept only if it
won on at least 5 of the 7 matches and did not lower the Metrica score.

## 5. The pipeline

```
data/tactical/*_atd.txt + .xml
  -> src/parse.py      arrays in metres, cached
  -> src/ball.py       drop points > 5 m from the local median, fill gaps up to 1 s with straight lines
  -> src/features.py   133 clues per frame (uses src/dead_ball.py and src/touches.py)
  -> src/model.py      H, A, D probabilities from models/possession.txt
  -> src/stage2.py     corrected probabilities from models/stage2.txt
  -> src/states.py     late-kickoff wait set to D; Viterbi per half, change cost 10,
                       possessions at least 1.4 s; X outside the halves
  -> evaluation/results/victor.csv   (repository root)
```

The 133 clues, in groups:

| Group | Clues |
|---|---|
| Ball proximity (per team) | Distance from ball to nearest player; players within 3 m and 8 m; each also as centred means over ±0.5, 1, 2 and 5 s |
| Who is closer | Share of frames where each team is nearer the ball, over the same four windows |
| Touches (`src/touches.py`) | Frames since the team's last touch and until its next one; its touches within ±2 s and ±5 s; how many touches in a row the team had at the last and next touch (capped at 10) |
| Ball motion | Velocity change, ball speed, distance from the touchline, ball seen now and over ±2 / ±5 s, length of the current gap, ball outside the pitch, ball still for 1 s |
| Team shape (per team) | Mean position along its attacking direction, how much it advanced in the last second, mean speed, spread, ball position along its attacking direction |
| Play state | Mean speed of all tracked players (2 s) and over ±5 s; the rule-based dead-ball flag |
| Pressure (per team) | Around the team's player nearest the ball: opponents within 2 m and 5 m, distance to the nearest opponent; centred means over ±0.5 to ±5 s |
| Team shape (per team, robust) | Defensive-line height (2nd/3rd deepest player), length, width and compactness, using quantiles and medians so a swapped or duplicated track barely moves them; the gap between the two lines; centred means over ±1 and ±5 s. These need no ball. |

A **touch** is a local peak in the ball's velocity change (after a 5-frame smoothing), in the top
15% of the match's changes, with a player within 3 m; it is credited to the nearest player's team.
The threshold is a quantile, so it adapts to how smooth each tracking system is.

## 6. Possession rules implemented

`CHALLENGE.md` defines possession as controlled possession, held until the other team gains
control. This is where our system draws the lines.

| Situation | What our system does |
|---|---|
| **Ball dead** | `D`, never owned by a team. The model learns "dead" from DFL's `BallStatus`. It is helped by a rule-based clue (`src/dead_ball.py`): dead where all players average under 1.2 m/s, or under 2.2 m/s while the ball is unseen for 2 s or more, outside the pitch by over 1 m, or still for 1 s; dead and live stretches under 3 s are then removed. |
| **Loose or contested ball** | Stays with the last team. The Viterbi cost of 10 per change means a switch only happens when the evidence for the other team holds over time; a few ambiguous frames never flip it. |
| **Touch without control, deflection, lost aerial duel** | Not a change. In the training labels every possession under 2 s is folded into the previous team's, so the model is taught that brief contacts do not change possession. The touch-run clues ("touches in a row by this team") act as the challenge's "consecutive passes" proxy for control. We have no ball height, so aerial duels are not detected as such; they are covered only by the same 2 s rule. |
| **Challenges / tackles** | A change is called only once the winning team's clues (closer to the ball, touching it, pushing forward) dominate for long enough to pay the change cost. The change point is where the decoder places it, not snapped to a specific touch. |
| **Restarts** | After a `D` stretch the decoder picks whichever team the clues favour, at the same cost of 10 as any other change. Throw-ins, corners and free kicks are not detected as separate events. |
| **Waiting for a kickoff** | A half can be graded from before its kickoff. `states.kickoff_wait` finds the last slow frame before the first 3 s of match pace (players' mean speed above 2 m/s); if that is more than 30 s into the half, everything before it is `D`. No frame numbers are used. |
| **Minimum possession** | The decoder never produces a possession shorter than 1.4 s (35 frames). Longer minimums scored higher on our relabelled answers but lower on DFL's original ones, so we kept the one that holds on both. |
| **Halves** | Each half is decoded on its own, so no possession crosses half-time. Frames outside the graded halves (from `games.py`) are `X`; we never abstain inside them, because on the ranking metric an `X` costs the same as repeating the last label and costs more on macroF1. |

## 7. Limitations and what we would fix

**Where it fails, from the test bed and the checks on the real match:**

- **Kickoffs in general.** The wait before a late kickoff is handled by a player-speed rule; a
  detector for the kickoff itself (ball on the centre spot, teams in their own halves) did not work.
- **Short possessions are under-found.** On the test bed we predict 251 possessions against 279 true
  ones per match; the misses are mostly short possessions the change cost absorbs.
- **Loose possession edges.** spellF1 on the test bed falls sharply as the required overlap rises: we
  usually find the right possession but place its start and end late or early. Snapping changes to
  detected touches did not help.
- **Tracking faults.** The damage now copies team swaps and duplicates at the real rates, and the
  model is trained on it, but it does not simulate a track jumping to an opponent mid-life. Clues made
  robust to duplicates (merging same-team tracks within 1 m) gave no gain and were dropped.
- **Domain gap.** Trained on 7 Bundesliga matches with a different tracking system and a different
  possession definition. The real ball goes missing in crowds and in the air, ours at random. We
  find fewer touches per live minute on the real match than on the test bed, and a large part of the
  model's gain comes from speed and dead-ball clues whose scale may differ in the real ATD.
- **Unused information.** Smart Tagging is only a check, not an input. The broadcast tracking and
  the video, which could fill ball gaps, are unused.
- **Aerial duels and deflections** are handled only by the 2 s rule, not by explicit detection.

**With more time, in order:**

1. Team calls first: on the test bed, perfect team calls would raise spellF1 far more than perfect
   dead-ball calls.
2. Tighten possession edges. Snapping changes to detected touches was tried and did not help.
3. A set-piece detector from the tracking, to sharpen restarts.
4. Use Smart Tagging phases as soft model input, if a way to measure its effect without labels can
   be found.
5. Fill ball gaps with a ball detector on the tactical video or with the broadcast tracking.

## 8. Reproduction

Python 3.11. All commands from the repository root. Nothing needs a GPU.

### 8.1 Regenerate the submission from the shipped model

1. Unpack the match data into `data/` exactly as in `DATA.md` (`data/tactical/BAR-ATH_tactical_atd.txt`,
   `..._atd.xml`, `..._smart_tagging.xml`, and so on).
2. Install the pinned packages:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
   pip install -r victor/requirements-run.txt
   ```

3. Run:

   ```bash
   python victor/run.py
   python evaluation/validate_submission.py evaluation/results/victor.csv
   ```

`run.py` parses the ATD into `cache/` on first use (`--rebuild-cache` forces it again), applies
`models/possession.txt` and `models/stage2.txt` and writes `evaluation/results/victor.csv`. Running it twice gives
byte-identical files; check with `certutil -hashfile evaluation\results\victor.csv SHA256` (Windows)
or `sha256sum` after each run. `python victor/run.py --name <team_name>` writes it under the upload name.

### 8.2 Retrain from scratch

1. Download the 14 IDSSE files from https://huggingface.co/datasets/pysport/idsse-data into
   `victor/external/idsse/` (about 2.6 GB). For each match id in
   `J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9`:

   ```
   DFL_02_01_matchinformation_DFL-COM-*_DFL-MAT-<match id>.xml
   DFL_04_03_positions_raw_observed_DFL-COM-*_DFL-MAT-<match id>.xml
   ```

2. Run, in order:

   ```bash
   python victor/train_model.py --cv            # first stage per held-out match, in cache/models/
   python victor/train_model.py --stage2 --cv   # second stage per held-out match
   python victor/score_external.py              # test-bed table: baselines, rules and current pipeline
   python victor/train_model.py                 # final first stage on all 7 -> models/possession.txt
   python victor/train_model.py --stage2        # final second stage -> models/stage2.txt
   python victor/run.py                         # submission -> evaluation/results/victor.csv
   ```

Notes:

- Training needs only the open matches; the competition data is not read.
- Converted open matches are cached in `cache/external/` on first use.
- Seeds are fixed everywhere (damage seeds, LightGBM `seed=0`, `deterministic=True`, row-wise
  histograms, 8 threads). Retraining on the same machine gives a byte-identical
  `models/possession.txt` and `models/stage2.txt`; on a different machine, floating-point differences in LightGBM could in
  principle change the model, which is why the trained model ships in `models/` and `run.py` uses it.
- `requirements-run.txt` pins what the pipeline needs and installs on Windows, macOS and Linux.
  `requirements-notebooks.txt` adds the notebook tooling; `requirements-lock.txt` is the full
  Windows freeze the work was done in.
- `python victor/score_external.py --detail` prints the full scorecard (per-class F1, confusion matrices,
  transition latency). Notebooks `04` to `07` in `notebooks/` walk through the pipeline, the
  experiments, the scorecard and a 2D replay; their outputs contain match data and are not
  published.

### 8.3 Code map

| File | Role |
|---|---|
| `run.py` | Data in, submission out |
| `train_model.py` | Trains the final or the per-match (`--cv`) models, first or second stage (`--stage2`) |
| `score_external.py` | Test bed: scores every version on the damaged open matches with the official scorer |
| `src/parse.py` | ATD text to cached arrays |
| `src/ball.py` | Ball cleaning |
| `src/dead_ball.py` | Rule-based dead-ball signal |
| `src/touches.py` | Touch detection and touch clues |
| `src/features.py` | The 133 clues; team swap |
| `src/model.py` | First-stage LightGBM training and scoring |
| `src/stage2.py` | Second-stage model on the first stage's probabilities |
| `src/states.py` | Viterbi decoder, `X` outside the halves |
| `src/team.py` | Nearest-team rule (baseline and rule-only version) |
| `src/external.py` | IDSSE conversion and label folding |
| `src/degrade.py` | Damage of the open tracking |
| `src/tags.py` | Smart Tagging agreement check |
| `src/report.py`, `src/animate.py` | Notebook plots and 2D replay; not used by `run.py` |
