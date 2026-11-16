"""Turn Metrica's public sample game 3 into a stand-in competition bundle, to compare pipelines.

    python solution/testbed_metrica3.py <out_dir>            # clean tracking
    python solution/testbed_metrica3.py <out_dir> --damage   # tracking made as poor as the real ATD

Then run any pipeline with MTAH_DATA_DIR=<out_dir> and score it against <out_dir>/gt.csv with the
official evaluation/possession_metrics.py.

Why game 3: the submitted models are trained on Bundesliga matches only, and the team's other
approaches trained on Metrica games 1-2, so game 3 is a fair held-out match. It is OUTSIDE data -- nothing here
touches the competition match.

  * tracking: written in the competition ATD line format, kick-offs placed on the real match's
    kick-off frames (11185 and 113742) so each half sits inside the graded blocks of games.py
  * answer key: from game 3's own events -- the team of the last pass / carry / recovery /
    set piece / shot / ball-lost owns the ball; dead from a ball out, foul, offside, hand ball or
    a shot out or in, until the next set piece. Only frames between each half's kick-off and its
    last event are graded; the rest are X
  * --damage uses only figures published in DATA.md: ball missing on ~41% of frames, in bursts;
    ~20.5 of 22 players tracked; plus position noise
  * Smart Tagging: an empty file (game 3 has none), so no pipeline gets tag help

Needs kloppy (pip install kloppy==3.18.0, as in requirements-starter.txt) to read game 3.
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from games import DATA_DIR, GAMES  # noqa: E402

BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3"
FILES = ("Sample_Game_3_metadata.xml", "Sample_Game_3_tracking.txt", "Sample_Game_3_events.json")
N = GAMES["BAR-ATH"]["n_video_frames"]
L, W = GAMES["BAR-ATH"]["pitch"]
KICKOFF_VIDEO = {1: 11185, 2: 113742}
OWNING = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST"}
FPS = 25


def fetch():
    folder = os.path.join(DATA_DIR, "external")
    os.makedirs(folder, exist_ok=True)
    for name in FILES:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            print("downloading", name)
            urllib.request.urlretrieve(BASE + "/" + name, path)
    return [os.path.join(folder, name) for name in FILES]


def subtypes(event):
    s = event.get("subtypes")
    if not s:
        return ""
    if isinstance(s, dict):
        s = [s]
    return " ".join(x["name"].upper() for x in s)


def main(out_dir, damage):
    meta_path, raw_path, events_path = fetch()
    from kloppy import metrica
    ds = metrica.load_tracking_epts(meta_data=meta_path, raw_data=raw_path, coordinates="metrica")
    df = ds.to_df(engine="pandas")
    home_id, away_id = (t.team_id for t in ds.metadata.teams)
    team_of = {p.player_id: (0 if t.team_id == home_id else 1)
               for t in ds.metadata.teams for p in t.players}
    pids = [c[:-2] for c in df.columns if c.endswith("_x") and c != "ball_x"]
    frames = df["frame_id"].to_numpy().astype(int)              # 1..n
    period = df["period_id"].to_numpy()
    px = df[[p + "_x" for p in pids]].to_numpy(dtype=np.float64).copy()
    py = df[[p + "_y" for p in pids]].to_numpy(dtype=np.float64).copy()
    bx = df["ball_x"].to_numpy(dtype=np.float64).copy()
    by = df["ball_y"].to_numpy(dtype=np.float64).copy()
    n = len(df)

    events = json.load(open(events_path))["data"]
    kick = {}
    for e in events:
        if e["type"]["name"] == "SET PIECE" and "KICK OFF" in subtypes(e) and e["period"] not in kick:
            kick[e["period"]] = e["start"]["frame"]

    rng = np.random.default_rng(3)
    if damage:
        def bursts(mean_on, mean_off, size):
            keep = np.ones(size, dtype=bool)
            t, on = 0, True
            while t < size:
                length = int(rng.exponential(mean_on if on else mean_off)) + 1
                if not on:
                    keep[t:t + length] = False
                t += length
                on = not on
            return keep
        base = np.mean(~np.isfinite(bx))                       # game 3 already misses some
        extra = max(0.41 - base, 0.0) / max(1.0 - base, 1e-9)  # top up to ~41% missing
        ball_keep = bursts(60, 60 * extra / max(1 - extra, 1e-9), n) if extra > 0             else np.ones(n, dtype=bool)
        bx[~ball_keep] = np.nan
        by[~ball_keep] = np.nan
        for j in range(px.shape[1]):                           # ~20.5 of 22 tracked
            keep = bursts(500, 37, n)
            px[~keep, j] = np.nan
            py[~keep, j] = np.nan
        print("ball missing on {:.1%} of frames".format(np.mean(~np.isfinite(bx))))

    def speed(x, y):
        s = np.full(x.shape, np.nan)
        s[2:-2] = np.hypot((x[4:] - x[:-4]) * L, (y[4:] - y[:-4]) * W) * FPS / 4
        return s
    ps = speed(px, py)                     # the ATD speed column is smooth: take it from
    bs = speed(bx, by)                     # the tracks before position noise is added
    if damage:
        px += rng.normal(0, 0.3 / L, px.shape)
        py += rng.normal(0, 0.3 / W, py.shape)
        bx += rng.normal(0, 0.1 / L, n)
        by += rng.normal(0, 0.1 / W, n)
        ps += rng.normal(0, 0.3, ps.shape)

    video = np.full(n, -1)
    for p, k in kick.items():
        m = period == p
        video[m] = KICKOFF_VIDEO[p] + (frames[m] - k)
    ok = (video >= 0) & (video < N)

    # ---- tracking in the competition line format -----------------------------------------------
    tac = os.path.join(out_dir, "tactical")
    os.makedirs(tac, exist_ok=True)
    stem = os.path.join(tac, "BAR-ATH_tactical")
    with open(stem + "_atd.xml", "w") as f:
        f.write('<Metadata><Teams><Team id="{}"><Name>Team A</Name></Team><Team id="{}"><Name>'
                'Team B</Name></Team></Teams><Players>'.format(home_id, away_id))
        for i, p in enumerate(pids):
            f.write('<Player id="{}" teamId="{}"><Name>Track {}</Name></Player>'.format(
                p, home_id if team_of[p] == 0 else away_id, i + 1))
        f.write("</Players></Metadata>")
    row_of_video = np.full(N, -1)
    row_of_video[video[ok]] = np.flatnonzero(ok)
    empty = ";".join(["NaN,NaN,NaN"] * len(pids))
    with open(stem + "_atd.txt", "w", newline="\n") as f:
        for v in range(N):
            r = row_of_video[v]
            if r < 0:
                f.write("{}:{}:NaN,NaN,NaN\n".format(v + 1, empty))
                continue
            groups = ["{:.5f},{:.5f},{:.2f}".format(px[r, j], py[r, j], ps[r, j])
                      if np.isfinite(px[r, j]) and np.isfinite(ps[r, j]) else "NaN,NaN,NaN"
                      for j in range(len(pids))]
            ball = ("{:.5f},{:.5f},{:.2f}".format(bx[r], by[r], bs[r])
                    if np.isfinite(bx[r]) and np.isfinite(bs[r]) else "NaN,NaN,NaN")
            f.write("{}:{}:{}\n".format(v + 1, ";".join(groups), ball))
    with open(stem + "_smart_tagging.xml", "w") as f:
        f.write("<file><ALL_INSTANCES></ALL_INSTANCES></file>")

    # ---- answer key from the events -------------------------------------------------------------
    events = sorted(events, key=lambda e: e["start"]["frame"])
    owner_mark = np.full(n + 2, -1)
    restarts, dead_starts, last_event = [], [], {}
    for e in events:
        t, sub = e["type"]["name"].upper(), subtypes(e)
        s, end = e["start"]["frame"], e["end"]["frame"] or e["start"]["frame"]
        team = 0 if (e.get("team") or {}).get("id") == home_id else 1
        last_event[e["period"]] = max(last_event.get(e["period"], 0), s)
        if t in OWNING:
            owner_mark[s] = team
        if t == "SET PIECE":
            restarts.append(s)
        if t == "BALL OUT" or (t == "SHOT" and ("OUT" in sub or "GOAL" in sub)):
            dead_starts.append(end)
        elif t == "FAULT RECEIVED" or (t == "BALL LOST" and ("OFFSIDE" in sub or "HAND BALL" in sub)):
            dead_starts.append(s)
    owner = np.full(n + 2, -1)
    cur = -1
    for i in range(n + 2):
        cur = owner_mark[i] if owner_mark[i] >= 0 else cur
        owner[i] = cur
    dead = np.zeros(n + 2, dtype=bool)
    restarts = np.sort(restarts)
    for d in dead_starts:
        k = np.searchsorted(restarts, d, side="right")
        if k < len(restarts) and restarts[k] - d <= 90 * FPS:
            dead[d:restarts[k]] = True
    gt = np.full(N, "X", dtype="<U1")
    for r in np.flatnonzero(ok):
        f_ = frames[r]
        if not (kick[period[r]] <= f_ <= last_event[period[r]]):
            continue
        gt[video[r]] = "D" if dead[f_] else ("H" if owner[f_] == 0 else "A")
    with open(os.path.join(out_dir, "gt.csv"), "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(gt))
    g = gt[gt != "X"]
    print("test bed written to {}: graded {} frames, H {:.0%} A {:.0%} D {:.0%}".format(
        out_dir, len(g), np.mean(g == "H"), np.mean(g == "A"), np.mean(g == "D")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--damage", action="store_true")
    a = ap.parse_args()
    main(a.out_dir, a.damage)
