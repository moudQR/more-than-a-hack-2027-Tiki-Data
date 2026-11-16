"""A 2D replay of the tracking as one self-contained HTML page, to open in any browser.

Shows every tracked player, the raw ball, the cleaned ball where the raw one is missing, detected
touches and the labels, frame by frame. Built from the cached arrays, so it needs no video decoder
and no ffmpeg.

The page for the provided match contains match data: keep it in cache/, which is not committed,
and out of cloud AI.
"""

import json

import numpy as np

import ball
import features
import touches

# Params
PITCH = (105.0, 68.0)
HOME_COLOUR, AWAY_COLOUR = "#a50044", "#1f6fd1"
STRIDE = 1  # every frame, 25 per second


def frames_json(data, blocks, start, stop, rows):
    """Compact per-frame data for the page. rows maps a label-row name to an array of labels."""
    clean = ball.clean(data["ball_xy"])
    near, _, _ = features.ball_distances(clean, data["player_xy"], data["team_of_slot"] == 0)
    touch_frames, touch_teams = touches.detect(clean, near["h"], near["a"], blocks)
    touch_of = dict(zip(touch_frames.tolist(), touch_teams.tolist()))

    def point(xy):
        return None if np.isnan(xy[0]) else [round(float(xy[0]), 2), round(float(xy[1]), 2)]

    team = data["team_of_slot"].tolist()
    out = []
    for frame in range(start, stop, STRIDE):
        players = [[slot, team[slot]] + point(xy)
                   for slot, xy in enumerate(data["player_xy"][frame]) if not np.isnan(xy[0])]
        out.append({"f": frame, "p": players, "b": point(data["ball_xy"][frame]),
                    "c": point(clean[frame]), "t": touch_of.get(frame),
                    "l": {name: str(labels[frame]) for name, labels in rows.items()}})
    return out


def write(path, data, blocks, start, seconds, rows, title):
    """Write the replay page for `seconds` from video frame `start`."""
    frames = frames_json(data, blocks, start, start + int(seconds * 25), rows)
    page = PAGE.replace("__TITLE__", title).replace("__DATA__", json.dumps(frames)) \
        .replace("__HOME__", HOME_COLOUR).replace("__AWAY__", AWAY_COLOUR)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)
    return path


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:16px}
canvas{background:#2e7d32;border-radius:6px;max-width:100%}
#bar{display:flex;gap:10px;align-items:center;margin:10px 0;flex-wrap:wrap}
button{background:#333;color:#eee;border:1px solid #555;border-radius:4px;padding:4px 10px;cursor:pointer}
input[type=range]{width:420px}
.key span{display:inline-block;margin-right:14px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
#labels{font-size:15px;margin-top:6px}
.lab{display:inline-block;min-width:22px;text-align:center;border-radius:3px;margin:0 8px 0 4px;font-weight:600}
</style></head><body>
<h3>__TITLE__</h3>
<canvas id="pitch" width="1050" height="680"></canvas>
<canvas id="strip" width="1050" height="40" style="background:#222;display:block;margin-top:6px"></canvas>
<div id="bar"><button id="play">Pause</button>
<input id="slider" type="range" min="0" value="0">
<span id="time"></span>
<label>speed <select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select></label></div>
<div class="key"><span><i class="dot" style="background:__HOME__"></i>home</span>
<span><i class="dot" style="background:__AWAY__"></i>away</span>
<span><i class="dot" style="background:#fff"></i>ball, tracked</span>
<span><i class="dot" style="border:2px solid #fff;width:8px;height:8px"></i>ball, filled gap</span>
<span><i class="dot" style="background:#ffeb3b"></i>touch</span>
<span>no ball dot: ball lost</span></div>
<div id="labels"></div>
<script>
const F=__DATA__, cv=document.getElementById('pitch'), g=cv.getContext('2d');
const st=document.getElementById('strip'), sg=st.getContext('2d');
const S=10, COL={H:'__HOME__',A:'__AWAY__',D:'#9e9e9e',X:'#444'}, TEAM=['__HOME__','__AWAY__'];
const slider=document.getElementById('slider'); slider.max=F.length-1;
let i=0, playing=true, flash=[];
const rowNames=Object.keys(F[0].l);
function pitch(){g.fillStyle='#2e7d32';g.fillRect(0,0,1050,680);g.strokeStyle='rgba(255,255,255,.8)';g.lineWidth=2;
 g.strokeRect(2,2,1046,676);g.beginPath();g.moveTo(525,0);g.lineTo(525,680);g.stroke();
 g.beginPath();g.arc(525,340,91.5,0,7);g.stroke();
 [[0,1],[1050,-1]].forEach(([x,s])=>{g.strokeRect(s>0?x:x-165,138.4,165,403.2);g.strokeRect(s>0?x:x-55,248.4,55,183.2);});}
function strip(){const w=1050/F.length;rowNames.forEach((n,r)=>{for(let k=0;k<F.length;k++){sg.fillStyle=COL[F[k].l[n]]||'#444';sg.fillRect(k*w,r*20,Math.ceil(w),18);}});}
function draw(){pitch();const fr=F[i];
 fr.p.forEach(([s,t,x,y])=>{g.fillStyle=TEAM[t];g.beginPath();g.arc(x*S,y*S,8,0,7);g.fill();g.fillStyle='#fff';g.font='9px sans-serif';g.fillText(s,x*S-5,y*S+3);});
 if(fr.t!==null&&fr.t!==undefined) flash.push({k:i,c:fr.c});
 flash=flash.filter(f=>i-f.k>=0&&i-f.k<12);
 flash.forEach(f=>{if(!f.c)return;g.strokeStyle='#ffeb3b';g.lineWidth=3;g.beginPath();g.arc(f.c[0]*S,f.c[1]*S,14+(i-f.k),0,7);g.stroke();});
 if(fr.b){g.fillStyle='#fff';g.beginPath();g.arc(fr.b[0]*S,fr.b[1]*S,5,0,7);g.fill();}
 else if(fr.c){g.strokeStyle='#fff';g.lineWidth=2;g.beginPath();g.arc(fr.c[0]*S,fr.c[1]*S,5,0,7);g.stroke();}
 document.getElementById('labels').innerHTML=rowNames.map(n=>n+':<span class="lab" style="background:'+(COL[fr.l[n]]||'#444')+'">'+fr.l[n]+'</span>').join('');
 const sec=(fr.f/25), m=Math.floor(sec/60), s=(sec%60).toFixed(1);
 document.getElementById('time').textContent='frame '+fr.f+'   video '+m+':'+s.padStart(4,'0');
 slider.value=i; strip(); sg.fillStyle='#fff'; sg.fillRect(i*1050/F.length-1,0,2,40);}
let last=0;function loop(t){const sp=parseFloat(document.getElementById('speed').value);
 if(playing&&t-last>=40/sp){i=(i+1)%F.length;last=t;draw();}requestAnimationFrame(loop);}
document.getElementById('play').onclick=e=>{playing=!playing;e.target.textContent=playing?'Pause':'Play';};
slider.oninput=e=>{i=+e.target.value;flash=[];draw();};
st.height=20*rowNames.length; draw(); requestAnimationFrame(loop);
</script></body></html>
"""
