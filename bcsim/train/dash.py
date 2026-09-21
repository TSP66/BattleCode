"""Live training dashboard.

Serves a page that polls the run's log.jsonl, so it tracks a run in progress.

    python -m train.dash --run ../runs/v0
    open http://localhost:877
"""

from __future__ import annotations

import argparse
import hmac
import http.server
import json
import pathlib
import secrets
import socket
import socketserver
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Battlecode self-play</title>
<style>
:root{
  color-scheme: light;
  --surface-1:#fcfcfb; --surface-2:#f4f4f2; --line:#e3e3df;
  --ink-1:#1a1a19; --ink-2:#55554f; --ink-3:#8a8a82;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
  --s6:#7d5bd0; --s7:#6b6b63;
  --grid:#ebebe7;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#232321; --line:#35352f;
  --ink-1:#f2f2ef; --ink-2:#b0b0a7; --ink-3:#7e7e76;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
  --s6:#9b7fe6; --s7:#9a9a90;
  --grid:#2c2c28;
}}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#232321; --line:#35352f;
  --ink-1:#f2f2ef; --ink-2:#b0b0a7; --ink-3:#7e7e76;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
  --s6:#9b7fe6; --s7:#9a9a90;
  --grid:#2c2c28;
}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-1);color:var(--ink-1);
  font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  padding:24px 16px 48px}
.wrap{max-width:1500px;margin:0 auto}
h1{font-size:19px;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:var(--ink-2);font-size:13px;margin-bottom:20px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--s3);
  margin-right:6px;vertical-align:1px}
.dot.stale{background:var(--ink-3)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:22px}
.tile{background:var(--surface-2);border:1px solid var(--line);border-radius:10px;padding:11px 13px}
.tile .k{color:var(--ink-3);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.tile .v{font-size:21px;font-weight:600;margin-top:3px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.card{background:var(--surface-2);border:1px solid var(--line);border-radius:10px;padding:13px 14px 8px}
.card h2{font-size:13px;margin:0;font-weight:600}
.card .note{color:var(--ink-3);font-size:11.5px;margin:2px 0 6px}
.legend{display:flex;flex-wrap:wrap;gap:11px;margin:5px 0 0;font-size:11.5px;color:var(--ink-2)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
svg{display:block;width:100%;height:auto;overflow:visible}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--line);
  border-radius:7px;padding:7px 9px;font-size:12px;box-shadow:0 5px 18px rgba(0,0,0,.14);
  opacity:0;transition:opacity .1s;z-index:9;font-variant-numeric:tabular-nums}
.tip b{font-weight:600}
.err{background:#e34948;color:#fff;padding:9px 12px;border-radius:8px;margin-bottom:14px}
h3{font-size:15px;margin:30px 0 4px}
.sec{color:var(--ink-2);font-size:12.5px;margin-bottom:12px}
.hm{overflow-x:auto;background:var(--surface-2);border:1px solid var(--line);border-radius:10px;padding:10px;margin-top:14px}
.hm table{border-collapse:separate;border-spacing:2px;font-size:11.5px;font-variant-numeric:tabular-nums}
.hm th{font-weight:500;color:var(--ink-2);padding:3px 6px;text-align:left;white-space:nowrap}
.hm td{padding:4px 6px;text-align:center;border-radius:4px;min-width:44px}
</style></head><body>
<div class="wrap">
  <h1>Battlecode self-play</h1>
  <div class="sub"><span class="dot" id="live"></span><span id="status">connecting…</span></div>
  <div id="err"></div>
  <div class="tiles" id="tiles"></div>
  <div class="grid" id="grid"></div>
  <h3>Yardsticks</h3>
  <div class="sec" id="evstatus">Score = (wins + ½ draws) / games against opponents that never train:
    scripted bots and older snapshots. Greedy play, unaugmented maps, both sides equally.</div>
  <div class="grid" id="evgrid"></div>
  <div class="hm" id="heat"></div>
</div>
<div class="tip" id="tip"></div>
<script>
const PAL = ['--s1','--s2','--s3','--s4','--s5','--s6','--s7'];
// each chart: title, note, series [{key,label}]; all series in one chart share a unit
const CHARTS = [
  {t:'Mean reward per turn', n:'weighted reward per closed transition', s:[{k:'reward_mean'}]},
  {t:'Return and value', n:'GAE return vs what the critic predicts',
   s:[{k:'return_mean',l:'return'},{k:'value_mean',l:'value'}]},
  {t:'Explained variance', n:'1 = critic explains the return, 0 = no better than the mean', s:[{k:'explained_var'}], zero:true},
  {t:'Policy loss', n:'clipped surrogate', s:[{k:'pg'}]},
  {t:'Value loss', n:'clipped MSE against the return', s:[{k:'v'}]},
  {t:'Entropy', n:'higher = still exploring', s:[{k:'ent'}]},
  {t:'Approx KL', n:'per update; a spike means the step was too large', s:[{k:'kl'}]},
  {t:'Clip fraction', n:'share of samples hitting the PPO clip', s:[{k:'clipfrac'}]},
  {t:'Episode length', n:'rounds survived, 500 is the cap', s:[{k:'rounds_mean'}]},
  {t:'Longest dragon', n:'mean of the longest surviving dragon per game', s:[{k:'longest'}]},
  {t:'Dragons per team', n:'mean alive at the end of a game', s:[{k:'units_mean'}]},
  {t:'Concentration', n:'leader length per dragon on the board; higher = one long dragon, not many short ones', s:[{k:'conc'}]},
  {t:'Reward components', n:'raw rate per closed transition, before weighting',
   s:[{k:'r_length_delta',l:'length Δ'},{k:'r_pearls',l:'pearls'},{k:'r_died',l:'deaths'},
      {k:'r_kills',l:'kills'},{k:'r_splits',l:'splits'}]},
  {t:'Team potentials', n:'raw per-turn change in team state, before weighting',
   s:[{k:'r_team_max',l:'our longest'},{k:'r_foe_max',l:'their longest'},
      {k:'r_team_len',l:'our total'},{k:'r_foe_len',l:'their total'}]},
  {t:'Action mix', n:'share of chosen actions', s:[{k:'sprint_frac',l:'sprint'},{k:'split_frac',l:'split'}]},
  {t:'Draw rate', n:'games ending with no winner', s:[{k:'draw_rate'}]},
  {t:'Portal usage', n:'share of turns that went through a portal', s:[{k:'r_portal'}]},
  {t:'Kills per game', n:'per team per game; a kill is an enemy dragon dying on your body or head',
   s:[{k:'kills_per_game',l:'kills'},{k:'deaths_per_game',l:'deaths, any cause'},
      {k:'headon_per_game',l:'head-on kills (killer died too)'}]},
  {t:'Trades', n:'segments per team per game: enemy length its kills removed vs its own length lost',
   s:[{k:'len_killed_per_game',l:'killed'},{k:'len_lost_per_game',l:'lost'}]},
  {t:'Better trader wins', n:'share of decided games won by the side with the better kill-minus-loss margin', s:[{k:'trade_win_rate'}]},
  {t:'Entropy coefficient', n:'weight on the entropy bonus; decays toward its floor', s:[{k:'ent_coef'}]},
  {t:'Small-map share', n:'games on maps of 16x16 or less (augmented included)', s:[{k:'small_map_frac'}]},
  {t:'Throughput', n:'dragon turns per second, end to end', s:[{k:'sps'}]},
  {t:'Usable transitions', n:'share of slots that trained; the rest are still open', s:[{k:'usable'}]},
];
const fmt = (v)=> v==null?'–':Math.abs(v)>=1e6?(v/1e6).toFixed(2)+'M':Math.abs(v)>=1000?
  (v/1000).toFixed(1)+'k':Math.abs(v)>=10?v.toFixed(1):Math.abs(v)>=1?v.toFixed(2):v.toFixed(4);
function ema(a,f){let o=[],p=null;for(const v of a){if(v==null){o.push(p);continue;}
  p=p==null?v:p*(1-f)+v*f;o.push(p);}return o;}
let rows=[];
function draw(){
  const grid=document.getElementById('grid');
  if(!grid.children.length) CHARTS.forEach((c,i)=>{
    const d=document.createElement('div'); d.className='card'; d.id='c'+i;
    d.innerHTML=`<h2>${c.t}</h2><div class="note">${c.n}</div><div class="plot"></div>`+
      (c.s.length>1?`<div class="legend">`+c.s.map((s,j)=>
        `<span><i style="background:var(${PAL[j%PAL.length]})"></i>${s.l||s.k}</span>`).join('')+`</div>`:'');
    grid.appendChild(d);
  });
  const x = rows.map(r=>r.total_turns/1e6);
  CHARTS.forEach((c,i)=>plot(document.querySelector('#c'+i+' .plot'), c, x, rows));
}
function plot(host,cfg,x,rows){
  const W=520,H=190,L=54,R=12,T=10,B=26;
  const series=cfg.s.map(s=>({...s,y:rows.map(r=>r[s.k]==null?null:+r[s.k])}));
  const all=series.flatMap(s=>s.y).filter(v=>v!=null&&isFinite(v));
  if(!all.length||x.length<2){host.innerHTML='<div style="color:var(--ink-3);padding:26px 0;font-size:12px">waiting for data…</div>';return;}
  let lo=Math.min(...all),hi=Math.max(...all);
  if(cfg.zero){lo=Math.min(lo,0);hi=Math.max(hi,0);}
  if(lo===hi){lo-=.5;hi+=.5;} const pad=(hi-lo)*.12; lo-=pad; hi+=pad;
  const xs=v=>L+(v-x[0])/((x[x.length-1]-x[0])||1)*(W-L-R);
  const ys=v=>T+(hi-v)/((hi-lo)||1)*(H-T-B);
  const ticks=[lo,(lo+hi)/2,hi];
  let g=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img">`;
  ticks.forEach(t=>{g+=`<line x1="${L}" x2="${W-R}" y1="${ys(t).toFixed(1)}" y2="${ys(t).toFixed(1)}"
    stroke="var(--grid)" stroke-width="1"/><text x="${L-7}" y="${(ys(t)+3.5).toFixed(1)}"
    text-anchor="end" font-size="10" fill="var(--ink-3)">${fmt(t)}</text>`;});
  if(cfg.zero&&lo<0&&hi>0) g+=`<line x1="${L}" x2="${W-R}" y1="${ys(0)}" y2="${ys(0)}" stroke="var(--ink-3)" stroke-width="1"/>`;
  [x[0],x[x.length-1]].forEach((v,j)=>{g+=`<text x="${xs(v)}" y="${H-8}" font-size="10"
    fill="var(--ink-3)" text-anchor="${j?'end':'start'}">${v.toFixed(1)}M</text>`;});
  series.forEach((s,j)=>{
    const col=`var(${PAL[j%PAL.length]})`, sm=ema(s.y,cfg.raw?1:.18);
    const path=(arr)=>{let d='',pen=false;arr.forEach((v,k)=>{if(v==null||!isFinite(v)){pen=false;return;}
      d+=(pen?'L':'M')+xs(x[k]).toFixed(1)+' '+ys(v).toFixed(1)+' ';pen=true;});return d;};
    g+=`<path d="${path(s.y)}" fill="none" stroke="${col}" stroke-width="1" opacity=".22"/>`;
    g+=`<path d="${path(sm)}" fill="none" stroke="${col}" stroke-width="2"
         stroke-linejoin="round" stroke-linecap="round"/>`;
    const last=[...sm].reverse().find(v=>v!=null);
    if(last!=null) g+=`<circle cx="${xs(x[x.length-1])}" cy="${ys(last)}" r="3" fill="${col}"
      stroke="var(--surface-2)" stroke-width="2"/>`;
  });
  g+=`<rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent" class="hit"/></svg>`;
  host.innerHTML=g;
  const svg=host.querySelector('svg'), tip=document.getElementById('tip');
  svg.addEventListener('pointermove',e=>{
    const b=svg.getBoundingClientRect(), px=(e.clientX-b.left)/b.width*W;
    let k=0,best=1e9; x.forEach((v,idx)=>{const d=Math.abs(xs(v)-px);if(d<best){best=d;k=idx;}});
    tip.innerHTML=`<b>${x[k].toFixed(2)}M turns</b>`+series.map((s,j)=>
      `<div><i style="display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;
       background:var(${PAL[j%PAL.length]})"></i>${s.l||cfg.t}: ${fmt(s.y[k])}</div>`).join('');
    tip.style.opacity=1; tip.style.left=Math.min(e.clientX+14,innerWidth-190)+'px';
    tip.style.top=(e.clientY+14)+'px';
  });
  svg.addEventListener('pointerleave',()=>tip.style.opacity=0);
}
function tiles(){
  const r=rows[rows.length-1]||{}, box=document.getElementById('tiles');
  const hrs=(r.elapsed||0)/3600;
  const items=[['total turns',fmt(r.total_turns)],['turns / sec',fmt(r.sps)],
    ['iteration',r.iter??'–'],['elapsed',hrs<1?((r.elapsed||0)/60).toFixed(0)+' min':hrs.toFixed(1)+' h'],
    ['mean reward',fmt(r.reward_mean)],['episode rounds',fmt(r.rounds_mean)],
    ['longest dragon',fmt(r.longest)],['entropy',fmt(r.ent)],
    ['score vs bots',evLast?fmt(evLast.bot_score):'–'],
    [evLast&&evLast.submitted_name?'vs '+evLast.submitted_name:'vs submitted',
     evLast&&evLast.submitted_score!=null?fmt(evLast.submitted_score):'–'],['kills / game',fmt(r.kills_per_game)]];
  box.innerHTML=items.map(([k,v])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
}
// Resolve relative to the current path so this works behind a proxy prefix
// (jupyter-server-proxy, /proxy/8770/) as well as at the root, and carry the
// access token through on every fetch.
const BASE = location.pathname.endsWith('/') ? location.pathname : location.pathname + '/';
function dataURL(){
  const q = new URLSearchParams(location.search);
  q.set('n','4000');
  return BASE + 'data?' + q.toString();
}
// ---- yardsticks: one row per evaluated snapshot
let evRows=[], evLast=null;
function evalCharts(){
  const names=[...new Set(evRows.flatMap(r=>Object.keys(r.summary||{})))];
  const bots=names.filter(n=>n.startsWith('bot:')), past=names.filter(n=>!n.startsWith('bot:'));
  const flat=evRows.map(r=>{const o={total_turns:r.total_turns,bot_score:r.bot_score,
    submitted_score:r.submitted_score};
    for(const [k,v] of Object.entries(r.summary||{})){o['s|'+k]=v.score;o['k|'+k]=v.kills;o['p|'+k]=v.portal;}
    return o;});
  const cs=[
    {t:'Score vs scripted bots', n:'one line per bot; 1 = wins every game', raw:true,
     s:bots.map(b=>({k:'s|'+b,l:b.slice(4)}))},
    {t:'Score vs past versions', n:'above 0.5 = better than it was; a fall back toward 0.5 means forgetting', raw:true,
     s:past.map(b=>({k:'s|'+b,l:b}))},
    {t:'Score vs last submitted', n:'the bar for the next upload: above 0.5 means it beats what is on the server', raw:true,
     s:[{k:'submitted_score',l:'vs last submitted'}]},
    {t:'Mean score vs bots', n:'saturates early; the past-version charts separate versions better', raw:true, s:[{k:'bot_score'}]},
    {t:'Kills per game vs bots', n:'the learner\'s kills, per game', raw:true, s:bots.map(b=>({k:'k|'+b,l:b.slice(4)}))},
    {t:'Portal usage vs bots', n:'share of the learner\'s turns through a portal', raw:true,
     s:bots.map(b=>({k:'p|'+b,l:b.slice(4)}))},
  ];
  const grid=document.getElementById('evgrid');
  grid.innerHTML='';
  cs.forEach((c,i)=>{
    const d=document.createElement('div'); d.className='card';
    d.innerHTML=`<h2>${c.t}</h2><div class="note">${c.n}</div><div class="plot"></div>`+
      (c.s.length>1?`<div class="legend">`+c.s.map((s,j)=>
        `<span><i style="background:var(${PAL[j%PAL.length]})"></i>${s.l||s.k}</span>`).join('')+`</div>`:'');
    grid.appendChild(d);
    plot(d.querySelector('.plot'), c, flat.map(r=>r.total_turns/1e6), flat);
  });
  heat();
}
function heat(){
  const box=document.getElementById('heat');
  if(!evLast||!evLast.cells){box.innerHTML='<div style="color:var(--ink-3);font-size:12px">no evaluation yet — run python -m train.yardstick</div>';return;}
  const opps=Object.keys(evLast.cells), maps=[...new Set(opps.flatMap(o=>Object.keys(evLast.cells[o])))].sort();
  // score 0 red, 0.5 neutral, 1 green, as a diverging ramp on the surface colour
  const col=v=>{const t=Math.max(-1,Math.min(1,(v-.5)*2));
    return t>=0?`color-mix(in srgb, var(--s3) ${Math.round(t*70)}%, var(--surface-1))`
               :`color-mix(in srgb, #e34948 ${Math.round(-t*70)}%, var(--surface-1))`;};
  let h=`<div style="font-size:12.5px;margin:0 0 6px"><b>Latest: ${(evLast.total_turns/1e6).toFixed(1)}M turns</b>
    <span style="color:var(--ink-3)"> · score per map, hover for the record · ${evLast.seconds}s</span></div>
    <table><tr><th></th>${maps.map(m=>`<th>${m}</th>`).join('')}<th>all</th></tr>`;
  for(const o of opps){
    h+=`<tr><th>${o}</th>`;
    for(const m of maps){const c=evLast.cells[o][m];
      h+=c?`<td style="background:${col(c.score)}" title="${c.win}W ${c.draw}D ${c.loss}L · kills ${c.kills} · longest ${c.longest} · rounds ${c.rounds}">${c.score.toFixed(2)}</td>`:'<td>–</td>';}
    const s=evLast.summary[o];
    h+=`<td style="background:${col(s.score)};font-weight:600" title="${s.n} games">${s.score.toFixed(2)}</td></tr>`;
  }
  box.innerHTML=h+'</table>';
}
async function pollEval(){
  try{
    const q=new URLSearchParams(location.search);
    const res=await fetch(BASE+'eval?'+q.toString(),{cache:'no-store'});
    if(!res.ok) return;
    const j=await res.json();
    if(j.rows.length!==evRows.length||!evLast){evRows=j.rows; evLast=j.last; evalCharts();}
  }catch(e){}
}
pollEval(); setInterval(pollEval,30000);
let lastLen=-1, lastChange=Date.now();
async function poll(){
  try{
    const res=await fetch(dataURL(),{cache:'no-store'});
    if(res.status===403){document.getElementById('err').innerHTML=
      '<div class="err">access token missing or wrong — open the URL the server printed</div>';return;}
    const j=await res.json();
    document.getElementById('err').innerHTML='';
    if(j.rows.length!==lastLen){lastLen=j.rows.length;lastChange=Date.now();}
    rows=j.rows; tiles(); draw();
    const idle=(Date.now()-lastChange)/1000;
    const live=document.getElementById('live');
    live.className='dot'+(idle>90?' stale':'');
    document.getElementById('status').textContent =
      `${j.run} · ${rows.length} iterations · ` + (idle>90?`no new data for ${idle|0}s`:'updating every 4s');
  }catch(e){
    document.getElementById('err').innerHTML=`<div class="err">cannot reach the dashboard server: ${e}</div>`;
  }
}
poll(); setInterval(poll,4000);
</script></body></html>
"""


def serve(run: pathlib.Path, port: int, host: str = "127.0.0.1",
          token: str = "") -> None:
    log = run / "log.jsonl"

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _allowed(self) -> bool:
            if not token:
                return True
            query = urllib.parse.urlparse(self.path).query
            given = urllib.parse.parse_qs(query).get("token", [""])[0]
            return hmac.compare_digest(given, token)

        def do_GET(self):
            if not self._allowed():
                self.send_response(403)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"missing or wrong ?token=\n")
                return
            # a proxy may forward its own prefix, so match on the tail
            tail = urllib.parse.urlparse(self.path).path.rstrip("/")
            if tail.endswith("eval"):
                rows = []
                if (run / "eval.jsonl").exists():
                    for line in (run / "eval.jsonl").read_text().splitlines():
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                last = rows[-1] if rows else None
                # per-map cells only for the newest; the curves need summaries
                slim = [{k: v for k, v in r.items() if k != "cells"} for r in rows]
                body = json.dumps({"rows": slim, "last": last}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
            elif tail.endswith("data"):
                rows = []
                if log.exists():
                    with log.open() as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                try:
                                    rows.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass          # a half-written final line
                body = json.dumps({"run": run.name, "rows": rows}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
            else:
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server((host, port), Handler) as srv:
        q = f"?token={token}" if token else ""
        where = socket.gethostname() if host not in ("127.0.0.1", "localhost") else "localhost"
        print(f"dashboard for {run} at http://{where}:{port}/{q}", flush=True)
        if host not in ("127.0.0.1", "localhost"):
            print(f"  bound to {host}: reachable by anything that can reach this host",
                  flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "runs/v0"))
    p.add_argument("--port", type=int, default=877)
    p.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 to reach it from another machine")
    p.add_argument("--token", default="auto",
                   help="'auto' generates one when binding externally, "
                        "'' disables the check")
    args = p.parse_args()
    tok = args.token
    if tok == "auto":
        tok = "" if args.host in ("127.0.0.1", "localhost") else secrets.token_urlsafe(16)
    serve(pathlib.Path(args.run), args.port, args.host, tok)
