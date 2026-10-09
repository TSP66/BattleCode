"""Live training dashboard.

Serves a page that polls the run's log.jsonl, so it tracks a run in progress.

    python -m train.dash --run ../runs/scratch_1001b
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
<title>Battlecode training</title>
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
  <h1>Battlecode training</h1>
  <div class="sub"><span class="dot" id="live"></span><span id="status">connecting…</span></div>
  <div id="err"></div>
  <div id="ratchet"></div>
  <div class="tiles" id="tiles"></div>
  <div class="grid" id="grid"></div>
  <h3>Gates</h3>
  <div class="sec" id="evstatus">Score = (wins + ½ draws) / games, greedy play on the gate maps, both sides equally.</div>
  <div class="grid" id="evgrid"></div>
  <div class="hm" id="heat"></div>
</div>
<div class="tip" id="tip"></div>
<script>
const PAL = ['--s1','--s2','--s3','--s4','--s5','--s6','--s7'];
// each chart: title, note, series [{key,label}]; all series in one chart share a unit
const CHARTS = [
  // what train/ratchet_ff_train.py logs (one row per iteration)
  {t:'Score in training', n:'learner score per opponent, from training games (sampled play; the gate is greedy)',
   prefix:'vs_', raw:false},
  {t:'How games are decided', n:'share of the last 4000 training games: elimination, else the round-500 verdict by queen length, longest dragon, total length (draw = all equal); agree = the verdict matches the real winner (should be ~1)',
   s:[{k:'end_elim',l:'elimination'},{k:'end_queen',l:'queen'},{k:'end_longest',l:'longest'},{k:'end_total',l:'total'},{k:'end_draw',l:'draw'},{k:'end_agree',l:'agree'}], lo:0, hi:1},
  {t:'Return and value', n:'GAE return vs what the critic predicts', s:[{k:'return_mean',l:'return'},{k:'value_mean',l:'value'}]},
  {t:'Explained variance', n:'1 = critic explains the return, 0 = no better than the mean (axis fixed to 0..1; values outside are drawn at the edge)', s:[{k:'explained_var'}], lo:0, hi:1},
  {t:'Critic MSE', n:'true-board critic, in units of its return scale', s:[{k:'critic_mse'}]},
  // the win/loss head (--wl-head, 2026-10-03): not yet driving the policy; switch when it beats phi
  {t:'Win/loss head: AUC', n:'does the head rank winners above losers? its prediction at the first turn on or after each round, over the last 8000 (game, team) results, decisive games only; 0.5 = chance. Compare with the phi chart',
   s:[{k:'wl_auc_r25',l:'round 25'},{k:'wl_auc_r100',l:'round 100'},{k:'wl_auc_r200',l:'round 200'},{k:'wl_auc_r350',l:'round 350'}], lo:0.5, hi:1},
  {t:'Phi AUC (the bar to beat)', n:'the same games and moments, ranked by the team\'s phi instead of the head',
   s:[{k:'phi_auc_r25',l:'round 25'},{k:'phi_auc_r100',l:'round 100'},{k:'phi_auc_r200',l:'round 200'},{k:'phi_auc_r350',l:'round 350'}], lo:0.5, hi:1},
  {t:'Win/loss head: Brier', n:'mean (P(win) - result)^2 with P = (head + 1) / 2, draws 0.5; 0.25 = always saying 50%, lower is better',
   s:[{k:'wl_brier_r25',l:'round 25'},{k:'wl_brier_r100',l:'round 100'},{k:'wl_brier_r200',l:'round 200'},{k:'wl_brier_r350',l:'round 350'}], lo:0, hi:0.3},
  {t:'Win/loss head: TD fit', n:'explained variance of its TD(0.98) targets (1 = fits them), and mean |prediction| (0 = no opinion yet)',
   s:[{k:'wl_ev',l:'explained var'},{k:'wl_absmean',l:'mean |pred|'}], lo:0, hi:1},
  // the blend (--wl-blend-turns, 2026-10-03): policy advantage = (1 - b) A_phi + b A_wl, each standardised, blend restandardised
  {t:'Advantage blend: who drives the policy', n:'b = the W/L weight (0 -> 1 over 1.2B turns); share = Cov(part, blend) / Var(blend), the two shares sum to 1 (at b = 0 the W/L share is 0)',
   s:[{k:'wl_blend',l:'b'},{k:'adv_share_phi',l:'phi share'},{k:'adv_share_wl',l:'W/L share'}], lo:0, hi:1},
  {t:'Advantage blend: variance', n:'each weighted part\'s own variance over the blend\'s ((1-b)^2 and b^2 over Var(blend)); the remainder is the cross term 2b(1-b)corr; mix sd = the blend\'s spread before it is rescaled to 1 (1 = parts agree, 0.71 = independent at b = 0.5)',
   s:[{k:'adv_var_phi',l:'phi'},{k:'adv_var_wl',l:'W/L'},{k:'adv_mix_sd',l:'mix sd'}], lo:0, hi:1.2},
  {t:'Advantage agreement', n:'correlation of the standardised phi and W/L advantages over the usable rows, and how often their signs agree (0.5 = unrelated)',
   s:[{k:'adv_corr',l:'correlation'},{k:'adv_sign_agree',l:'sign agrees'}], lo:-0.2, hi:1},
  {t:'Advantage tails', n:'share of the sum of squares in the top 1% of rows (a normal: ~0.10; higher = a few rows dominate the step)',
   s:[{k:'adv_phi_tail',l:'phi'},{k:'adv_wl_tail',l:'W/L'}], lo:0, hi:0.6},
  {t:'Advantage raw spread', n:'sd of each advantage before standardising, in its own unit (phi points vs a +-1 result): noise trend, not comparable across the two',
   s:[{k:'adv_phi_sd',l:'phi'},{k:'adv_wl_sd',l:'W/L'}]},
  {t:'Mean reward per turn', n:'Phi change per learner turn', s:[{k:'reward_mean'}]},
  {t:'Entropy', n:'policy entropy (nats); higher = still exploring', s:[{k:'ent'}]},
  {t:'Entropy coefficient', n:'0.1 falling linearly to 0 over 2B run turns', s:[{k:'ent_coef'}]},
  {t:'KL to teacher', n:'KL to the last promoted version (coef 0.15); axis capped at 0.15, higher values drawn at the top', s:[{k:'kl_teacher'}], lo:0, hi:0.15},
  {t:'Approx KL', n:'per update; a spike means the step was too large', s:[{k:'kl'}]},
  {t:'Clip fraction', n:'share of samples hitting the PPO clip', s:[{k:'clipfrac'}]},
  {t:'Ratio check', n:'|ratio - 1| on the first minibatch; aborts above 0.05', s:[{k:'ratio0'}]},
  {t:'Policy loss', n:'clipped surrogate', s:[{k:'pg'}]},
  {t:'Clip by temperature band', n:'tb0..tb4: learner temperature bands (edges 0.15/0.25/0.35/0.45)',
   s:[{k:'tb0_clip',l:'<0.15'},{k:'tb1_clip',l:'0.15-0.25'},{k:'tb2_clip',l:'0.25-0.35'},{k:'tb3_clip',l:'0.35-0.45'},{k:'tb4_clip',l:'>0.45'}]},
  {t:'Entropy by temperature band', n:'policy entropy per learner temperature band',
   s:[{k:'tb0_ent',l:'<0.15'},{k:'tb1_ent',l:'0.15-0.25'},{k:'tb2_ent',l:'0.25-0.35'},{k:'tb3_ent',l:'0.35-0.45'},{k:'tb4_ent',l:'>0.45'}]},
  {t:'Temperatures', n:'learner range top (falls 0.5 to 0.25 over 2B turns), learner mean, opponents mean',
   s:[{k:'temp_hi',l:'learner top'},{k:'temp_learn_mean',l:'learner mean'},{k:'temp_opp_mean',l:'opponent mean'}]},
  {t:'Time per iteration', n:'seconds collecting, optimising, critic', s:[{k:'t_roll',l:'rollout'},{k:'t_opt',l:'optimise'},{k:'t_crit',l:'critic'}]},
  {t:'Throughput', n:'dragon turns per second, end to end', s:[{k:'sps'}]},
  {t:'GPU memory', n:'GB', s:[{k:'gpu_gb'}]},
  {t:'Usable transitions', n:'share of slots that trained; the rest are still open', s:[{k:'usable'}]},
];
const fmt = (v)=> v==null?'–':Math.abs(v)>=1e6?(v/1e6).toFixed(2)+'M':Math.abs(v)>=1000?
  (v/1000).toFixed(1)+'k':Math.abs(v)>=10?v.toFixed(1):Math.abs(v)>=1?v.toFixed(2):v.toFixed(4);
function ema(a,f){let o=[],p=null;for(const v of a){if(v==null){o.push(p);continue;}
  p=p==null?v:p*(1-f)+v*f;o.push(p);}return o;}
let rows=[], EVENTS=[];
// a restart that rewinds the turn counter (a reset to an older anchor, a
// dropped candidate) supersedes every row at or past the turn it restarts
// from: plot only the line the run is on now. log.jsonl keeps them all.
function lineage(rs){
  const out=[];
  for(const r of rs){
    if(r.total_turns!=null) while(out.length&&out[out.length-1].total_turns>=r.total_turns) out.pop();
    out.push(r);
  }
  return out;
}
function draw(){
  const grid=document.getElementById('grid');
  if(!grid.children.length) CHARTS.forEach((c,i)=>{
    c.s=c.s||[];
    const d=document.createElement('div'); d.className='card'; d.id='c'+i;
    d.innerHTML=`<h2>${c.t}</h2><div class="note">${c.n}</div><div class="plot"></div>`+
      (c.s.length>1?`<div class="legend">`+c.s.map((s,j)=>
        `<span><i style="background:var(${PAL[j%PAL.length]})"></i>${s.l||s.k}</span>`).join('')+`</div>`:'');
    grid.appendChild(d);
  });
  const cur = lineage(rows);
  const x = cur.map(r=>r.total_turns/1e6);
  CHARTS.forEach((c,i)=>{
    if(c.prefix){
      // series discovered from the log: one per key with this prefix
      const keys=[...new Set(rows.flatMap(r=>Object.keys(r).filter(k=>k.startsWith(c.prefix)&&!(c.exclude||[]).includes(k))))].sort();
      const label=k=>{const t=k.slice(c.prefix.length);
        return c.prefix==='ev_'?(t==='0'?'self-play':'opponent '+t):t;};
      if(JSON.stringify(keys)!==JSON.stringify(c._keys||[])){
        c._keys=keys; c.s=keys.map(k=>({k,l:label(k)}));
        const card=document.getElementById('c'+i); let lg=card.querySelector('.legend');
        if(!lg){lg=document.createElement('div');lg.className='legend';card.appendChild(lg);}
        lg.innerHTML=c.s.map((s,j)=>`<span><i style="background:var(${PAL[j%PAL.length]})"></i>${s.l}</span>`).join('');
      }
    }
    plot(document.querySelector('#c'+i+' .plot'), c, x, cur);
  });
}
function plot(host,cfg,x,rows){
  const W=520,H=190,L=54,R=12,T=10,B=26;
  const fixed=cfg.lo!=null&&cfg.hi!=null;   // a fixed axis: values outside it are drawn at its edge
  const clip=v=>fixed?Math.min(cfg.hi,Math.max(cfg.lo,v)):v;
  const series=cfg.s.map(s=>({...s,y:rows.map(r=>r[s.k]==null?null:clip(+r[s.k]))}));
  const all=series.flatMap(s=>s.y).filter(v=>v!=null&&isFinite(v));
  if(!all.length||x.length<2){host.innerHTML='<div style="color:var(--ink-3);padding:26px 0;font-size:12px">waiting for data…</div>';return;}
  let lo=Infinity,hi=-Infinity;   // a loop, not Math.min(...all): spreading 100k+ points overflows the stack
  for(const v of all){if(v<lo)lo=v;if(v>hi)hi=v;}
  if(fixed){lo=cfg.lo;hi=cfg.hi;}
  else{
    if(cfg.zero){lo=Math.min(lo,0);hi=Math.max(hi,0);}
    if(lo===hi){lo-=.5;hi+=.5;} const pad=(hi-lo)*.12; lo-=pad; hi+=pad;
  }
  const xs=v=>L+(v-x[0])/((x[x.length-1]-x[0])||1)*(W-L-R);
  const ys=v=>T+(hi-v)/((hi-lo)||1)*(H-T-B);
  const ticks=[lo,(lo+hi)/2,hi];
  let g=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img">`;
  ticks.forEach(t=>{g+=`<line x1="${L}" x2="${W-R}" y1="${ys(t).toFixed(1)}" y2="${ys(t).toFixed(1)}"
    stroke="var(--grid)" stroke-width="1"/><text x="${L-7}" y="${(ys(t)+3.5).toFixed(1)}"
    text-anchor="end" font-size="10" fill="var(--ink-3)">${fmt(t)}</text>`;});
  if(cfg.zero&&lo<0&&hi>0) g+=`<line x1="${L}" x2="${W-R}" y1="${ys(0)}" y2="${ys(0)}" stroke="var(--ink-3)" stroke-width="1"/>`;
  // run events (events.jsonl): a dashed marker at the turn each one restarted from
  if(cfg.events!==false){
    const byX={};
    for(const e of EVENTS){const v=e.total_turns/1e6; if(v<x[0]||v>x[x.length-1]) continue;
      const k=xs(v).toFixed(1); (byX[k]=byX[k]||[]).push(e);}
    for(const [px,es] of Object.entries(byX)){
      g+=`<line x1="${px}" x2="${px}" y1="${T}" y2="${H-B}" stroke="var(--ink-3)" stroke-width="1"
        stroke-dasharray="3 3"><title>${es.map(e=>e.when+' '+e.label).join('\n')}</title></line>`;
    }
  }
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
    const near=cfg.events===false?[]:EVENTS.filter(e=>Math.abs(xs(e.total_turns/1e6)-px)<6);
    tip.innerHTML=`<b>${x[k].toFixed(2)}M turns</b>`+near.map(e=>`<div style="color:var(--ink-2)">⟲ ${e.when} ${e.label}</div>`).join('')+series.map((s,j)=>
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
    ['generation',r.gen??'–'],['segment',r.segment??'–'],['iteration',r.iter??'–'],
    ['segment time',hrs<1?((r.elapsed||0)/60).toFixed(0)+' min':hrs.toFixed(1)+' h'],
    ['explained var',fmt(r.explained_var)],['entropy',fmt(r.ent)],['ent coef',fmt(r.ent_coef)],
    ['KL to teacher',fmt(r.kl_teacher)],['clip',fmt(r.clipfrac)],['lr',fmtLR(r.lr)]];
  if(r.wl_auc_r100!=null) items.push(['W/L AUC r100',fmt(r.wl_auc_r100)],['phi AUC r100',fmt(r.phi_auc_r100)]);
  if(r.wl_blend!=null) items.push(['W/L blend b',fmt(r.wl_blend)],['W/L adv share',fmt(r.adv_share_wl)],['adv corr',fmt(r.adv_corr)]);
  for(const k of Object.keys(r).filter(k=>k.startsWith('vs_'))) items.push(['vs '+k.slice(3),fmt(r[k])]);
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
  const flat=evRows.map(r=>{const o={total_turns:r.total_turns};
    for(const [k,v] of Object.entries(r.summary||{})) o['s|'+k]=v.score;
    return o;});
  const cs=[{t:'Gate score vs each version', n:'one point per gate; promotion needs >= 0.6 against every one', raw:true,
     s:names.map(b=>({k:'s|'+b,l:b}))}];
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
  if(!evLast||!evLast.cells){box.innerHTML='<div style="color:var(--ink-3);font-size:12px">no gate yet</div>';return;}
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
// a learning rate in 3 significant figures (the cosine schedule makes long floats): 2.00e-4
const fmtLR=v=>(v==null||isNaN(+v))?'–':(+v).toExponential(2).replace('e-','e-');
// ---- ratchet (train/ratchet.py): state.json + one row per gate decision
async function pollRatchet(){
  try{
    const q=new URLSearchParams(location.search);
    const res=await fetch(BASE+'ratchet?'+q.toString(),{cache:'no-store'});
    if(!res.ok) return;
    const j=await res.json(); if(!j.state) return;
    const st=j.state, gs=j.gens, box=document.getElementById('ratchet');
    const fmtT=t=>new Date(t*1000).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
    const evs=(j.events||[]).map(e=>({...e,when:fmtT(e.time)}));
    if(JSON.stringify(evs)!==JSON.stringify(EVENTS)){EVENTS=evs; draw();}
    const vcol={promote:'var(--s3)',extend:'var(--s4)',discard:'#e34948'};
    const cand=st.cand?`gen ${st.gen} · segment ${st.segment}`:'between candidates';
    const tile=(k,v)=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`;
    let h=`<h3 style="margin-top:0">Ratchet</h3><div class="sec">One learner trains continuously; every segment ends in a gate played greedily
      against the last version (${gs.length?gs[gs.length-1].n_anchor:528} games) and every earlier one. Promote at &ge; 0.6 against all of them,
      otherwise train on. A promoted version joins the opponent pool and becomes the KL teacher.</div>
      <div class="tiles">${tile('anchor',st.anchor_name)}${tile('training',cand)}${tile('promotions',st.promotions)}
      ${tile('gates',gs.length)}${tile('learning rate',fmtLR(rows&&rows.length&&rows[rows.length-1].lr!=null?rows[rows.length-1].lr:st.lr))}${tile('experiment turns',fmt(st.turns))}</div>`;
    if(st.anchor_scores&&Object.keys(st.anchor_scores).length){
      h+=`<div class="sec">Last promotion's gate scores (training opponents are weighted toward the low ones; &ge; 0.95 leaves training): `+
        Object.entries(st.anchor_scores).map(([k,v])=>`<b>${k}</b> ${v.toFixed(2)}`).join(' · ')+`</div>`;
    }
    if(EVENTS.length){
      h+=`<div class="sec"><b>Restarts</b> (dashed markers on the charts; the charts plot only the rows after the newest rewind):`+
        `<ul style="margin:4px 0 0;padding-left:18px">`+[...EVENTS].reverse().map(e=>
        `<li><span style="color:var(--ink-3)">${e.when} · ${fmt(e.total_turns)} turns</span> — ${e.label}</li>`).join('')+`</ul></div>`;
    }
    if(gs.length){
      h+=`<div class="grid"><div class="card" id="rgate"><h2>Gate: lowest score vs any version</h2>
        <div class="note">one point per gate; the second line is the 0.6 promotion bar</div><div class="plot"></div></div></div>`;
      const names=[...new Set(gs.flatMap(g=>Object.keys(g.scores)))].filter(n=>n!=='anchor');
      h+=`<div class="hm"><table><tr><th>time</th><th>gate</th><th>turns</th><th>anchor</th><th>vs anchor</th>`+
        names.map(n=>`<th>${n}</th>`).join('')+`<th>lowest</th><th>verdict</th></tr>`;
      const ncol=names.length+6;
      const tl=[...gs.map(g=>({t:g.time,g})),...EVENTS.map(e=>({t:e.time,e}))].sort((a,b)=>b.t-a.t);
      for(const {g,e} of tl){
        if(e){h+=`<tr><th>${new Date(e.time*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</th>
          <td colspan="${ncol}" style="text-align:left;color:var(--ink-2)">⟲ restart: ${e.label}</td></tr>`;continue;}
        const t=new Date(g.time*1000);
        h+=`<tr><th>${t.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</th><th>${g.tag}</th><td>${fmt(g.total_turns)}</td><td>${g.anchor_name}</td>
          <td style="font-weight:600">${g.vs_anchor.toFixed(3)}</td>`+
          names.map(n=>{const v=g.scores[n], b=(g.anchor_scores||{})[n];
            return v==null?'<td>–</td>':`<td title="anchor ${b!=null?b.toFixed(2):'–'}">${v.toFixed(2)}${b!=null?
              ` <span style="color:var(--ink-3)">(${(v-b>=0?'+':'')+(v-b).toFixed(2)})</span>`:''}</td>`;}).join('')+
          `<td>${g.worst_score!=null?g.worst_score.toFixed(2)+' '+g.worst_score_vs:'–'}</td>
          <td style="color:${vcol[g.verdict]};font-weight:600">${g.verdict}</td></tr>`;
      }
      h+='</table></div>';
    }
    const sl=j.sl||[];
    if(sl.length){
      // one row per supervised pass: accuracy = the greedy move is a correct one; held = turns it did not train on
      const K=['blank','pearl','queen_kill','keep_wall','late_suicide','trapped_queen'];
      const ab=(r,part,k)=>{const b=((r.before||{})[part]||{})[k], a=((r.after||{})[part]||{})[k];
        if(!a) return '<td>–</td>';
        const d=a.acc-b.acc, c=d>0.005?'var(--s3)':(d<-0.005?'#e34948':'var(--ink-3)');
        return `<td title="P(correct) ${b.p.toFixed(2)} → ${a.p.toFixed(2)}, n ${a.n}">${b.acc.toFixed(2)} → <b>${a.acc.toFixed(2)}</b>`+
          ` <span style="color:${c}">(${(d>=0?'+':'')+d.toFixed(2)})</span></td>`;};
      h+=`<h3>Supervised pass (perfect play)</h3><div class="sec">After every gate: a few small steps toward the moves that are known
        to be right, generated fresh on random maps (never the training maps), stopped if the policy drifts (KL on ordinary turns). Accuracy = the greedy move
        is a correct one, before → after the pass; <b>held</b> = turns it did not train on, <b>in-sample</b> = the ones it did.</div>`;
      h+=`<div class="hm"><table><tr><th>time</th><th>after gate</th><th>found</th><th>steps</th><th>KL</th>`+
        K.map(k=>`<th>held ${k}</th>`).join('')+K.map(k=>`<th>in-sample ${k}</th>`).join('')+`</tr>`;
      for(const r of [...sl].reverse()){
        const t=new Date(r.time*1000);
        h+=`<tr><th>${t.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</th><th>${r.tag}</th>
          <td>${K.map(k=>(r.found||{})[k]||0).join(' / ')}</td>
          <td title="${r.stopped||''}">${r.steps}/${r.steps_max}${r.stopped?' ⏹':''}</td><td>${(r.kl_held||0).toFixed(4)}</td>`+
          K.map(k=>ab(r,'held',k)).join('')+K.map(k=>ab(r,'train',k)).join('')+`</tr>`;
      }
      h+=`</table><div class="note" style="margin-top:6px">found = blank / pearl / queen kill / keep wall / late suicide / trapped queen;
        ⏹ = stopped early at the KL limit (hover for why)</div></div>`;
    }
    box.innerHTML=h;
    if(gs.length){
      const rows_=gs.map((g,i)=>({x:i+1,v:g.worst_score!=null?g.worst_score:g.vs_anchor,bar:0.6}));
      plot(document.querySelector('#rgate .plot'),{t:'lowest score',raw:true,events:false,
        s:[{k:'v',l:'lowest score'},{k:'bar',l:'promotion bar'}]}, rows_.map(r=>r.x), rows_);
    }
  }catch(e){}
}
pollRatchet(); setInterval(pollRatchet,20000);
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

    def turn0() -> int:
        """The run's turn origin, from dash.json {"turn0": N}: a fresh start that
        continues an old checkpoint's turn count still plots from 0."""
        try:
            return int(json.loads((run / "dash.json").read_text()).get("turn0", 0))
        except (OSError, ValueError, AttributeError):
            return 0

    def shift(r, t0, keys=("total_turns",)):
        if not t0 or not isinstance(r, dict):
            return r
        r = dict(r)
        for k in keys:
            if isinstance(r.get(k), (int, float)):
                r[k] -= t0
        return r

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
            t0 = turn0()
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
                rows = [shift(r, t0) for r in rows]
                last = rows[-1] if rows else None
                slim = [{k: v for k, v in r.items() if k != "cells"} for r in rows]
                body = json.dumps({"rows": slim, "last": last}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
            elif tail.endswith("ratchet"):
                state, gens = None, []
                if (run / "state.json").exists():
                    try:
                        state = json.loads((run / "state.json").read_text())
                    except json.JSONDecodeError:
                        pass
                if (run / "gens.jsonl").exists():
                    for line in (run / "gens.jsonl").read_text().splitlines():
                        try:
                            gens.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                events = []
                if (run / "events.jsonl").exists():
                    for line in (run / "events.jsonl").read_text().splitlines():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                # the supervised passes on perfect play (train/perfect_play.py, ratchet --sl)
                sl = []
                if (run / "sl.jsonl").exists():
                    for line in (run / "sl.jsonl").read_text().splitlines():
                        try:
                            sl.append({k: v for k, v in json.loads(line).items() if k != "kl_trace"})
                        except json.JSONDecodeError:
                            pass
                state = shift(state, t0, ("total_turns", "turns"))
                gens = [shift(g, t0) for g in gens]
                events = [shift(e, t0) for e in events]
                sl = [shift(r, t0) for r in sl]
                body = json.dumps({"state": state, "gens": gens, "events": events, "sl": sl}).encode()
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
                rows = [shift(r, t0) for r in rows]
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
    p.add_argument("--run", default=str(ROOT / "runs/scratch_1001b"))
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
