"""The ratchet: short PPO segments from a gated anchor that can only get stronger.

ft1-ft7 all ran one long fine-tune and either froze (a leash too tight to
move) or found a gain and then lost it (ft6 113M -> 125M). Here nothing is
kept unless it wins a fixed, large match:

  1. the ANCHOR is the best policy so far (it starts as ft6's 113M snapshot);
  2. a CANDIDATE trains from it for one segment (ratchet_train.py: result-only
     team advantage, frozen pretrained critic, KL to the anchor);
  3. the GATE plays the candidate greedily against the anchor (--anchor-reps x
     96 games, both sides of every map) and against every league member (96
     games each);
  4. PROMOTE if it scores >= --promote vs the anchor and loses no more than
     --max-drop against any league member relative to the anchor's own score.
     The old anchor joins the league (as a training opponent too).
     EXTEND (another segment from the candidate, same teacher) if it failed
     but is not worse: score >= --extend-min and no league regression, up to
     --max-segments segments.
     DISCARD otherwise; after --lr-patience discards in a row the LR halves
     (down to --lr-min).

Training opponents are weighted towards the league members the anchor does
worst against: w = max(1 - score, 0.15) ** 2 from the anchor's gate scores.

Everything the dashboard shows lands in --run: log.jsonl (training, one row
per iteration across all segments), eval.jsonl (each gate, in yardstick's
format) and gens.jsonl (one row per gate decision). state.json is the
supervisor's memory: kill it, start it again with the same --run and it picks
up where it was (a segment interrupted mid-way resumes from its latest.pt).
A file named STOP in --run makes it exit at the next decision point.

    python -m train.ratchet run --run ../runs/ratchet
    python -m train.ratchet gate --cand a.pt --anchor b.pt --league "x=c.pt" --out g.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
BCSIM = ROOT / "bcsim"
PY = "/usr/bin/python3"                          # the torch that can drive the GPU
ANCHOR = "anchor"

DEFAULT_LEAGUE = [
    ("submitted v10", ROOT / "runs/submitted/v10.pt"),
    ("v9", ROOT / "runs/submitted/v9.pt"),
    ("sss_r3", ROOT / "runs/anchors/sss_r3_bc_64x4.pt"),
    ("sabotage", ROOT / "runs/anchors/sabotage_bc_64x4.pt"),
    ("shink_r1", ROOT / "runs/anchors/shink_r1_bc_64x4.pt"),
    ("vibing_r4", ROOT / "runs/anchors/vibing_r4_bc_64x4.pt"),
]


# ------------------------------------------------------------------ gate
def gate(a) -> None:
    """Plays the candidate against the anchor (repeated, for a big sample) and
    the league; writes a yardstick-style row with the anchor reps merged."""
    import numpy as np
    import torch
    sys.path.insert(0, str(BCSIM))
    import bcsim
    from train.yardstick import evaluate, greedy, load_net

    dev = torch.device("cuda")
    maps = bcsim.load_maps(a.maps)
    map_names = [f.stem for f in sorted(pathlib.Path(a.maps).glob("*.map"))]
    league = [tuple(x.split("=", 1)) for x in a.league.split(",") if x]
    specs = []
    if a.anchor:
        specs += [(f"{ANCHOR}#{i}", a.anchor) for i in range(a.anchor_reps)]
    specs += league
    n_envs = len(specs) * len(maps) * 2 * max(1, a.games // 2)
    net, _ = load_net(a.cand, dev)
    opps = []
    # a captured graph does not hold its network: keep every one alive here, or
    # a replay reads freed weights (illegal memory access)
    nets = [net]
    for name, path in specs:
        onet, _ = load_net(path, dev)
        nets.append(onet)
        opps.append({"name": name, "act": greedy(onet, dev, n_envs), "path": path})
    res = evaluate(greedy(net, dev, n_envs), opps, maps, map_names, games=a.games,
                   threads=a.threads, seed=a.seed, max_seconds=a.max_seconds)

    # merge the anchor repeats into one opponent
    cells, summary = res["cells"], res["summary"]
    reps = [k for k in cells if k.startswith(ANCHOR + "#")]
    if reps:
        merged = {}
        for k in reps:
            for m, c in cells.pop(k).items():
                d = merged.setdefault(m, {"n": 0, "win": 0, "draw": 0, "loss": 0,
                                          "rounds": 0.0, "kills": 0.0, "deaths": 0.0,
                                          "longest": 0.0})
                for f in ("rounds", "kills", "deaths", "longest"):
                    d[f] += c[f] * c["n"]
                for f in ("n", "win", "draw", "loss"):
                    d[f] += c[f]
            summary.pop(k)
        tot = {"n": 0, "win": 0, "draw": 0, "kills": 0.0}
        for m, d in merged.items():
            g = max(d["n"], 1)
            for f in ("rounds", "kills", "deaths", "longest"):
                d[f] = round(d[f] / g, 3)
            d["score"] = round((d["win"] + 0.5 * d["draw"]) / g, 4)
            d["portal"] = 0.0
            for f in ("n", "win", "draw"):
                tot[f] += d[f]
            tot["kills"] += d["kills"] * d["n"]
        g = max(tot["n"], 1)
        cells = {ANCHOR: merged, **cells}
        summary = {ANCHOR: {"n": tot["n"], "score": round((tot["win"] + 0.5 * tot["draw"]) / g, 4),
                            "win": round(tot["win"] / g, 4), "draw": round(tot["draw"] / g, 4),
                            "kills": round(tot["kills"] / g, 3), "portal": 0.0}, **summary}
    row = {"cells": cells, "summary": summary, "seconds": res["seconds"],
           "ckpt": a.cand, "anchor_ckpt": a.anchor,
           "opponents": {n: p for n, p in specs if not n.startswith(ANCHOR + "#")}}
    if a.anchor:
        row["opponents"][ANCHOR] = a.anchor
    sub = [k for k in summary if k.startswith("submitted ")]
    if sub:
        row["submitted_name"], row["submitted_score"] = sub[0], summary[sub[0]]["score"]
    tmp = pathlib.Path(a.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    tmp.replace(a.out)
    print(json.dumps({k: (v["score"], v["n"]) for k, v in summary.items()}), flush=True)


# ------------------------------------------------------------------ supervisor
class Supervisor:
    def __init__(self, a):
        self.a = a
        self.run = pathlib.Path(a.run).resolve()
        self.run.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run / "state.json"
        self.logf = open(self.run / "ratchet.log", "a")
        # stop it by this pid, never by a command-line pattern
        import os
        (self.run / "supervisor.pid").write_text(str(os.getpid()))

    # -- bookkeeping
    def say(self, msg: str) -> None:
        line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.s, indent=1))
        tmp.replace(self.state_path)

    def append(self, name: str, row: dict) -> None:
        with open(self.run / name, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    def init_state(self) -> None:
        if self.state_path.exists():
            self.s = json.loads(self.state_path.read_text())
            self.say(f"resumed: gen {self.s['gen']} segment {self.s['segment']}, "
                     f"anchor {self.s['anchor']}, {self.s['turns'] / 1e6:.1f}M turns")
            return
        (self.run / "anchors").mkdir(exist_ok=True)
        a0 = self.run / "anchors" / "gen0.pt"
        shutil.copy(self.a.start, a0)
        self.s = {"gen": 1, "segment": 0, "anchor": str(a0), "anchor_name": "gen0 (ft6 113M)",
                  "anchor_scores": None, "league": [[n, str(p)] for n, p in DEFAULT_LEAGUE],
                  "cand": None, "turns": 0, "lr": self.a.lr, "discards": 0,
                  "seed": 1, "failures": 0, "promotions": 0}
        self.save()
        self.say(f"new ratchet from {self.a.start}")

    # -- subprocesses
    def call(self, args: list[str], log: pathlib.Path) -> int:
        with open(log, "a") as fh:
            fh.write(f"\n$ {' '.join(args)}\n")
            fh.flush()
            p = subprocess.run(args, cwd=BCSIM, stdout=fh, stderr=subprocess.STDOUT)
        return p.returncode

    def run_gate(self, cand: str, anchor: str | None, tag: str) -> dict | None:
        out = self.run / "gates" / f"{tag}.json"
        out.parent.mkdir(exist_ok=True)
        if out.exists():
            return json.loads(out.read_text())
        league = ",".join(f"{n}={p}" for n, p in self.s["league"])
        args = [PY, "-u", "-m", "train.ratchet", "gate", "--cand", cand, "--league", league,
                "--out", str(out), "--games", str(self.a.games),
                "--anchor-reps", str(self.a.anchor_reps), "--seed", str(self.s["seed"] * 7 + 3),
                "--max-seconds", str(self.a.gate_seconds)]
        if anchor:
            args += ["--anchor", anchor]
        for attempt in range(3):
            self.say(f"gate {tag}: attempt {attempt + 1}")
            rc = self.call(args, self.run / "gate.out")
            if rc == 0 and out.exists():
                return json.loads(out.read_text())
            self.say(f"gate {tag} failed with exit code {rc}")
            time.sleep(30)
        return None

    def weights(self) -> tuple[list[str], list[str], list[float]]:
        """Training opponents: the anchor plus the league, weighted to the
        members the anchor scores worst against."""
        sc = self.s["anchor_scores"] or {}
        names = [ANCHOR] + [n for n, _ in self.s["league"]]
        paths = [self.s["anchor"]] + [p for _, p in self.s["league"]]
        w = [max(1.0 - sc.get(n, 0.5), 0.15) ** 2 for n in names]
        return names, paths, w

    def stop_requested(self) -> bool:
        if (self.run / "STOP").exists():
            self.say("STOP file found: exiting at this decision point")
            return True
        return False

    # -- one segment
    def cand_turns(self, path: pathlib.Path) -> int:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return int(ck.get("cand_turns", 0)) if ck.get("ratchet") else 0

    def log_turns(self) -> int:
        """The newest total_turns in log.jsonl (so a crashed segment's turns
        still move the dashboard's x axis on)."""
        log = self.run / "log.jsonl"
        if not log.exists():
            return 0
        with open(log, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 65536))
            for line in reversed(fh.read().decode(errors="ignore").splitlines()):
                try:
                    return int(json.loads(line)["total_turns"])
                except (ValueError, KeyError):
                    continue
        return 0

    def train_segment(self) -> str:
        """Runs (or resumes) the current segment. Returns 'ok', 'abort' or 'crash'.

        The candidate's own counter (cand_turns in its checkpoints) says how far
        it is; this segment ends at (segment + 1) * --segment-turns of it."""
        s, c = self.s, self.s["cand"]
        cdir = pathlib.Path(c["dir"])
        final = cdir / "final.pt"
        if c.get("segment_done") == s["segment"] and final.exists():
            return "ok"
        names, paths, w = self.weights()
        target = (s["segment"] + 1) * self.a.segment_turns
        for attempt in range(3):
            latest = cdir / "latest.pt"
            if latest.exists():
                init, cont = str(latest), True
            else:
                init, cont = c["init"], c["cont"]
            done = self.cand_turns(pathlib.Path(init)) if cont else 0
            left = target - done
            if left <= 0:
                left = 1                          # one iteration still writes final.pt
            if latest.exists():
                self.say(f"resuming from {latest} ({done / 1e6:.1f}M candidate turns done)")
            args = [PY, "-u", "-m", "train.ratchet_train", "--init", init,
                    "--teacher", s["anchor"], "--out", str(cdir),
                    "--log", str(self.run / "log.jsonl"), "--turns", str(left),
                    "--turn-base", str(c["turns_before_exp"] + done), "--gen", str(s["gen"]),
                    "--segment", str(s["segment"]), "--lr", str(s["lr"]),
                    "--seed", str(s["seed"] * 1000 + s["segment"] * 10 + attempt),
                    "--opponents", ",".join(paths), "--opp-names", ",".join(names),
                    "--opp-weights", ",".join(f"{x:.4f}" for x in w),
                    "--self-frac", str(self.a.self_frac), "--kl-coef", str(self.a.kl_coef)]
            if cont:
                args.append("--continue")
            if final.exists():
                final.unlink()                    # the previous segment's, kept as segN.pt
            self.say(f"train gen {s['gen']} segment {s['segment']} (attempt {attempt + 1}): "
                     f"lr {s['lr']}, opponents " +
                     ", ".join(f"{n} {x / sum(w):.2f}" for n, x in zip(names, w)))
            t0 = time.time()
            rc = self.call(args, self.run / "train.out")
            if rc == 0 and final.exists():
                s["turns"] = c["turns_before_exp"] + self.cand_turns(final)
                c["segment_done"] = s["segment"]
                self.save()
                self.say(f"segment done in {(time.time() - t0) / 60:.0f} min")
                return "ok"
            s["turns"] = max(s["turns"], self.log_turns())
            self.save()
            if rc == 3:
                self.say("segment aborted by a safety check (see train.out)")
                return "abort"
            self.say(f"training exited with code {rc}; retrying in 60s")
            time.sleep(60)
        return "crash"

    # -- the loop
    def loop(self) -> None:
        self.init_state()
        s = self.s
        if s["anchor_scores"] is None:
            row = self.run_gate(s["anchor"], None, "gen0_baseline")
            if row is None:
                raise SystemExit("baseline gate failed three times")
            s["anchor_scores"] = {k: v["score"] for k, v in row["summary"].items()}
            self.append("eval.jsonl", {**row, "total_turns": 0, "gen": 0, "kind": "baseline"})
            self.say("baseline: " + ", ".join(f"{k} {v:.3f}" for k, v in s["anchor_scores"].items()))
            self.save()
        while True:
            if self.stop_requested():
                return
            if s["cand"] is None:
                cdir = self.run / "cands" / f"g{s['gen']:03d}_s{s['seed']}"
                s["cand"] = {"dir": str(cdir), "init": s["anchor"], "cont": False,
                             "turns_before": 0, "turns_before_exp": s["turns"]}
                s["segment"] = 0
                self.save()
            status = self.train_segment()
            cdir = pathlib.Path(s["cand"]["dir"])
            if status != "ok":
                s["failures"] += 1
                self.decide_discard(f"training {status}")
                if s["failures"] >= 4:
                    self.say("4 failed segments in a row: stopping for a human")
                    return
                continue
            s["failures"] = 0
            final = cdir / "final.pt"
            # keep each segment's end point: final.pt is overwritten by an extension
            seg_ck = cdir / f"seg{s['segment']}.pt"
            if not seg_ck.exists():
                shutil.copy(final, seg_ck)
            tag = f"g{s['gen']:03d}_s{s['seed']}_seg{s['segment']}"
            row = self.run_gate(str(seg_ck), s["anchor"], tag)
            if row is None:
                s["failures"] += 1
                self.decide_discard("gate failed to run")
                continue
            self.decide(row, seg_ck, tag)

    def decide(self, row: dict, ck: pathlib.Path, tag: str) -> None:
        s, a = self.s, self.a
        summ = {k: v["score"] for k, v in row["summary"].items()}
        vs_anchor = summ[ANCHOR]
        drops = {n: round(s["anchor_scores"][n] - summ[n], 4) for n in summ
                 if n != ANCHOR and n in s["anchor_scores"]}
        worst = max(drops.items(), key=lambda kv: kv[1]) if drops else ("-", 0.0)
        regress = worst[1] > a.max_drop
        if vs_anchor >= a.promote and not regress:
            verdict = "promote"
        elif vs_anchor >= a.extend_min and not regress and s["segment"] + 1 < a.max_segments:
            verdict = "extend"
        else:
            verdict = "discard"
        g = {"time": time.time(), "gen": s["gen"], "segment": s["segment"], "seed": s["seed"],
             "tag": tag, "total_turns": s["turns"], "vs_anchor": vs_anchor,
             "n_anchor": row["summary"][ANCHOR]["n"], "worst_drop": worst[1],
             "worst_drop_vs": worst[0], "verdict": verdict, "lr": s["lr"],
             "anchor_name": s["anchor_name"], "scores": summ, "anchor_scores": s["anchor_scores"],
             "gate_seconds": row["seconds"]}
        self.append("gens.jsonl", g)
        self.append("eval.jsonl", {**row, "total_turns": s["turns"], "gen": s["gen"],
                                   "segment": s["segment"], "kind": "gate", "verdict": verdict})
        self.say(f"GATE {tag}: {vs_anchor:.3f} vs anchor over {g['n_anchor']} games; worst drop "
                 f"{worst[1]:+.3f} ({worst[0]}) -> {verdict.upper()}")
        if verdict == "promote":
            new = self.run / "anchors" / f"gen{s['gen']}.pt"
            shutil.copy(ck, new)
            s["league"].append([s["anchor_name"].split(" ")[0], s["anchor"]])
            # past anchors beyond the newest --keep-anchors leave the league (and
            # the gate), so a gate stays ~20 minutes
            past = [e for e in s["league"] if e[0].startswith("gen")]
            for e in past[:-a.keep_anchors]:
                s["league"].remove(e)
            # the old anchor's score against the new one is the gate's, seen from the other side
            new_scores = {k: v for k, v in summ.items() if k != ANCHOR}
            new_scores[s["anchor_name"].split(" ")[0]] = vs_anchor
            s.update(anchor=str(new), anchor_name=f"gen{s['gen']}", anchor_scores=new_scores,
                     gen=s["gen"] + 1, cand=None, discards=0, promotions=s["promotions"] + 1)
            s["seed"] += 1
            self.save()
        elif verdict == "extend":
            s["cand"].update(init=str(ck), cont=True)
            s["segment"] += 1
            self.save()
        else:
            self.decide_discard(f"gate {vs_anchor:.3f}, worst drop {worst[1]:+.3f}")

    def decide_discard(self, why: str) -> None:
        s, a = self.s, self.a
        s["discards"] += 1
        self.say(f"discard candidate {s['cand']['dir'] if s['cand'] else '-'}: {why}")
        if s["discards"] >= a.lr_patience and s["lr"] > a.lr_min:
            s["lr"] = max(s["lr"] / 2, a.lr_min)
            s["discards"] = 0
            self.say(f"{a.lr_patience} discards in a row: lr -> {s['lr']}")
        s["cand"], s["segment"] = None, 0
        s["seed"] += 1
        self.save()


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", default=str(ROOT / "runs/ratchet"))
    r.add_argument("--start", default=str(ROOT / "runs/ft6/snapshots/turns_112721920.pt"))
    r.add_argument("--segment-turns", type=int, default=50_000_000)
    r.add_argument("--max-segments", type=int, default=3)
    r.add_argument("--promote", type=float, default=0.55)
    r.add_argument("--extend-min", type=float, default=0.48)
    r.add_argument("--max-drop", type=float, default=0.10,
                   help="largest allowed fall vs a league member (2 SE at 96 games each)")
    r.add_argument("--lr", type=float, default=3e-5)
    r.add_argument("--lr-min", type=float, default=7.5e-6)
    r.add_argument("--lr-patience", type=int, default=2)
    r.add_argument("--kl-coef", type=float, default=0.5)
    r.add_argument("--self-frac", type=float, default=0.2)
    r.add_argument("--games", type=int, default=12, help="gate games per (opponent, map)")
    r.add_argument("--anchor-reps", type=int, default=4, help="anchor played this many times over")
    r.add_argument("--gate-seconds", type=float, default=3600)
    r.add_argument("--keep-anchors", type=int, default=3, help="past anchors kept in the league")
    g = sub.add_parser("gate")
    g.add_argument("--cand", required=True)
    g.add_argument("--anchor", default="")
    g.add_argument("--league", default="")
    g.add_argument("--out", required=True)
    g.add_argument("--maps", default=str(ROOT / "runs/ft3/maps"))
    g.add_argument("--games", type=int, default=12)
    g.add_argument("--anchor-reps", type=int, default=4)
    g.add_argument("--threads", type=int, default=16)
    g.add_argument("--seed", type=int, default=12345)
    g.add_argument("--max-seconds", type=float, default=3600)
    a = p.parse_args()
    if a.cmd == "gate":
        gate(a)
    else:
        Supervisor(a).loop()


if __name__ == "__main__":
    main()
