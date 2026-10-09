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
import math
import re
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
BCSIM = ROOT / "bcsim"
PY = "/usr/bin/python3"                          # the torch that can drive the GPU
ANCHOR = "anchor"

# Every entry must give the 708 scalars the env now writes (bc_memory.hpp): the
# clones are native, the older nets are zero-padded copies from
# train/migrate_scalars.py, which play identically to their originals.
#
# gen4 is here because the user asked (2026-09-23) to keep it as a frozen
# opponent across the switch to memory, and 192,000 turns of play confirm the
# widened copy is move-for-move the same policy.
#
# Refreshed for the 2026-09-23 restart: v9, sss_r3 and vibing_r4 left, being
# clones of teams that have since fallen down the ladder; devtest_2050 and v12
# joined, and they are the two strongest models in that day's round robin after
# sponge (which is the seed, so it is already in via --self-frac).
DEFAULT_LEAGUE = [
    ("gen4", ROOT / "runs/anchors708/gen4.pt"),
    ("devtest_2050", ROOT / "runs/i2/devtest_2050/best.pt"),
    ("v12", ROOT / "runs/submitted/v12.pt"),
    ("submitted v10", ROOT / "runs/anchors708/v10.pt"),
    ("sabotage", ROOT / "runs/anchors708/sabotage_bc_64x4.pt"),
    ("shink_r1", ROOT / "runs/anchors708/shink_r1_bc_64x4.pt"),
    # The team that is top of the ladder NOW, cloned from its current submission
    # (3952, held-out action accuracy 0.8035). `sabotage` above is the same team's
    # submission 546 -- their oldest -- so it stayed in as a fixed yardstick while
    # this one answers the question that matters: can we beat what is actually
    # winning. A run started before this clone existed keeps its own league, which
    # is stored in its state file, so nothing already measured moves.
    ("sabotage_3952", ROOT / "runs/imitate_sab_3952/best.pt"),
]


# ------------------------------------------------------------------ gate

def _is_lstm(path: str) -> bool:
    """Whether a checkpoint is the LSTM policy, which trains through
    ratchet_lstm_train.py (per-dragon state, 14x14 grid) instead of ratchet_train.py."""
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)["args"].get("arch") in ("lstm", "ff", "ffl")


def gate(a) -> None:
    """Plays the candidate against the anchor (repeated, for a big sample) and
    the league; writes a yardstick-style row with the anchor reps merged."""
    import numpy as np
    import torch
    sys.path.insert(0, str(BCSIM))
    if getattr(a, "s2", False):          # sonar v2 + the self-kill: its own library, before bcsim loads
        import os
        lib = "libbcvec_s2_g15.so" if getattr(a, "grid", 14) == 15 else "libbcvec_s2.so"
        if getattr(a, "portal", False):    # BC_PORTALREP (2026-10-02): 15x15 only
            if getattr(a, "grid", 14) != 15:
                raise SystemExit("--portal needs --grid 15")
            lib = "libbcvec_s2_g15p.so"
        os.environ["BCSIM_LIB"] = str(BCSIM / "bcsim" / lib)
    import bcsim
    from train.yardstick import evaluate, greedy, load_net

    dev = torch.device("cuda")
    # maps the gate skips (training keeps them): --exclude-maps, plus the run's
    # gate_exclude.txt, which a live supervisor's gates pick up without a restart
    skip = {x for x in a.exclude_maps.split(",") if x}
    ex = pathlib.Path(a.out).resolve().parent.parent / "gate_exclude.txt"
    if ex.exists():
        skip |= set(ex.read_text().split())
    files = [f for f in sorted(pathlib.Path(a.maps).glob("*.map")) if f.stem not in skip]
    maps = bcsim.load_maps([str(f) for f in files])
    map_names = [f.stem for f in files]
    if skip:
        print(f"  gate skips {', '.join(sorted(skip))}", flush=True)
    league = [tuple(x.split("=", 1)) for x in a.league.split(",") if x]
    specs = []
    if a.anchor:
        specs += [(f"{ANCHOR}#{i}", a.anchor) for i in range(a.anchor_reps)]
    specs += league
    n_envs = len(specs) * len(maps) * 2 * max(1, a.games // 2)
    net, cck = load_net(a.cand, dev)
    lstm_any = [cck["args"].get("arch") in ("lstm", "ff", "ffl")]

    def player(n_, ck_):
        # an LSTM policy (train/lstm_net.py) keeps a state per dragon and reads the
        # 14x14 grid, so it cannot go through greedy()
        if ck_["args"].get("arch") in ("lstm", "ff", "ffl"):
            from train.distill_lstm import LSTMGreedy
            f_ = LSTMGreedy(n_, dev, n_envs)
        else:
            f_ = greedy(n_, dev)
        # sonar v2 (--s2): its own action count, and whether it speaks the team packet
        f_.n_act = int(ck_["args"].get("n_actions", 48))
        f_.grid_model = int(ck_["args"].get("grid", 14))     # cropped from a bigger simulator grid
        if getattr(n_, "temp_in", False):
            # a temperature-conditioned policy played greedy is told the lowest temperature it trained at
            n_.temp_default.fill_(float(ck_["args"].get("temp_min", ck_["args"].get("temp", 0.4))))
        f_.speaks = bool(ck_["args"].get("sonar2", False))
        f_.keep_probs = f_.speaks
        return f_

    opps = []
    # a captured graph does not hold its network: keep every one alive here, or
    # a replay reads freed weights (illegal memory access)
    nets = [net]
    shared: dict = {}          # --fast: one player per checkpoint, so the anchor repeats are one call
    for name, path in specs:
        if a.fast and path in shared:
            act_, is_lstm = shared[path]
        else:
            onet, ock = load_net(path, dev)
            nets.append(onet)
            act_, is_lstm = player(onet, ock), ock["args"].get("arch") in ("lstm", "ff", "ffl")
            shared[path] = (act_, is_lstm)
        lstm_any.append(is_lstm)
        opps.append({"name": name, "act": act_, "path": path})
    # the LSTM policies were trained with our sonar (four-way broadcast) and read
    # its echoes, so any gate that includes one is played with sonar on
    cand_ = player(net, cck)
    cand_.temp = float(getattr(a, "cand_temp", 0.0))     # train/panel.py --temp: the candidate only
    res = evaluate(cand_, opps, maps, map_names, games=a.games,
                   threads=a.threads, seed=a.seed, max_seconds=a.max_seconds,
                   memchan=a.memchan, sonar=any(lstm_any) and not a.memchan, fast=a.fast,
                   skip_done=a.skip_done)

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
    row = {"cells": cells, "summary": summary, "seconds": res["seconds"], "steps": res.get("steps"),
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
            if "explore" not in self.s:           # a run from before the decaying rate
                self.s["explore"] = self.a.explore
            # a new --segment-turns applies from the current segment on (user, 2026-10-01: 40M -> 80M
            # once training got faster): the current segment keeps its start and gets the new length
            c_ = self.s.get("cand")
            if c_ and c_.get("seg_turns") and c_["seg_turns"] != self.a.segment_turns:
                start = c_.get("seg_base_turns", 0) + (self.s["segment"] - c_.get("seg_base_index", 0)) * c_["seg_turns"]
                self.say(f"segment length {c_['seg_turns'] / 1e6:.0f}M -> {self.a.segment_turns / 1e6:.0f}M "
                         f"from segment {self.s['segment']} (starts at {start / 1e6:.1f}M candidate turns)")
                c_.update(seg_base_turns=start, seg_base_index=self.s["segment"], seg_turns=self.a.segment_turns)
                self.save()
            # the shared map changes what every net in the league reads, so
            # turning it on or off mid-experiment would make the gate scores
            # before and after incomparable and the promotions meaningless
            if bool(self.s.get("memchan", False)) != bool(self.a.memchan):
                raise SystemExit(
                    f"this run was started with --memchan {self.s.get('memchan', False)} "
                    f"and you passed {bool(self.a.memchan)}. That moves memfar for every "
                    "net in the league, so the gate scores would not be comparable: "
                    "start a new --run directory instead")
            return
        (self.run / "anchors").mkdir(exist_ok=True)
        a0 = self.run / "anchors" / "gen0.pt"
        shutil.copy(self.a.start, a0)
        # the label is display only, but the FIRST token becomes the league key
        # when gen0 is promoted (see _promote), so it must stay "gen0"
        seed_tag = pathlib.Path(self.a.start).parent.name or pathlib.Path(self.a.start).stem
        self.s = {"gen": 1, "segment": 0, "anchor": str(a0), "anchor_name": f"gen0 ({seed_tag})",
                  "anchor_scores": None, "league": [[n, str(p)] for n, p in self.league0()],
                  "cand": None, "turns": 0, "lr": self.a.lr, "discards": 0,
                  "seed": 1, "failures": 0, "promotions": 0,
                  "memchan": bool(self.a.memchan), "explore": self.a.explore}
        self.save()
        self.say(f"new ratchet from {self.a.start}")

    def league0(self) -> list:
        """The league a NEW run starts with: --league-file (name=path per line, # comments)
        or DEFAULT_LEAGUE. For an LSTM run the names must be the ones the v8 critic was
        pretrained under (critic_v8 ids.json), since that is how members find their slot."""
        f = getattr(self.a, "league_file", "")
        if not f:
            return DEFAULT_LEAGUE
        out = []
        for line in pathlib.Path(f).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                n, _, p = line.partition("=")
                out.append((n.strip(), pathlib.Path(p.strip())))
        return out

    # -- subprocesses
    def call(self, args: list[str], log: pathlib.Path) -> int:
        with open(log, "a") as fh:
            fh.write(f"\n$ {' '.join(args)}\n")
            fh.flush()
            p = subprocess.run(args, cwd=BCSIM, stdout=fh, stderr=subprocess.STDOUT)
        return p.returncode

    def run_gate(self, cand: str, anchor: str | None, tag: str, league_only: list | None = None,
                 maps: str | None = None, games: int | None = None) -> dict | None:
        out = self.run / "gates" / f"{tag}.json"
        out.parent.mkdir(exist_ok=True)
        if out.exists():
            return json.loads(out.read_text())
        league = ",".join(f"{n}={p}" for n, p in (league_only if league_only is not None else self.gate_league()))
        args = [PY, "-u", "-m", "train.ratchet", "gate", "--cand", cand, "--league", league,
                "--out", str(out), "--games", str(games or self.a.games),
                "--anchor-reps", str(self.a.anchor_reps), "--seed", str(self.s["seed"] * 7 + 3),
                "--max-seconds", str(self.a.gate_seconds)]
        if self.a.memchan:
            args.append("--memchan")
        if getattr(self.a, "s2", False):
            args.append("--s2")
        if getattr(self.a, "portal", False):
            args.append("--portal")
        if getattr(self.a, "grid", 14) != 14:
            args += ["--grid", str(self.a.grid)]
        if maps:
            args += ["--maps", maps]
        elif getattr(self.a, "gate_maps", ""):
            args += ["--maps", self.a.gate_maps]
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

    def side_gates(self, cand: str, tag: str, row: dict) -> None:
        """(user, 2026-10-05) Map-restricted league members (state opp_maps: the distilled top teams)
        are left out of the main gate and played here, each only on its own maps (the current server
        copies, maps-server-1003), --side-games games a map; their scores go into the row (so they set
        the training weights) but never into the promotion rule (decide)."""
        om = self.s.get("opp_maps", {}) or {}
        for name, path in self.s["league"]:
            if name not in om:
                continue
            mdir = self.run / "opp_maps" / name
            mdir.mkdir(parents=True, exist_ok=True)
            for m_ in om[name]:
                link = mdir / f"{m_}.map"
                if not link.exists():
                    link.symlink_to(ROOT / "maps-server-1003" / f"{m_}.map")
            side = self.run_gate(cand, None, f"{tag}__{name}", league_only=[(name, path)], maps=str(mdir),
                                 games=getattr(self.a, "side_games", 16))
            if side is None:
                self.say(f"side gate vs {name} failed: its score is not updated")
                continue
            row["summary"][name] = side["summary"][name]
            row["cells"][name] = side["cells"][name]
            self.say(f"side gate {tag} vs {name} on {len(om[name])} maps: {side['summary'][name]['score']:.3f} "
                     f"over {side['summary'][name]['n']} games")

    @staticmethod
    def is_version(name: str) -> bool:
        """A past version of the learner: this run's genN, or an earlier run's
        <prefix>_genN (ratchet_v9's league holds ratchet_v8_sab's as v8sab_genN)."""
        return re.search(r"(^|_)gen\d+$", name) is not None

    def gate_league(self) -> list:
        """Who the gate plays besides the anchor. With --gate-versions K: every
        assessment bot (the members that are not past versions) and only the newest
        K past versions (user, 2026-09-28: a gate against every old version is too
        slow). Training still plays the whole league (weights)."""
        om = self.s.get("opp_maps", {}) or {}
        league = [e for e in self.s["league"] if e[0] not in om]     # map-restricted: side_gates
        only = [x for x in (getattr(self.a, "gate_league", "") or "").split(",") if x]
        if only:                          # --gate-league: exactly these members (quick checks)
            return [e for e in league if e[0] in only]
        k = getattr(self.a, "gate_versions", 0) or 0
        if k <= 0:
            return league
        versions = [e for e in league if self.is_version(e[0])]
        keep = {id(e) for e in versions[-k:]}
        return [e for e in league if not self.is_version(e[0]) or id(e) in keep]

    def weights(self) -> tuple[list[str], list[str], list[float], float]:
        """Training opponents: the anchor plus the league, weighted to the
        members the anchor scores worst against; and the self-play share.

        Self-play is --self-frac of the envs, plus, for every retired league
        member, its even share of the rest (1/len(league) of 1 - --self-frac),
        up to --self-frac-max: an opponent the anchor has outgrown hands its
        games to self-play rather than to the members still in training."""
        sc = self.s["anchor_scores"] or {}
        names = [ANCHOR] + [n for n, _ in self.s["league"]]
        paths = [self.s["anchor"]] + [p for _, p in self.s["league"]]
        if getattr(self.a, "league_weighting", "relative") == "absolute":
            # Nobody ever leaves training (user, 2026-09-28: forgetting). Each member
            # gets an ABSOLUTE share of games: (1 - --self-frac) / --weight-members,
            # scaled by w / w(0.5) with w = (1 - score)^2 + floor. As the anchor's
            # score against a member goes to 1 its share goes to ~0 but never to 0;
            # self-play takes whatever the league does not. The anchor always counts
            # as 0.5. If members the anchor LOSES to would push the league past
            # 1 - --self-frac, the league is scaled back to exactly that.
            floor = getattr(self.a, "weight_floor", 0.0025)
            base = self.a.self_frac
            w_ref = 0.25 + floor
            # a fixed budget per member, NOT (1 - base) / len(names): otherwise every
            # beaten old version added would shrink the share of the ones still hard
            even = (1.0 - base) / getattr(self.a, "weight_members", 10)
            share = [even * ((1.0 - (0.5 if n == ANCHOR else sc.get(n, 0.5))) ** 2 + floor) / w_ref
                     for n in names]
            total = sum(share)
            if total > 1.0 - base:
                share = [x * (1.0 - base) / total for x in share]
            self_frac = 1.0 - sum(share)
            if getattr(self.a, "train_versions_only", False):
                # (user, 2026-10-01: throughput) training plays only this run's own versions --
                # the anchor and past genN -- which share the learner's cheap architecture; the
                # rest of the league (LSTMs, flat nets) stays in the gate. The league's budget,
                # 1 - --self-frac, goes to the versions in proportion to their shares.
                keep = [i for i, n in enumerate(names) if n == ANCHOR or re.fullmatch(r"gen\d+", n)]
                ks = sum(share[i] for i in keep)
                names, paths = [names[i] for i in keep], [paths[i] for i in keep]
                share = [share[i] * (1.0 - base) / ks for i in keep]
                self_frac = base
            return names, paths, share, self_frac
        w = [max(1.0 - sc.get(n, 0.5), 0.15) ** 2 for n in names]
        # Retired: a member the anchor already beats at --retire-at or better
        # teaches nothing, so it gets no training games. It stays in the GATE, so
        # a regression against it is still caught, and since the weights are
        # recomputed from each new anchor's gate it comes back by itself if the
        # score ever falls under the line. The anchor itself is never retired.
        cut = getattr(self.a, "retire_at", 0.0) or 0.0
        if cut > 0:
            retired = [n for n in names[1:] if sc.get(n, 0.0) >= cut]
            if retired:
                self.say(f"retired from training (anchor scores >= {cut}): " + ", ".join(
                    f"{n} {sc[n]:.2f}" for n in retired))
            # --train-exclude (user, 2026-10-07: throughput): named members get no training games
            # either, whatever their score (each one is a forward per rollout step)
            excl = [n for n in names[1:] if n in set(x for x in getattr(self.a, "train_exclude", "").split(",") if x)
                    and n not in retired]
            if excl:
                self.say("excluded from training (--train-exclude): " + ", ".join(excl))
            # their share goes to the other opponents, not to self-play (unlike a retirement's)
            w = [0.0 if n in retired or n in excl else x for n, x in zip(names, w)]
            base = self.a.self_frac
            n_league = max(len(names) - 1, 1)
            self_frac = min(base + (1.0 - base) * len(retired) / n_league,
                            max(base, getattr(self.a, "self_frac_max", base)))
        else:
            self_frac = self.a.self_frac
        keep = [i for i, x in enumerate(w) if x > 0]
        return [names[i] for i in keep], [paths[i] for i in keep], [w[i] for i in keep], self_frac

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
        names, paths, w, self_frac = self.weights()
        # per candidate: ones created before this was stored ran 50M segments. seg_base_* rebase
        # the count when --segment-turns changed mid-candidate (see init_state)
        target = c.get("seg_base_turns", 0) + (s["segment"] + 1 - c.get("seg_base_index", 0)) * c.get("seg_turns", 50_000_000)
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
            lstm = _is_lstm(s["anchor"])
            lstm_extra = (["--start-name", self.a.start_name] if lstm and getattr(self.a, "start_name", "") else [])
            if lstm and getattr(self.a, "ent", None) is not None:
                lstm_extra += ["--ent", str(self.a.ent)]
            if lstm and getattr(self.a, "ent_anneal_turns", 0.0) > 0:
                lstm_extra += ["--ent-anneal-turns", str(self.a.ent_anneal_turns),
                               "--ent-anneal-start", str(getattr(self.a, "ent_anneal_start", 0.0))]
            if lstm and getattr(self.a, "kl_coef_end", -1.0) >= 0 and getattr(self.a, "kl_anneal_turns", 0.0) > 0:
                lstm_extra += ["--kl-coef-end", str(self.a.kl_coef_end), "--kl-anneal-turns",
                               str(self.a.kl_anneal_turns), "--kl-anneal-start", str(self.a.kl_anneal_start)]
            if lstm and getattr(self.a, "alpha", None) is not None:
                lstm_extra += ["--alpha", str(self.a.alpha)]
            if lstm and getattr(self.a, "temp_lo", -1.0) > 0:
                lstm_extra += ["--temp-lo", str(self.a.temp_lo), "--temp-hi-start", str(self.a.temp_hi_start),
                               "--temp-hi-end", str(self.a.temp_hi_end),
                               "--temp-anneal-turns", str(self.a.temp_anneal_turns)]
            if lstm and getattr(self.a, "opp_temp_max", -1.0) >= 0:
                lstm_extra += ["--opp-temp-max", str(self.a.opp_temp_max), "--opp-greedy", str(self.a.opp_greedy)]
            if lstm and getattr(self.a, "temp_cond", 0):
                lstm_extra += ["--temp-cond", str(self.a.temp_cond)]
            if lstm and getattr(self.a, "scal_in", 0):
                lstm_extra += ["--scal-in", str(self.a.scal_in)]
            if lstm and getattr(self.a, "ident_in", 0):
                lstm_extra += ["--ident-in", str(self.a.ident_in)]
            if lstm and getattr(self.a, "ahist_in", 0):
                lstm_extra += ["--ahist-in", str(self.a.ahist_in)]
            if lstm and getattr(self.a, "wl_head", 0):
                lstm_extra += ["--wl-head", str(self.a.wl_head), "--wl-lam", str(self.a.wl_lam),
                               "--wl-coef", str(self.a.wl_coef)]
            if lstm and getattr(self.a, "critic_view", 0):
                lstm_extra += ["--critic-view", str(self.a.critic_view), "--critic-frac", str(self.a.critic_frac)]
                if getattr(self.a, "critic_init", ""):
                    # used only while --init carries no critic (the first segment): later ones carry their own
                    lstm_extra += ["--critic-init", self.a.critic_init]
            if lstm and getattr(self.a, "lr_cosine_turns", 0.0) > 0:
                lstm_extra += ["--lr-cosine-turns", str(self.a.lr_cosine_turns)]
            if lstm and getattr(self.a, "critic_warmup", 0):
                lstm_extra += ["--critic-warmup", str(self.a.critic_warmup)]
            import torch
            ff = torch.load(s["anchor"], map_location="cpu", weights_only=False)["args"].get("arch") in ("ff", "ffl")
            module = "train.ratchet_ff_train" if ff else ("train.ratchet_lstm_train" if lstm else "train.ratchet_train")
            if getattr(self.a, "wl_blend_turns", 0.0) > 0:
                if not ff:
                    raise SystemExit("--wl-blend-turns needs a feed-forward run (ratchet_ff_train)")
                lstm_extra += ["--wl-blend-turns", str(self.a.wl_blend_turns),
                               "--wl-blend-start", str(self.a.wl_blend_start)]
            args = [PY, "-u", "-m", module, "--init", init,
                    "--teacher", s["anchor"], "--out", str(cdir),
                    "--log", str(self.run / "log.jsonl"), "--turns", str(left),
                    "--turn-base", str(c["turns_before_exp"] + done), "--gen", str(s["gen"]),
                    "--segment", str(s["segment"]), "--lr", str(s["lr"]),
                    "--seed", str(s["seed"] * 1000 + s["segment"] * 10 + attempt),
                    "--opponents", ",".join(paths), "--opp-names", ",".join(names),
                    "--opp-weights", ",".join(f"{x:.4f}" for x in w),
                    *(["--opp-maps", ";".join(f"{n_}={'+'.join(ms_)}" for n_, ms_ in s.get("opp_maps", {}).items()
                                              if n_ in names)]
                      if any(n_ in names for n_ in s.get("opp_maps", {})) else []),
                    "--self-frac", f"{self_frac:.4f}", "--kl-coef", str(self.a.kl_coef),
                    *(["--max-kl", str(self.a.max_kl)] if getattr(self.a, "max_kl", 0.0) > 0 else []),
                    *(["--memchan"] if self.a.memchan and not lstm else []),
                    *(["--maps", self.a.train_maps] if self.a.train_maps else []),
                    *(["--live-maps", self.a.live_maps, "--live-share", str(self.a.live_share)]
                      if self.a.live_maps and self.a.live_share > 0 else []),
                    *(["--gen-maps", self.a.gen_maps, "--gen-share", str(self.a.gen_share),
                       "--gen-per-map", str(self.a.gen_per_map)]
                      if getattr(self.a, "gen_maps", "") else []),
                    *(["--pearl-hotspots"] if getattr(self.a, "pearl_hotspots", False) else []),
                    # the run's CURRENT rate, which shrinks after every gate, so an
                    # extended candidate explores less in its next segment too
                    "--explore", str(s.get("explore", c.get("explore", 0.0))),
                    *(["--critic", self.a.critic] if lstm and getattr(self.a, "critic", "") else []),
                    *(["--s2"] if getattr(self.a, "s2", False) else []),
                    *(["--portal"] if getattr(self.a, "portal", False) else []),
                    "--temp", str(getattr(self.a, "temp", 1.0)),
                    *lstm_extra]
            if cont:
                args.append("--continue")
            if final.exists():
                final.unlink()                    # the previous segment's, kept as segN.pt
            self.say(f"train gen {s['gen']} segment {s['segment']} (attempt {attempt + 1}): "
                     f"lr {s['lr']}, explore {s.get('explore', 0.0):.4f}, self-play {self_frac:.2f}, opponents " +
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
        if s["anchor_scores"] is None and not s["league"]:
            s["anchor_scores"] = {}               # a run from scratch: no league to baseline against
            self.save()
        if s["anchor_scores"] is None:
            # a re-baseline (anchor_scores reset to null, e.g. after the gate maps
            # change) must not reuse the first baseline's cached gate file
            tag = "gen0_baseline" if s["gen"] <= 1 and s["promotions"] == 0 else f"{s['anchor_name']}_rebaseline"
            row = self.run_gate(s["anchor"], None, tag)
            if row is None:
                raise SystemExit("baseline gate failed three times")
            s["anchor_scores"] = {k: v["score"] for k, v in row["summary"].items()}
            self.append("eval.jsonl", {**row, "total_turns": s["turns"], "gen": s["gen"] if s["promotions"] else 0,
                                       "kind": "baseline"})
            self.say("baseline: " + ", ".join(f"{k} {v:.3f}" for k, v in s["anchor_scores"].items()))
            self.save()
        while True:
            if self.stop_requested():
                return
            if s["cand"] is None:
                self.new_cand()
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
            self.side_gates(str(seg_ck), tag, row)
            self.decide(row, seg_ck, tag)
            if getattr(self.a, "sl", False):
                self.sl_pass(tag)

    def new_cand(self) -> None:
        s = self.s
        cdir = self.run / "cands" / f"g{s['gen']:03d}_s{s['seed']}"
        # --cand-init: the run's first candidate starts there instead of at the anchor (a run
        # from scratch: the anchor is the uniform-random player, the candidate random weights)
        first = s["promotions"] == 0 and getattr(self.a, "cand_init", "")
        s["cand"] = {"dir": str(cdir), "init": self.a.cand_init if first else s["anchor"], "cont": False,
                     "turns_before": 0, "turns_before_exp": s["turns"],
                     # fixed per candidate, so an extension keeps its settings
                     "explore": s.get("explore", self.a.explore),   # at creation; training uses s["explore"]
                     "seg_turns": self.a.segment_turns}
        s["segment"] = 0
        self.save()

    def sl_pass(self, tag: str) -> None:
        """--sl (user, 2026-10-02, supervised_learning.md): right after every gate, a short supervised
        pass on perfect play (train/perfect_play.py) over the checkpoint the next segment starts from;
        that segment then starts from the result. Its KL anchor is the unmodified checkpoint, and the
        anchor/league keep the gated (unmodified) versions. Never before the first RL segment (user: it
        could collapse a policy that has learnt nothing yet). A failed pass is logged and skipped:
        training goes on from the unmodified checkpoint."""
        s, a = self.s, self.a
        if s["turns"] <= 0:
            return
        if s["cand"] is None:
            self.new_cand()
        c = s["cand"]
        if c.get("sl_tag") == tag:                 # done already (a restarted supervisor)
            return
        src = c["init"]
        # --sl-eval-only (user, 2026-10-02): the accuracy numbers, no backprop; the segment starts from src
        eval_only = getattr(a, "sl_eval_only", False)
        out = pathlib.Path(c["dir"]) / f"{'sleval' if eval_only else 'sl'}_{tag}.pt"
        if getattr(a, "temp_lo", -1.0) > 0:        # the learner's temperature range at this point of the run
            f = min(1.0, max(0.0, s["turns"] / a.temp_anneal_turns))
            lo, hi = a.temp_lo, a.temp_hi_start + (a.temp_hi_end - a.temp_hi_start) * f
        else:
            lo = hi = getattr(a, "temp", 1.0)
        args = [PY, "-u", "-m", "train.perfect_play", "--init", src, "--out", str(out),
                "--log", str(self.run / "sl.jsonl"), "--tag", tag, "--gen", str(s["gen"]),
                "--total-turns", str(s["turns"]), "--seed", str(s["seed"] * 1000 + s["segment"] * 10 + 7),
                "--steps", str(0 if eval_only else a.sl_steps), "--batch", str(a.sl_batch), "--lr", str(a.sl_lr),
                "--kl-coef", str(a.sl_kl_coef), "--max-kl", str(a.sl_max_kl),
                "--temp-lo", f"{lo:.4f}", "--temp-hi", f"{hi:.4f}"]      # its own random maps, never ours
        self.say(f"supervised pass after {tag}: perfect play on {src}")
        t0 = time.time()
        rc = self.call(args, self.run / "sl.out")
        c["sl_tag"] = tag                          # once per gate, whatever happened
        if eval_only:
            self.say(f"perfect-play evaluation done in {(time.time() - t0) / 60:.1f} min (rc {rc}, no training): "
                     f"next segment starts from {src}")
        elif rc == 0 and out.exists():
            c["init"] = str(out)
            self.say(f"supervised pass done in {(time.time() - t0) / 60:.1f} min: next segment starts from {out}")
        else:
            self.say(f"supervised pass failed (exit code {rc}, see sl.out): next segment starts from {src}")
        self.save()

    def decide(self, row: dict, ck: pathlib.Path, tag: str) -> None:
        s, a = self.s, self.a
        summ = {k: v["score"] for k, v in row["summary"].items()}
        vs_anchor = summ[ANCHOR]
        # map-restricted members (side_gates) set training weights only: never promotion or drops
        restricted = set(s.get("opp_maps", {}) or {})
        summ_gate = {k: v for k, v in summ.items() if k not in restricted}
        drops = {n: round(s["anchor_scores"][n] - summ_gate[n], 4) for n in summ_gate
                 if n != ANCHOR and n in s["anchor_scores"]}
        # a drop counts only past --drop-z standard errors of the difference (user,
        # 2026-09-30): at 66 games a cell, two equal policies differ by more than 0.10
        # against at least one of nine opponents ~70% of the time. The anchor's score
        # was measured with the same games per cell, so both sides use this gate's n.
        def limit(n_):
            k = max(int(row["summary"][n_].get("n", 0)), 1)
            p0, p1 = s["anchor_scores"][n_], summ[n_]
            return max(a.max_drop, getattr(a, "drop_z", 0.0) * math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / k))
        limits = {n: round(limit(n), 4) for n in drops}
        worst = max(drops.items(), key=lambda kv: kv[1] - limits[kv[0]]) if drops else ("-", 0.0)
        regress = bool(drops) and worst[1] > limits[worst[0]]
        # --promote-streak N (user, 2026-09-30): promote only after N gates in a row at or
        # over --promote against the same anchor, so one lucky quick check cannot move the teacher
        s["streak"] = s.get("streak", 0) + 1 if vs_anchor >= a.promote else 0
        streak_ok = s["streak"] >= max(1, getattr(a, "promote_streak", 1))
        promote_all = getattr(a, "promote_all", 0.0) or 0.0
        if promote_all > 0:
            # (user, 2026-10-01) the only rule: at least --promote-all against the last version AND every
            # earlier one; otherwise train on (no discards, no segment limit)
            worst_score = min(summ_gate.items(), key=lambda kv: kv[1])
            verdict = "promote" if worst_score[1] >= promote_all else "extend"
        elif vs_anchor >= a.promote and not regress and streak_ok:
            verdict = "promote"
        elif (vs_anchor >= a.extend_min and s["segment"] + 1 < a.max_segments
              and (not regress or getattr(a, "regress_extends", False))):
            # --regress-extends (user, 2026-09-28): a drop against a gated opponent
            # blocks promotion but does not kill the candidate while it has segments left
            verdict = "extend"
        else:
            verdict = "discard"
        g = {"time": time.time(), "gen": s["gen"], "segment": s["segment"], "seed": s["seed"],
             "tag": tag, "total_turns": s["turns"], "vs_anchor": vs_anchor,
             "n_anchor": row["summary"][ANCHOR]["n"], "worst_drop": worst[1],
             "worst_drop_vs": worst[0], "worst_drop_limit": limits.get(worst[0], a.max_drop) if drops else None,
             "verdict": verdict, "lr": s["lr"],
             "anchor_name": s["anchor_name"], "scores": summ, "anchor_scores": s["anchor_scores"],
             "gate_seconds": row["seconds"]}
        self.append("gens.jsonl", g)
        self.append("eval.jsonl", {**row, "total_turns": s["turns"], "gen": s["gen"],
                                   "segment": s["segment"], "kind": "gate", "verdict": verdict})
        if promote_all > 0:
            g["worst_score"], g["worst_score_vs"] = worst_score[1], worst_score[0]
            self.say(f"GATE {tag}: {vs_anchor:.3f} vs the last version over {g['n_anchor']} games; lowest "
                     f"{worst_score[1]:.3f} ({worst_score[0]}), needs {promote_all} against all -> {verdict.upper()}")
        else:
            self.say(f"GATE {tag}: {vs_anchor:.3f} vs anchor over {g['n_anchor']} games; worst drop "
                     f"{worst[1]:+.3f} ({worst[0]}, limit {limits.get(worst[0], a.max_drop):.3f}) -> {verdict.upper()}")
        # exploration shrinks after every gate (user, 2026-09-28): once the policy uses
        # its sprints well, random moves only make its games harder. Each branch below
        # saves the state.
        if s.get("explore", 0.0) > 0:
            new_eps = s["explore"] * a.explore_decay
            self.say(f"explore {s['explore']:.4f} -> {new_eps:.4f}")
            s["explore"] = new_eps
        if verdict == "promote":
            new = self.run / "anchors" / f"gen{s['gen']}.pt"
            shutil.copy(ck, new)
            s["league"].append([s["anchor_name"].split(" ")[0], s["anchor"]])
            # past anchors beyond the newest --keep-anchors leave the league (and
            # the gate), so a gate stays ~20 minutes
            past = [e for e in s["league"] if e[0].startswith("gen")]
            for e in past[:-a.keep_anchors]:
                s["league"].remove(e)
            # the old anchor's score against the new one is the gate's, seen from the other side.
            # Members the gate no longer plays (--gate-versions) keep their last measured
            # score: a missing one would read as 0.5 and hand them a full share of games.
            new_scores = {k: v for k, v in (s["anchor_scores"] or {}).items()
                          if k != ANCHOR and k not in summ}
            new_scores.update({k: v for k, v in summ.items() if k != ANCHOR})
            new_scores[s["anchor_name"].split(" ")[0]] = vs_anchor
            s["streak"] = 0
            old_cand = s["cand"]
            s.update(anchor=str(new), anchor_name=f"gen{s['gen']}", anchor_scores=new_scores,
                     gen=s["gen"] + 1, cand=None, discards=0, promotions=s["promotions"] + 1)
            if promote_all > 0:
                # one continuous learner (user, 2026-10-01): it trains on from the promoted weights, optimizer
                # and critic included, now with the new version as its KL teacher and in its league
                done = self.cand_turns(ck)
                s["cand"] = {"dir": str(self.run / "cands" / f"g{s['gen']:03d}_s{s['seed'] + 1}"),
                             "init": str(ck), "cont": True, "turns_before": 0,
                             "turns_before_exp": old_cand["turns_before_exp"],
                             "explore": s.get("explore", 0.0), "seg_turns": a.segment_turns,
                             "seg_base_turns": done, "seg_base_index": 0}
                s["segment"] = 0
            if s["lr"] != a.lr:
                self.say(f"promotion: lr {s['lr']} -> {a.lr}")
            s["lr"] = a.lr                        # a halved LR lasts until the next promotion
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
    r.add_argument("--segment-turns", type=int, default=100_000_000,
                   help="turns per segment for NEW candidates (~48 min at 35k turns/s)")
    r.add_argument("--max-segments", type=int, default=3)
    r.add_argument("--regress-extends", action="store_true",
                   help="a candidate over --max-drop against some opponent is extended (if it clears "
                        "--extend-min vs the anchor and has segments left) instead of discarded; "
                        "it still cannot be promoted")
    r.add_argument("--promote", type=float, default=0.55)
    r.add_argument("--promote-all", type=float, default=0.0,
                   help="(user, 2026-10-01) the only rule when > 0: promote at a score of at least this against "
                        "the last version and every earlier one in the gate; otherwise train on (never discard); "
                        "a promoted learner trains on from its own weights")
    r.add_argument("--cand-init", default="", help="the run's first candidate starts here instead of at --start "
                   "(a run from scratch: --start is the uniform-random player, this the random weights)")
    r.add_argument("--grid", type=int, default=14, choices=(14, 15), help="with --s2: the gate's simulator grid")
    r.add_argument("--portal", action="store_true",
                   help="with --s2 --grid 15: the BC_PORTALREP simulator (packets carry the portal report, grid "
                        "channels 39-42 the portal planes) for training and gates; older policies read zeros there")
    r.add_argument("--alpha", type=float, default=None, help="feed-forward runs: discount per round (critic horizon "
                   "1 / (1 - alpha)); unset = the trainer's default")
    r.add_argument("--ent-anneal-turns", type=float, default=0.0,
                   help="feed-forward runs: --ent falls linearly to 0 over this many run turns; 0 = constant")
    r.add_argument("--ent-anneal-start", type=float, default=0.0,
                   help="feed-forward runs: the run turn the --ent anneal counts from")
    r.add_argument("--kl-coef-end", type=float, default=-1.0,
                   help="feed-forward runs: --kl-coef moves linearly to this over --kl-anneal-turns run turns from "
                        "--kl-anneal-start (user, 2026-10-06); -1 = constant")
    r.add_argument("--kl-anneal-turns", type=float, default=0.0)
    r.add_argument("--kl-anneal-start", type=float, default=0.0)
    r.add_argument("--sl", action="store_true",
                   help="(user, 2026-10-02) after every gate, a short supervised pass on perfect play "
                        "(train/perfect_play.py) over the checkpoint the next segment starts from")
    r.add_argument("--sl-eval-only", action="store_true",
                   help="with --sl: only measure perfect-play accuracy after each gate, no training steps")
    r.add_argument("--sl-steps", type=int, default=32)
    r.add_argument("--sl-batch", type=int, default=32)
    r.add_argument("--sl-lr", type=float, default=1e-5)
    r.add_argument("--sl-kl-coef", type=float, default=0.02)
    r.add_argument("--sl-max-kl", type=float, default=0.02,
                   help="the pass stops once its KL to the unmodified policy on ordinary turns passes this")
    r.add_argument("--promote-streak", type=int, default=1,
                   help="gates in a row at or over --promote (against the same anchor) needed to promote")
    r.add_argument("--gate-league", default="",
                   help="comma list: the gate plays only these league members besides the anchor "
                        "(training still plays the whole league)")
    r.add_argument("--extend-min", type=float, default=0.48)
    r.add_argument("--max-drop", type=float, default=0.10,
                   help="smallest allowed limit on a fall vs a league member")
    r.add_argument("--drop-z", type=float, default=2.0,
                   help="a fall counts only past this many standard errors of the difference "
                        "(both scores at this gate's games per cell); the limit is the larger of the two")
    r.add_argument("--lr", type=float, default=3e-5)
    r.add_argument("--lr-cosine-turns", type=float, default=0.0,
                   help="feed-forward runs (user, 2026-10-02): the policy LR follows a cosine from --lr to 0 over "
                        "this many run turns, per iteration in the trainer; 0 = constant")
    r.add_argument("--lr-min", type=float, default=7.5e-6)
    r.add_argument("--lr-patience", type=int, default=2)
    r.add_argument("--kl-coef", type=float, default=0.5)
    r.add_argument("--max-kl", type=float, default=0.0,
                   help="trainer aborts above this KL to the teacher; 0 = the trainer's default (0.1)")
    r.add_argument("--self-frac", type=float, default=0.2, help="self-play share of envs, before retirements")
    r.add_argument("--ent", type=float, default=None,
                   help="LSTM runs: entropy bonus coefficient; unset = ratchet_lstm_train's default (0.001)")
    r.add_argument("--temp-lo", type=float, default=-1.0, help="feed-forward runs: learner temperature per game, "
                   "uniform in [--temp-lo, hi], hi falling --temp-hi-start -> --temp-hi-end over --temp-anneal-turns "
                   "run turns (user, 2026-10-01: 0.1, 0.5 -> 0.25 over 2e9); -1 = fixed --temp")
    r.add_argument("--temp-hi-start", type=float, default=0.5)
    r.add_argument("--temp-hi-end", type=float, default=0.25)
    r.add_argument("--temp-anneal-turns", type=float, default=2e9)
    r.add_argument("--opp-temp-max", type=float, default=-1.0, help="league opponents: greedy with probability "
                   "--opp-greedy, else uniform in [0, --opp-temp-max]; -1 = all at --temp")
    r.add_argument("--opp-greedy", type=float, default=0.1)
    r.add_argument("--temp-cond", type=int, default=0, help="1 = the policy is told its temperature")
    r.add_argument("--scal-in", type=int, default=0,
                   help="1 = ffl policies read five scalars in their first dense layer (ff_net scal_in), bolted "
                        "on as zero columns when the --init lacks them")
    r.add_argument("--ident-in", type=int, default=0,
                   help="1 = ffl policies also read birth round / 500, sin(id / 7), sin(id / 43) in their first "
                        "dense layer (ff_net ident_in), bolted on as zero columns when the --init lacks them")
    r.add_argument("--side-games", type=int, default=16,
                   help="games per map in each map-restricted member's side gate (state opp_maps)")
    r.add_argument("--ahist-in", type=int, default=0,
                   help="1 = ffl policies also read their own decayed action history, excluding the last "
                        "action (ff_net ahist_in, grid plane 57), bolted on as zero columns when the --init lacks them")
    r.add_argument("--wl-head", type=int, default=0,
                   help="1 = the true-board critic also learns a win/loss head (TD(lambda), undiscounted), "
                        "logged as wl_* (AUC / Brier at rounds 25-350 against phi); it does not drive the policy")
    r.add_argument("--wl-lam", type=float, default=0.98)
    r.add_argument("--wl-coef", type=float, default=0.5)
    r.add_argument("--wl-blend-turns", type=float, default=0.0,
                   help="feed-forward runs with --wl-head: the policy's advantage blends in the win/loss head's, "
                        "b rising 0 -> 1 over this many run turns from --wl-blend-start (ratchet_ff_train)")
    r.add_argument("--wl-blend-start", type=float, default=0.0)
    r.add_argument("--critic-view", type=int, default=0,
                   help="feed-forward runs: the value is the true-board critic (train/cview.py) with a W x W crop; "
                        "0 = the PrivValue head (ratchet_ff_train --critic-view)")
    r.add_argument("--critic-frac", type=float, default=0.5, help="with --critic-view: share of learner turns it trains on")
    r.add_argument("--critic-init", default="", help="with --critic-view: the pretrained critic for the run's first segment")
    r.add_argument("--self-frac-max", type=float, default=0.85,
                   help="each retired league member adds its share of the rest to self-play, up to this")
    r.add_argument("--memchan", action="store_true",
                   help="train and gate with the team's shared map (train/memfeat.py). "
                        "Both halves get it or neither: it moves memfar, so a candidate "
                        "trained with it and gated without would be measured on "
                        "different features from the ones it learned on")
    r.add_argument("--explore", type=float, default=0.0,
                   help="uniform-exploration share at the start of a NEW run (both trainers' "
                        "--explore); kept in state.json and multiplied by --explore-decay after "
                        "every gate. Gates play the argmax, so they never explore")
    r.add_argument("--explore-decay", type=float, default=0.75)
    r.add_argument("--critic-warmup", type=int, default=0,
                   help="LSTM runs: critic-only iterations at a new candidate, or when the critic is swapped")
    r.add_argument("--temp", type=float, default=1.0,
                   help="policy temperature for every policy in training (ratchet_lstm_train --temp); "
                        "gates always play the argmax")
    r.add_argument("--s2", action="store_true",
                   help="the next-generation simulator (sonar v2 team packet, the self-kill, 43-channel "
                        "grid) for training and gates; the start must be a checkpoint trained with it")
    r.add_argument("--critic", default="", help="LSTM runs: the pretrained v8 critic (critic_v8 "
                   "pretrained.pt); empty = ratchet_lstm_train's default")
    r.add_argument("--games", type=int, default=12, help="gate games per (opponent, map)")
    r.add_argument("--anchor-reps", type=int, default=4, help="anchor played this many times over")
    r.add_argument("--gate-seconds", type=float, default=3600)
    r.add_argument("--train-versions-only", action="store_true",
                   help="training opponents are only the anchor and this run's past genN (absolute "
                        "weighting); the other league members are played by the gate only")
    r.add_argument("--keep-anchors", type=int, default=3, help="past anchors kept in the league; "
                   "0 = keep every one (training never drops them; see --gate-versions)")
    r.add_argument("--gate-versions", type=int, default=0, help="the gate plays the assessment bots "
                   "and only this many of the newest past versions; 0 = the whole league")
    r.add_argument("--league-weighting", choices=["relative", "absolute"], default="relative",
                   help="absolute: every member keeps a share of games that shrinks to ~0 (never 0) "
                        "as the anchor's score against it goes to 1; use with --retire-at 0")
    r.add_argument("--weight-members", type=int, default=10,
                   help="absolute weighting: a member at score 0.5 gets (1 - --self-frac) / this")
    r.add_argument("--weight-floor", type=float, default=0.0025,
                   help="absolute weighting: added to (1 - score)^2, so no member's share is ever 0")
    # map supply. Training may use a wider pool than the ladder is played on (see
    # MAPS_PROPOSAL.md); gating stays on the live rotation, so a gate score always
    # means "strength on the maps we are scored on".
    r.add_argument("--train-maps", default="", help="map pool for training; empty = ratchet_train's default")
    r.add_argument("--live-maps", default="", help="with --live-share, the maps to hold at that share")
    r.add_argument("--live-share", type=float, default=0.0, help="0 = every base map weighted equally")
    r.add_argument("--gen-maps", default="",
                   help="generated maps (train/loong_mapgen.py) to mix into training; empty = none")
    r.add_argument("--gen-share", type=float, default=0.65, help="with --gen-maps, their share of sampling")
    r.add_argument("--gen-per-map", type=int, default=3)
    r.add_argument("--pearl-hotspots", action="store_true",
                   help="official maps' variants get symmetric pearl hotspots")
    r.add_argument("--gate-maps", default="", help="maps the gate plays on; empty = the gate's default")
    r.add_argument("--league-file", default="", help="NEW runs: the league, one name=path per line")
    r.add_argument("--retire-at", type=float, default=0.8,
                   help="a league member the anchor scores at least this against gets no training "
                        "games (it stays in the gate); 0 = never")
    r.add_argument("--train-exclude", default="", help="comma list of league members that get no "
                   "training games, as if retired (the gate plays only --gate-versions anyway)")
    r.add_argument("--start-name", default="", help="LSTM runs: the start policy's name in the v8 "
                   "critic's league (critic_v8 ids.json), which seeds the learner's critic slot")
    g = sub.add_parser("gate")
    g.add_argument("--cand", required=True)
    g.add_argument("--anchor", default="")
    g.add_argument("--league", default="")
    g.add_argument("--out", required=True)
    g.add_argument("--maps", default=str(ROOT / "runs/ft3/maps"))
    g.add_argument("--exclude-maps", default="", help="comma list of map names the gate skips")
    g.add_argument("--games", type=int, default=12)
    g.add_argument("--anchor-reps", type=int, default=4)
    g.add_argument("--threads", type=int, default=16)
    g.add_argument("--skip-done", action="store_true", help=argparse.SUPPRESS)   # tests: see evaluate
    # ON by default since 2026-10-01 (user: faster checks). The 09-28 slowdown was Pool.release_env
    # scanning every live key: finished games, played with action 0, end every few steps and each
    # end called forget() on every player. Fixed (pools index keys by env; finished envs skip
    # forget) and measured solo on ratchet_ff3's real check: 355 s -> 230 s. In fp32
    # (BC_EVAL_FP32=1) fast played exactly the old loop's games; in bf16 the batches differ, so
    # the games are the same in distribution, not bit-identical.
    g.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True,
                   help="yardstick.evaluate's fast path: games already counted are no longer "
                        "computed, and the anchor repeats are one player (--no-fast: the old loop)")
    g.add_argument("--seed", type=int, default=12345)
    g.add_argument("--max-seconds", type=float, default=3600)
    g.add_argument("--memchan", action="store_true")
    g.add_argument("--s2", action="store_true")
    g.add_argument("--grid", type=int, default=14, choices=(14, 15))
    g.add_argument("--portal", action="store_true")
    a = p.parse_args()
    if a.cmd == "gate":
        gate(a)
    else:
        Supervisor(a).loop()


if __name__ == "__main__":
    main()
