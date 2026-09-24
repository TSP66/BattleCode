"""Fixed yardsticks: win rates against scripted bots and past versions.

Nothing here is trained on. Self-play reward says little about whether a
policy is getting better, since its opponent improves at the same rate. So a
snapshot is played against opponents that stay put:

  * the scripted bots of cpp/bc_bots.hpp, each built around one idea (pearls,
    splitting, blocking, portals, running away, random), so a weak row points
    at a kind of play the policy cannot handle;
  * earlier snapshots of the same run, which is the check on forgetting: a win
    rate against an older self that falls back toward 50% means lost ground.

Every (opponent, map) cell plays both sides equally, on the unaugmented maps.
The learner picks greedily, as the deployed bot does.

Run it next to training; it waits for snapshots and evaluates each new one:

    python -m train.yardstick --run runs/v1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                                    # noqa: E402
from train.net import ActorCritic, PyramidActorCritic, masked_logits   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root

COLS = bcsim.EpisodeStats.COLUMNS
C_WIN, C_ROUNDS, C_MAP = COLS.index("winner"), COLS.index("rounds"), COLS.index("map")
PORTAL = bcsim.REWARD_COMPS.index("portal")


def load_net(path: str | pathlib.Path, dev: torch.device) -> tuple[ActorCritic, dict]:
    """`args["arch"]` picks the architecture; absent means the flat net, which
    is every checkpoint written before the pyramid existed.

    The checkpoint is read onto the host and the net moved to the device once it
    is full, rather than reading the weights straight onto the device and
    copying them into place there. That ordering matters: a net whose parameters
    were filled by a device-to-device `load_state_dict` makes a later
    `torch.cuda.graph` capture hand back memory that faults on replay. Measured
    on 2026-09-24 -- nine nets built by the constructor and captured replay
    fine, the same nine loaded from checkpoints raise `an illegal memory access
    was encountered` on the first replay, and loading via the host is what fixes
    it. Dropping the checkpoint dict before capture does not, so it is the
    device-side copy and not a live reference to the loaded tensors.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    hidden = next(v for k, v in ck["net"].items() if k.endswith("fuse.0.weight")).shape[0]
    # The scalar width comes from the checkpoint, never from the env. The env's
    # row grows as features are appended (708 -> 713 with the sonar echoes) and
    # every older net has to go on reading exactly the columns it was trained
    # on, or the frozen league stops being frozen.
    n_scalars = next(v for k, v in ck["net"].items()
                     if k.endswith("scalar.0.weight")).shape[1]
    if a.get("arch") == "convlstm":
        from train.recurrent import RecurrentActorCritic
        net = RecurrentActorCritic(bcsim.N_CHANNELS, bcsim.WIDE_CH, bcsim.N_ACTIONS,
                                   near_width=a["near_width"], near_blocks=a["near_blocks"],
                                   hid_ch=a["hid_ch"], side=a.get("side", bcsim.WIDE_SIDE),
                                   hidden=hidden, n_scalars=n_scalars)
    elif a.get("arch") == "pyramid":
        net = PyramidActorCritic(bcsim.N_CHANNELS, bcsim.WIDE_CH, bcsim.N_ACTIONS,
                                 near_width=a["near_width"], near_blocks=a["near_blocks"],
                                 wide_width=a["wide_width"], wide_blocks=a["wide_blocks"],
                                 wide_side=bcsim.WIDE_SIDE, hidden=hidden,
                                 n_scalars=n_scalars)
    else:
        net = ActorCritic(bcsim.N_CHANNELS, n_scalars, bcsim.N_ACTIONS,
                          width=a["width"], blocks=a["blocks"], hidden=hidden)
    net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["net"].items()})
    net.to(dev).eval()
    return net, ck


def greedy(net, dev, max_batch: int = 0):
    """Argmax policy as a callable on numpy rows.

    `max_batch` is accepted and ignored. It used to capture the forward pass as
    a CUDA graph and replay it with the rows padded out, because an evaluation
    step is one dragon turn and launch latency dominates. That path silently
    produced wrong actions and has been removed.

    What it did, measured on 2026-09-24: gen1 against gen4 over 72 games on
    maps-live scores **0.5972** eagerly and **0.0000** with graphs, and the
    graph learner acts on 46k rows against its opponent's 513k because its
    swarm never grows -- it is playing badly from the first turn, not losing
    late. A mirror match (one net, the same callable on both sides, so only one
    graph is ever captured) scores exactly 0.5000, and the corruption appears
    once a *second* graph is captured: capture calls empty_cache(), and the
    graph captured earlier replays against memory it no longer owns. Nothing
    cheap fixed it -- not a shared memory pool, not thread_local capture, not a
    static output tensor, not warming up on the capture stream, not disabling
    the autocast weight cache.

    The cost of losing it is small and the cost of keeping it was every number
    this harness produced: 72 games on nine maps takes 154s eagerly.
    """
    wants = getattr(net, "wants_wide", False)

    def act(local, scalar, mask, wide=None):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = net(torch.from_numpy(local).to(dev), torch.from_numpy(scalar).to(dev),
                            torch.from_numpy(wide).to(dev) if wants else None)
            m = torch.from_numpy(mask).to(dev).bool()
            return masked_logits(logits.float(), m).argmax(dim=1).to(torch.int32).cpu().numpy()
    act.wants_wide = wants
    return act


def _call(fn, obs, rows, wide=None):
    """An act callable on the chosen rows. A stateful one (a policy with
    memory, see clone_eval.py) gets the whole observation and the row mask,
    since it has to know which dragon of which game each row is."""
    if getattr(fn, "stateful", False):
        # A recurrent policy needs the planes as well as the observation; an
        # older stateful one (clone_eval) does not, so it is asked.
        if getattr(fn, "wants_wide", False):
            return fn.rows(obs, rows, wide)
        return fn.rows(obs, rows)
    if getattr(fn, "wants_wide", False):
        return fn(obs.local[rows], obs.scalar[rows], obs.mask[rows], wide[rows])
    return fn(obs.local[rows], obs.scalar[rows], obs.mask[rows])


def evaluate(learner, opponents: list[dict], maps: list[str], map_names: list[str],
             games: int = 16, threads: int = 8, seed: int = 12345,
             max_seconds: float = 900.0, progress: float = 0.0,
             sonar: bool = False) -> dict:
    """Plays `games` per (opponent, map) cell, half on each side.

    opponents: {"name", "bot": index} for a scripted bot, or {"name", "act":
    callable} for a network. Returns {opponent: {map: stats}} plus totals.
    """
    # One env per game, all at once. A step is a single dragon turn, so a game
    # between two swarms of ~30 dragons is ~15k steps; played back to back
    # they would add up, played side by side only the longest one counts.
    per_side = 1
    layout = [(o, mi, side) for o in range(len(opponents))
              for mi in range(len(maps)) for side in (0, 1)
              for _ in range(max(1, games // 2))]
    n = len(layout)
    any_wide = any(getattr(f, "wants_wide", False)
                   for f in [learner] + [o.get("act") for o in opponents] if f is not None)
    # sonar has to match what the learner was trained with, or its five echo
    # scalars arrive as zeros it has never seen. It is symmetric -- every dragon
    # in the env broadcasts -- so the frozen opponents also see a non-zero
    # num_msgs, which is why a league measured with sonar on is not directly
    # comparable with one measured without it.
    print(f"  sonar {'on' if sonar else 'off'}", flush=True)
    env = bcsim.BattlecodeVecEnv(maps, num_envs=n, num_threads=threads, seed=seed,
                                 closure_capacity=max(8192, n * 160), wide=any_wide,
                                 sonar=sonar)
    learner_team = np.array([side for _, _, side in layout], np.int8)
    opp_of = np.array([o for o, _, _ in layout])
    for i, (o, mi, side) in enumerate(layout):
        bot = opponents[o].get("bot")
        env.set_opponent(i, team=(1 - side) if bot is not None else -1,
                         bot=bot if bot is not None else 0, map_index=mi)
    net_opps = [o for o, spec in enumerate(opponents) if spec.get("act") is not None]
    is_bot = np.array([opponents[o].get("bot") is not None for o in opp_of])

    stateful = [f for f in [learner] + [o.get("act") for o in opponents]
                if getattr(f, "stateful", False)]
    done = np.zeros(n, np.int64)
    results: list[list] = [[] for _ in range(n)]
    turns = np.zeros(n, np.int64)          # learner turns, for portal usage
    portal = np.zeros(n, np.float64)
    obs = env.reset()
    t0 = time.perf_counter()
    last_report = t0
    while (done < per_side).any():
        if progress and time.perf_counter() - last_report > progress:
            last_report = time.perf_counter()
            print(f"  {int((done >= per_side).sum())}/{n} games done, "
                  f"{last_report - t0:.0f}s", flush=True)
        if time.perf_counter() - t0 > max_seconds:
            print(f"  eval hit its {max_seconds:.0f}s limit, "
                  f"{int((done < per_side).sum())} envs short", flush=True)
            break
        mine = obs.team == learner_team
        acts = np.zeros(n, np.int32)
        if mine.any():
            acts[mine] = _call(learner, obs, mine, env.wide)
        for o in net_opps:
            rows = (~mine) & (opp_of == o)
            if rows.any():
                acts[rows] = _call(opponents[o]["act"], obs, rows, env.wide)
        # portal usage is only counted against bots: there every closure is the
        # learner's, whereas against a network both sides' turns close
        turns += mine & (done < per_side) & is_bot
        obs, closures, eps = env.step(acts)
        if len(closures.env):
            keep = is_bot[closures.env] & (done[closures.env] < per_side)
            np.add.at(portal, closures.env[keep], closures.comps[keep, PORTAL])
        for row in eps.rows:
            e = int(row[0])
            for fn in stateful:
                fn.forget(e)            # the env has already started its next game
            if done[e] >= per_side:
                continue
            done[e] += 1
            results[e].append(row.copy())

    out: dict = {}
    for i, (o, mi, side) in enumerate(layout):
        name, mname = opponents[o]["name"], map_names[mi]
        cell = out.setdefault(name, {}).setdefault(mname, {
            "n": 0, "win": 0, "draw": 0, "loss": 0, "rounds": 0.0, "kills": 0.0,
            "deaths": 0.0, "longest": 0.0, "portal_turns": 0.0, "turns": 0})
        me, them = ("a", "b") if side == 0 else ("b", "a")
        for r in results[i]:
            d = dict(zip(COLS, r.tolist()))
            cell["n"] += 1
            cell["win"] += int(d["winner"] == side)
            cell["draw"] += int(d["winner"] < 0)
            cell["loss"] += int(d["winner"] == 1 - side)
            cell["rounds"] += d["rounds"]
            cell["kills"] += d[f"{me}_kills"]
            cell["deaths"] += d[f"{me}_deaths"]
            cell["longest"] += d[f"{me}_longest"]
        cell["portal_turns"] += float(portal[i])
        cell["turns"] += int(turns[i])
    # per-cell means and a score where a draw is half a win
    summary = {}
    for name, by_map in out.items():
        tot = {"n": 0, "win": 0, "draw": 0, "loss": 0, "kills": 0.0, "portal_turns": 0.0,
               "turns": 0}
        for mname, c in by_map.items():
            for k in tot:
                tot[k] += c[k]
            g = max(c["n"], 1)
            for k in ("rounds", "kills", "deaths", "longest"):
                c[k] = round(c[k] / g, 3)
            c["score"] = round((c["win"] + 0.5 * c["draw"]) / g, 4)
            c["portal"] = round(c.pop("portal_turns") / max(c.pop("turns"), 1), 5)
        g = max(tot["n"], 1)
        summary[name] = {"n": tot["n"], "score": round((tot["win"] + 0.5 * tot["draw"]) / g, 4),
                         "win": round(tot["win"] / g, 4), "draw": round(tot["draw"] / g, 4),
                         "kills": round(tot["kills"] / g, 3),
                         "portal": round(tot["portal_turns"] / max(tot["turns"], 1), 5)}
    return {"cells": out, "summary": summary,
            "seconds": round(time.perf_counter() - t0, 1)}


# ------------------------------------------------------------------ worker
SNAP_RE = re.compile(r"turns_(\d+)\.pt$")


def snapshots(run: pathlib.Path) -> list[tuple[int, pathlib.Path]]:
    d = run / "snapshots"
    if not d.exists():
        return []
    out = []
    for p in d.glob("turns_*.pt"):
        m = SNAP_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def last_submitted(manifest: pathlib.Path) -> tuple[str, pathlib.Path] | None:
    """The most recent entry of runs/submitted/submitted.jsonl, which
    wasmprobe/submit.sh appends to on every upload."""
    if not manifest.exists():
        return None
    last = None
    for line in manifest.read_text().splitlines():
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            pass
    if not last:
        return None
    # entries may be absolute paths from another machine; the copy sits beside the manifest
    ckpt = pathlib.Path(last["ckpt"])
    if not ckpt.exists():
        ckpt = manifest.parent / ckpt.name
    if not ckpt.exists():
        return None
    return f"submitted {last['version']}", ckpt


def past_selves(snaps: list[tuple[int, pathlib.Path]], now: int, lags: list[int],
                anchors: list[str], manifest: pathlib.Path | None = None
                ) -> list[tuple[str, pathlib.Path]]:
    """Older versions to play: the last submitted bot first (the bar a new
    upload has to clear), then the snapshot nearest each lag behind `now`, the
    run's first snapshot, and any fixed anchor files."""
    picked: dict[str, pathlib.Path] = {}
    sub = last_submitted(manifest) if manifest else None
    if sub:
        picked[sub[0]] = sub[1]
    older = [(t, p) for t, p in snaps if t < now]
    for lag in lags:
        if not older:
            break
        want = now - lag
        if want <= 0:
            continue                # the run is not that old yet
        t, p = min(older, key=lambda tp: abs(tp[0] - want))
        if abs(t - want) <= lag // 2:
            picked[f"self -{lag / 1e6:.0f}M"] = p
    if older:
        picked["run start"] = older[0][1]
    for a in anchors:
        picked["anchor " + pathlib.Path(a).stem] = pathlib.Path(a)
    return list(picked.items())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "runs/v1"))
    p.add_argument("--maps", default=str(ROOT / "maps"))
    # deliberately light: a noisy number every so often is all it is for
    p.add_argument("--games", type=int, default=4, help="per (opponent, map) cell")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--sonar", choices=["auto", "on", "off"], default="auto",
                   help="auto takes it from each checkpoint's own --sonar")
    p.add_argument("--every-minutes", type=float, default=15.0,
                   help="start an evaluation of the newest snapshot this often")
    p.add_argument("--lags", default="50e6,200e6",
                   help="play the snapshot this many turns older, comma separated")
    p.add_argument("--anchors", default=str(ROOT / "runs/anchors"),
                   help="fixed checkpoints to always play: files or directories of .pt, "
                        "comma separated")
    p.add_argument("--submitted", default=str(ROOT / "runs/submitted/submitted.jsonl"),
                   help="manifest of uploaded versions; the newest is always an opponent")
    p.add_argument("--ckpt", default="", help="evaluate this one file and exit")
    p.add_argument("--dump", default="", help="with --ckpt: append the full result here")
    p.add_argument("--poll", type=float, default=30.0)
    p.add_argument("--max-seconds", type=float, default=900.0,
                   help="per evaluation; unfinished games are dropped, so raise it next "
                        "to a training run")
    p.add_argument("--gpu-frac", type=float, default=0.0,
                   help="cap this process's share of GPU memory (next to a training run)")
    a = p.parse_args()
    if a.gpu_frac:
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac)

    dev = torch.device("cuda")
    run = pathlib.Path(a.run)
    maps = bcsim.load_maps(a.maps)
    map_names = [f.stem for f in sorted(pathlib.Path(a.maps).glob("*.map"))]
    lags = [int(float(x)) for x in a.lags.split(",") if x]
    anchors = []
    for x in filter(None, a.anchors.split(",")):
        path = pathlib.Path(x)
        anchors += [str(f) for f in sorted(path.glob("*.pt"))] if path.is_dir() else \
            ([x] if path.exists() else [])
    out_path = run / "eval.jsonl"

    def one(path: pathlib.Path, turns: int) -> dict:
        net, ck = load_net(path, dev)
        opps = [{"name": f"bot:{b}", "bot": i} for i, b in enumerate(bcsim.BOTS)]
        nets = []
        for name, pp in past_selves(snapshots(run), turns, lags, anchors,
                                    pathlib.Path(a.submitted)):
            try:
                nets.append((name, pp, load_net(pp, dev)[0]))
            except Exception as ex:                 # an incompatible old layout
                print(f"  skipping {name}: {ex}", flush=True)
        # every graph is captured at the full env count, the most rows a step has
        n_envs = (len(opps) + len(nets)) * len(maps) * 2 * max(1, a.games // 2)
        for name, pp, onet in nets:
            opps.append({"name": name, "act": greedy(onet, dev), "path": str(pp)})
        res = evaluate(greedy(net, dev), opps, maps, map_names, games=a.games,
                       threads=a.threads, max_seconds=a.max_seconds,
                       sonar=ck.get("args", {}).get("sonar", False)
                       if a.sonar == "auto" else a.sonar == "on")
        row = {"total_turns": turns, "iter": ck.get("iter"), "ckpt": str(path),
               "time": time.time(), "opponents": {o["name"]: o.get("path", "") for o in opps},
               **res}
        bots = [v["score"] for k, v in res["summary"].items() if k.startswith("bot:")]
        row["bot_score"] = round(float(np.mean(bots)), 4) if bots else None
        sub = [k for k in res["summary"] if k.startswith("submitted ")]
        if sub:
            row["submitted_name"] = sub[0]
            row["submitted_score"] = res["summary"][sub[0]]["score"]
        return row

    if a.ckpt:
        row = one(pathlib.Path(a.ckpt), int(torch.load(a.ckpt, map_location="cpu",
                                                         weights_only=False).get("total_turns", 0)))
        print(json.dumps(row["summary"], indent=1))
        if a.dump:
            with open(a.dump, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        return

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["total_turns"])
            except (json.JSONDecodeError, KeyError):
                pass
    last_start = 0.0
    print(f"yardstick watching {run / 'snapshots'}; bots {bcsim.BOTS}", flush=True)
    while True:
        wait = last_start + a.every_minutes * 60 - time.time()
        todo = [(t, pp) for t, pp in snapshots(run) if t not in done]
        if not todo or wait > 0:
            time.sleep(max(min(a.poll, wait), 1.0))
            continue
        last_start = time.time()
        # behind? skip to the newest, the dashboard only needs the trend
        t, path = todo[-1]
        for tt, _ in todo[:-1]:
            done.add(tt)
        print(f"evaluating {path.name}", flush=True)
        try:
            row = one(path, t)
        except Exception as ex:
            print(f"  failed: {ex!r}", flush=True)
            done.add(t)
            continue
        with out_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        done.add(t)
        s = "  ".join(f"{k} {v['score']:.2f}" for k, v in row["summary"].items())
        print(f"  {t / 1e6:.0f}M turns in {row['seconds']}s | {s}", flush=True)


if __name__ == "__main__":
    main()
