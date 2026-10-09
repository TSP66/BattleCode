"""Critic pretraining data (user, 2026-10-01): one format for every source of games.

Sources: round-robin / generated games between our policies and clones (train/roundrobin_all.py
--record), at per-game temperatures, and re-simulated server replays (observed = 1, both teams
greedy, T = 0). A clone is recorded under its own name; train/critic_pretrain.py maps names to
team identities (a clone shares its original's).

A shard directory holds
    samples.npz  a random share of turns (both teams): cview (n, stride) uint8, priv (n, PRIV_COUNT),
                 game, team, round, step
    turns.npz    EVERY turn of every recorded game: game, team, round, step, phi (the acting team's
                 v8 potential, Phi's terms summed) -- the returns are built from these
    games.jsonl  one row per game: game, side names / temperatures, observed, map, source, winner,
                 rounds, finished
Game ids are unique within a shard; critic_pretrain.py prefixes them with the shard.

Returns (`returns`): ratchet_ff_train's reward exactly -- the change in the team's Phi between its
consecutive turns, discounted alpha per round, the game's final Phi (this team's sign) as the last
step -- for several alphas at once (user: horizons 20 / 10 / 5 / 3 rounds).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

import bcsim


class Recorder:
    def __init__(self, out: str | pathlib.Path, keep: float, seed: int = 0, cview_w: int = 27):
        self.out = pathlib.Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        if any(self.out.glob("*.npz")):
            raise SystemExit(f"{self.out} already holds a shard: give a new directory")
        self.keep, self.cview_w = float(keep), int(cview_w)
        self.rng = np.random.default_rng(seed)
        self.turns = {k: [] for k in ("game", "team", "round", "step", "phi")}
        self.samples = {k: [] for k in ("cview", "priv", "game", "team", "round", "step")}
        self.meta: dict[int, dict] = {}
        self.gfile = open(self.out / "games.jsonl", "w")
        self.n_kept = 0

    def start(self, game: int, **meta) -> None:
        """A game begins: sides (names), temps, observed, map, source -- whatever describes it."""
        self.meta[int(game)] = {"game": int(game), **meta}

    def turn(self, game: np.ndarray, obs, cview: np.ndarray, rows: np.ndarray, step: int) -> None:
        """Before the step: the acting dragon of every env in `rows` (a bool mask or indices)."""
        idx = np.flatnonzero(rows) if rows.dtype == bool else np.asarray(rows)
        if not len(idx):
            return
        phi = obs.priv[idx, bcsim.PRIV_BASE:].sum(1).astype(np.float32)
        g = game[idx].astype(np.int64)
        tm = obs.team[idx].astype(np.int8)
        rd = obs.round[idx].astype(np.int16)
        self.turns["game"].append(g)
        self.turns["team"].append(tm)
        self.turns["round"].append(rd)
        self.turns["step"].append(np.full(len(idx), step, np.int32))
        self.turns["phi"].append(phi)
        k = self.rng.random(len(idx)) < self.keep
        if k.any():
            ki = idx[k]
            self.samples["cview"].append(cview[ki].copy())
            self.samples["priv"].append(obs.priv[ki].astype(np.float32))
            self.samples["game"].append(g[k])
            self.samples["team"].append(tm[k])
            self.samples["round"].append(rd[k])
            self.samples["step"].append(np.full(int(k.sum()), step, np.int32))
            self.n_kept += int(k.sum())

    def end(self, game: int, winner: int, rounds: int, finished: bool = True) -> None:
        m = self.meta.pop(int(game))
        m.update(winner=int(winner), rounds=int(rounds), finished=bool(finished))
        self.gfile.write(json.dumps(m) + "\n")

    def close(self) -> None:
        for g in list(self.meta):                          # games cut off (time limit): kept, marked
            self.end(g, -2, -1, finished=False)
        self.gfile.close()
        cat = lambda d: {k: np.concatenate(v) if v else np.zeros(0) for k, v in d.items()}   # noqa: E731
        np.savez(self.out / "turns.npz", **cat(self.turns))
        s = cat(self.samples)
        if not len(s["game"]):
            s["cview"] = np.zeros((0, bcsim.cview_layout(self.cview_w)["stride"]), np.uint8)
        np.savez(self.out / "samples.npz", cview_w=np.array(self.cview_w), **s)
        print(f"recorded {self.n_kept:,} samples, {sum(len(x) for x in self.turns['game']):,} turns -> {self.out}",
              flush=True)


def returns(turns: dict, games: dict, alphas: list[float]) -> dict:
    """Per turn (aligned with `turns`), the discounted Phi return for each alpha; NaN for a turn of a
    game that did not finish. `games` maps game -> its games.jsonl row (finished flag)."""
    g_, tm_, rd_, st_, ph_ = (np.asarray(turns[k]) for k in ("game", "team", "round", "step", "phi"))
    out = {a: np.full(len(g_), np.nan) for a in alphas}
    o = np.lexsort((st_, g_))                              # each game's turns in order
    gs = g_[o]
    cuts = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1, len(o)]
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        ix = o[c0:c1]
        gm = games.get(int(g_[ix[0]]))
        if gm is None or not gm.get("finished", False):
            continue
        fin = ix[-1]                                       # the game's last recorded turn
        for team in (0, 1):
            jt = ix[tm_[ix] == team]
            if not len(jt):
                continue
            ph, rd = ph_[jt].astype(np.float64), rd_[jt].astype(np.float64)
            q = ph_[fin] if tm_[fin] == team else -ph_[fin]        # the final Phi, this team's sign
            r = np.empty(len(jt))
            r[:-1] = ph[1:] - ph[:-1]
            r[-1] = 0.0 if jt[-1] == fin else q - ph[-1]
            for a in alphas:
                w = a ** (rd - rd[0])
                out[a][jt] = np.cumsum((w * r)[::-1])[::-1] / w
    return out
