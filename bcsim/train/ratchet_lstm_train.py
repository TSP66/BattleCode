"""One ratchet training segment for the LSTM policy (train/lstm_net.py).

Reward (the user's call, 2026-09-25): the TEAM's potential change, v8's Phi
(REWARDS.md, kappa 1) read from the engine's privileged row -- Phi at the team's
next turn minus Phi now, discounted --alpha per ROUND (0.95: ~20 rounds; 0.96 on 28-29 Sep). No
result term, and nothing about whether the acting dragon survives. A team's last
turn of a game takes the game's final observed Phi (v8's terms are antisymmetric,
so Phi for one team is minus Phi for the other). critic_v8.phi_returns is the
same target, offline.

Critic: train/critic_net.BoardCritic regressing that return directly, with Phi
among its inputs (not as an offset: the -Phi anchor measured ill-conditioned).
It starts from critic_v8 pretrain (replays + league games) and keeps training
here, separate from the policy, conditioned on both teams' slots (the learner's
seeded from the league agent it started as; anchors share it) and on the PPO
iteration.

Advantages: GAE-lambda along each team's own turns, bootstrapped by the critic; lambda,
like alpha, is per ROUND (2026-09-29), so the credit horizon does not depend on how many
dragons a team has.
Rows whose team has no later turn inside the rollout (and whose game did not
end) are masked out of the loss but stay in the BPTT chunks for the state.

The rest is ratchet_train.py's: KL to the anchor (stored at rollout time; the
teacher is frozen and carries its own per-dragon state), PPO clip, the same
aborts (exit 3) and the final.pt / latest.pt contract, so ratchet.py drives it.
Sonar is ours (four-way broadcast, protocol 3).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# --s2: the next-generation simulator (sonar v2 team packet, the self-kill as action 48,
# 43-channel grid), which has to be chosen before bcsim loads its library
os.environ.setdefault("BCSIM_LIB", str(pathlib.Path(__file__).resolve().parents[1] / "bcsim" /
                                       ("libbcvec_priv_s2.so" if "--s2" in sys.argv else "libbcvec_priv.so")))

import bcsim                                    # noqa: E402
from train import augment                       # noqa: E402
from train import d4 as D4                      # noqa: E402
from train.critic_net import N_BINARY_PLANES, BoardCritic, TeamSlots, board_pack, board_unpack  # noqa: E402
from train.critic_v8 import GAMMA, IDS, KAPPA, LEARNER_ID  # noqa: E402
from train.distill_lstm import Pool, chunks_of  # noqa: E402
from train.lstm_net import LSTMPolicy          # noqa: E402
from train.net import masked_logits             # noqa: E402
from train.yardstick import load_net            # noqa: E402
from train.sonar2 import intents_from_probs     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
ABORT = 3
ARCH_KEYS = ("c1", "b1", "c2", "b2", "squeeze", "embed", "hidden", "layers")


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="LSTM policy to start from")
    p.add_argument("--teacher", required=True, help="KL teacher (an LSTM checkpoint): the anchor")
    p.add_argument("--continue", dest="cont", action="store_true")
    p.add_argument("--critic", default=str(ROOT / "runs/critic_v8/pretrained.pt"),
                   help="v8 critic: critic_v8 pretrained.pt, or a previous segment's final.pt")
    p.add_argument("--start-name", default="", help="the league name the start policy was "
                   "pretrained under (critic_v8 ids.json); its slot seeds the learner's")
    p.add_argument("--opponents", default="")
    p.add_argument("--opp-names", default="")
    p.add_argument("--opp-weights", default="")
    p.add_argument("--self-frac", type=float, default=0.2)
    p.add_argument("--out", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--turns", type=float, default=50e6)
    p.add_argument("--turn-base", type=int, default=0)
    p.add_argument("--gen", type=int, default=0)
    p.add_argument("--segment", type=int, default=0)
    p.add_argument("--maps", default=str(ROOT / "maps-all"))
    p.add_argument("--live-maps", default="")
    p.add_argument("--live-share", type=float, default=0.0)
    p.add_argument("--gen-maps", default="",
                   help="directory of generated maps (train/loong_mapgen.py); empty = none")
    p.add_argument("--gen-share", type=float, default=0.65,
                   help="with --gen-maps, the generated maps' share of sampling")
    p.add_argument("--gen-per-map", type=int, default=3, help="augmented variants per generated map")
    p.add_argument("--pearl-hotspots", action="store_true",
                   help="official maps' variants get symmetric pearl hotspots (augment.OFFICIAL_AUG)")
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--batch-chunks", type=int, default=512)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--lr-cosine-turns", type=float, default=0.0,
                   help="ratchet_ff_train (user, 2026-10-02): the policy's LR follows a cosine from --lr at run turn 0 "
                        "to 0 at this many run turns (--turn-base included, so it spans segments); 0 = constant")
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--critic-sample", type=int, default=8, help="1 in N learner turns keeps its "
                   "board for the critic's own update")
    p.add_argument("--critic-warmup", type=int, default=0, help="iterations at the start of a "
                   "NEW segment that train only the critic")
    p.add_argument("--alpha", type=float, default=0.95,
                   help="discount per ROUND of the team's potential changes (0.95; 0.96 on 28-29 Sep)")
    p.add_argument("--lam", type=float, default=0.95,
                   help="GAE lambda per ROUND, like alpha (2026-09-29; before, it was applied per team "
                        "MOVE, so with 30 dragons it decayed 0.95^30 = 0.21 a round)")
    p.add_argument("--vf-coef", type=float, default=0.5, help="ratchet_ff_train (--critic own): weight of "
                   "the value head's loss in the shared update")
    p.add_argument("--vf-trunk", action="store_true", help="ratchet_ff_train: let the value loss train "
                   "the policy's trunk too (off: the head reads detached features)")
    p.add_argument("--ppo-batch", type=int, default=16384, help="ratchet_ff_train: rows per PPO optimizer step (the LSTM trainer's 512 chunks x 32)")
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--kl-coef", type=float, default=0.4)
    p.add_argument("--ent", type=float, default=0.001)
    p.add_argument("--ent-anneal-turns", type=float, default=0.0,
                   help="ratchet_ff_train: --ent falls linearly to 0 over this many RUN turns (--turn-base on), "
                        "so the schedule runs on across segments and generations; 0 = constant")
    p.add_argument("--ent-anneal-start", type=float, default=0.0,
                   help="ratchet_ff_train: the run turn the --ent anneal counts from (user, 2026-10-06: from a gate)")
    p.add_argument("--kl-coef-end", type=float, default=-1.0,
                   help="ratchet_ff_train: --kl-coef moves linearly to this over --kl-anneal-turns run turns from "
                        "--kl-anneal-start, then stays; -1 = constant --kl-coef")
    p.add_argument("--kl-anneal-turns", type=float, default=0.0)
    p.add_argument("--kl-anneal-start", type=float, default=0.0)
    p.add_argument("--temp-lo", type=float, default=-1.0,
                   help="ratchet_ff_train: learner teams draw their temperature per game, uniform in [--temp-lo, hi], hi "
                        "falling linearly from --temp-hi-start to --temp-hi-end over the run's first --temp-anneal-turns; "
                        "-1 = everyone at --temp (as before)")
    p.add_argument("--temp-hi-start", type=float, default=0.5)
    p.add_argument("--temp-hi-end", type=float, default=0.25)
    p.add_argument("--temp-anneal-turns", type=float, default=2e9, help="run turns (with --turn-base) over which hi falls")
    p.add_argument("--opp-temp-max", type=float, default=-1.0,
                   help="ratchet_ff_train: league opponents draw a temperature per game, greedy with probability "
                        "--opp-greedy, else uniform in [0, --opp-temp-max]; -1 = all at --temp")
    p.add_argument("--opp-greedy", type=float, default=0.1)
    p.add_argument("--temp-cond", type=int, default=0,
                   help="ratchet_ff_train: 1 = the policy is told its temperature (ff_net FFLPolicy temp_in; a policy "
                        "without the input gets it as a zero column)")
    p.add_argument("--scal-in", type=int, default=0,
                   help="ratchet_ff_train: 1 = five scalars into the first dense layer (ff_net FFLPolicy scal_in: "
                        "round/500 and its square, length, (alive/unit_limit)^2, is_queen; a policy without them "
                        "gets them as zero columns)")
    p.add_argument("--ident-in", type=int, default=0,
                   help="ratchet_ff_train: 1 = three identity scalars into the first dense layer (ff_net FFLPolicy "
                        "ident_in: birth round / 500, sin(id / 7), sin(id / 43), off grid planes 54-56; a policy "
                        "without them gets them as zero columns)")
    p.add_argument("--opp-maps", default="",
                   help="name=map+map;... -- those opponents play only on those official maps (ratchet_ff_train)")
    p.add_argument("--ahist-in", type=int, default=0,
                   help="1 = ffl policies also read their own action history excluding the last action "
                        "(ff_net ahist_in: a(t-2) 0.5, a(t-3) 0.25, ... per action id, off grid plane 57 of a "
                        "BC_AHIST build), bolted on as zero columns when the --init lacks them")
    p.add_argument("--wl-head", type=int, default=0,
                   help="ratchet_ff_train with --critic-view: 1 = the critic also learns a win/loss head (cview "
                        "CViewCritic wl: undiscounted result, +1 / -1, draw 0) by TD(lambda), added fresh when the "
                        "critic lacks it; it does not drive the policy (user, 2026-10-03: for a later sparse reward)")
    p.add_argument("--wl-lam", type=float, default=0.98, help="the win/loss head's TD lambda, per round")
    p.add_argument("--wl-coef", type=float, default=0.5, help="the win/loss head's loss weight")
    p.add_argument("--wl-blend-turns", type=float, default=0.0,
                   help="ratchet_ff_train --wl-head: the policy's advantage becomes (1 - b) A_phi + b A_wl, each "
                        "standardised first and the blend restandardised, b rising linearly 0 -> 1 over this many "
                        "run turns from --wl-blend-start (user, 2026-10-03: 1.2B); 0 = phi only, A_wl logged")
    p.add_argument("--wl-blend-start", type=float, default=0.0, help="the run turn at which the blend is 0")
    p.add_argument("--critic-view", type=int, default=0,
                   help="ratchet_ff_train: the value is train/cview.CViewCritic on the simulator's true-board "
                        "view (a W x W crop about the acting head, wrapped, plus the pooled board), W odd; 0 = "
                        "the PrivValue head on the policy's features, ONLY to continue a run that has one (ff3)")
    p.add_argument("--critic-init", default="",
                   help="ratchet_ff_train --critic-view: a pretrained critic (train/critic_pretrain.py) for a run whose "
                        "--init carries none; its head for --alpha is the one PPO trains")
    p.add_argument("--critic-frac", type=float, default=0.5,
                   help="ratchet_ff_train --critic-view: share of learner turns kept for the critic's training "
                        "(every turn still gets a value for GAE)")
    p.add_argument("--explore", type=float, default=0.0,
                   help="share of the learner's sampling spread uniformly over legal moves "
                        "(0.02 planned for the run after ratchet_v8_sab)")
    p.add_argument("--max-kl", type=float, default=0.1)
    p.add_argument("--min-ent", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aug-per-map", type=int, default=48)
    p.add_argument("--aug-original-share", type=float, default=0.25)
    p.add_argument("--temp", type=float, default=1.0,
                   help="policy temperature for EVERY policy in training (user, 2026-09-30): the learner's "
                        "rollouts and its PPO update, the KL teacher, and the league opponents all use "
                        "softmax(logits / temp). Below 1 the trained policy plays close to its argmax, which is "
                        "what the bot deploys (ratchet_v11 at 1.0 improved sampled play 0.755 but not greedy, 0.435)")
    p.add_argument("--s2", action="store_true",
                   help="BC_SONAR2 simulator: the learner's team speaks the v2 packet with its own sampling "
                        "probabilities (both teams in self-play), as do opponents trained with it; older "
                        "opponents play their own 48 actions and 38 grid channels")
    p.add_argument("--portal", action="store_true",
                   help="(ratchet_ff_train, with --s2 and a 15x15 learner) the BC_PORTALREP simulator "
                        "(make -C bcsim s2g15p): packets carry the portal report instead of the move "
                        "probabilities, grid channels 39-42 are the portal planes; --init must be a portal "
                        "policy (args portal, train/portal_init.py)")
    return p.parse_args()


class Actor:
    """A frozen opponent that samples its moves; LSTM ones carry their own states."""

    def __init__(self, path: str, dev, n_slots: int):
        self.net, ck = load_net(path, dev)
        for q in self.net.parameters():
            q.requires_grad_(False)
        self.lstm = ck["args"].get("arch") in ("lstm", "ff", "ffl")
        self.wants_wide = getattr(self.net, "wants_wide", False)
        # grows rather than recycling another dragon's state when a split-heavy game
        # outruns n_slots (the gate died of exactly this, 27-28 Sep)
        self.pool = (Pool(n_slots, self.net.layers, self.net.hidden, dev, grow=True,
                          no_action=getattr(self.net, "no_action", 48)) if self.lstm else None)
        self.dev = dev
        self.speaks = bool(ck["args"].get("sonar2", False))       # sonar v2 (--s2)
        # the grid this policy was trained on: in a bigger simulator grid it reads the centre crop
        # (as LSTMGreedy.grid_model; both span -G/2 .. G-G/2-1 about the head)
        self.grid_model = int(ck["args"].get("grid", 14))
        self.last_probs = None
        self.fwd = None              # ratchet_ff_train: a torch.compiled forward of self.net (2026-10-05)

    def act(self, idx, obs, grid_t, local_t, scalar_t, wide_t, mask_t, temp=None, stage=None):
        """`temp`: None = everyone at TEMP; else a (len(idx),) tensor of per-row temperatures,
        0 = greedy (ratchet_ff_train --opp-temps). A temperature-conditioned net is told it."""
        if self.lstm:
            G_ = grid_t.shape[-1]
            if G_ > self.grid_model:
                off = G_ // 2 - self.grid_model // 2
                grid_t = grid_t[..., off:off + self.grid_model, off:off + self.grid_model]
            elif G_ < self.grid_model:
                raise RuntimeError(f"a {self.grid_model}x{self.grid_model} opponent in a {G_}x{G_} simulator")
            sl = self.pool.get(idx, obs.uid[idx], stage)
            s = stage(sl) if stage is not None else torch.as_tensor(sl, device=self.dev)
            st = [(self.pool.h[l, s], self.pool.c[l, s]) for l in range(self.net.layers)]
            net_ = self.fwd if self.fwd is not None else self.net
            if getattr(self.net, "temp_in", False):
                logits, _, new = net_(grid_t, self.pool.prev[s], st, temp=TEMP if temp is None else temp)
            else:
                logits, _, new = net_(grid_t, self.pool.prev[s], st)
            for l, (h, c) in enumerate(new):
                self.pool.h[l, s] = h.float()
                self.pool.c[l, s] = c.float()
        else:
            logits, _ = self.net(local_t, scalar_t, wide_t if self.wants_wide else None)
        # a 48-action opponent in the 49-action simulator plays its own legal moves
        ml = masked_logits(logits.float(), mask_t[:, :logits.shape[1]])
        if getattr(self.net, "uniform", False):
            # the uniform-random player (train/make_random.py): uniform over the legal moves at any
            # temperature, greedy rows included
            ml = masked_logits(torch.zeros_like(logits.float()), mask_t[:, :logits.shape[1]])
            probs = ml.softmax(1)
            a = torch.multinomial(probs, 1).squeeze(1)
        elif temp is None:
            probs = (ml / TEMP).softmax(1)
            a = torch.multinomial(probs, 1).squeeze(1)
        else:
            t = temp.float().to(ml.device)
            greedy = t <= 0
            probs = (ml / t.clamp(min=1e-3)[:, None]).softmax(1)
            a = torch.where(greedy, ml.argmax(1), torch.multinomial(probs, 1).squeeze(1))
            # a greedy row's packet (sonar v2) carries the move it makes, with certainty
            # one-hot by comparison: F.one_hot range-checks on the device, a sync per call (2026-10-05)
            hot = (torch.arange(ml.shape[1], device=a.device)[None, :] == a[:, None]).to(probs.dtype)
            probs = torch.where(greedy[:, None], hot, probs)
        if self.speaks:
            self.last_probs = probs
        if self.lstm:
            self.pool.prev[s] = a
        return a

    def release(self, e, u):
        if self.pool is not None:
            self.pool.release(e, u)

    def release_env(self, e):
        if self.pool is not None:
            self.pool.release_env(e)


def board_float(b: torch.Tensor) -> torch.Tensor:
    """The env's uint8 board as the critic reads it (board_unpack's scaling)."""
    x = b.float()
    x[:, N_BINARY_PLANES:] /= 255.0
    return x


def load_critic(path: str, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = BoardCritic(bcsim.N_CHANNELS, ck["n_scalars"], 1, ck["board_ch"], n_priv=ck["n_priv"],
                      n_phi=ck["n_phi"], resid_scale=ck.get("resid_scale", 1.0))
    net.load_state_dict(ck["critic"])
    return net.to(dev), ck


TEMP = 1.0          # --temp, set in main()


def main() -> None:
    global TEMP
    a = parse()
    TEMP = float(a.temp)
    if a.critic_view:
        raise SystemExit("--critic-view is ratchet_ff_train's (feed-forward policies); the LSTM trainer has its board critic")
    if a.wl_blend_turns:
        raise SystemExit("--wl-blend-turns is ratchet_ff_train's (it needs the win/loss head)")
    if not hasattr(bcsim.env._lib, "bcv_bind_priv"):
        raise SystemExit("needs libbcvec_priv.so (make -C bcsim)")
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.backends.cudnn.benchmark = True
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(a.log, "a")

    # ---- policy, teacher
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    if ck["args"].get("arch") != "lstm":
        raise SystemExit(f"{a.init} is not an LSTM policy checkpoint (feed-forward ones: ratchet_ff_train)")
    arch = {k: ck["args"][k] for k in ARCH_KEYS}
    arch2 = {"in_ch": ck["args"].get("in_ch", 38), "n_actions": ck["args"].get("n_actions", 48)}
    speaks = bool(ck["args"].get("sonar2", False))
    if speaks != a.s2:
        raise SystemExit(f"{a.init}: trained {'with' if speaks else 'without'} sonar v2, run "
                         f"{'with' if speaks else 'without'} --s2")
    policy = LSTMPolicy(**arch, **arch2)
    policy.load_state_dict(ck["net"])
    policy.to(dev)
    popt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0, eps=1e-5)
    done0 = 0
    critic_state = None
    if a.cont and ck.get("ratchet"):
        if "opt" in ck:
            popt.load_state_dict(ck["opt"])
            for g in popt.param_groups:
                g["lr"] = a.lr
        done0 = int(ck.get("cand_turns", 0))
        critic_state = ck.get("critic")
    elif ck.get("ratchet") and ck.get("critic") is not None:
        critic_state = ck["critic"]       # a promoted anchor carries the critic it trained with
    del ck
    teacher, tck = load_net(a.teacher, dev)
    if tck["args"].get("arch") != "lstm":
        raise SystemExit("the KL teacher must be an LSTM checkpoint (the anchor)")
    for q in teacher.parameters():
        q.requires_grad_(False)

    # ---- critic
    critic, cck = load_critic(a.critic, dev)
    slots = TeamSlots(cck["slots"])
    # A carried critic (a promoted anchor's, or this candidate's own on --continue) is
    # used only if it descends from the pretrained critic this run was given. gen10 of
    # ratchet_v8_sab carries a critic fitted to the OLD reward in a 64-slot table, and
    # starting a new run from it would silently throw away the new pretrain.
    critic_src = str(pathlib.Path(a.critic).resolve())
    if critic_state is not None and critic_state.get("source") != critic_src:
        print(f"ignoring the critic carried in {a.init} (from {critic_state.get('source', 'an older run')}): "
              f"using {critic_src}", flush=True)
        critic_state = None
    if cck.get("alpha") is not None and abs(cck["alpha"] - a.alpha) > 1e-9:
        print(f"WARNING: critic {a.critic} was fitted to alpha {cck['alpha']}, training uses "
              f"{a.alpha}: its values are for the other horizon until it re-fits", flush=True)
    reg = cck.get("ids") or (json.loads(IDS.read_text()) if IDS.exists() else {})
    learner_slot = slots.slot(LEARNER_ID, add=True)
    if critic_state is not None:
        critic.load_state_dict(critic_state["net"])
        slots = TeamSlots(critic_state["slots"])
        learner_slot = slots.slot(LEARNER_ID, add=True)
    elif a.start_name and a.start_name in reg:
        src = slots.slot(reg[a.start_name])
        with torch.no_grad():
            critic.team_emb.weight[learner_slot] = critic.team_emb.weight[src]
        print(f"learner's critic slot seeded from {a.start_name} (slot {src})", flush=True)
    # how the critic was fitted (critic_v8 pretrain --drop-local / --d4): it has to be fed
    # the same way here, or it values inputs it never saw
    c_drop_local, c_d4 = bool(cck.get("drop_local", False)), bool(cck.get("d4", False))
    # a candidate that keeps going with a critic it did not carry gets the warmup too
    fresh_critic = critic_state is None
    print(f"critic {a.critic}: board {cck['board_ch']} planes, alpha {cck.get('alpha')}"
          f"{', local window zeroed' if c_drop_local else ''}{', D4 in its updates' if c_d4 else ''}"
          f"{'' if fresh_critic else ', carried state'}", flush=True)
    copt = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr, weight_decay=0.01)
    if critic_state is not None and "opt" in critic_state and a.cont:
        try:
            copt.load_state_dict(critic_state["opt"])
            # a 64-slot optimizer state against the 256-slot table fails at the first step
            for p_ in critic.parameters():
                st = copt.state.get(p_, {})       # keyed by parameter, not position
                if "exp_avg" in st and st["exp_avg"].shape != p_.shape:
                    raise ValueError("critic optimizer state from a smaller team table")
        except ValueError as e:
            copt = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr, weight_decay=0.01)
            print(f"critic optimizer restarted: {e}", flush=True)

    N, T, L = a.envs, a.steps, a.chunk
    slots_per = N * 160
    opp_paths = [x for x in a.opponents.split(",") if x]
    opp_names = [x for x in a.opp_names.split(",") if x] or [pathlib.Path(x).parent.name for x in opp_paths]
    opps = [Actor(x, dev, slots_per) for x in opp_paths]
    # the anchor and past anchors (gen1, gen2, ...) are earlier versions of the
    # learner, so they share its slot; league members keep their pretrained one
    opp_slot = [learner_slot if (n_ == "anchor" or (n_.startswith("gen") and n_[3:].isdigit()))
                else (slots.slot(reg[n_]) if n_ in reg else 0) for n_ in opp_names]
    for n_, s_ in zip(opp_names, opp_slot):
        if s_ == 0:
            print(f"WARNING: opponent {n_} has no critic slot (not in the critic's registry): "
                  "its games are valued as an unknown team's", flush=True)
    opp_p = None
    if opps:
        wts = [float(x) for x in a.opp_weights.split(",")] if a.opp_weights else [1.0] * len(opps)
        if len(wts) != len(opps) or len(opp_names) != len(opps):
            raise SystemExit("--opp-weights/--opp-names need one entry per opponent")
        opp_p = np.array(wts) / sum(wts)
    print(f"gen {a.gen} segment {a.segment}: LSTM policy {arch} from {a.init}; teacher {a.teacher} "
          f"(KL {a.kl_coef}); reward: team potential change (v8 Phi, kappa {KAPPA}), alpha {a.alpha}/round, lambda {a.lam}; lr {a.lr}, critic lr "
          f"{a.critic_lr}; self-play {a.self_frac}; opponents "
          + ", ".join(f"{n}{'' if s else ' (no critic slot)'} {w:.2f}"
                      for n, s, w in zip(opp_names, opp_slot, opp_p if opps else [])), flush=True)

    # ---- env
    texts, map_w, map_names = augment.training_pool(
        a.maps, a.aug_per_map, a.seed, a.aug_original_share, a.live_maps, a.live_share,
        a.gen_maps, a.gen_share, a.gen_per_map, a.pearl_hotspots,
        log=lambda m: print(m, flush=True))
    any_wide = any(o.wants_wide for o in opps)
    env = bcsim.BattlecodeVecEnv(texts, num_envs=N, num_threads=a.threads, seed=a.seed,
                                 closure_capacity=max(8192, N * 160), privileged=True, board=True,
                                 wide=any_wide, sonar=True, grid=True)
    env.set_map_weights(map_w)
    env.set_potential_gamma(GAMMA)
    env.set_reward_v8(True, KAPPA)
    C, G, A = env.grid.shape[1], env.grid.shape[2], bcsim.N_ACTIONS
    n_sc = cck["n_scalars"]

    slot = np.zeros(N, np.int64)
    learner = np.full(N, -1, np.int8)

    def assign(envs: np.ndarray) -> None:
        if not len(envs):
            return
        fz = rng.random(len(envs)) >= a.self_frac if opps else np.zeros(len(envs), bool)
        pick = rng.choice(len(opps), len(envs), p=opp_p) if opps else np.zeros(len(envs), np.int64)
        slot[envs] = np.where(fz, 1 + pick, 0)
        learner[envs] = np.where(fz, rng.integers(0, 2, len(envs)), -1)
        if a.s2:
            # who speaks the v2 packet in each new game: both teams in self-play; else the
            # learner's team, and the opponent's if that agent was trained with it
            for e_ in envs.tolist():
                if learner[e_] < 0:
                    env.set_sonar2(e_, (0, 1))
                else:
                    o_ = opps[slot[e_] - 1]
                    env.set_sonar2(e_, (int(learner[e_]),) + ((1 - int(learner[e_]),) if o_.speaks else ()))

    assign(np.arange(N))
    lpool = Pool(slots_per, policy.layers, policy.hidden, dev, no_action=policy.no_action)
    tpool = Pool(slots_per, teacher.layers, teacher.hidden, dev, no_action=teacher.no_action)
    ctx1 = torch.ones(N, 1, device=dev)

    # ---- rollout buffers
    b_grid = torch.zeros(T, N, C, G, G, dtype=torch.float16, device=dev)
    b_mask = torch.zeros(T, N, A, dtype=torch.bool, device=dev)
    b_action = torch.zeros(T, N, dtype=torch.long, device=dev)
    b_prev = torch.zeros(T, N, dtype=torch.long, device=dev)
    b_logp = torch.zeros(T, N, device=dev)
    b_tlogp = torch.zeros(T, N, A, dtype=torch.float16, device=dev)
    b_h = torch.zeros(T, N, policy.layers, policy.hidden, dtype=torch.float16, device=dev)
    b_c = torch.zeros(T, N, policy.layers, policy.hidden, dtype=torch.float16, device=dev)
    b_value = torch.zeros(T, N, device=dev)
    # the team chain: every row's potential (from its own team's side), team and
    # round, the same team's next turn in the same game, and for a team's last
    # turn of a finished game, the game's final row
    phi_all = np.zeros((T, N), np.float32)
    team_all = np.zeros((T, N), np.int8)
    rnd_all = np.zeros((T, N), np.int32)
    nxt_team = np.full((T, N), -1, np.int64)
    end_row = np.full((T, N), -1, np.int64)
    last_team = np.full((N, 2), -1, np.int64)
    envs_ix = np.arange(N)
    life_cpu = np.full((T, N), -1, np.int64)
    games = collections.deque(maxlen=4000)
    cbuf = {k: [] for k in ("row", "local", "scalar", "priv", "bits", "tail", "ts", "tf", "rnd")}

    obs = env.reset()
    turns_per_iter = N * T
    n_iters = max(1, math.ceil(a.turns / turns_per_iter))
    t_start = time.perf_counter()

    def save(path: pathlib.Path, it: int, cand: int) -> None:
        ckpt = {"net": policy.state_dict(), "opt": popt.state_dict(), "ratchet": True,
                "iter": it, "total_turns": cand, "cand_turns": cand,
                "critic": {"net": critic.state_dict(), "opt": copt.state_dict(), "slots": slots.as_dict(),
                           "source": critic_src},
                "args": {**vars(a), "arch": "lstm", **arch, **arch2, "sonar2": speaks}}
        tmp = path.with_suffix(".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)

    def abort(why: str) -> None:
        print(f"ABORT: {why}", flush=True)
        log_file.write(json.dumps({"gen": a.gen, "segment": a.segment, "abort": why,
                                   "total_turns": a.turn_base}) + "\n")
        log_file.flush()
        sys.exit(ABORT)

    for it in range(n_iters):
        policy.eval()
        critic.eval()
        nxt_team.fill(-1)
        end_row.fill(-1)
        last_team.fill(-1)                             # a rollout starts every chain afresh
        life_cpu.fill(-1)
        for k in cbuf:
            cbuf[k].clear()
        t0 = time.perf_counter()
        for t in range(T):
            learn = (learner < 0) | (obs.team == learner)
            grid = torch.from_numpy(env.grid).to(dev, non_blocking=True)
            local = torch.from_numpy(obs.local).to(dev, non_blocking=True)
            scalar = torch.from_numpy(obs.scalar).to(dev, non_blocking=True)
            mask = torch.from_numpy(obs.mask).to(dev, non_blocking=True).bool()
            wide = torch.from_numpy(env.wide).to(dev, non_blocking=True) if any_wide else None
            action = torch.zeros(N, dtype=torch.long, device=dev)
            li = np.flatnonzero(learn)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                if len(li):
                    lt = torch.as_tensor(li, device=dev)
                    ls = lpool.get(li, obs.uid[li])
                    s = torch.as_tensor(ls, device=dev)
                    prev = lpool.prev[s]
                    st = [(lpool.h[l, s], lpool.c[l, s]) for l in range(policy.layers)]
                    b_h[t, lt] = torch.stack([h for h, _ in st], 1).half()
                    b_c[t, lt] = torch.stack([c for _, c in st], 1).half()
                    logits, _, new = policy(grid[lt], prev, st)
                    lp_all = F.log_softmax(masked_logits(logits.float() / TEMP, mask[lt]), 1)
                    if a.explore > 0:
                        # Behaviour = (1 - eps) * policy + eps * uniform over legal moves
                        # (the learner only; opponents never explore). As in
                        # ratchet_train.py, the recorded logp below is the POLICY's, not
                        # the mixture's: an exact importance weight would shrink a
                        # collapsed action's gradient to nothing (v23 gives six 3-step
                        # sprints p ~ 0 though they survive 76-87%, 2026-09-28), so it
                        # could never come back. The bias is bounded by the PPO clip.
                        legal = mask[lt].float()
                        # a dragon with no legal move: every action, as masked_logits does
                        legal = torch.where(legal.sum(1, keepdim=True) > 0, legal, torch.ones_like(legal))
                        q = (1 - a.explore) * lp_all.exp() + a.explore * legal / legal.sum(1, keepdim=True)
                        act = torch.multinomial(q, 1).squeeze(1)
                    else:
                        act = torch.multinomial(lp_all.exp(), 1).squeeze(1)
                    if a.s2:
                        env.intent[li] = intents_from_probs(lp_all.exp())
                    for l, (h, c) in enumerate(new):
                        lpool.h[l, s] = h.float()
                        lpool.c[l, s] = c.float()
                    ts_ = torch.as_tensor(tpool.get(li, obs.uid[li]), device=dev)
                    tst = [(tpool.h[l, ts_], tpool.c[l, ts_]) for l in range(teacher.layers)]
                    tlog, _, tnew = teacher(grid[lt], prev, tst)
                    for l, (h, c) in enumerate(tnew):
                        tpool.h[l, ts_] = h.float()
                        tpool.c[l, ts_] = c.float()
                    b_tlogp[t, lt] = F.log_softmax(masked_logits(tlog.float() / TEMP, mask[lt]), 1).half()
                    b_logp[t, lt] = lp_all.gather(1, act[:, None]).squeeze(1)
                    b_prev[t, lt] = prev
                    lpool.prev[s] = act
                    tpool.prev[ts_] = act
                    action[lt] = act
                    life_cpu[t, li] = lpool.life[ls]
                    # the critic's value of this turn, for GAE, from the learner's side
                    board = board_float(torch.from_numpy(env.board[li]).to(dev))
                    opp_k = slot[li] - 1
                    tf = np.where(learner[li] < 0, learner_slot,
                                  np.array([opp_slot[k] if k >= 0 else learner_slot for k in opp_k]))
                    _, v = critic(torch.zeros_like(local[lt]) if c_drop_local else local[lt],
                                  scalar[lt, :n_sc], ctx1[:len(li)],
                                  torch.from_numpy(obs.priv[li]).to(dev), board,
                                  torch.full((len(li),), learner_slot, device=dev),
                                  torch.as_tensor(tf, device=dev),
                                  torch.full((len(li),), float(it), device=dev),
                                  torch.from_numpy(obs.round[li]).to(dev).float())
                    b_value[t, lt] = v.float()
                for k, o in enumerate(opps):
                    oi = np.flatnonzero(~learn & (slot == k + 1))
                    if len(oi):
                        ot = torch.as_tensor(oi, device=dev)
                        action[ot] = o.act(oi, obs, grid[ot], local[ot], scalar[ot],
                                           wide[ot] if wide is not None else None, mask[ot])
                        if a.s2 and o.speaks:
                            env.intent[oi] = intents_from_probs(o.last_probs)
            # a sample of the learner's turns keeps everything the critic's update needs
            keep_c = li[rng.random(len(li)) < 1.0 / a.critic_sample]
            if len(keep_c):
                bits, tail = board_pack(env.board[keep_c])
                opp_k = slot[keep_c] - 1
                cbuf["row"].append(t * N + keep_c)
                cbuf["local"].append(obs.local[keep_c].astype(np.float16))
                cbuf["scalar"].append(obs.scalar[keep_c, :n_sc].copy())
                cbuf["priv"].append(obs.priv[keep_c].copy())
                cbuf["bits"].append(bits)
                cbuf["tail"].append(tail)
                cbuf["ts"].append(np.full(len(keep_c), learner_slot))
                cbuf["tf"].append(np.where(learner[keep_c] < 0, learner_slot,
                                           np.array([opp_slot[k] if k >= 0 else learner_slot for k in opp_k])))
                cbuf["rnd"].append(obs.round[keep_c].copy())
            b_grid[t] = grid.half()
            b_mask[t] = mask
            b_action[t] = action
            phi_all[t] = obs.priv[:, bcsim.PRIV_BASE:].sum(1)
            team_all[t] = obs.team
            rnd_all[t] = obs.round
            tm_ = obs.team.astype(np.int64)
            prev_t = last_team[envs_ix, tm_]
            has = prev_t >= 0
            nxt_team.flat[prev_t[has]] = t * N + envs_ix[has]
            last_team[envs_ix, tm_] = t * N + envs_ix
            obs, closures, eps = env.step(action.to(torch.int32).cpu().numpy())
            if len(closures.env):
                for e_, u_, d_ in zip(closures.env.tolist(), closures.uid.tolist(), closures.done.tolist()):
                    if d_:
                        lpool.release(e_, u_)
                        tpool.release(e_, u_)
                        for o in opps:
                            o.release(e_, u_)
            if len(eps.rows):
                e = eps.rows[:, 0].astype(np.int64)
                w = eps.rows[:, bcsim.EpisodeStats.COLUMNS.index("winner")].astype(np.int64)
                for ee, ww in zip(e.tolist(), w.tolist()):
                    if slot[ee] > 0:
                        games.append((int(slot[ee]), 0.5 if ww < 0 else float(ww == learner[ee])))
                    # each team's last turn of this game gets the game's final row
                    for x in (0, 1):
                        m_ = last_team[ee, x]
                        if m_ >= 0:
                            end_row.flat[m_] = t * N + ee
                    last_team[ee] = -1
                    lpool.release_env(ee)
                    tpool.release_env(ee)
                    for o in opps:
                        o.release_env(ee)
                assign(e)
        t_roll = time.perf_counter() - t0

        # ---- GAE along each team's own turns: reward = the team's potential change,
        # discounted alpha per round (critic_v8.phi_returns is the same target, offline)
        v_all = b_value.reshape(-1).cpu().numpy()
        ph, tmf, rf = phi_all.reshape(-1), team_all.reshape(-1), rnd_all.reshape(-1)
        nx, er = nxt_team.reshape(-1), end_row.reshape(-1)
        rows_l = np.flatnonzero(life_cpu.reshape(-1) >= 0)
        is_l = life_cpu.reshape(-1) >= 0
        adv = np.zeros(T * N, np.float32)
        usable = np.zeros(T * N, bool)
        rw = np.zeros(T * N, np.float32)
        # reverse time order: a team's next turn is always a later row, so it is done first
        for row in rows_l[::-1]:
            n_ = nx[row]
            if n_ >= 0 and is_l[n_]:
                r = ph[n_] - ph[row]
                disc = a.alpha ** (rf[n_] - rf[row])
                delta = r + disc * v_all[n_] - v_all[row]
                # lambda per ROUND, like alpha: moves in the same round share the chain
                lam_r = a.lam ** (rf[n_] - rf[row])
                adv[row] = delta + (disc * lam_r * adv[n_] if usable[n_] else 0.0)
                usable[row] = True
                rw[row] = r
            elif er[row] >= 0:
                e_r = er[row]
                q = ph[e_r] if tmf[e_r] == tmf[row] else -ph[e_r]
                r = 0.0 if e_r == row else q - ph[row]
                adv[row] = r - v_all[row]             # nothing after the game ends
                usable[row] = True
                rw[row] = r
        ret = adv + v_all
        n_use = int(usable.sum())
        if n_use < 4096:
            print(f"it {it}: only {n_use} usable rows, skipped", flush=True)
            continue
        u_ix = np.flatnonzero(usable)
        ev = float(1.0 - np.var(ret[u_ix] - v_all[u_ix]) / (np.var(ret[u_ix]) + 1e-8))
        adv_t = torch.from_numpy(adv).to(dev)
        mu, sd = adv_t[torch.from_numpy(u_ix).to(dev)].mean(), adv_t[torch.from_numpy(u_ix).to(dev)].std() + 1e-8
        adv_n = (adv_t - mu) / sd
        use_t = torch.from_numpy(usable).to(dev)

        # ---- critic update on the sampled turns that were scored
        critic.train()
        t0 = time.perf_counter()
        c_loss = 0.0
        if cbuf["row"]:
            crow = np.concatenate(cbuf["row"])
            ok = usable[crow]
            if ok.sum() > 256:
                cat = lambda k: np.concatenate(cbuf[k])[ok]      # noqa: E731
                cr = torch.from_numpy(ret[crow[ok]]).to(dev)
                cl_, cs_, cp_ = (torch.from_numpy(cat(k)).to(dev) for k in ("local", "scalar", "priv"))
                cb_, ct_ = torch.from_numpy(cat("bits")).to(dev), torch.from_numpy(cat("tail")).to(dev)
                cts, ctf = torch.from_numpy(cat("ts")).to(dev), torch.from_numpy(cat("tf")).to(dev)
                crd = torch.from_numpy(cat("rnd")).to(dev).float()
                nb_ = 0
                for _ in range(2):
                    perm = torch.randperm(len(cr), device=dev)
                    for s0 in range(0, len(cr), 1024):
                        ix = perm[s0:s0 + 1024]
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            bd = board_unpack(cb_[ix], ct_[ix], bcsim.BOARD_CH, 64, device=dev)
                            loc_, sc_ = cl_[ix].float(), cs_[ix].float()
                            if c_d4:
                                # a random flip/rotation per position, as it was pretrained
                                fx_, fy_, tr_ = D4.random_syms(len(ix), dev)
                                Wm = torch.round(sc_[:, D4.SC_MW] * 64).long()
                                Hm = torch.round(sc_[:, D4.SC_MH] * 64).long()
                                bd[:, 14:18] *= 255.0
                                bd = D4.board(bd, Wm, Hm, fx_, fy_, tr_)
                                bd[:, 14:18] /= 255.0
                                loc_, sc_ = D4.local(loc_, fx_, fy_, tr_), D4.scalars(sc_, fx_, fy_, tr_)
                            if c_drop_local:
                                loc_ = torch.zeros_like(loc_)
                            _, v = critic(loc_, sc_, torch.ones(len(ix), 1, device=dev),
                                          cp_[ix].float(), bd,
                                          cts[ix].long(), ctf[ix].long(),
                                          torch.full((len(ix),), float(it), device=dev), crd[ix])
                        loss = F.mse_loss(v.float(), cr[ix])
                        copt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                        copt.step()
                        c_loss += loss.item()
                        nb_ += 1
                c_loss /= max(nb_, 1)
        t_crit = time.perf_counter() - t0

        # ---- one PPO epoch over the learner's turns, regrouped by dragon
        st = {"pg": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0, "kl_teacher": 0.0}
        nb = 0
        t0 = time.perf_counter()
        warm = (not a.cont or fresh_critic) and it < a.critic_warmup
        if not warm:
            flat_life = life_cpu.reshape(-1)
            ch_local = chunks_of(flat_life[rows_l], L)
            ch = torch.as_tensor(np.where(ch_local >= 0, rows_l[np.maximum(ch_local, 0)], -1), device=dev)
            fg = b_grid.reshape(T * N, C, G, G)
            fm, fa, fp = b_mask.reshape(T * N, A), b_action.reshape(-1), b_prev.reshape(-1)
            fl, ft = b_logp.reshape(-1), b_tlogp.reshape(T * N, A)
            fh = b_h.reshape(T * N, policy.layers, policy.hidden)
            fc = b_c.reshape(T * N, policy.layers, policy.hidden)
            policy.train()
            perm = torch.randperm(len(ch), device=dev)
            for b0 in range(0, len(ch), a.batch_chunks):
                cidx = ch[perm[b0:b0 + a.batch_chunks]]
                present = cidx >= 0
                rows = cidx[present]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x = policy.encode(fg[rows].float(), fp[rows])
                X = torch.zeros(*cidx.shape, x.shape[1], device=dev, dtype=x.dtype)
                X[present] = x
                first = cidx[:, 0]
                state = [(fh[first, l].float(), fc[first, l].float()) for l in range(policy.layers)]
                outs = []
                for k in range(cidx.shape[1]):
                    lg, _, state = policy.step(X[:, k].float(), state)
                    outs.append(lg)
                logits = torch.stack(outs, 1)[present]
                keep = use_t[rows]
                if not keep.any():
                    continue
                logits, rows = logits[keep], rows[keep]
                lp_all = F.log_softmax(masked_logits(logits / TEMP, fm[rows]), 1)
                logp = lp_all.gather(1, fa[rows][:, None]).squeeze(1)
                ent = -(lp_all.exp() * lp_all).nan_to_num(0.0).sum(1)
                ratio = (logp - fl[rows]).exp()
                mb = adv_n[rows]
                pg = -torch.min(ratio * mb, ratio.clamp(1 - a.clip, 1 + a.clip) * mb).mean()
                t_lp = ft[rows].float()
                kl_t = (t_lp.exp() * (t_lp - lp_all)).nan_to_num(0.0).sum(1).mean()
                loss = pg - a.ent * ent.mean() + a.kl_coef * kl_t
                if not torch.isfinite(loss):
                    abort(f"non-finite loss at iteration {it}")
                popt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                popt.step()
                with torch.no_grad():
                    st["pg"] += pg.item()
                    st["ent"] += ent.mean().item()
                    st["kl"] += (fl[rows] - logp).mean().item()
                    st["clipfrac"] += ((ratio - 1).abs() > a.clip).float().mean().item()
                    st["kl_teacher"] += kl_t.item()
                nb += 1
            for k in st:
                st[k] /= max(nb, 1)
            if not all(torch.isfinite(q).all() for q in policy.parameters()):
                abort(f"non-finite parameters at iteration {it}")
        t_opt = time.perf_counter() - t0

        cand = done0 + (it + 1) * turns_per_iter
        row = {"gen": a.gen, "segment": a.segment, "iter": it,
               "total_turns": a.turn_base + (it + 1) * turns_per_iter, "cand_turns": cand, "lr": a.lr,
               "critic_warmup": warm, "explore": a.explore, "sps": turns_per_iter / (t_roll + t_crit + t_opt),
               "t_roll": round(t_roll, 2), "t_crit": round(t_crit, 2), "t_opt": round(t_opt, 2),
               "usable": round(n_use / max(len(rows_l), 1), 3), "explained_var": round(ev, 4),
               "critic_mse": round(c_loss, 5), "return_mean": float(ret[u_ix].mean()),
               "value_mean": float(v_all[u_ix].mean()), "reward_mean": float(rw[u_ix].mean()),
               "pool_exhausted": lpool.exhausted, "elapsed": round(time.perf_counter() - t_start, 1),
               **{k: round(v, 5) for k, v in st.items()}}
        if games:
            g = np.array(games)
            for k, name in enumerate(opp_names):
                m = g[:, 0] == k + 1
                if m.sum() >= 30:
                    row[f"vs_{name}"] = round(float(g[m, 1].mean()), 4)
                    row[f"n_{name}"] = int(m.sum())
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()
        if it % 5 == 0 or it == n_iters - 1:
            vs = " ".join(f"{k[3:]} {v:.2f}" for k, v in row.items() if k.startswith("vs_"))
            print(f"g{a.gen}.{a.segment} it {it:4d}/{n_iters} | {cand / 1e6:6.1f}M | {row['sps']:>7,.0f} t/s | "
                  f"ev {ev:.3f} cmse {c_loss:.4f}{' WARMUP' if warm else ''} | ent {st['ent']:.3f} "
                  f"klT {st['kl_teacher']:.4f} clip {st['clipfrac']:.3f} | {vs}", flush=True)
        if not warm:
            if st["kl_teacher"] > a.max_kl:
                abort(f"KL to teacher {st['kl_teacher']:.4f} > {a.max_kl}")
            if st["ent"] < a.min_ent:
                abort(f"entropy {st['ent']:.4f} < {a.min_ent}")
        if it % 10 == 9:
            save(out / "latest.pt", it, cand)
    save(out / "final.pt", n_iters - 1, done0 + n_iters * turns_per_iter)
    (out / "latest.pt").unlink(missing_ok=True)
    print(f"segment done in {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
