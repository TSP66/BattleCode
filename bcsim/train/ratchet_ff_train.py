"""One ratchet training segment for a feed-forward grid policy (train/ff_net.py: ff, ffl).

ratchet_lstm_train.py without the recurrence and without the board critic
(user, 2026-10-01: speed). Same contract with ratchet.py (arguments, log.jsonl rows,
final.pt / latest.pt, exit 3 on abort), same reward (v8 Phi changes of the team,
alpha and GAE lambda per ROUND), same KL to the anchor, temperature, sonar v2 and
league opponents (which may still be LSTMs: train/ratchet_lstm_train.Actor).

The critic is another output of the policy (--critic own): train/ff_net.PrivValue on
the policy's last hidden layer plus privileged inputs -- the engine's privileged row
and which opponent the game is against. The policy never reads those; its logits
depend on the grid and the previous action only. By default the head reads the
features detached, so only the policy loss trains the trunk (--vf-trunk shares it:
ratchet_ff1 did, and regressed). A fresh head (the start from a clone) gets --critic-warmup
iterations that train only the head, on detached features, before the policy moves.

--critic-view W (user, 2026-10-01) replaces that head with train/cview.CViewCritic, a separate
network on the simulator's TRUE board: a W x W crop about the acting head (wrapped, like the
policy's window) plus the whole board pooled to 16 x 16, with the privileged row, the opponent
id and both teams' temperatures -- never the policy's features (user). The
simulator writes that view as one bit-packed byte row per turn (bcsim cview, ~6 KB at W 27);
every learner turn gets a value for GAE, and a random --critic-frac of them is kept on the GPU
for the critic's own training, inside the same minibatches as the policy's.

Per-game temperatures (user, 2026-10-01: the greedy-vs-sampled gap). Each team in each game
draws its own at the start of the game: a learner team uniform in [--temp-lo, hi], where hi
falls linearly from --temp-hi-start to --temp-hi-end over the run's first --temp-anneal-turns
(the RUN's turns, --turn-base included, so the schedule spans segments and restarts); a league
opponent is greedy (0) with probability --opp-greedy, else uniform in [0, --opp-temp-max]
(user, 2026-10-01: 0.1 .. 0.5 -> 0.25 over 2B turns; 10% greedy, else 0 .. 0.4). Every learner row
keeps its temperature, and the update uses THAT temperature for the PPO ratio (so a row is only
ever compared with itself), weights the policy loss by T / --temp (the 1/T gradient scale would
otherwise let cold rows dominate), and measures the KL to the teacher at --temp for every row
(one regulariser strength). --temp-cond 1 also tells the policy its temperature (ff_net
FFLPolicy temp_in, added as a zero column: unchanged until trained). The critic reads both
teams' temperatures. Defaults (--temp-lo -1, --opp-temp-max -1) are the fixed-temperature trainer.

PPO updates flat minibatches of the learner's usable turns (no BPTT chunks): one optimizer
step per --ppo-batch rows (16,384, as the LSTM trainer), accumulated over MICRO-row pieces.

Throughput (user, 2026-10-01): the envs are two halves, each its own simulator; one half
steps in a worker thread while the GPU picks the other's moves. The grid is bound to
page-locked memory, the local window is copied only for flat-net opponents, and the
policy/teacher forwards are torch.compiled (FF_COMPILE=0 turns that off). Measured on
ratchet_ff3's gen2 with the versions-only league: 28.8k -> 39.5k turns/s.
"""

from __future__ import annotations

import collections
import concurrent.futures
import ctypes
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _sim_grid() -> int:
    """The simulator's grid is the learner's (--init): 14, or 15 for the 15x15 policies (2026-10-01)."""
    if "--init" not in sys.argv:
        return 14
    ck_ = torch.load(sys.argv[sys.argv.index("--init") + 1], map_location="cpu", weights_only=False)
    return int(ck_["args"].get("grid", 14))


SIM_GRID = _sim_grid()
if SIM_GRID not in (14, 15) or (SIM_GRID == 15 and "--s2" not in sys.argv):
    raise SystemExit(f"no simulator build for a {SIM_GRID}x{SIM_GRID} learner{'' if '--s2' in sys.argv else ' without --s2'}")
# --portal (2026-10-02): the BC_PORTALREP build, 15x15 sonar v2 only
SIM_PORTAL = "--portal" in sys.argv
if SIM_PORTAL and (SIM_GRID != 15 or "--s2" not in sys.argv):
    raise SystemExit("--portal needs --s2 and a 15x15 learner (the only BC_PORTALREP build is s2g15p)")
os.environ.setdefault("BCSIM_LIB", str(pathlib.Path(__file__).resolve().parents[1] / "bcsim" /
                                       ("libbcvec_priv_s2_g15p.so" if SIM_PORTAL else
                                        {14: "libbcvec_priv_s2.so", 15: "libbcvec_priv_s2_g15.so"}[SIM_GRID]
                                        if "--s2" in sys.argv else "libbcvec_priv.so")))

import bcsim                                    # noqa: E402
import train.ratchet_lstm_train as RL           # noqa: E402
from train import augment                       # noqa: E402
from train.critic_v8 import GAMMA, KAPPA        # noqa: E402
from train.distill_lstm import Pool             # noqa: E402
from train.fast_gae import GAE                  # noqa: E402
from train.ff_net import AHIST_PLANE, FF_KEYS, LSTM_KEYS, N_IDENT_FEATS, N_SCAL_FEATS, N_TEMP_FEATS, PrivValue, build  # noqa: E402
from train.lstm_net import N_CH_ID, N_CH_Q, widen_stem          # noqa: E402
from train.net import masked_logits             # noqa: E402
from train.ratchet_lstm_train import ABORT, Actor, parse  # noqa: E402
from train.sonar2 import intents_from_probs     # noqa: E402
from train.yardstick import load_net            # noqa: E402

N_OPP_IDS = 64      # opponent-embedding rows; id 0 = self-play
TEMP_BANDS = torch.tensor([0.15, 0.25, 0.35, 0.45])   # per-temperature-band stats: tb0 < 0.15 <= tb1 < 0.25 ... <= tb4
FF_ENVS = 2048
COMPILE = os.environ.get("FF_COMPILE", "1") == "1"   # torch.compile the rollout forwards
# the league opponents' forwards too (user, 2026-10-05: throughput -- eager, each cost ~2.5 ms of kernel
# launches a half-step for ~0.4 ms of GPU work). Not bit-exact with eager (fused kernels round differently
# in bf16), as the learner's compiled rollout forward is not; FF_CHECK_OPP=1 measures the difference.
COMPILE_OPP = COMPILE and os.environ.get("FF_COMPILE_OPP", "0") == "1"   # off: no speed-up measured (2026-10-05)
CHECK_OPP = os.environ.get("FF_CHECK_OPP", "") == "1"
OPP_DIFF: list = []
PROF = os.environ.get("FF_PROF", "") == "1"         # section timers (profiling only)
# diagnostics only (2026-10-06): per optimizer step, the policy gradient of the PPO term, the teacher KL
# term and the entropy term separately (two half-batches each, so a term's noise-free norm is
# sqrt(<g_A, g_B>)), raw and Adam-preconditioned, one JSON row per step to this file. Off = unchanged.
GRADPROBE = os.environ.get("FF_GRADPROBE", "")
PT = collections.defaultdict(float)


def rf_(name):
    """A torch.profiler label (profiling runs only)."""
    import contextlib
    return torch.profiler.record_function(name) if PROF else contextlib.nullcontext()
MICRO = int(os.environ.get("FF_MICRO", "4096"))   # rows per forward/backward piece of an optimizer batch (FF_MICRO: diagnostics)


class Stage:
    """Page-locked staging for the rollout's small host -> GPU copies (user, 2026-10-05: throughput).

    torch.as_tensor(numpy, device=cuda) and a blocking .to(cuda) each wait for the GPU to drain its queue
    (16k cudaStreamSynchronize an iteration, 4.3 s), so the CPU and GPU took turns. Here an array is
    copied into a pinned buffer and sent with a non-blocking DMA instead: the same bytes reach the GPU in
    the same stream order, so every kernel sees the same inputs. The buffer is reused from the start
    by reset() at the top of each act(), which is safe because act() ends in a sync (.cpu())."""

    def __init__(self, nbytes: int, dev):
        self.buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        self.off, self.dev = 0, dev

    def reset(self) -> None:
        self.off = 0

    def __call__(self, x) -> torch.Tensor:
        x = np.ascontiguousarray(x)
        dt = torch.from_numpy(x[:0].reshape(-1)).dtype
        o = (self.off + 63) // 64 * 64
        if o + x.nbytes > self.buf.numel():
            raise RuntimeError(f"Stage: {o + x.nbytes} bytes in an act(), the buffer holds {self.buf.numel()}")
        t = self.buf[o:o + x.nbytes].view(dt).view(x.shape)
        t.numpy()[...] = x
        self.off = o + x.nbytes
        return t.to(self.dev, non_blocking=True)


def auc(score: np.ndarray, pos: np.ndarray) -> float:
    """Area under the ROC curve: P(a random positive scores above a random negative), ties half."""
    order = np.argsort(score, kind="mergesort")
    s_sorted = score[order]
    ranks = np.empty(len(score), np.float64)
    i = 0
    while i < len(score):                                  # average ranks over ties
        j = i
        while j + 1 < len(score) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_p = int(pos.sum())
    n_n = len(pos) - n_p
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def main() -> None:
    a = parse()
    RL.TEMP = TEMP = float(a.temp)               # Actor (league opponents) reads RL.TEMP
    if a.critic != "own":
        raise SystemExit("ratchet_ff_train trains its own value head: run with --critic own")
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.backends.cudnn.benchmark = True
    if os.environ.get("FF_DET") == "1":                   # exactness tests only: deterministic kernels
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(os.environ.get("FF_DET_ALG") == "1", warn_only=True)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(a.log, "a")

    # ---- policy + value head, teacher
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    arch_name = ck["args"].get("arch")
    if arch_name not in ("ff", "ffl"):
        raise SystemExit(f"{a.init} is not a feed-forward policy (arch {arch_name})")
    arch = {k: ck["args"][k] for k in (FF_KEYS if arch_name == "ff" else LSTM_KEYS)}
    # grid too (2026-10-01): a 15x15 policy saved without it would be rebuilt as 14x14
    arch2 = {"in_ch": ck["args"].get("in_ch", 38), "n_actions": ck["args"].get("n_actions", 48),
             "grid": int(ck["args"].get("grid", 14)),
             "temp_in": bool(ck["args"].get("temp_in", False)), "temp_ref": float(ck["args"].get("temp_ref", 0.4)),
             "scal_in": bool(ck["args"].get("scal_in", False)), "ident_in": bool(ck["args"].get("ident_in", False)),
             "ahist_in": bool(ck["args"].get("ahist_in", False))}
    if bcsim.PORTAL_BUILD != a.portal:
        raise SystemExit(f"{os.environ['BCSIM_LIB']}: {'a' if bcsim.PORTAL_BUILD else 'not a'} portal build, "
                         f"run {'with' if bcsim.PORTAL_BUILD else 'without'} --portal")
    if bool(ck["args"].get("portal", False)) != a.portal:
        raise SystemExit(f"{a.init}: {'a' if ck['args'].get('portal') else 'not a'} portal policy, run "
                         f"{'with' if ck['args'].get('portal') else 'without'} --portal "
                         "(train/portal_init.py makes one from an older policy)")
    speaks = bool(ck["args"].get("sonar2", False))
    if speaks != a.s2:
        raise SystemExit(f"{a.init}: trained {'with' if speaks else 'without'} sonar v2, run "
                         f"{'with' if speaks else 'without'} --s2")
    # the learner reads the simulator's whole grid (2026-10-01: the queen channels, 43 -> 54): a
    # narrower --init gets zero weights on the new channels, so it starts out playing exactly as it was.
    # Never past 54: the identity planes after them (2026-10-03) only feed the MLP (--ident-in).
    widened_from, cnn_ch = None, min(bcsim.GRID_CH, N_CH_Q)
    if arch2["in_ch"] < cnn_ch:
        if arch_name != "ffl":
            raise SystemExit(f"{a.init}: widening a {arch_name} policy's input is not supported (ffl only)")
        widened_from, arch2["in_ch"] = arch2["in_ch"], cnn_ch
        print(f"learner input widened {widened_from} -> {cnn_ch} channels (zero weights on the new ones)",
              flush=True)
    policy = build({"arch": arch_name, **arch, **arch2, "portal": a.portal})
    policy.load_state_dict(widen_stem(ck["net"], arch2["in_ch"]))
    policy.to(dev)
    # ---- per-game temperatures
    temp_vary = a.temp_lo > 0                     # the learner's temperature schedule is on
    opp_vary = a.opp_temp_max >= 0                # league opponents draw theirs
    if temp_vary and not (0.0 < a.temp_lo <= a.temp_hi_end <= a.temp_hi_start and a.temp_anneal_turns > 0):
        raise SystemExit(f"learner temperatures: need 0 < --temp-lo <= --temp-hi-end <= --temp-hi-start and "
                         f"--temp-anneal-turns > 0 (got {a.temp_lo}, {a.temp_hi_end}, {a.temp_hi_start}, "
                         f"{a.temp_anneal_turns})")
    if opp_vary and not 0.0 <= a.opp_greedy <= 1.0:
        raise SystemExit(f"--opp-greedy must be in [0, 1], got {a.opp_greedy}")
    temp_min = a.temp_lo if temp_vary else TEMP

    def temp_hi(turns: float) -> float:
        """The top of the learner's range after `turns` of the run."""
        f = min(1.0, max(0.0, turns / a.temp_anneal_turns))
        return a.temp_hi_start + (a.temp_hi_end - a.temp_hi_start) * f
    cur_hi = [temp_hi(a.turn_base) if temp_vary else TEMP]

    def lr_at(turns: float) -> float:
        """The policy's LR after `turns` of the run: --lr, or with --lr-cosine-turns T the cosine
        --lr * (1 + cos(pi * min(turns / T, 1))) / 2 (user, 2026-10-02: T = 20B turns). The value
        head's LR stays --critic-lr."""
        if a.lr_cosine_turns <= 0:
            return a.lr
        f = min(1.0, max(0.0, turns / a.lr_cosine_turns))
        return a.lr * 0.5 * (1.0 + math.cos(math.pi * f))

    def ent_at(turns: float) -> float:
        """The entropy coefficient after `turns` of the run: --ent, falling linearly to 0 over
        --ent-anneal-turns (user, 2026-10-01: 0.1 -> 0 over ~2B turns), or constant."""
        if a.ent_anneal_turns <= 0:
            return a.ent
        # counted from --ent-anneal-start (user, 2026-10-06: a fresh 0.005 -> 0 over 3B from a gate)
        return a.ent * min(1.0, max(0.0, 1.0 - (turns - a.ent_anneal_start) / a.ent_anneal_turns))
    ent_c = [ent_at(a.turn_base)]

    def wl_blend_at(turns: float) -> float:
        """The win/loss advantage's weight b in the policy's advantage after `turns` of the run: 0 before
        --wl-blend-start, rising linearly to 1 over --wl-blend-turns (user, 2026-10-03: 1.2B), 0 when off."""
        if a.wl_blend_turns <= 0:
            return 0.0
        return min(1.0, max(0.0, (turns - a.wl_blend_start) / a.wl_blend_turns))
    wl_b = [wl_blend_at(a.turn_base)]
    if a.wl_blend_turns > 0:
        print(f"advantage: (1 - b) A_phi + b A_wl, each standardised, blend restandardised; b {wl_b[0]:.4f} now, "
              f"0 at {a.wl_blend_start / 1e9:.4f}B -> 1 at {(a.wl_blend_start + a.wl_blend_turns) / 1e9:.4f}B run turns",
              flush=True)
    added_temp_input = False
    if a.temp_cond:
        if arch_name != "ffl":
            raise SystemExit("--temp-cond needs an ffl policy")
        if not policy.temp_in:
            policy.add_temp_input()
            added_temp_input = True
        if abs(policy.temp_ref - TEMP) > 1e-9:
            raise SystemExit(f"the policy's temperature reference is {policy.temp_ref}, this run's --temp is {TEMP}")
    elif getattr(policy, "temp_in", False):
        raise SystemExit(f"{a.init} is temperature-conditioned: run with --temp-cond 1")
    # --scal-in (user, 2026-10-03): bolted on at a gate as zero columns, so the policy plays exactly as gated
    added_scal_input = False
    if a.scal_in:
        if arch_name != "ffl":
            raise SystemExit("--scal-in needs an ffl policy")
        if not policy.scal_in:
            mlp0_cols = policy.mlp[0].in_features
            policy.add_scal_input()
            added_scal_input = True
    elif getattr(policy, "scal_in", False):
        raise SystemExit(f"{a.init} reads the scalar inputs: run with --scal-in 1")
    arch2["scal_in"] = bool(getattr(policy, "scal_in", False))
    # --ident-in (user, 2026-10-03): birth round / 500, sin(id / 7), sin(id / 43) into the first dense
    # layer, the same way: zero columns (after the scalars, before the temperature) bolted on at a gate
    added_ident_input = False
    if a.ident_in:
        if arch_name != "ffl":
            raise SystemExit("--ident-in needs an ffl policy")
        if bcsim.GRID_CH < N_CH_ID:
            raise SystemExit(f"--ident-in: {os.environ.get('BCSIM_LIB', 'the simulator')} has {bcsim.GRID_CH} "
                             f"grid channels, not {N_CH_ID} (rebuild bcsim)")
        if not policy.ident_in:
            mlp0_cols_i = policy.mlp[0].in_features
            policy.add_ident_input()
            added_ident_input = True
    elif getattr(policy, "ident_in", False):
        raise SystemExit(f"{a.init} reads the identity inputs: run with --ident-in 1")
    arch2["ident_in"] = bool(getattr(policy, "ident_in", False))
    # --ahist-in (user, 2026-10-05): the dragon's own decayed action history excluding the last action
    # (grid plane 57, a BC_AHIST build), n_actions more zero columns after the identity ones
    added_ahist_input = False
    if a.ahist_in:
        if arch_name != "ffl":
            raise SystemExit("--ahist-in needs an ffl policy")
        if bcsim.GRID_CH <= AHIST_PLANE:
            raise SystemExit(f"--ahist-in: {os.environ.get('BCSIM_LIB', 'the simulator')} has {bcsim.GRID_CH} "
                             f"grid channels: no action-history plane {AHIST_PLANE} (make -C bcsim s2g15p)")
        if not policy.ahist_in:
            mlp0_cols_h = policy.mlp[0].in_features
            policy.add_ahist_input()
            added_ahist_input = True
    elif getattr(policy, "ahist_in", False):
        raise SystemExit(f"{a.init} reads the action history: run with --ahist-in 1")
    arch2["ahist_in"] = bool(getattr(policy, "ahist_in", False))
    arch2["temp_in"] = bool(getattr(policy, "temp_in", False))
    arch2["temp_ref"] = float(getattr(policy, "temp_ref", 0.4))
    arch2["temp_min"] = temp_min
    feat_dim = policy.dense.out_features if arch_name == "ff" else arch["hidden"]
    vhead = PrivValue(feat_dim, bcsim.PRIV_COUNT, N_OPP_IDS).to(dev)
    opp_ids: dict[str, int] = {}
    fresh_head = True
    if ck.get("vhead") is not None:
        vhead.load_state_dict(ck["vhead"]["net"])
        opp_ids = dict(ck["vhead"]["opp_ids"])
        fresh_head = False
    use_cv = a.critic_view > 0
    if not use_cv and ck.get("vhead") is None:
        # the PrivValue head reads the policy's features: only to carry on a run that already has one (ff3)
        raise SystemExit("a new value head must be the true-board critic (--critic-view W): the PrivValue head "
                         "reads the policy's features, which no new critic may (user, 2026-10-01)")
    critic = None
    added_wl = False
    if a.wl_head and not use_cv:
        raise SystemExit("--wl-head needs the true-board critic (--critic-view W)")
    if a.wl_blend_turns < 0 or (a.wl_blend_turns > 0 and not a.wl_head):
        raise SystemExit("--wl-blend-turns needs --wl-head 1 (and must be >= 0)")
    if use_cv:
        from train.cview import IDENT_UNKNOWN, CViewCritic
        if not 0.0 < a.critic_frac <= 1.0:
            raise SystemExit(f"--critic-frac must be in (0, 1], got {a.critic_frac}")
        # the critic: carried in --init (a later segment), else --critic-init (critic_pretrain.py), else fresh
        ck_c, c_from = ck.get("critic"), a.init
        if ck_c is None and a.critic_init:
            ck_c, c_from = torch.load(a.critic_init, map_location="cpu", weights_only=False)["critic"], a.critic_init
        if ck_c is not None:
            if ck_c["spec"].get("kind") != "cview2" or ck_c["spec"]["w"] != a.critic_view:
                raise SystemExit(f"{c_from} carries a critic {ck_c['spec']}; this run asks for a cview2 critic, "
                                 f"crop {a.critic_view}")
            critic = CViewCritic.from_spec(ck_c["spec"]).to(dev)
            critic.load_state_dict(ck_c["net"])
            c_names, c_slots = dict(ck_c.get("names", {})), dict(ck_c.get("slots", {}))
            fresh_head = False
        else:
            critic = CViewCritic(a.critic_view, bcsim.PRIV_COUNT, alphas=[a.alpha]).to(dev)
            c_names, c_slots = {}, {}
            fresh_head = True
        # --wl-head (user, 2026-10-03): the win/loss head, added fresh to a critic without one
        if a.wl_head and critic.wl is None:
            critic.add_wl_head()
            added_wl = True
        elif critic.wl is not None and not a.wl_head:
            raise SystemExit(f"{c_from}: the critic has a win/loss head: run with --wl-head 1")
        try:
            hv = critic.head(a.alpha)                  # the head PPO trains: this run's discount
        except ValueError as ex:
            raise SystemExit(f"{c_from}: {ex}")
    vnet = critic if use_cv else vhead           # whichever value network this run trains

    def vscale() -> torch.Tensor:
        """The trained head's return scale (a view: in-place updates reach the buffer)."""
        return critic.ret_scale[hv] if use_cv else vhead.ret_scale
    # the win/loss head's parameters are a group of their own (the third), so an optimiser state saved
    # before it existed still loads: the group is added after the load when the head is new
    wl_on = use_cv and critic.wl is not None
    wl_params = list(critic.wl.parameters()) if wl_on else []
    wl_ids = {id(p_) for p_ in wl_params}
    popt = torch.optim.AdamW([{"params": policy.parameters(), "lr": a.lr},
                              {"params": [p_ for p_ in vnet.parameters() if id(p_) not in wl_ids], "lr": a.critic_lr}]
                             + ([{"params": wl_params, "lr": a.critic_lr}] if wl_on and not added_wl else []),
                             weight_decay=0.0, eps=1e-5)
    done0 = 0
    if a.cont and ck.get("ratchet"):
        if "opt" in ck and not fresh_head and not added_temp_input and widened_from is None:
            # Adam's moments for mlp.0.weight get zero columns where the new inputs went (before the
            # temperature ones), so every other weight keeps its optimiser state; (columns before, added)
            inserts = ([(mlp0_cols, N_SCAL_FEATS)] if added_scal_input else []) + \
                      ([(mlp0_cols_i, N_IDENT_FEATS)] if added_ident_input else []) + \
                      ([(mlp0_cols_h, policy.n_actions)] if added_ahist_input else [])
            if inserts:
                pi = [p_.data_ptr() for p_ in policy.parameters()].index(policy.mlp[0].weight.data_ptr())
                st = ck["opt"]["state"].get(pi)
                if st is not None:
                    for key in ("exp_avg", "exp_avg_sq"):
                        m = st[key]
                        if m.shape[1] != inserts[0][0]:
                            raise SystemExit(f"{a.init}: optimiser state for mlp.0.weight is {tuple(m.shape)}")
                        for cols, k in inserts:
                            n0 = cols - (N_TEMP_FEATS if policy.temp_in else 0)
                            m = torch.cat([m[:, :n0], m.new_zeros(m.shape[0], k), m[:, n0:]], 1)
                        st[key] = m
            popt.load_state_dict(ck["opt"])
            popt.param_groups[0]["lr"], popt.param_groups[1]["lr"] = a.lr, a.critic_lr
            for g_ in popt.param_groups[2:]:
                g_["lr"] = a.critic_lr
        done0 = int(ck.get("cand_turns", 0))
    if added_wl:
        popt.add_param_group({"params": wl_params, "lr": a.critic_lr})
    del ck
    teacher, tck = load_net(a.teacher, dev)
    for q in teacher.parameters():
        q.requires_grad_(False)
    # (user, 2026-10-01) no KL to the uniform-random player: a from-scratch run's teacher is version 0
    # until the first promotion, and pulling towards uniform play is not "not forgetting". The KL (and
    # its --max-kl abort) starts with the first promoted version.
    kl_coef = 0.0 if getattr(teacher, "uniform", False) else a.kl_coef
    if kl_coef != a.kl_coef:
        print(f"teacher {a.teacher} is the uniform-random player: KL to it off (was {a.kl_coef}), "
              f"--max-kl not enforced", flush=True)

    def kl_at(turns: float) -> float:
        """The teacher-KL coefficient after `turns` of the run: --kl-coef, moving linearly to --kl-coef-end
        over --kl-anneal-turns from --kl-anneal-start (user, 2026-10-06: 0.01 -> 0.005 over 3B), or constant."""
        if kl_coef == 0.0 or a.kl_coef_end < 0 or a.kl_anneal_turns <= 0:
            return kl_coef
        f = min(1.0, max(0.0, (turns - a.kl_anneal_start) / a.kl_anneal_turns))
        return kl_coef + (a.kl_coef_end - kl_coef) * f
    kl_c = [kl_at(a.turn_base)]

    # wider and shorter than the LSTM trainer's 512 x 512 (same turns per iteration): a feed-forward
    # step is cheap, so per-step overhead dominates; --envs/--steps given explicitly still win
    if "--envs" not in sys.argv:
        a.envs = FF_ENVS
    if "--steps" not in sys.argv:
        a.steps = 512 * 512 // a.envs
    N, T = a.envs, a.steps
    slots_per = N * 40                           # per-dragon pools grow on demand
    opp_paths = [x for x in a.opponents.split(",") if x]
    opp_names = [x for x in a.opp_names.split(",") if x] or [pathlib.Path(x).parent.name for x in opp_paths]
    # --opp-maps (user, 2026-10-05): "name=map+map;name=..." -- that opponent plays only on those official
    # maps (by name; "<name>_2", the server's newer copy, counts as <name>). Drawn at its weight as usual;
    # drawn for a game on any other map, the game is redrawn from self-play and the unrestricted opponents.
    opp_maps: dict[str, set[str]] = {}
    for part in [x for x in a.opp_maps.split(";") if x]:
        n_, _, ms_ = part.partition("=")
        opp_maps[n_] = {x for x in ms_.split("+") if x}
        if n_ not in opp_names or not opp_maps[n_]:
            raise SystemExit(f"--opp-maps {part!r}: no opponent named {n_!r} or no maps")
    opps = [Actor(x, dev, slots_per) for x in opp_paths]
    # stable ids for the value head's opponent embedding, kept across segments in the checkpoint;
    # the anchor and past generations are earlier learners, like self-play (id 0)
    for n_ in opp_names:
        if n_ not in opp_ids and not (n_ == "anchor" or (n_.startswith("gen") and n_[3:].isdigit())):
            opp_ids[n_] = min(len(opp_ids) + 1, N_OPP_IDS - 1)
    opp_vid = np.array([opp_ids.get(n_, 0) for n_ in opp_names], np.int64)
    if use_cv:
        # identities for the critic (both teams): name -> identity (critic_pretrain's map; a clone is its
        # original team) -> slot. The learner has a slot of its own, new to a pretrained table unless it
        # is there already, started as a copy of the start policy's identity; past selves: see ident_slot.
        def is_past_self(n_):
            return n_ == "anchor" or (n_.startswith("gen") and n_[3:].isdigit())
        if "learner" not in c_slots:
            free = [k for k in range(1, critic.n_slots) if k not in set(c_slots.values())]
            if not free:
                raise SystemExit("the critic's identity table is full: no slot for the learner")
            c_slots["learner"] = free[0]
            start_ident = c_names.get(getattr(a, "start_name", "") or "", None)
            with torch.no_grad():
                if start_ident in c_slots:
                    critic.ident.weight[free[0]] = critic.ident.weight[c_slots[start_ident]]
        learner_slot = c_slots["learner"]
        # the uniform-random player (train/make_random.py) is no earlier learner, whatever its league
        # name: it gets an identity of its own
        if any(getattr(o.net, "uniform", False) for o in opps) and "uniform_random" not in c_slots:
            free = [k for k in range(1, critic.n_slots) if k not in set(c_slots.values())]
            if not free:
                raise SystemExit("the critic's identity table is full: no slot for the random player")
            c_slots["uniform_random"] = free[0]

        # (user, 2026-10-02) every past generation gets an identity of its own: the learner wins ~90%
        # against gen1 and ~50% against itself, so one shared slot hid a real difference in value.
        # Keyed by the generation's real name ("anchor" is resolved from its file, anchors/genN.pt),
        # so a generation keeps its slot through later segments; a new one starts as a copy of the
        # learner's identity (it was the learner when it was promoted).
        def past_name(n_, p_):
            stem = pathlib.Path(p_).stem
            if n_ == "anchor" and stem.startswith("gen") and stem[3:].isdigit():
                return stem
            return n_

        def ident_slot(n_, o_, p_):
            if getattr(o_.net, "uniform", False):
                return c_slots["uniform_random"]
            if is_past_self(n_):
                key = f"past:{past_name(n_, p_)}"
                if key not in c_slots:
                    free = [k for k in range(1, critic.n_slots) if k not in set(c_slots.values())]
                    if not free:
                        raise SystemExit(f"the critic's identity table is full: no slot for {key}")
                    c_slots[key] = free[0]
                    with torch.no_grad():
                        critic.ident.weight[free[0]] = critic.ident.weight[learner_slot]
                return c_slots[key]
            known = c_slots.get(c_names.get(n_, f"agent:{n_}"))
            if known is not None:
                return known
            if n_ in opp_maps:
                # (user, 2026-10-05) a distilled opponent (map-restricted, --opp-maps) gets an identity of
                # its own, started as a copy of the learner's: it is the learner fine-tuned on a top team
                key = f"agent:{n_}"
                free = [k for k in range(1, critic.n_slots) if k not in set(c_slots.values())]
                if not free:
                    raise SystemExit(f"the critic's identity table is full: no slot for {key}")
                c_slots[key] = free[0]
                with torch.no_grad():
                    critic.ident.weight[free[0]] = critic.ident.weight[learner_slot]
                return c_slots[key]
            return IDENT_UNKNOWN
        opp_slot = np.array([ident_slot(n_, o_, p_) for n_, o_, p_ in zip(opp_names, opps, opp_paths)], np.int64)
        unknown = [n_ for n_, k in zip(opp_names, opp_slot) if k == IDENT_UNKNOWN]
        print(f"critic identities: learner slot {learner_slot}; " +
              ", ".join(f"{n_} -> {k}" for n_, k in zip(opp_names, opp_slot)) +
              (f"; UNKNOWN to the critic: {unknown}" if unknown else ""), flush=True)
    opp_p = None
    if opps:
        wts = [float(x) for x in a.opp_weights.split(",")] if a.opp_weights else [1.0] * len(opps)
        if len(wts) != len(opps) or len(opp_names) != len(opps):
            raise SystemExit("--opp-weights/--opp-names need one entry per opponent")
        opp_p = np.array(wts) / sum(wts)
    vdesc = (f"true-board critic, crop {a.critic_view} + pooled board "
             f"+ privileged row + opponent id + temperatures, trained on {a.critic_frac:.0%} of learner turns"
             if use_cv else "policy features + privileged row + opponent id")
    print(f"gen {a.gen} segment {a.segment}: feed-forward policy {arch_name} {arch} from {a.init}; teacher "
          f"{a.teacher} (KL {kl_coef}); value: {vdesc} "
          f"({'fresh' if fresh_head else 'carried'}), vf {a.vf_coef}; reward v8 Phi, kappa {KAPPA}, alpha "
          f"{a.alpha}/round, lambda {a.lam}; lr {a.lr}"
          f"{f' cosine to 0 at {a.lr_cosine_turns / 1e9:g}B run turns (now {lr_at(a.turn_base):.3g})' if a.lr_cosine_turns > 0 else ''}"
          f", head lr {a.critic_lr}; temp {TEMP}"
          f"{f' (uniform {a.temp_lo} .. {cur_hi[0]:.3f}, top falling to {a.temp_hi_end} at {a.temp_anneal_turns / 1e9:g}B turns)' if temp_vary else ''}"
          f"{f', opponents {a.opp_greedy:.0%} greedy else uniform 0 .. {a.opp_temp_max}' if opp_vary else ''}"
          f"{', policy told its temperature' + (' (new input)' if added_temp_input else '') if a.temp_cond else ''}"
          f"{', scalar inputs' + (' (new, zero columns)' if added_scal_input else '') if a.scal_in else ''}"
          f"{', identity inputs' + (' (new, zero columns)' if added_ident_input else '') if a.ident_in else ''}"
          f"{', action history' + (' (new, zero columns)' if added_ahist_input else '') if a.ahist_in else ''}; self-play "
          f"{a.self_frac}; opponents " + ", ".join(f"{n} {w:.2f}" for n, w in zip(opp_names, opp_p if opps else [])),
          flush=True)

    # ---- env: no board (no board critic), privileged row for Phi and the value head
    texts, map_w, map_names = augment.training_pool(
        a.maps, a.aug_per_map, a.seed, a.aug_original_share, a.live_maps, a.live_share,
        a.gen_maps, a.gen_share, a.gen_per_map, a.pearl_hotspots,
        log=lambda m: print(m, flush=True))
    any_wide = any(o.wants_wide for o in opps)
    # Two halves, each its own simulator: one half steps on the CPU (a worker thread; ctypes
    # drops the GIL inside the C step) while the GPU picks the other half's moves (user,
    # 2026-10-01: throughput). Envs are numbered globally, half h holding h*NH .. h*NH+NH-1.
    if N % 2:
        raise SystemExit("--envs must be even (two halves)")
    NH = N // 2
    halves, grid_pin, cv_pin = [], [], []
    for h in range(2):
        e_ = bcsim.BattlecodeVecEnv(texts, num_envs=NH, num_threads=a.threads, seed=a.seed + 7919 * h,
                                    closure_capacity=max(8192, NH * 160), privileged=True, board=False,
                                    wide=any_wide, sonar=True, grid=True)
        e_.set_map_weights(map_w)
        e_.set_potential_gamma(GAMMA)
        e_.set_reward_v8(True, KAPPA)
        # the grid into page-locked memory, so its copy to the GPU is a real DMA (34 MB a
        # half-step as float32; the C side keeps whatever buffer it was last bound to)
        pin = torch.zeros(e_.grid.shape, dtype=torch.float32, pin_memory=True)
        bcsim.env._lib.bcv_bind_grid(ctypes.c_void_p(e_._h), ctypes.c_void_p(pin.data_ptr()))
        e_.grid = pin.numpy()
        grid_pin.append(pin)
        if use_cv:
            # the critic view, page-locked too; copied in act() before the step that overwrites it
            # (act's closing .cpu() waits for every copy it queued)
            cpin = torch.zeros((NH, critic.lay.stride), dtype=torch.uint8, pin_memory=True)
            e_.bind_cview(cpin.numpy(), a.critic_view)
            cv_pin.append(cpin)
        halves.append(e_)
    # each half its own opponent players (their per-dragon pools are keyed by local env index)
    opps_h = [opps, [Actor(x, dev, slots_per) for x in opp_paths]]
    C, G, A = halves[0].grid.shape[1], halves[0].grid.shape[2], bcsim.N_ACTIONS
    if arch2["grid"] != G:
        raise SystemExit(f"the learner reads a {arch2['grid']}x{arch2['grid']} grid, the simulator gives {G}x{G}")
    t_grid = int(tck["args"].get("grid", 14))
    if t_grid > G:
        raise SystemExit(f"the teacher reads a {t_grid}x{t_grid} grid, the simulator gives {G}x{G}")
    t_off = G // 2 - t_grid // 2              # a smaller-grid teacher reads the centre crop

    def t_crop(g_):
        return g_ if t_grid == G else g_[..., t_off:t_off + t_grid, t_off:t_off + t_grid]
    # a teacher with fewer actions (49 against a 15x15 learner's 51): the learner's previous action is one
    # it may not know (its "no action" instead), and its logits are padded so the extra actions get
    # probability 0 -- the KL then puts no pressure on them either way
    t_nact = int(getattr(teacher, "n_actions", A))
    t_noact = int(getattr(teacher, "no_action", t_nact))
    if t_nact > A:
        raise SystemExit(f"the teacher has {t_nact} actions, the simulator {A}")
    from train.net import NEG as _NEG

    def t_prev(p_):
        return p_ if t_nact == A else torch.where(p_ >= t_nact, torch.full_like(p_, t_noact), p_)

    def t_pad(lg):
        return lg if lg.shape[1] == A else torch.cat([lg, lg.new_full((lg.shape[0], A - lg.shape[1]), _NEG)], 1)
    P = bcsim.PRIV_COUNT

    slot = np.zeros(N, np.int64)
    learner = np.full(N, -1, np.int8)
    team_temp = np.full((N, 2), TEMP, np.float32)           # each game's two teams' temperatures

    def assign(h: int, envs: np.ndarray) -> None:
        """New games in half h's local envs."""
        if not len(envs):
            return
        g_ = envs + h * NH
        fz = rng.random(len(envs)) >= a.self_frac if opps else np.zeros(len(envs), bool)
        pick = rng.choice(len(opps), len(envs), p=opp_p) if opps else np.zeros(len(envs), np.int64)
        if restricted.any() and fz.any():
            bad = fz & restricted[pick]
            if bad.any():
                cur = halves[h].env_maps()[envs]
                for j in np.flatnonzero(bad):
                    if map_base[cur[j]] in opp_maps[opp_names[pick[j]]]:
                        continue
                    n_redraw[0] += 1
                    # off its maps: redraw from self-play and the unrestricted opponents, in proportion
                    if rng.random() * (a.self_frac + (1.0 - a.self_frac) * unres_p.sum()) < a.self_frac:
                        fz[j] = False
                    else:
                        pick[j] = rng.choice(len(opps), p=unres_p / unres_p.sum())
        slot[g_] = np.where(fz, 1 + pick, 0)
        learner[g_] = np.where(fz, rng.integers(0, 2, len(envs)), -1)
        if temp_vary or opp_vary:
            for ge in g_.tolist():
                for tm in (0, 1):
                    if learner[ge] < 0 or learner[ge] == tm:
                        team_temp[ge, tm] = rng.uniform(a.temp_lo, cur_hi[0]) if temp_vary else TEMP
                    elif opp_vary:
                        team_temp[ge, tm] = 0.0 if rng.random() < a.opp_greedy else rng.uniform(0.0, a.opp_temp_max)
                    else:
                        team_temp[ge, tm] = TEMP
        if a.s2:
            for e_, ge in zip(envs.tolist(), g_.tolist()):
                if learner[ge] < 0:
                    halves[h].set_sonar2(e_, (0, 1))
                else:
                    o_ = opps[slot[ge] - 1]
                    halves[h].set_sonar2(e_, (int(learner[ge]),) + ((1 - int(learner[ge]),) if o_.speaks else ()))

    # --opp-maps: which opponents are restricted, each map's base name, the unrestricted weights
    restricted = np.array([n_ in opp_maps for n_ in opp_names], bool) if opps else np.zeros(0, bool)
    map_base = [x[:-2] if x.endswith("_2") else x for x in map_names]
    unres_p = np.where(restricted, 0.0, opp_p) if opps else np.zeros(0)
    n_redraw = [0]
    if restricted.any():
        for n_, ms_ in opp_maps.items():
            miss = ms_ - set(map_base)
            if miss:
                raise SystemExit(f"--opp-maps {n_}: not in the training maps: {sorted(miss)}")
        print("map-restricted opponents: " + "; ".join(f"{n_} on {'+'.join(sorted(m_))}" for n_, m_ in opp_maps.items()),
              flush=True)
    for h in range(2):
        assign(h, np.arange(NH))
    # prev action per dragon (no recurrent state: layers = 0), keyed by GLOBAL env; the teacher reads the same prev
    lpool = Pool(slots_per, 0, 1, dev, grow=True, no_action=policy.no_action)

    # ---- rollout buffers
    b_grid = torch.zeros(T, N, C, G, G, dtype=torch.float16, device=dev)
    b_mask = torch.zeros(T, N, A, dtype=torch.bool, device=dev)
    b_action = torch.zeros(T, N, dtype=torch.long, device=dev)
    b_prev = torch.zeros(T, N, dtype=torch.long, device=dev)
    b_logp = torch.zeros(T, N, device=dev)
    b_tlogp = torch.zeros(T, N, A, dtype=torch.float16, device=dev)
    b_value = torch.zeros(T, N, device=dev)
    b_temp = torch.full((T, N), TEMP, device=dev)          # the temperature each learner row sampled at
    b_priv = torch.zeros(T, N, P, device=dev)
    b_opp = torch.zeros(T, N, dtype=torch.long, device=dev)
    phi_all = np.zeros((T, N), np.float32)
    team_all = np.zeros((T, N), np.int8)
    rnd_all = np.zeros((T, N), np.int32)
    nxt_team = np.full((T, N), -1, np.int64)
    end_row = np.full((T, N), -1, np.int64)
    last_team = np.full((N, 2), -1, np.int64)
    is_learn = np.zeros((T, N), bool)
    games = collections.deque(maxlen=4000)
    # how each game was decided (user, 2026-10-03, for the dash): elimination, else the round-500 verdict's first
    # differing level -- queen length, longest dragon, total length -- else a draw. Each game's queens and totals
    # are the last ones a turn saw (priv cols 0-9, at most one dragon turn stale); longest comes exact from the
    # episode row. end_state per GLOBAL env: a_queen, b_queen, a_total, b_total.
    END_KINDS = ("elim", "queen", "longest", "total", "draw")
    # the win/loss head (--wl-head): its value per learner turn, and each game's winner at the row key end_row
    # uses (step * N + env). For the dash: per GLOBAL env and team, the head's prediction and the team's phi at
    # the first turn on or after each of WL_ROUNDS (NaN until then), on the GPU (no sync in the rollout); a
    # game's are snapshotted when it ends and scored against the result once an iteration.
    WL_ROUNDS = (25, 100, 200, 350)
    end_win = np.full((T, N), -2, np.int8)
    if wl_on:
        b_wl = torch.zeros(T, N, device=dev)
        wl_at = torch.full((N, 2, 2, len(WL_ROUNDS)), float("nan"), device=dev)    # env, team, (head, phi), round
        wl_rounds_t = torch.tensor(WL_ROUNDS, device=dev)
        wl_stage: list = []                                  # (snapshot (2, 2, K) on the GPU, winner)
        wl_games = collections.deque(maxlen=8000)            # (team's head preds (K,), phi (K,), z), per game and team
    end_state = np.zeros((N, 4), np.int64)
    ends = collections.deque(maxlen=4000)                    # (kind index, verdict agrees with the winner)
    if use_cv:
        # the critic's training rows: a random --critic-frac of learner turns, packed in arrival order;
        # cv_slot maps a buffer row (t * N + env) to its place here, -1 = not kept
        CV_CAP = int(math.ceil(a.critic_frac * T * N))
        b_cv = torch.zeros(CV_CAP, critic.lay.stride, dtype=torch.uint8, device=dev)
        b_cvtemp = torch.zeros(CV_CAP, 2, dtype=torch.float16, device=dev)
        b_cvid = torch.zeros(CV_CAP, 2, dtype=torch.long, device=dev)
        cv_slot = torch.full((T * N,), -1, dtype=torch.long, device=dev)
        cv_n = [0, 0]                                   # kept this iteration, dropped (buffer full)
        print(f"critic view: crop {a.critic_view}, {critic.lay.stride} bytes a row; training buffer "
              f"{CV_CAP:,} rows ({CV_CAP * critic.lay.stride / 2**30:.2f} GB)", flush=True)

    obs_h = [e_.reset() for e_ in halves]
    turns_per_iter = N * T
    n_iters = max(1, math.ceil(a.turns / turns_per_iter))
    t_start = time.perf_counter()
    if fresh_head:
        if use_cv and len(critic.alphas) > 1:
            raise SystemExit("a fresh critic has one head")
        vnet.set_priv_stats(torch.from_numpy(np.concatenate([o_.priv for o_ in obs_h])).to(dev))

    def save(path: pathlib.Path, it: int, cand: int) -> None:
        ckpt = {"net": policy.state_dict(), "opt": popt.state_dict(), "ratchet": True,
                "iter": it, "total_turns": cand, "cand_turns": cand,
                **({"critic": {"net": critic.state_dict(), "spec": critic.spec(), "names": c_names,
                               "slots": c_slots}} if use_cv
                   else {"vhead": {"net": vhead.state_dict(), "opp_ids": opp_ids}}),
                "args": {**vars(a), "arch": arch_name, **arch, **arch2, "sonar2": speaks}}
        tmp = path.with_suffix(".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)

    def abort(why: str) -> None:
        print(f"ABORT: {why}", flush=True)
        log_file.write(json.dumps({"gen": a.gen, "segment": a.segment, "abort": why,
                                   "total_turns": a.turn_base}) + "\n")
        log_file.flush()
        sys.exit(ABORT)

    stepper = concurrent.futures.ThreadPoolExecutor(max_workers=1)   # one step in flight, in order

    def _timed_step(h_, acts_):
        _s0 = time.perf_counter()
        r_ = halves[h_].step(acts_)
        if PROF:
            PT["sim_step"] += time.perf_counter() - _s0
        return r_
    gix_h = [np.arange(NH) + h * NH for h in range(2)]
    flat_opps = any(not o.lstm for o in opps)

    stage = Stage(64 << 20, dev)

    def act(h: int, t: int) -> np.ndarray:
        """Half h's moves at rollout step t; records its rows of the buffers."""
        _pa = time.perf_counter()
        stage.reset()
        env_, obs, off, gix = halves[h], obs_h[h], h * NH, gix_h[h]
        cols = slice(off, off + NH)
        learn = (learner[cols] < 0) | (obs.team == learner[cols])
        grid = grid_pin[h].to(dev, non_blocking=True)
        # the local window and scalars only for opponents that read them (flat nets)
        local = stage(obs.local) if flat_opps else None
        scalar = stage(obs.scalar) if flat_opps else None
        mask = stage(obs.mask).bool()
        priv = stage(obs.priv)
        wide = stage(env_.wide) if any_wide else None
        action = torch.zeros(NH, dtype=torch.long, device=dev)
        _p0 = time.perf_counter()
        li = np.flatnonzero(learn)
        is_learn[t, cols] = learn
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if len(li):
                lt = stage(li)
                gt = lt + off
                s_ = stage(lpool.get(li + off, obs.uid[li], stage))
                prev = lpool.prev[s_]
                # each row's own temperature (its game's, its team's)
                tl = stage(team_temp[li + off, obs.team[li].astype(np.int64)])
                with rf_("R_learner"):
                    feat = f_features(grid[lt], prev, tl) if policy.temp_in else f_features(grid[lt], prev)
                    logits = policy.pi(feat)
                lp_all = F.log_softmax(masked_logits(logits.float() / tl[:, None], mask[lt]), 1)
                if a.explore > 0:
                    legal = mask[lt].float()
                    legal = torch.where(legal.sum(1, keepdim=True) > 0, legal, torch.ones_like(legal))
                    q = (1 - a.explore) * lp_all.exp() + a.explore * legal / legal.sum(1, keepdim=True)
                    act_ = torch.multinomial(q, 1).squeeze(1)
                else:
                    act_ = torch.multinomial(lp_all.exp(), 1).squeeze(1)
                if a.s2 and not a.portal:      # a portal build's packets carry no intents
                    env_.intent[li] = intents_from_probs(lp_all.exp())
                with rf_("R_teacher"):
                    tlog = t_logits(grid[lt], prev, tl)
                # the teacher at the reference temperature, whatever this row sampled at (one KL strength)
                b_tlogp[t, gt] = F.log_softmax(masked_logits(tlog.float() / TEMP, mask[lt]), 1).half()
                b_logp[t, gt] = lp_all.gather(1, act_[:, None]).squeeze(1)
                b_temp[t, gt] = tl
                b_prev[t, gt] = prev
                sl = slot[li + off]
                ov = stage(np.where(sl > 0, opp_vid[np.maximum(sl - 1, 0)], 0)
                           if len(opp_vid) else np.zeros(len(li), np.int64))
                b_opp[t, gt] = ov
                if use_cv:
                    cv_rows = cv_pin[h].to(dev, non_blocking=True)[lt]
                    # this team's and the other's sampling temperature (0 = a greedy opponent)
                    lteam = obs.team[li].astype(np.int64)
                    temp = stage(np.stack([team_temp[li + off, lteam], team_temp[li + off, 1 - lteam]], 1))
                    sl_ = slot[li + off]
                    ident = stage(np.stack([np.full(len(li), learner_slot, np.int64),
                                            np.where(sl_ > 0, opp_slot[np.maximum(sl_ - 1, 0)], learner_slot)], 1))
                    if wl_on:
                        with rf_("R_critic"):
                            raw_, wv_ = critic.raw_wl(cv_rows, priv[lt], ident, temp)
                        b_value[t, gt] = (raw_[:, hv] * critic.ret_scale[hv]).float()
                        b_wl[t, gt] = wv_.float()
                        # the first prediction (and phi) on or after each dash round, per env and team
                        e_g = stage(li + off)
                        tm_g = stage(lteam)
                        r_g = stage(obs.round[li])
                        cur = wl_at[e_g, tm_g]                                      # (n, 2, K)
                        now = torch.stack([wv_.float(), priv[lt][:, bcsim.PRIV_BASE:].sum(1).float()], 1)
                        take = torch.isnan(cur) & (r_g[:, None, None] >= wl_rounds_t)
                        wl_at[e_g, tm_g] = torch.where(take, now[:, :, None].expand_as(cur), cur)
                    else:
                        b_value[t, gt] = critic(cv_rows, priv[lt], ident, temp)[:, hv].float()
                    keep = np.flatnonzero(rng.random(len(li)) < a.critic_frac)
                    room = CV_CAP - cv_n[0]
                    if len(keep) > room:
                        cv_n[1] += len(keep) - room
                        keep = keep[:room]
                    if len(keep):
                        kt_ = stage(keep)
                        dst = slice(cv_n[0], cv_n[0] + len(keep))
                        b_cv[dst] = cv_rows[kt_]
                        b_cvtemp[dst] = temp[kt_].half()
                        b_cvid[dst] = ident[kt_]
                        cv_slot[t * N + gt[kt_]] = torch.arange(dst.start, dst.stop, device=dev)
                        cv_n[0] += len(keep)
                else:
                    b_value[t, gt] = vhead(feat, priv[lt], ov).float()
                lpool.prev[s_] = act_
                action[lt] = act_
            _p1 = time.perf_counter()
            sl_all = slot[cols]
            for k, o in enumerate(opps_h[h]):
                oi = np.flatnonzero(~learn & (sl_all == k + 1))
                if len(oi):
                    ot = stage(oi)
                    with rf_(f"R_opp{k}"):
                      action[ot] = o.act(oi, obs, grid[ot], local[ot] if local is not None else None,
                                       scalar[ot] if scalar is not None else None,
                                       wide[ot] if wide is not None else None, mask[ot],
                                       temp=(stage(team_temp[oi + off, obs.team[oi].astype(np.int64)])
                                             if opp_vary else None), stage=stage)
                    if a.s2 and o.speaks and not a.portal:
                        env_.intent[oi] = intents_from_probs(o.last_probs)
        _p2 = time.perf_counter()
        b_grid[t, cols] = grid.half()
        b_mask[t, cols] = mask
        b_action[t, cols] = action
        b_priv[t, cols] = priv
        phi_all[t, cols] = obs.priv[:, bcsim.PRIV_BASE:].sum(1)
        pv_ = np.rint(np.expm1(obs.priv[:, [0, 3, 8, 9]].astype(np.float64) * 5.0)).astype(np.int64)
        b_side = obs.team.astype(bool)                      # the acting dragon is team B: its "own" columns are B's
        end_state[cols, 0] = np.where(b_side, pv_[:, 3], pv_[:, 2])
        end_state[cols, 1] = np.where(b_side, pv_[:, 2], pv_[:, 3])
        end_state[cols, 2] = np.where(b_side, pv_[:, 1], pv_[:, 0])
        end_state[cols, 3] = np.where(b_side, pv_[:, 0], pv_[:, 1])
        team_all[t, cols] = obs.team
        rnd_all[t, cols] = obs.round
        tm_ = obs.team.astype(np.int64)
        prev_t = last_team[gix, tm_]
        has = prev_t >= 0
        nxt_team.flat[prev_t[has]] = t * N + gix[has]
        last_team[gix, tm_] = t * N + gix
        _p3 = time.perf_counter()
        out_ = action.to(torch.int32).cpu().numpy()
        if PROF:
            _p4 = time.perf_counter()
            PT["act_pre"] += _p0 - _pa; PT["act_learner"] += _p1 - _p0; PT["act_opps"] += _p2 - _p1
            PT["act_tail"] += _p3 - _p2; PT["act_sync"] += _p4 - _p3
        return out_

    def post(h: int, obs, closures, eps, t: int) -> None:
        """Half h's step (taken at rollout step t) has returned: pools, finished games, new games."""
        off = h * NH
        obs_h[h] = obs
        if len(closures.env):
            for e_, u_, d_ in zip(closures.env.tolist(), closures.uid.tolist(), closures.done.tolist()):
                if d_:
                    lpool.release(e_ + off, u_)
                    for o in opps_h[h]:
                        o.release(e_, u_)
        if len(eps.rows):
            e = eps.rows[:, 0].astype(np.int64)
            w = eps.rows[:, bcsim.EpisodeStats.COLUMNS.index("winner")].astype(np.int64)
            ci = bcsim.EpisodeStats.COLUMNS.index
            for k_, (el, ww) in enumerate(zip(e.tolist(), w.tolist())):
                ee = el + off
                er = eps.rows[k_]
                au, bu = int(er[ci("a_units")]), int(er[ci("b_units")])
                aq, bq, at, bt = end_state[ee].tolist()
                al, bl = int(er[ci("a_longest")]), int(er[ci("b_longest")])
                if au == 0 or bu == 0:
                    kind, pred = 0, (-1 if au == bu else int(au == 0))
                elif aq != bq:
                    kind, pred = 1, int(bq > aq)
                elif al != bl:
                    kind, pred = 2, int(bl > al)
                elif at != bt:
                    kind, pred = 3, int(bt > at)
                else:
                    kind, pred = 4, -1
                ends.append((kind, pred == ww))
                end_win.flat[t * N + ee] = ww
                if wl_on:
                    wl_stage.append((wl_at[ee].clone(), ww))
                    wl_at[ee] = float("nan")
                if slot[ee] > 0:
                    games.append((int(slot[ee]), 0.5 if ww < 0 else float(ww == learner[ee])))
                for x in (0, 1):
                    m_ = last_team[ee, x]
                    if m_ >= 0:
                        end_row.flat[m_] = t * N + ee
                last_team[ee] = -1
                lpool.release_env(ee)
                for o in opps_h[h]:
                    o.release_env(el)
            assign(h, e)

    if COMPILE_OPP:
        def checked(net_, cf_):
            """FF_CHECK_OPP: run both, keep eager's result, record compiled vs eager."""
            def f_(*a_, **k_):
                lc_ = cf_(*a_, **k_)[0].float()
                out_ = net_(*a_, **k_)
                le_ = out_[0].float()
                pe_, pc_ = (le_ / TEMP).log_softmax(1), (lc_ / TEMP).log_softmax(1)
                OPP_DIFF.append(((le_ - lc_).abs().max(), (pe_.exp() - pc_.exp()).abs().max(),
                                 (pe_.exp() * (pe_ - pc_)).sum(1).mean(), (le_.argmax(1) != lc_.argmax(1)).float().mean()))
                return out_
            return f_
        for o_ in opps_h[0] + opps_h[1]:
            if o_.lstm and not getattr(o_.net, "uniform", False):
                cf_ = torch.compile(o_.net, dynamic=True)
                o_.fwd = checked(o_.net, cf_) if CHECK_OPP else cf_
    t_cond = bool(getattr(teacher, "temp_in", False))      # a temperature-conditioned anchor is told it too
    # compiled forwards for the rollout (--compile): at these batch sizes the per-kernel
    # launch cost is a real share of a step
    f_train = torch.compile(policy.features, dynamic=False) if COMPILE else policy.features
    if COMPILE:
        f_features = torch.compile(policy.features, dynamic=True)
        _t_fwd = torch.compile(teacher.forward, dynamic=True)
        t_logits = ((lambda g_, p_, t_: t_pad(_t_fwd(t_crop(g_), t_prev(p_), None, t_)[0])) if t_cond  # noqa: E731
                    else (lambda g_, p_, t_: t_pad(_t_fwd(t_crop(g_), t_prev(p_))[0])))
    else:
        f_features = policy.features
        t_logits = ((lambda g_, p_, t_: t_pad(teacher(t_crop(g_), t_prev(p_), None, t_)[0])) if t_cond  # noqa: E731
                    else (lambda g_, p_, t_: t_pad(teacher(t_crop(g_), t_prev(p_))[0])))

    def temp_summary() -> dict:
        """What the games in flight drew: learner teams' range and mean, opponents' greedy share and mean."""
        if not (temp_vary or opp_vary):
            return {}
        own = np.zeros((N, 2), bool)
        own[:, 0] = (learner < 0) | (learner == 0)
        own[:, 1] = (learner < 0) | (learner == 1)
        lt_, ot_ = team_temp[own], team_temp[~own]
        out = {"temp_learn_mean": round(float(lt_.mean()), 4), "temp_learn_min": round(float(lt_.min()), 4),
               "temp_learn_max": round(float(lt_.max()), 4)}
        if len(ot_):
            out.update(temp_opp_mean=round(float(ot_.mean()), 4), temp_opp_greedy=round(float((ot_ == 0).mean()), 4),
                       temp_opp_max=round(float(ot_.max()), 4))
        return out

    PROF_IT = int(os.environ.get("FF_PROF_TORCH", "-1"))   # profile this iteration with torch.profiler
    gae = GAE(a.alpha, a.lam, a.wl_lam if wl_on else 0.0)
    temp_bands_dev = TEMP_BANDS.to(dev)
    band_ids = torch.arange(len(TEMP_BANDS) + 1, device=dev)
    for it in range(n_iters):
        if it == PROF_IT:
            from torch.profiler import ProfilerActivity, profile
            _prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            _prof.__enter__()
        policy.eval()
        vnet.eval()
        if temp_vary:
            cur_hi[0] = temp_hi(a.turn_base + it * turns_per_iter)    # new games draw from [temp_lo, cur_hi]
        ent_c[0] = ent_at(a.turn_base + it * turns_per_iter)
        kl_c[0] = kl_at(a.turn_base + it * turns_per_iter)
        wl_b[0] = wl_blend_at(a.turn_base + it * turns_per_iter)
        lr_now = lr_at(a.turn_base + it * turns_per_iter)
        popt.param_groups[0]["lr"] = lr_now
        if use_cv:
            cv_slot.fill_(-1)
            cv_n[0] = cv_n[1] = 0
        nxt_team.fill(-1)
        end_row.fill(-1)
        end_win.fill(-2)
        last_team.fill(-1)
        is_learn.fill(False)
        t0 = time.perf_counter()
        fut: list = [None, None]
        for t in range(T):
            for h in range(2):
                if fut[h] is not None:
                    _q0 = time.perf_counter()
                    res_ = fut[h][0].result()
                    _q1 = time.perf_counter()
                    post(h, *res_, fut[h][1])
                    _q2 = time.perf_counter()
                    if PROF:
                        PT["wait_sim"] += _q1 - _q0; PT["post"] += _q2 - _q1
                acts = act(h, t)
                fut[h] = (stepper.submit(_timed_step, h, acts), t)
        for h in range(2):                       # the last steps' results, before GAE
            post(h, *fut[h][0].result(), fut[h][1])
        t_roll = time.perf_counter() - t0
        if PROF:
            print("PROF roll " + " ".join(f"{k} {v:.2f}" for k, v in sorted(PT.items())), flush=True)
            PT.clear()
        if CHECK_OPP and OPP_DIFF:
            d_ = torch.tensor([[float(x) for x in r_] for r_ in OPP_DIFF])
            print(f"OPPCHECK {len(d_)} calls: max |dlogit| {d_[:, 0].max():.4f}, max |dprob| {d_[:, 1].max():.5f} "
                  f"(mean of per-call max {d_[:, 1].mean():.5f}), KL mean {d_[:, 2].mean():.2e} max {d_[:, 2].max():.2e}, "
                  f"greedy-argmax flips {d_[:, 3].mean():.5f}", flush=True)
            OPP_DIFF.clear()

        # ---- GAE along each team's own turns (as ratchet_lstm_train, Phi reward)
        t0 = time.perf_counter()
        v_all = b_value.reshape(-1).cpu().numpy()
        ph, tmf, rf = phi_all.reshape(-1), team_all.reshape(-1), rnd_all.reshape(-1)
        nx, er = nxt_team.reshape(-1), end_row.reshape(-1)
        is_l = is_learn.reshape(-1)
        rows_l = np.flatnonzero(is_l)
        if wl_on:
            # the win/loss head: undiscounted TD(lambda), no reward until the game's result z (+1 / -1, draw 0)
            vw = b_wl.reshape(-1).cpu().numpy()
            ew = end_win.reshape(-1)
        # the backward pass along each team's turns, compiled (train/fast_gae.py: bit for bit the old Python loop)
        adv, usable, rw, adv_w = gae(rows_l, nx, er, is_l, ph, tmf, rf, v_all, wl_on,
                                     vw if wl_on else None, ew if wl_on else None)
        ret = adv + v_all
        n_use = int(usable.sum())
        if n_use < 4096:
            print(f"it {it}: only {n_use} usable rows, skipped", flush=True)
            continue
        u_ix = np.flatnonzero(usable)
        ev = float(1.0 - np.var(ret[u_ix] - v_all[u_ix]) / (np.var(ret[u_ix]) + 1e-8))
        # the head's output unit follows the returns' spread (EMA; set outright on a fresh head)
        sd_now = float(np.std(ret[u_ix])) + 1e-6
        with torch.no_grad():
            if fresh_head and it == 0:
                vscale().fill_(sd_now)
            else:
                vscale().mul_(0.95).add_(0.05 * sd_now)
        adv_t = torch.from_numpy(adv).to(dev)
        u_t = torch.from_numpy(u_ix).to(dev)
        mu, sd = adv_t[u_t].mean(), adv_t[u_t].std() + 1e-8
        adv_n = (adv_t - mu) / sd
        ret_t = torch.from_numpy(ret).to(dev)
        if wl_on:
            ret_w = adv_w + vw
            ret_w_t = torch.from_numpy(ret_w).to(dev)
            wl_ev = float(1.0 - np.var(ret_w[u_ix] - vw[u_ix]) / (np.var(ret_w[u_ix]) + 1e-8))
            # the blend (--wl-blend-turns, user 2026-10-03): A_phi and A_wl are each standardised over this
            # iteration's usable rows (their raw units differ: phi points vs a +-1 result), mixed as
            # (1 - b) A_phi + b A_wl, and the mix standardised again, so the policy step keeps one size whatever
            # b is and however the two agree (two independent unit parts at b = 0.5 would give sd 0.71).
            # Logged every iteration, b = 0 included, so the dash shows what the blend would do.
            aw_t = torch.from_numpy(adv_w).to(dev)
            aw_n = (aw_t - aw_t[u_t].mean()) / (aw_t[u_t].std() + 1e-8)
            b_ = wl_b[0]
            mix = (1.0 - b_) * adv_n + b_ * aw_n
            ap_u, aw_u, mx_u = adv_n[u_t], aw_n[u_t], mix[u_t]
            v_mix = float(mx_u.var()) + 1e-12

            def tail(x):                                  # share of sum(x^2) in the top 1% |x| (normal: ~0.10)
                q_ = x.float().abs()
                return float((q_.topk(max(1, len(q_) // 100)).values ** 2).sum() / ((q_ ** 2).sum() + 1e-12))
            blend_row = {
                "wl_blend": round(b_, 5),
                "opp_map_redraws": n_redraw[0],
                "adv_phi_sd": round(float(sd), 5), "adv_wl_sd": round(float(aw_t[u_t].std()), 5),
                "adv_corr": round(float(torch.corrcoef(torch.stack([ap_u, aw_u]))[0, 1]), 4),
                "adv_sign_agree": round(float(((ap_u > 0) == (aw_u > 0)).float().mean()), 4),
                # each part's share of the blend: Cov(part, blend) / Var(blend), the two sum to 1
                "adv_share_phi": round(float(((1.0 - b_) * ap_u * mx_u).mean()) / v_mix, 4),
                "adv_share_wl": round(float((b_ * aw_u * mx_u).mean()) / v_mix, 4),
                # each part's own variance over the blend's (the rest is the cross term 2 b (1 - b) corr)
                "adv_var_phi": round((1.0 - b_) ** 2 * float(ap_u.var()) / v_mix, 4),
                "adv_var_wl": round(b_ ** 2 * float(aw_u.var()) / v_mix, 4),
                "adv_mix_sd": round(v_mix ** 0.5, 4),
                "adv_phi_tail": round(tail(ap_u), 4), "adv_wl_tail": round(tail(aw_u), 4),
                "adv_phi_maxz": round(float(ap_u.abs().max()), 2), "adv_wl_maxz": round(float(aw_u.abs().max()), 2)}
            if b_ > 0:
                adv_n = (mix - mx_u.mean()) / (v_mix ** 0.5 + 1e-8)
            del aw_t, aw_n, mix, ap_u, aw_u, mx_u
        t_gae = time.perf_counter() - t0

        # ---- update: flat minibatches of usable learner turns
        st = {"pg": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0, "kl_teacher": 0.0}
        nb = 0
        c_loss, nc = 0.0, 0
        w_loss = 0.0
        cv_rows_used = 0
        t0 = time.perf_counter()
        warm = fresh_head and it < a.critic_warmup
        fg = b_grid.reshape(T * N, C, G, G)
        fm, fa, fp = b_mask.reshape(T * N, A), b_action.reshape(-1), b_prev.reshape(-1)
        fl, ft = b_logp.reshape(-1), b_tlogp.reshape(T * N, A)
        fpriv, fopp = b_priv.reshape(T * N, P), b_opp.reshape(-1)
        ftemp = b_temp.reshape(-1)
        tb = {}                                           # per temperature band: rows, clipped, kl, entropy
        acc = torch.zeros(7, dtype=torch.float64, device=dev)   # c_loss, w_loss, pg, ent, kl, clipfrac, kl_teacher
        tb_acc = torch.zeros(len(TEMP_BANDS) + 1, 4, dtype=torch.float64, device=dev)
        nonfinite = torch.zeros((), dtype=torch.bool, device=dev)
        ratio0 = None                                     # |ratio - 1| on the first piece: rollout vs update logits
        policy.train()
        vnet.train()
        for _ in range(2 if warm else 1):
            perm = u_t[torch.randperm(len(u_t), device=dev)]
            if COMPILE and not warm and len(perm) >= MICRO:
                # whole pieces only: one shape for the compiled trunk (a ragged last piece made it
                # recompile every iteration); the dropped remainder is < MICRO random rows
                perm = perm[:len(perm) // MICRO * MICRO]
            if use_cv:
                # each piece's critic rows, found for the whole permutation at once (one sync, not a boolean
                # index per piece): pos_cv lists the kept rows' places in perm in order, cut[k] where piece k's
                # start among them -- rows[sel] and cs[sel] exactly, in the same order
                cs_all = cv_slot[perm]
                pos_cv = torch.nonzero(cs_all >= 0).squeeze(1)
                starts = [b0_ + m0_ for b0_ in range(0, len(perm), a.ppo_batch)
                          for m0_ in range(0, min(a.ppo_batch, len(perm) - b0_), MICRO)]
                cut = torch.searchsorted(pos_cv, torch.tensor(starts + [len(perm)], device=dev)).tolist()
            pc = 0                                            # piece counter
            for b0 in range(0, len(perm), a.ppo_batch):
                # one optimizer step per --ppo-batch rows (16384: the LSTM trainer's 512 chunks x 32),
                # accumulated over pieces of MICRO rows so the trunk's activations fit
                batch = perm[b0:b0 + a.ppo_batch]
                popt.zero_grad(set_to_none=True)
                if GRADPROBE and not warm:
                    gpar = [q_ for q_ in policy.parameters() if q_.requires_grad]
                    gacc = {k_: [torch.zeros(sum(q_.numel() for q_ in gpar), device=dev) for _h in range(2)]
                            for k_ in ("pg", "kl", "ent")}
                for m0 in range(0, len(batch), MICRO):
                    rows = batch[m0:m0 + MICRO]
                    share = len(rows) / len(batch)
                    pcur, pc = pc, pc + 1
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        with torch.set_grad_enabled(not warm):      # warmup: the head only, no trunk graph
                            tr = ftemp[rows]
                            ff_ = f_train if not warm else policy.features
                            feat = ff_(fg[rows].float(), fp[rows], tr) if policy.temp_in else ff_(fg[rows].float(), fp[rows])
                        # the head reads detached features unless --vf-trunk: the value loss does not
                        # reshape the policy's trunk (ratchet_ff1 shared it and its greedy play fell
                        # 0.49 -> 0.34 vs the anchor in two checks while its KL looked like v12's)
                        vfeat = feat if (a.vf_trunk and not warm) else feat.detach()
                        if use_cv:
                            # the critic trains on the rows of this piece that were kept for it
                            pk_ = pos_cv[cut[pcur]:cut[pcur + 1]]
                            v_rows = perm[pk_]
                            csel = cs_all[pk_]
                            if len(v_rows) and wl_on:
                                v_full, w_pred = critic.raw_wl(b_cv[csel], fpriv[v_rows], b_cvid[csel],
                                                               b_cvtemp[csel])
                                v_raw = v_full[:, hv]
                            elif len(v_rows):
                                v_raw = critic.raw(b_cv[csel], fpriv[v_rows], b_cvid[csel],
                                                   b_cvtemp[csel])[:, hv]
                        else:
                            v_rows = rows
                            v_raw = vhead.raw(vfeat, fpriv[rows], fopp[rows])
                    vloss = (F.mse_loss(v_raw.float(), ret_t[v_rows] / vscale()) if len(v_rows)
                             else torch.zeros((), device=dev))
                    # the win/loss head trains on the same critic rows; its value reaches the policy only
                    # through the blended advantage (--wl-blend-turns), never through this loss
                    wloss = (F.mse_loss(w_pred.float(), ret_w_t[v_rows]) if wl_on and len(v_rows)
                             else torch.zeros((), device=dev))
                    if warm:
                        loss = vloss + a.wl_coef * wloss
                    else:
                        logits = policy.pi(feat.float())
                        ml_ = masked_logits(logits, fm[rows])
                        # the ratio at the temperature the row was sampled at: a row is only compared with itself
                        lp_all = F.log_softmax(ml_ / tr[:, None], 1)
                        logp = lp_all.gather(1, fa[rows][:, None]).squeeze(1)
                        ent = -(lp_all.exp() * lp_all).nan_to_num(0.0).sum(1)
                        ratio = (logp - fl[rows]).exp()
                        mb = adv_n[rows]
                        # d log pi_T / d logits scales as 1/T: weight by T / TEMP so cold rows do not dominate
                        pg = -((tr / TEMP) * torch.min(ratio * mb, ratio.clamp(1 - a.clip, 1 + a.clip) * mb)).mean()
                        t_lp = ft[rows].float()
                        # the KL to the teacher at the reference temperature for every row: one strength
                        lp_ref = lp_all if not temp_vary else F.log_softmax(ml_ / TEMP, 1)
                        kl_t = (t_lp.exp() * (t_lp - lp_ref)).nan_to_num(0.0).sum(1).mean()
                        # the entropy's gradient also scales as 1/T: the same T / TEMP weight as pg (user,
                        # 2026-10-02: unweighted, cold rows got up to TEMP / T = 4x the bonus and every
                        # band sat at the same entropy)
                        ent_w = ((tr / TEMP) * ent).mean()
                        loss = pg - ent_c[0] * ent_w + kl_c[0] * kl_t + a.vf_coef * vloss + a.wl_coef * wloss
                    # checked once per optimizer step (before popt.step), not per piece: each check is a sync
                    nonfinite = nonfinite | ~torch.isfinite(loss.detach())
                    if ratio0 is None and not warm:
                        # before any step this iteration the policy is the one that played: the ratio must
                        # be 1 up to bf16 noise. Far from it means rows are scored at the wrong temperature.
                        ratio0 = float((ratio.detach() - 1).abs().mean())
                        if ratio0 > 0.05:
                            abort(f"first-piece |ratio - 1| = {ratio0:.4f} at iteration {it}: the update does not "
                                  f"reproduce the rollout's probabilities")
                    if GRADPROBE and not warm:
                        # each term's gradient on this piece, into the half of the batch it belongs to
                        half = int(m0 >= len(batch) // 2)
                        for k_, term in (("pg", pg), ("kl", kl_c[0] * kl_t), ("ent", -ent_c[0] * ent_w)):
                            gs = torch.autograd.grad(term * share * 2, gpar, retain_graph=True, allow_unused=True)
                            gacc[k_][half] += torch.cat([(g_ if g_ is not None else torch.zeros_like(q_)).reshape(-1)
                                                         for g_, q_ in zip(gs, gpar)]).float()
                    if loss.requires_grad:                  # warmup piece with no critic rows: nothing to learn
                        (loss * share).backward()
                    with torch.no_grad():
                        # on the GPU, in float64, in the order the old per-piece .item() sums ran: read once an
                        # iteration (each .item() was a sync; ~30 a piece)
                        acc[0] += share * vloss.detach().double() * vscale().double() ** 2
                        if wl_on:
                            acc[1] += share * wloss.detach().double()
                        cv_rows_used += len(v_rows)
                        if not warm:
                            acc[2] += share * pg.detach().double()
                            acc[3] += share * ent.mean().double()
                            acc[4] += share * (fl[rows] - logp).mean().double()
                            acc[5] += share * ((ratio - 1).abs() > a.clip).float().mean().double()
                            acc[6] += share * kl_t.detach().double()
                            if temp_vary:
                                band = torch.bucketize(tr, temp_bands_dev)
                                clipped = ((ratio - 1).abs() > a.clip).float()
                                # (rows, bands); a comparison, not F.one_hot (its range check is a sync)
                                oh = (band[:, None] == band_ids[None, :]).float()
                                # per band: rows, clipped, kl, entropy (float32 sums per piece, as before)
                                tb_acc += torch.stack([oh.sum(0), clipped @ oh, (fl[rows] - logp) @ oh,
                                                       ent @ oh], 1).double()
                if bool(nonfinite):
                    abort(f"non-finite loss at iteration {it}")
                if GRADPROBE and not warm:
                    # Adam's per-weight scale 1 / (sqrt(v_hat) + eps), from the state before this step
                    st0 = [popt.state.get(q_, {}) for q_ in gpar]
                    # (a weight Adam has never stepped gets scale 0: it is left out of the preconditioned norms)
                    pre = (torch.cat([((s_["exp_avg_sq"] / (1 - 0.999 ** float(s_["step"]))).sqrt() + 1e-5)
                                      .reciprocal().reshape(-1) if "exp_avg_sq" in s_
                                      else torch.zeros(q_.numel(), device=dev) for s_, q_ in zip(st0, gpar)])
                           if any("exp_avg_sq" in s_ for s_ in st0) else None)
                    tot = [sum(gacc[k_][h_] for k_ in gacc) for h_ in range(2)]
                    vecs = dict(gacc, total=tot)
                    prow = {"iter": it, "step": nc, "rows": len(batch), "kl_teacher": float(kl_t),
                            "ent_coef": ent_c[0], "kl_coef": kl_c[0]}
                    for tag, sc in (("raw", None), ("adam", pre)):
                        if tag == "adam" and sc is None:
                            continue
                        h = {k_: [v_[0] * sc, v_[1] * sc] if sc is not None else v_ for k_, v_ in vecs.items()}
                        full = {k_: (v_[0] + v_[1]) / 2 for k_, v_ in h.items()}
                        for k_ in h:
                            prow[f"{tag}_{k_}_norm"] = float(full[k_].norm())          # mean over the batch
                            ab = float(h[k_][0] @ h[k_][1])                            # signal^2 estimate
                            prow[f"{tag}_{k_}_sig"] = math.copysign(abs(ab) ** 0.5, ab)
                            prow[f"{tag}_{k_}_halfcos"] = ab / max(float(h[k_][0].norm() * h[k_][1].norm()), 1e-30)
                        for k1, k2 in (("pg", "kl"), ("pg", "ent"), ("kl", "ent")):
                            prow[f"{tag}_cos_{k1}_{k2}"] = float(F.cosine_similarity(full[k1], full[k2], 0))
                            # the signal-only cosine: cross-half products drop each term's own noise
                            x12 = float(h[k1][0] @ h[k2][1] + h[k1][1] @ h[k2][0]) / 2
                            d12 = (abs(float(h[k1][0] @ h[k1][1])) * abs(float(h[k2][0] @ h[k2][1]))) ** 0.5
                            prow[f"{tag}_sigcos_{k1}_{k2}"] = x12 / max(d12, 1e-30)
                    with open(GRADPROBE, "a") as f_:
                        f_.write(json.dumps(prow) + "\n")
                    if os.environ.get("FF_GRADPROBE_VEC"):    # the full per-step vectors, for cross-step products
                        vd = pathlib.Path(os.environ["FF_GRADPROBE_VEC"])
                        vd.mkdir(parents=True, exist_ok=True)
                        torch.save({k_: ((v_[0] + v_[1]) / 2).half().cpu() for k_, v_ in gacc.items() if k_ != "ent"}
                                   | ({"pre": pre.half().cpu()} if pre is not None else {})
                                   | {"iter": it, "step": nc, "rows": len(batch)},
                                   vd / f"g_{it:03d}_{nc:03d}.pt")
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_([p_ for p_ in vnet.parameters() if id(p_) not in wl_ids], 1.0)
                if wl_on:
                    torch.nn.utils.clip_grad_norm_(wl_params, 1.0)
                if warm:
                    for q_ in policy.parameters():
                        q_.grad = None                  # the head only
                popt.step()
                nc += 1
                nb += 0 if warm else 1
        acc_h, tb_h = acc.tolist(), tb_acc.tolist()
        c_loss, w_loss = acc_h[0], acc_h[1]
        for k, v_ in zip(("pg", "ent", "kl", "clipfrac", "kl_teacher"), acc_h[2:]):
            st[k] = v_
        for k_, d_ in enumerate(tb_h):
            if d_[0] > 0:
                tb[k_] = [int(round(d_[0])), d_[1], d_[2], d_[3]]
        for k in st:
            st[k] /= max(nb, 1)
        c_loss /= max(nc, 1)
        w_loss /= max(nc, 1)
        if not all(torch.isfinite(q).all() for q in policy.parameters()):
            abort(f"non-finite parameters at iteration {it}")
        if not all(torch.isfinite(q).all() for q in vnet.parameters()):
            abort(f"non-finite value-network parameters at iteration {it}")
        t_opt = time.perf_counter() - t0

        cand = done0 + (it + 1) * turns_per_iter
        row = {"gen": a.gen, "segment": a.segment, "iter": it,
               "total_turns": a.turn_base + (it + 1) * turns_per_iter, "cand_turns": cand, "lr": lr_now,
               "critic_warmup": warm, "explore": a.explore, "sps": turns_per_iter / (t_roll + t_gae + t_opt),
               "t_roll": round(t_roll, 2), "t_crit": round(t_gae, 2), "t_opt": round(t_opt, 2),
               "usable": round(n_use / max(len(rows_l), 1), 3), "explained_var": round(ev, 4),
               "critic_mse": round(c_loss, 5), "return_mean": float(ret[u_ix].mean()),
               "value_mean": float(v_all[u_ix].mean()), "reward_mean": float(rw[u_ix].mean()),
               "ret_scale": round(float(vscale()), 5),
               "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "ratio0": None if ratio0 is None else round(ratio0, 5),
               **({"temp_hi": round(cur_hi[0], 4)} if temp_vary else {}), "ent_coef": round(ent_c[0], 6), "kl_coef": round(kl_c[0], 6),
               **temp_summary(),
               **{f"tb{k_}_{name}": round(v_ / d_[0], 5) if name != "n" else d_[0]
                  for k_, d_ in sorted(tb.items()) for name, v_ in zip(("n", "clip", "kl", "ent"), d_)},
               "critic_rows": cv_rows_used,
               **({"cv_kept": cv_n[0], "cv_dropped": cv_n[1]} if use_cv else {}),
               "pool_exhausted": lpool.exhausted, "elapsed": round(time.perf_counter() - t_start, 1),
               **{k: round(v, 5) for k, v in st.items()}}
        if games:
            g = np.array(games)
            for k, name in enumerate(opp_names):
                m = g[:, 0] == k + 1
                if m.sum() >= 30:
                    row[f"vs_{name}"] = round(float(g[m, 1].mean()), 4)
                    row[f"n_{name}"] = int(m.sum())
        if wl_on:
            # the games that ended this iteration, scored per team: the head's and phi's AUC (decisive games)
            # and the head's Brier score, at each dash round the game reached
            if wl_stage:
                snaps = torch.stack([s_ for s_, _ in wl_stage]).cpu().numpy()        # (g, team, head/phi, K)
                for (_, ww_), sn in zip(wl_stage, snaps):
                    for x in (0, 1):
                        if not np.isnan(sn[x, 0]).all():
                            wl_games.append((sn[x, 0], sn[x, 1], 0.0 if ww_ < 0 else (1.0 if ww_ == x else -1.0)))
                wl_stage.clear()
            row.update(blend_row)
            row.update(wl_ev=round(wl_ev, 4), wl_mse=round(w_loss, 5), wl_mean=round(float(vw[u_ix].mean()), 4),
                       wl_absmean=round(float(np.abs(vw[u_ix]).mean()), 4), wl_games=len(wl_games))
            if len(wl_games) >= 100:
                P_ = np.array([g_[0] for g_ in wl_games])
                H_ = np.array([g_[1] for g_ in wl_games])
                Z_ = np.array([g_[2] for g_ in wl_games])
                for k_, K_ in enumerate(WL_ROUNDS):
                    m_ = ~np.isnan(P_[:, k_])
                    d_ = m_ & (Z_ != 0)
                    if d_.sum() >= 50 and 0 < (Z_[d_] > 0).sum() < d_.sum():
                        row[f"wl_auc_r{K_}"] = round(auc(P_[d_, k_], Z_[d_] > 0), 4)
                        row[f"phi_auc_r{K_}"] = round(auc(H_[d_, k_], Z_[d_] > 0), 4)
                        # Brier on P(win) = (pred + 1) / 2 against (z + 1) / 2 (a draw is 0.5)
                        row[f"wl_brier_r{K_}"] = round(float(np.mean(((P_[m_, k_] - Z_[m_]) / 2) ** 2)), 4)
                        row[f"wl_n_r{K_}"] = int(m_.sum())
        if len(ends) >= 100:
            en = np.array(ends)
            for k_, name in enumerate(END_KINDS):
                row[f"end_{name}"] = round(float((en[:, 0] == k_).mean()), 4)
            row["end_agree"] = round(float(en[:, 1].mean()), 4)   # the classification's verdict == the real winner
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()
        if it % 5 == 0 or it == n_iters - 1:
            wl_txt = f" wl_ev {wl_ev:.3f} b {wl_b[0]:.3f} corr {blend_row['adv_corr']:.2f}" if wl_on else ""
            vs = " ".join(f"{k[3:]} {v:.2f}" for k, v in row.items() if k.startswith("vs_"))
            print(f"g{a.gen}.{a.segment} it {it:4d}/{n_iters} | {cand / 1e6:6.1f}M | {row['sps']:>7,.0f} t/s "
                  f"(roll {t_roll:.1f}s gae {t_gae:.1f}s opt {t_opt:.1f}s) | ev {ev:.3f} cmse {c_loss:.4f}"
                  f"{wl_txt}"
                  f"{' WARMUP' if warm else ''} | ent {st['ent']:.3f} klT {st['kl_teacher']:.4f} "
                  f"clip {st['clipfrac']:.3f} | {vs}", flush=True)
        if not warm:
            if kl_coef > 0 and st["kl_teacher"] > a.max_kl:
                abort(f"KL to teacher {st['kl_teacher']:.4f} > {a.max_kl}")
            if st["ent"] < a.min_ent:
                abort(f"entropy {st['ent']:.4f} < {a.min_ent}")
        if it == PROF_IT:
            torch.cuda.synchronize()
            _prof.__exit__(None, None, None)
            ka = _prof.key_averages()
            print(ka.table(sort_by="self_cpu_time_total", row_limit=45), flush=True)
            print(ka.table(sort_by="self_cuda_time_total", row_limit=30), flush=True)
            _prof.export_chrome_trace(str(out / "trace.json"))
            print(f"PROF trace -> {out / 'trace.json'}", flush=True)
        if it % 10 == 9:
            save(out / "latest.pt", it, cand)
    save(out / "final.pt", n_iters - 1, done0 + n_iters * turns_per_iter)
    (out / "latest.pt").unlink(missing_ok=True)
    print(f"segment done in {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
