"""Turns the env's interleaved agent transitions into PPO batches.

Agents act one at a time and are born and killed mid-episode, so a reward does
not arrive on the step that earned it. The env hands back `closures`: a reward
and a done flag tagged with the agent (`uid`) whose earlier action they belong
to. This module matches each closure to the slot where that action was
recorded, links each agent's slots into a chain, and runs GAE along the chain.

Everything is vectorised: a python dict keyed by agent would cost more than the
simulator itself.
"""

from __future__ import annotations

import numpy as np
import torch

ID_BITS = 12           # uid = (episode << 12) | dragon_id, as bc_vec.hpp builds it
ID_MASK = (1 << ID_BITS) - 1
MAX_IDS = 1 << ID_BITS


class Rollout:
    def __init__(self, steps: int, num_envs: int, n_channels: int, window: int,
                 n_scalars: int, n_actions: int, device: torch.device,
                 store_dtype: torch.dtype = torch.float16):
        self.T, self.N = steps, num_envs
        self.device = device
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)
        self.local = z(steps, num_envs, n_channels, window, window, dt=store_dtype)
        self.scalar = z(steps, num_envs, n_scalars, dt=store_dtype)
        self.mask = torch.zeros(steps, num_envs, n_actions, dtype=torch.bool, device=device)
        self.action = z(steps, num_envs, dt=torch.int64)
        self.logp = z(steps, num_envs)
        self.value = z(steps, num_envs)
        self.reward = z(steps, num_envs)
        self.done = z(steps, num_envs, dt=torch.bool)
        self.closed = z(steps, num_envs, dt=torch.bool)
        self.nxt = torch.full((steps, num_envs), -1, dtype=torch.int32, device=device)

        # CPU-side bookkeeping: which slot is each agent's open transition in
        self.slot = np.full((num_envs, MAX_IDS), -1, dtype=np.int32)
        self.slot_uid = np.zeros((num_envs, MAX_IDS), dtype=np.int64)
        self.envs = np.arange(num_envs, dtype=np.int64)
        self.nxt_cpu = np.full((steps, num_envs), -1, dtype=np.int32)
        self.reward_cpu = np.zeros((steps, num_envs), dtype=np.float32)
        self.done_cpu = np.zeros((steps, num_envs), dtype=np.bool_)
        self.closed_cpu = np.zeros((steps, num_envs), dtype=np.bool_)
        self.t = 0
        self.orphans = 0          # closures for agents that never acted
        self.overwrites = 0       # a slot reused before it closed: must stay 0

        # pinned staging, so the host copy overlaps the GPU instead of blocking
        pin = lambda *s, dt: torch.zeros(*s, dtype=dt, pin_memory=True)
        self._p_local = pin(num_envs, n_channels, window, window, dt=torch.float32)
        self._p_scalar = pin(num_envs, n_scalars, dt=torch.float32)
        self._p_mask = pin(num_envs, n_actions, dt=torch.uint8)
        self._p_local_np = self._p_local.numpy()
        self._p_scalar_np = self._p_scalar.numpy()
        self._p_mask_np = self._p_mask.numpy()

    def begin(self) -> None:
        self.t = 0
        self.slot.fill(-1)
        self.nxt_cpu.fill(-1)
        self.reward_cpu.fill(0.0)
        self.done_cpu.fill(False)
        self.closed_cpu.fill(False)
        self.orphans = 0
        self.overwrites = 0
        self.comp_sum = np.zeros(self.reward_cpu.shape[0:0] + (0,), np.float64)
        self.n_closed = 0
        self.comp_total: np.ndarray | None = None

    def stage(self, obs):
        """Copies the env's reusable buffers into pinned memory and uploads.

        The env overwrites `obs` in place on the next step, so this copy is not
        optional.
        """
        np.copyto(self._p_local_np, obs.local)
        np.copyto(self._p_scalar_np, obs.scalar)
        np.copyto(self._p_mask_np, obs.mask)
        local = self._p_local.to(self.device, non_blocking=True)
        scalar = self._p_scalar.to(self.device, non_blocking=True)
        mask = self._p_mask.to(self.device, non_blocking=True).bool()
        return local, scalar, mask

    def record(self, t: int, staged, obs, action, logp, value, learn=None) -> None:
        """Stores the observation acted on and the action taken, at slot t.

        `learn` (bool per env) marks the rows that train; the others (a frozen
        opponent's turns) are stored but never linked, so they never close and
        never enter a batch, and their closures count as orphans.
        """
        local, scalar, mask = staged
        self.local[t].copy_(local, non_blocking=True)
        self.scalar[t].copy_(scalar, non_blocking=True)
        self.mask[t].copy_(mask, non_blocking=True)
        self.action[t] = action
        self.logp[t] = logp
        self.value[t] = value

        envs = self.envs if learn is None else self.envs[learn]
        uid = obs.uid if learn is None else obs.uid[learn]
        ids = (uid & ID_MASK).astype(np.int64)
        prev = self.slot[envs, ids]
        live = prev >= 0
        if live.any():
            # the previous transition of this agent must already have closed
            unclosed = live & ~self.closed_cpu[prev.clip(0), envs]
            self.overwrites += int(unclosed.sum())
            self.nxt_cpu[prev[live], envs[live]] = t
        self.slot[envs, ids] = t
        self.slot_uid[envs, ids] = uid

    def close(self, closures, weights: np.ndarray) -> None:
        """Applies finished transitions: reward into their slot, done flag set."""
        if len(closures.uid) == 0:
            return
        env = closures.env.astype(np.int64)
        ids = (closures.uid & ID_MASK).astype(np.int64)
        slot = self.slot[env, ids]
        ok = (slot >= 0) & (self.slot_uid[env, ids] == closures.uid)
        self.orphans += int((~ok).sum())
        if not ok.any():
            return
        slot, env = slot[ok], env[ok]
        comps = closures.comps[ok]
        # raw component rates, so pearls/turn and deaths/turn stay readable
        # whatever the weights are set to
        if self.comp_total is None:
            self.comp_total = comps.sum(axis=0, dtype=np.float64)
        else:
            self.comp_total += comps.sum(axis=0, dtype=np.float64)
        self.n_closed += int(ok.sum())
        rew = (comps @ weights).astype(np.float32)
        # several closures can land on one slot only if an agent acted twice
        # without closing, which `overwrites` would have caught; add anyway
        np.add.at(self.reward_cpu, (slot, env), rew)
        self.done_cpu[slot, env] |= closures.done[ok].astype(bool)
        self.closed_cpu[slot, env] = True

    def finish(self, gamma: float, lam: float):
        """Uploads bookkeeping and runs GAE along each agent's chain of slots.

        A transition trains only if it closed and either ended the agent's
        episode or has a successor to bootstrap from.
        """
        self.reward.copy_(torch.from_numpy(self.reward_cpu))
        self.done.copy_(torch.from_numpy(self.done_cpu))
        self.closed.copy_(torch.from_numpy(self.closed_cpu))
        self.nxt.copy_(torch.from_numpy(self.nxt_cpu))

        T, N = self.T, self.N
        adv = torch.zeros(T, N, device=self.device)
        nxt = self.nxt.long()
        has_next = nxt >= 0
        cont = (~self.done) & has_next          # bootstrap from the next slot
        valid = self.closed & (self.done | has_next)

        idx = torch.arange(N, device=self.device)
        for t in range(T - 1, -1, -1):
            nx = nxt[t]
            safe = nx.clamp(min=0)
            v_next = torch.where(cont[t], self.value[safe, idx], torch.zeros_like(adv[t]))
            a_next = torch.where(cont[t], adv[safe, idx], torch.zeros_like(adv[t]))
            delta = self.reward[t] + gamma * v_next - self.value[t]
            adv[t] = delta + gamma * lam * a_next
        adv = torch.where(valid, adv, torch.zeros_like(adv))
        ret = adv + self.value
        return adv, ret, valid

    def flat_batch(self, adv, ret, valid):
        sel = valid.reshape(-1).nonzero(as_tuple=True)[0]
        flat = lambda x: x.reshape(self.T * self.N, *x.shape[2:])
        return {
            "local": flat(self.local)[sel],
            "scalar": flat(self.scalar)[sel],
            "mask": flat(self.mask)[sel],
            "action": flat(self.action)[sel],
            "logp": flat(self.logp)[sel],
            "value": flat(self.value)[sel],
            "adv": adv.reshape(-1)[sel],
            "ret": ret.reshape(-1)[sel],
        }
