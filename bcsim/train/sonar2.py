"""Sonar v2 on the python side (cpp/bc_sonar2.hpp is the definition).

`intents_from_probs` turns a policy's action distribution into the four numbers the
packet carries: P(split), P(sprint), P(first step left), P(first step right), which
the caller writes into `env.intent` before `env.step`. `intents_from_turn` does the
same for a replayed move (certain: one-hot), for cloning a team whose own sonar we
ignore. `decode` is the codec in python, for tests and analysis.

Codec ids (bc_vec.hpp decode_for): 0-2 one step, 3-11 two, 12-38 three, 39-47 split.
The first step's turn is the lowest base-3 digit: 0 straight, 1 left, 2 right.
"""

from __future__ import annotations

import numpy as np

N_ACTIONS = 48
_first = np.zeros(N_ACTIONS, np.int64)
for _a in range(39):
    _first[_a] = (_a if _a < 3 else (_a - 3 if _a < 12 else _a - 12)) % 3
SPLIT = np.zeros(N_ACTIONS, np.float32); SPLIT[39:] = 1
SPRINT = np.zeros(N_ACTIONS, np.float32); SPRINT[3:39] = 1
LEFT = np.zeros(N_ACTIONS, np.float32); LEFT[:39] = _first[:39] == 1
RIGHT = np.zeros(N_ACTIONS, np.float32); RIGHT[:39] = _first[:39] == 2
_M = np.stack([SPLIT, SPRINT, LEFT, RIGHT], 1)          # (48, 4)


_M49 = np.concatenate([_M, np.zeros((1, 4), np.float32)])   # the self-kill (id 48) counts as none
# BC_XSPLIT builds: ids 49, 50 are splits (k = len - 2, len - 3)
_M51 = np.concatenate([_M49, np.array([[1, 0, 0, 0], [1, 0, 0, 0]], np.float32)])
# BC_FARSPRINT builds: ids 51-74 are far sprints (2026-10-02; that build's packets carry no intents,
# BC_PORTALREP, but the evaluation still computes them): a sprint, first step unknown here
_M75 = np.concatenate([_M51, np.tile(np.array([[0, 1, 0, 0]], np.float32), (24, 1))])
_MS = {48: _M, 49: _M49, 51: _M51, 75: _M75}


def intent_matrix(width: int) -> np.ndarray:
    """(width, 4): the P(split, sprint, left, right) each action id contributes."""
    return _MS[width]


def intents_from_probs(probs) -> np.ndarray:
    """(n, 48, 49, 51 or 75) action probabilities (numpy or torch) -> (n, 4) float32."""
    M = _MS[probs.shape[-1]]
    try:
        import torch
        if isinstance(probs, torch.Tensor):
            # outside autocast: a bf16 product would come back as bf16, which numpy cannot read
            with torch.autocast(probs.device.type, enabled=False):
                return (probs.float() @ torch.as_tensor(M, device=probs.device)).float().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(probs, np.float32) @ M


def intents_from_actions(ids: np.ndarray) -> np.ndarray:
    """Codec action ids -> one-hot intents, (n, 4)."""
    return _M51[np.clip(np.asarray(ids, np.int64), 0, 50)].copy()


def intent_from_turn(kind: int, dirs, facing: int) -> tuple[float, float, float, float]:
    """A replayed move, which may be outside the codec (sprints of 4+ steps)."""
    if kind == 1:
        return (1.0, 0.0, 0.0, 0.0)
    if kind != 0 or not len(dirs):
        return (0.0, 0.0, 0.0, 0.0)
    d0 = int(dirs[0])
    left = float(d0 == (facing + 3) % 4)
    right = float(d0 == (facing + 1) % 4)
    return (0.0, float(len(dirs) >= 2), left, right)


# ---------------------------------------------------------------- codec, python
KEY = 0xC2B2AE3D27D4EB4F
M64 = (1 << 64) - 1


def _mix(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def tag(team: int, rnd: int, low56: int) -> int:
    k = KEY ^ ((team & 1) << 63) ^ ((rnd & 0xFFFFFFFF) << 20)
    return _mix(k ^ _mix(low56 & 0x00FFFFFFFFFFFFFF)) >> 56


def decode(v: int, team: int, rnd: int, portal: bool = False) -> dict | None:
    """The packet, or None unless its tag checks for `team` at `rnd` or `rnd - 1`. Layout as
    cpp/bc_sonar2.hpp (2026-10-01: bit 0 = the sender is the queen, bit 20 = the enemy named
    is the queen, P(split) 5 bits at 21, P(sprint) 4 bits at 16, the queen sighting's age at 14,
    P(left) / P(right) 4 bits at 8 / 4, a relayed report's came-from direction at 2). portal=True:
    the BC_PORTALREP layout, whose portal report replaces the probabilities and the facing."""
    low = v & 0x00FFFFFFFFFFFFFF
    sent = next((r for r in (rnd, rnd - 1) if r >= 0 and tag(team, r, low) == v >> 56), None)
    if sent is None:
        return None
    enemy = bool(low & 2)
    if portal:
        # BC_PORTALREP (2026-10-02): the probability and facing bits carry the portal report
        q = {"round": sent, "hx": low >> 50 & 63, "hy": low >> 44 & 63, "len": low >> 38 & 63,
             "queen": bool(low & 1), "enemy": enemy, "enemy_queen": enemy and bool(low >> 20 & 1),
             "queen_age": (low >> 14 & 3) if enemy and (low >> 20 & 1) else 0,
             "came_from": (low >> 2 & 3) if enemy and (low >> 20 & 1) else 0,
             "ex": low >> 32 & 63, "ey": low >> 26 & 63, "portal": bool(low >> 13 & 1)}
        if q["portal"]:
            q.update(px=low >> 4 & 63, py=(low >> 12 & 1) << 5 | (low >> 21 & 31), pdir=low >> 10 & 3,
                     pbox=low >> 16 & 3, ppearls=low >> 18 & 3)
        return q
    return {"round": sent, "hx": low >> 50 & 63, "hy": low >> 44 & 63, "len": low >> 38 & 63,
            "queen": bool(low & 1), "enemy": enemy, "enemy_queen": enemy and bool(low >> 20 & 1),
            "queen_age": (low >> 14 & 3) if enemy and (low >> 20 & 1) else 0,
            "came_from": (low >> 2 & 3) if enemy and (low >> 20 & 1) else 0,
            "ex": low >> 32 & 63, "ey": low >> 26 & 63,
            "p_split": (low >> 21 & 31) / 31, "p_sprint": (low >> 16 & 15) / 15,
            "facing": low >> 12 & 3, "p_left": (low >> 8 & 15) / 15, "p_right": (low >> 4 & 15) / 15}
