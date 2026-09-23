"""Batched Battlecode environment for PyTorch.

One step advances every game by a single dragon turn, which is what the real
engine does: dragons act one at a time, in id order, each seeing the board as
the dragons before it left it.

Because agents act in turn and are born and killed mid-episode, a step does not
hand back a neat (reward, done) row per env. Instead it returns:

  * `obs`  - what the dragon about to act can see, one row per env;
  * `closures` - transitions that just finished, each naming the agent (`uid`)
    whose earlier observation it belongs to, its reward components and whether
    its episode ended.

`RolloutBuffer` stitches those together into PPO batches.
"""

from __future__ import annotations

import ctypes
import pathlib
from dataclasses import dataclass, field

import numpy as np

# BCSIM_LIB points at another build (tests use it for instrumented ones)
_LIB_PATH = pathlib.Path(__import__("os").environ.get(
    "BCSIM_LIB", pathlib.Path(__file__).resolve().parent / "libbcvec.so"))
_lib = ctypes.CDLL(str(_LIB_PATH))

_lib.bcv_create.restype = ctypes.c_void_p
_lib.bcv_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                            ctypes.c_int, ctypes.c_int, ctypes.c_ulonglong,
                            ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.c_char_p, ctypes.c_int]
_lib.bcv_destroy.argtypes = [ctypes.c_void_p]
_lib.bcv_bind.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 8
_lib.bcv_reset.argtypes = [ctypes.c_void_p]
_lib.bcv_step.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 6
_lib.bcv_step_codec.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 3
_lib.bcv_closures.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 4 + [ctypes.c_int]
_lib.bcv_episodes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_lib.bcv_layout.argtypes = [ctypes.c_void_p]
_lib.bcv_set_map_weights.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.bcv_set_env_opponent.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 4
_lib.bcv_set_potential_gamma.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.bcv_ep_cols.restype = ctypes.c_int
_lib.bcv_bot_count.restype = ctypes.c_int
_lib.bcv_bot_name.restype = ctypes.c_char_p
_lib.bcv_bot_name.argtypes = [ctypes.c_int]

_layout = (ctypes.c_int * 8)()
_lib.bcv_layout(_layout)
N_CHANNELS, WINDOW, N_SCALARS, MAX_MSGS, N_ACTIONS, N_REWARD_COMPS, MAX_STEPS, N_MOVES = _layout
EP_COLS = _lib.bcv_ep_cols()
# scripted opponents for evaluation, see cpp/bc_bots.hpp
BOTS = [_lib.bcv_bot_name(i).decode() for i in range(_lib.bcv_bot_count())]

CHANNELS = ["pearl", "pearl_time", "never_spawn", "self_head", "self_body",
            "ally_head", "ally_body", "enemy_head", "enemy_body",
            "face_fwd", "face_right", "face_back", "face_left",
            "kelp_fwd", "kelp_right", "kelp_back", "kelp_left",
            "portal_fwd", "portal_right", "portal_back", "portal_left",
            "self_index", "self_tail"]
# The BASE scalars, indices 0-13. N_SCALARS above is the full row the env
# writes: these, then mem (676) and memfar (18) from cpp/bc_memory.hpp. The base
# ones never move, so a net trained before memory widens with zero columns and
# plays identically (train/migrate_scalars.py).
SCALARS = ["round", "length", "length_raw", "units", "face_n", "face_e", "face_s", "face_w",
           "head_x", "head_y", "map_w", "map_h", "num_msgs", "team_b"]
REWARD_COMPS = ["length_delta", "pearls", "sprint_cost", "split_cost", "died",
                "kills", "enemy_deaths", "ally_deaths", "win", "lose", "draw",
                "final_length", "splits",
                "team_len", "team_max", "foe_len", "foe_max",
                "portal", "eliminated", "team_units", "foe_units"]
assert len(REWARD_COMPS) == N_REWARD_COMPS, "REWARD_COMPS is out of step with RW_COUNT"

# A sensible starting point: grow, stay alive, win. Override per experiment.
DEFAULT_REWARD_WEIGHTS = {
    "length_delta": 0.10,
    "pearls": 0.20,
    "died": -1.0,
    "kills": 0.5,
    "win": 5.0,
    "lose": -5.0,
    "final_length": 0.05,
}


def reward_vector(weights: dict[str, float] | None = None) -> np.ndarray:
    w = dict(DEFAULT_REWARD_WEIGHTS if weights is None else weights)
    out = np.zeros(N_REWARD_COMPS, dtype=np.float32)
    for name, value in w.items():
        if name not in REWARD_COMPS:
            raise KeyError(f"unknown reward component {name!r}; have {REWARD_COMPS}")
        out[REWARD_COMPS.index(name)] = value
    return out


@dataclass
class Observation:
    local: np.ndarray      # (num_envs, C, 7, 7) float32
    scalar: np.ndarray     # (num_envs, S) float32
    msgs: np.ndarray       # (num_envs, MAX_MSGS) uint32, raw sonar values
    mask: np.ndarray       # (num_envs, N_ACTIONS) uint8, 1 = allowed
    uid: np.ndarray        # (num_envs,) int64, identifies the acting agent
    dragon_id: np.ndarray  # (num_envs,) int32
    team: np.ndarray       # (num_envs,) int8
    round: np.ndarray      # (num_envs,) int32
    # (num_envs, PRIV_COUNT) float32 privileged global features, only with
    # BattlecodeVecEnv(privileged=True); for critics that never ship
    priv: np.ndarray | None = None


@dataclass
class Closures:
    """Transitions that ended during the last step."""
    env: np.ndarray        # (n,) int32
    uid: np.ndarray        # (n,) int64
    comps: np.ndarray      # (n, N_REWARD_COMPS) float32
    done: np.ndarray       # (n,) int8, 1 when that agent's episode is over

    def reward(self, weights: np.ndarray) -> np.ndarray:
        return self.comps @ weights


@dataclass
class EpisodeStats:
    rows: np.ndarray = field(default_factory=lambda: np.zeros((0, EP_COLS), np.int32))

    COLUMNS = ["env", "rounds", "winner", "a_longest", "b_longest", "a_units", "b_units", "map",
               "a_deaths", "b_deaths", "a_kills", "b_kills",
               "a_len_lost", "b_len_lost", "a_len_killed", "b_len_killed",
               "a_headon", "b_headon"]

    def as_dicts(self) -> list[dict]:
        return [dict(zip(self.COLUMNS, row.tolist())) for row in self.rows]


class BattlecodeVecEnv:
    """Many Battlecode games, stepped one dragon turn at a time."""

    def __init__(self, maps: list[str], num_envs: int = 64, num_threads: int = 8,
                 seed: int = 0, egocentric: bool = True, random_pearl_seed: bool = True,
                 max_rounds: int = 500, closure_capacity: int | None = None,
                 privileged: bool = False, board: bool = False):
        if not maps:
            raise ValueError("need at least one map")
        blob = b"".join(m.encode() for m in maps)
        lengths = (ctypes.c_int * len(maps))(*[len(m.encode()) for m in maps])
        err = ctypes.create_string_buffer(512)
        self._h = _lib.bcv_create(blob, lengths, len(maps), num_envs, num_threads, seed,
                                  int(egocentric), int(random_pearl_seed), max_rounds,
                                  err, 512)
        if not self._h:
            raise ValueError(err.value.decode())
        self.num_envs = num_envs
        self.num_actions = N_ACTIONS

        self._local = np.zeros((num_envs, N_CHANNELS, WINDOW, WINDOW), np.float32)
        self._scalar = np.zeros((num_envs, N_SCALARS), np.float32)
        self._msgs = np.zeros((num_envs, MAX_MSGS), np.uint32)
        self._mask = np.zeros((num_envs, N_ACTIONS), np.uint8)
        self._uid = np.zeros(num_envs, np.int64)
        self._dragon = np.zeros(num_envs, np.int32)
        self._team = np.zeros(num_envs, np.int8)
        self._round = np.zeros(num_envs, np.int32)
        _lib.bcv_bind(ctypes.c_void_p(self._h), *[a.ctypes.data for a in
                      (self._local, self._scalar, self._msgs, self._mask,
                       self._uid, self._dragon, self._team, self._round)])
        self._priv = None
        if privileged:
            # only newer builds have it; a plain run never looks the symbol up
            if not hasattr(_lib, "bcv_bind_priv"):
                raise RuntimeError(f"{_LIB_PATH.name} has no privileged features; "
                                   "build libbcvec_priv.so and set BCSIM_LIB")
            _lib.bcv_priv_count.restype = ctypes.c_int
            _lib.bcv_bind_priv.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self._priv = np.zeros((num_envs, _lib.bcv_priv_count()), np.float32)
            _lib.bcv_bind_priv(ctypes.c_void_p(self._h), self._priv.ctypes.data)
        self.board = None
        if board:
            # (num_envs, BOARD_CH, BOARD_MAX, BOARD_MAX) uint8, the map in the
            # top-left corner, relative to the acting dragon's team
            if not hasattr(_lib, "bcv_bind_board"):
                raise RuntimeError(f"{_LIB_PATH.name} has no board export; "
                                   "build libbcvec_priv.so and set BCSIM_LIB")
            shape = (ctypes.c_int * 2)()
            _lib.bcv_board_shape(shape)
            _lib.bcv_bind_board.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.board = np.zeros((num_envs, shape[0], shape[1], shape[1]), np.uint8)
            _lib.bcv_bind_board(ctypes.c_void_p(self._h), self.board.ctypes.data)

        cap = closure_capacity or max(1024, num_envs * 140)
        self._cl_env = np.zeros(cap, np.int32)
        self._cl_uid = np.zeros(cap, np.int64)
        self._cl_comps = np.zeros((cap, N_REWARD_COMPS), np.float32)
        self._cl_done = np.zeros(cap, np.int8)
        self._cap = cap
        self._ep = np.zeros((num_envs * 4, EP_COLS), np.int32)
        self.num_maps = len(maps)

        # scratch for structured actions
        self._a_kind = np.zeros(num_envs, np.int8)
        self._a_steps = np.zeros(num_envs, np.int8)
        self._a_dirs = np.zeros((num_envs, MAX_STEPS), np.int8)
        self._a_split = np.zeros(num_envs, np.int16)
        self._a_send = np.zeros(num_envs, np.int8)
        self._a_sonar = np.zeros(num_envs, np.uint32)

    # -------------------------------------------------- lifecycle
    def close(self) -> None:
        if getattr(self, "_h", None):
            _lib.bcv_destroy(ctypes.c_void_p(self._h))
            self._h = None

    def __del__(self):
        self.close()

    def reset(self) -> Observation:
        _lib.bcv_reset(ctypes.c_void_p(self._h))
        return self.observation()

    def observation(self) -> Observation:
        return Observation(self._local, self._scalar, self._msgs, self._mask,
                           self._uid, self._dragon, self._team, self._round, self._priv)

    # -------------------------------------------------- configuration
    def set_map_weights(self, weights) -> None:
        """Relative sampling weight per map, used from each env's next episode."""
        w = np.ascontiguousarray(weights, dtype=np.float64)
        if w.shape != (self.num_maps,):
            raise ValueError(f"expected {self.num_maps} weights, got {w.shape}")
        _lib.bcv_set_map_weights(ctypes.c_void_p(self._h), w.ctypes.data)

    def probe(self, env_index: int) -> tuple[np.ndarray, np.ndarray]:
        """What every codec move would do for env_index's acting dragon, tried
        on a copy of the game: (before, per_move). before = enemy dragons
        alive, the longest enemy's length, our dragons alive; per_move is
        (N_MOVES, PROBE_FIELDS), see VecEnv::probe in cpp/bc_vec.hpp."""
        if not hasattr(_lib, "_probe_ready"):
            _lib.bcv_probe.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            _lib.bcv_probe_fields.restype = ctypes.c_int
            _lib._probe_ready = _lib.bcv_probe_fields()
        f = _lib._probe_ready
        out = np.zeros(3 + N_MOVES * f, np.int32)
        _lib.bcv_probe(ctypes.c_void_p(self._h), env_index, out.ctypes.data)
        return out[:3], out[3:].reshape(N_MOVES, f)

    def last_deaths(self, env_index: int) -> np.ndarray:
        """(n, 4) rows (dragon id, reason, killer id or -1, team) of the
        deaths env_index's last action caused."""
        _lib.bcv_last_deaths.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        _lib.bcv_last_deaths.restype = ctypes.c_int
        out = np.zeros((64, 4), np.int32)
        n = _lib.bcv_last_deaths(ctypes.c_void_p(self._h), env_index, out.ctypes.data, 64)
        return out[:min(n, 64)].copy()

    def set_potential_gamma(self, gamma: float) -> None:
        """Team potentials become gamma * phi(s') - phi(s); 1 = plain differences."""
        _lib.bcv_set_potential_gamma(ctypes.c_void_p(self._h), float(gamma))

    def set_opponent(self, env_index: int, team: int = -1, bot: int | str = 0,
                     map_index: int = -1) -> None:
        """Evaluation: `team` (0 A, 1 B, -1 none) is played by scripted `bot`
        inside the env, so observations only ever come from the other team;
        `map_index` pins the map (-1 samples). Call reset() afterwards."""
        if isinstance(bot, str):
            bot = BOTS.index(bot)
        _lib.bcv_set_env_opponent(ctypes.c_void_p(self._h), env_index, team, bot, map_index)

    # -------------------------------------------------- stepping
    def step(self, action_ids: np.ndarray, send_sonar: np.ndarray | None = None,
             sonar: np.ndarray | None = None) -> tuple[Observation, Closures, EpisodeStats]:
        """Steps with codec action ids (see `codec.py` for what they mean)."""
        ids = np.ascontiguousarray(action_ids, dtype=np.int32)
        if ids.shape != (self.num_envs,):
            raise ValueError(f"expected {self.num_envs} actions, got {ids.shape}")
        send = self._a_send if send_sonar is None else np.ascontiguousarray(send_sonar, np.int8)
        value = self._a_sonar if sonar is None else np.ascontiguousarray(sonar, np.uint32)
        _lib.bcv_step_codec(ctypes.c_void_p(self._h), ids.ctypes.data,
                            send.ctypes.data, value.ctypes.data)
        return self.observation(), self._closures(), self._episodes()

    def step_raw(self, kind: np.ndarray, n_steps: np.ndarray, dirs: np.ndarray,
                 split_k: np.ndarray, send_sonar: np.ndarray | None = None,
                 sonar: np.ndarray | None = None) -> tuple[Observation, Closures, EpisodeStats]:
        """Steps with structured actions, for action spaces of your own design.

        kind: 0 move, 1 split, 2 suicide. dirs holds absolute directions
        (0 N, 1 E, 2 S, 3 W), the first `n_steps` of each row being used.
        """
        np.copyto(self._a_kind, kind)
        np.copyto(self._a_steps, n_steps)
        np.copyto(self._a_dirs, dirs)
        np.copyto(self._a_split, split_k)
        send = self._a_send if send_sonar is None else np.ascontiguousarray(send_sonar, np.int8)
        value = self._a_sonar if sonar is None else np.ascontiguousarray(sonar, np.uint32)
        _lib.bcv_step(ctypes.c_void_p(self._h), self._a_kind.ctypes.data,
                      self._a_steps.ctypes.data, self._a_dirs.ctypes.data,
                      self._a_split.ctypes.data, send.ctypes.data, value.ctypes.data)
        return self.observation(), self._closures(), self._episodes()

    # -------------------------------------------------- results
    def _closures(self) -> Closures:
        n = _lib.bcv_closures(ctypes.c_void_p(self._h), self._cl_env.ctypes.data,
                              self._cl_uid.ctypes.data, self._cl_comps.ctypes.data,
                              self._cl_done.ctypes.data, self._cap)
        if n > self._cap:
            raise RuntimeError(f"closure buffer too small: {n} > {self._cap}; "
                               "pass a larger closure_capacity")
        return Closures(self._cl_env[:n], self._cl_uid[:n], self._cl_comps[:n], self._cl_done[:n])

    def _episodes(self) -> EpisodeStats:
        n = _lib.bcv_episodes(ctypes.c_void_p(self._h), self._ep.ctypes.data, len(self._ep))
        return EpisodeStats(self._ep[:min(n, len(self._ep))])


def load_maps(paths: list[str] | str) -> list[str]:
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for p in paths:
        path = pathlib.Path(p)
        if path.is_dir():
            out += [f.read_text() for f in sorted(path.glob("*.map"))]
        else:
            out.append(path.read_text())
    return out
