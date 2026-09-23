"""Fast, engine-exact Battlecode simulator with a PyTorch-facing PPO interface."""

from .env import (BattlecodeVecEnv, Closures, EpisodeStats, Observation,
                  CHANNELS, SCALARS, REWARD_COMPS, DEFAULT_REWARD_WEIGHTS,
                  N_ACTIONS, N_CHANNELS, N_SCALARS, N_REWARD_COMPS, WINDOW, MAX_MSGS,
                  load_maps, reward_vector, BOTS, EP_COLS, WIDE_CH, WIDE_SIDE,
                  BOARD_CH, BOARD_MAX)

__all__ = ["BattlecodeVecEnv", "Closures", "EpisodeStats", "Observation",
           "CHANNELS", "SCALARS", "REWARD_COMPS", "DEFAULT_REWARD_WEIGHTS",
           "N_ACTIONS", "N_CHANNELS", "N_SCALARS", "N_REWARD_COMPS", "WINDOW", "MAX_MSGS",
           "load_maps", "reward_vector", "BOTS", "EP_COLS", "WIDE_CH", "WIDE_SIDE",
           "BOARD_CH", "BOARD_MAX"]
