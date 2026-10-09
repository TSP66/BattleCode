"""Fast, engine-exact Battlecode simulator with a PyTorch-facing PPO interface."""

from .env import (BattlecodeVecEnv, Closures, EpisodeStats, Observation,
                  CHANNELS, SCALARS, REWARD_COMPS, DEFAULT_REWARD_WEIGHTS,
                  N_ACTIONS, N_CHANNELS, N_SCALARS, N_REWARD_COMPS, WINDOW, MAX_MSGS,
                  SONAR_DIRS, NUM_MSGS_CAP,
                  PRIV_COUNT, PRIV_BASE, N_PHI_TERMS, PHI_COMPS,
                  load_maps, reward_vector, BOTS, EP_COLS, WIDE_CH, WIDE_SIDE,
                  BOARD_CH, BOARD_MAX, GRID_CH, GRID_SIDE, PORTAL_BUILD, cview_layout)

__all__ = ["BattlecodeVecEnv", "Closures", "EpisodeStats", "Observation",
           "CHANNELS", "SCALARS", "REWARD_COMPS", "DEFAULT_REWARD_WEIGHTS",
           "N_ACTIONS", "N_CHANNELS", "N_SCALARS", "N_REWARD_COMPS", "WINDOW", "MAX_MSGS",
           "SONAR_DIRS", "NUM_MSGS_CAP",
           "PRIV_COUNT", "PRIV_BASE", "N_PHI_TERMS", "PHI_COMPS",
           "load_maps", "reward_vector", "BOTS", "EP_COLS", "WIDE_CH", "WIDE_SIDE",
           "BOARD_CH", "BOARD_MAX", "GRID_CH", "GRID_SIDE", "PORTAL_BUILD", "cview_layout"]
