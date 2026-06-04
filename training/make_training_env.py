## training/make_training_env.py
# --------------------------------------------------------------
# Creates a headless PettingZoo-compatible parallel environment
# for MARL training.
#
# This module instantiates TAP
# environment via GameManager and wraps it with PettingZoo’s
# `aec_to_parallel` utility.
#
# Sources for implementation:
# - PettingZoo documentation – AEC API:
# https://pettingzoo.farama.org/api/aec/
# - PettingZoo documentation – Parallel API:
# https://pettingzoo.farama.org/api/parallel/
# --------------------------------------------------------------

from pettingzoo.utils.conversions import aec_to_parallel
from controller.game_manager import GameManager
from marl_tap_mdp import MARL_TAPMDP
from team.score import Score
from maps.map import Map
from config import Config

def make_training_env():
    """
    Instantiates the TAP game logic and wraps it in a PettingZoo parallel environment for MARL training.

    Returns:
        parallel_env (ParallelEnv): A PettingZoo-compatible environment.
    """
    # Configure grid size for training the smaller 11x15 map.
    rows, cols = 11, 15
    # Build dependencies for the GameManager
    score_manager = Score()
    game_map = Map(rows, cols)      # Create a Map
    # Instantiate GameManager headless (no UI), with placeholder team keys
    game_manager = GameManager(
        score=score_manager,
        turn_label=None,
        score_label=None,
        status_panel=None,
        log_panel=None,
        team_by_side={"drop_p1": None, "drop_p2": None},  # placeholder team keys
        dice_panel=None,
        game_map=game_map,                                # add the Map
        calculate_dimensions_func=None
    )
    # MARL_TAPMDP implements the AEC (agent-environment cycle) API
    aec_env = MARL_TAPMDP(game_manager, grid_size=(rows, cols))
    # GameManager sometimes needs direct access to SARL env method get_obs
    game_manager.env = aec_env.env   # link GameManager back to inner SARL_TAPMDP
    # Convert AEC to Parallel for training loops that expect dict step()
    # Many MARL libraries expect dict-based step() using the Parallel API
    parallel_env = aec_to_parallel(aec_env)

    return parallel_env
