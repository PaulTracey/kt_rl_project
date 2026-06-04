# marl_tap_mdp.py
# --------------------------------------------------------------
# MARL RL environment module
#
# PettingZoo-compatible environment for MARL in TAP.
#
# Wraps SARL_TAPMDP into an AECEnv interface with alternating agents,
# "player_0" and "player_1") tracking per-agent rewards, terminations, and observations.
#
# Sources for implementation:
# - PettingZoo AEC API Documentation:
# https://pettingzoo.farama.org/api/aec/
# - PettingZoo Custom Environment Tutorial:
# https://pettingzoo.farama.org/tutorials/gymnasium_conversion/custom_environment/
#
# Modifications in this file:
# - Implements the PettingZoo AEC (Agent-Environment Cycle) API.
# - Manages turn order between two agents using an agent_selector.
# - Tracks per-agent rewards, terminations, and observations.
# - Includes custom logic to buffer end-of-episode signals, ensuring
#   synchronous termination for all agents at the end of a round.
#
# --------------------------------------------------------------
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import numpy as np
from sarl_tap_mdp import SARL_TAPMDP
from loguru import logger

class MARL_TAPMDP(AECEnv):
    """A PettingZoo AEC environment for multi-agent training in TAP."""
    # Metadata required by PettingZoo
    metadata = {
        "render_modes": ["human"],
        "name": "killteam_marl_v0",
        "is_parallelizable": True # Means multiple env instances can run in parallel.
    }

    def __init__(self, game_manager, grid_size=(11, 15), max_cycles=1000):
        """
        Initialise the MARL environment.

        Args:
           game_manager: the shared TAP GameManager.
           grid_size: 11 x 15 or 22 x 30 grid.
           max_cycles: Hard cap on steps per episode.
        """
        super().__init__()
        # Render mode
        self.render_mode = "human"

        # Store external configuration.
        self.game_manager = game_manager
        self.grid_size = grid_size
        self.max_cycles = max_cycles

        # Instantiate SARL_TAPMDP for logic shared by both agents, with PettingZoo managing turn order for IPPO.
        self.env = SARL_TAPMDP(game_manager, grid_size=grid_size)

        # PettingZoo agent
        # Two agents alternate turns in a fixed order through agent_selector.
        self.agents = ["player_0", "player_1"]
        # Master list of all agents. Used to restore `self.agents` on reset.
        self.possible_agents = self.agents[:]
        # Create the turn-management iterator that cycles through active agents.
        self._agent_selector = agent_selector(self.agents)
        # Set the first agent to act in the episode.
        self.agent_selection = self._agent_selector.next()

        # Per-agent data tracking
        # Dictionaries to store state, rewards, and diagnostics for each agent.

        # Stores the last action taken by agents.
        self._actions = {a: None for a in self.agents}
        # Stores the reward each agent from its last action.
        self.rewards = {a: 0.0 for a in self.agents}
        # Tracks the total reward accumulated by each agent during the episode.
        self._cumulative_rewards = {a: 0.0 for a in self.agents}

        self.infos = {a: {} for a in self.agents}

        # Public done flags seen by the wrapper
        self.terminations = {a: False for a in self.agents}
        self.truncations  = {a: False for a in self.agents}
        self.dones        = {a: False for a in self.agents}

        # Buffer done flags until end-of-cycle
        self._pending_term = {a: False for a in self.agents}
        self._pending_trun = {a: False for a in self.agents}
        self._pending_episode_over = False

        # PettingZoo requires separate action and observation spaces for each agent.
        self.steps = 0
        self.action_spaces = {a: self.env.action_space for a in self.agents}
        self.observation_spaces = {a: self.env.observation_space for a in self.agents}

    def reset(self, seed=None, options=None):
        """
        Resets the environment for a new episode.
        """
        self.steps = 0
        # Reset the single-agent game environment.
        self.env.reset()
        # Restore list of active agents for the new episode.
        self.agents = self.possible_agents[:]
        # Reset the agent selector to start the turn order from beginning.
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        # Reset all per-agent tracking dictionaries to their initial state.
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations  = {a: False for a in self.agents}
        self.dones        = {a: False for a in self.agents}
        # Reset the custom end-of-episode signal buffers.
        self._pending_term = {a: False for a in self.agents}
        self._pending_trun = {a: False for a in self.agents}
        self._pending_episode_over = False
        # Return the initial observation for each agent.
        observations = {a: self.observe(a) for a in self.agents}
        return observations, {}

    def observe(self, agent):
        """
        Returns the observation of the agent.
        """
        # Get observation from the underlying single-agent environment.
        obs = self.env.get_observation()
        logger.debug(f"[OBS] {len(obs)} values for agent {agent}: {obs[:20]}...")
        return obs

    def observation_space(self, agent):
        """
        PettingZoo API method: returns the observation space for a single agent.
        """
        return self.observation_spaces[agent]

    def _finalize_cycle_if_needed(self):
        """Applies buffered termination signals at the end of a full agent cycle."""
        if self._pending_episode_over:
            # If the episode is flagged to end, mark all agents as terminated.
            for agent in self.agents:
                self.terminations[agent] = True

            self.agents = []  # Clear the list of active agents for the next episode.
            return True  # Signal that the episode has officially ended.
        else:
            # If the episode is not over, publish any per-agent done flags from the buffer.
            for agent in self.agents:
                self.terminations[agent] = self._pending_term.get(agent, False)
                self.truncations[agent] = self._pending_trun.get(agent, False)

            # Clear the buffers for the next cycle.
            self._pending_term.clear()
            self._pending_trun.clear()
            return False  # Signal that the episode is continuing.

    def step(self, action):
        """
        Processes one step for the currently selected agent.
        """
        agent = self.agent_selection

        # 1) If this agent is already DONE (published at a previous cycle end),
        #    PettingZoo requires _was_dead_step(None).
        if self.dones.get(agent, False):
            if action is not None:
                logger.debug("Ignoring action for terminated agent.")
            self._was_dead_step(None)
            if self._agent_selector.is_last():
                ended = self._finalize_cycle_if_needed()
                # If the episode is still ongoing after the last agent's turn,
                # select the next agent to start the new cycle.
                if not ended:
                    self.agent_selection = self._agent_selector.next()
            else:
                self.agent_selection = self._agent_selector.next()
            return

        # 2) If the episode is ending, take no-operation steps
        #    for the remaining agents to allow the current round to finish.
        if self._pending_episode_over:
            self.rewards[agent] = 0.0
            if self._agent_selector.is_last():
                ended = self._finalize_cycle_if_needed()
                if not ended:
                    self.agent_selection = self._agent_selector.next()
            else:
                self.agent_selection = self._agent_selector.next()
            return

        # 3) Normal live agent step
        self.rewards[agent] = 0.0
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.rewards[agent] = reward
        self.infos[agent] = info
        self._cumulative_rewards[agent] += reward
        self.steps += 1

        # Buffer per-agent dones (do NOT publish mid-cycle)
        self._pending_term[agent] = bool(terminated)
        self._pending_trun[agent] = bool(truncated)

        # Episode-level end signal (buffered until cycle end)
        if terminated or truncated or self.steps >= self.max_cycles or getattr(self.game_manager, "game_over", False):
            self._pending_episode_over = True

        # 4) End-of-cycle finalize, else advance round-robin
        if self._agent_selector.is_last():
            ended = self._finalize_cycle_if_needed()
            if not ended:
                self.agent_selection = self._agent_selector.next()
        else:
            self.agent_selection = self._agent_selector.next()

    def render(self):
        """
        Renders the environment.
        """
        return self.env.render()

    def close(self):
        """
        Closes the environment.
        """
        self.env.close()

    def state(self):
        """
        Returns the global state of the environment.
        """
        return {a: self.observe(a) for a in self.agents}
