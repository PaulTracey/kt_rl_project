# sarl_tap_mdp.py
# --------------------------------------------------------------
# single-agent RL environment module
#
# A single-agent RL environment for TAP.
# Developed using the Gymnasium template for MDPs.
#
# Used for:
#   - Vectorises the board state and unit context into observations
#   - Maps discrete agent actions to game operations
#   - Applies shaped rewards for objectives, combat, and positioning
#   - Tracks per-episode events to avoid double counting
# --------------------------------------------------------------

# NOTE: This logic duplicates parts of GameManager.
# For the prototype, RL requires automatic action without UI input.
# Future work refactoring could unify some of the repeated solutions and
# reduce duplication.


import gymnasium as gym
from gymnasium import spaces
import numpy as np
from loguru import logger
from team.combat import attack_with_weapon
from team.team import Team


class SARL_TAPMDP(gym.Env):
    """
    Single-Agent Reinforcement Learning environment over the TAP grid.
    """
    # Show rendering in environments
    metadata = {'render.modes': ['human']}

    def __init__(self, game_manager, grid_size=(11, 15), max_units=6):
        """
        Initialise the environment.
        Args:
          game_manager: Reference to the running TAP GameManager.
          grid_size: The (rows, cols) of the grid.
          max_units: The max number of units per team.
        """

        # Call parent constructor to initialise Gym.Env
        super(SARL_TAPMDP, self).__init__()
        # Store game manager reference and environment configuration
        self.game_manager = game_manager # link to TAP game manager
        self.grid_rows, self.grid_cols = grid_size
        self.max_units = max_units  # max units encoded in obs

        # Step-level state
        self.current_unit_pos = (0, 0)  # fallback pos for current unit
        self.step_count = 0   # Step counter

        # Tracking for event detection across steps
        self.last_unit_state = {
            "was_on_objective": False,
            "incapacitated_enemies": set(),
            "enemy_ids_last_step": set(),
        }
        # -----------------------------
        # Action space (Discrete 12)
        # -----------------------------
        # 0-7:  8 adjacent moves
        # 8:    shoot
        # 9:    melee
        # 10:   fall back
        # 11:   pass
        self.action_space = spaces.Discrete(12)

        # -----------------------------
        # Observation space (flat vector)
        # -----------------------------
        # grid_rows*grid_cols         board occupancy
        # max_units                   friendly wounds
        # max_units*2                 friendly positions (x, y)
        # max_units*2                 enemy positions (x, y)
        # 1                           current turn
        # 1                           selected unit remaining APL
        obs_size = (
            self.grid_rows * self.grid_cols +
            self.max_units +
            self.max_units * 2 +
            self.max_units * 2 +
            1 +
            1
        )
        # Observation space: 1D float32 vector (shape=(obs_size,)).
        self.observation_space = spaces.Box(low=0, high=100, shape=(obs_size,), dtype=np.float32)
        # Initialise state (latest observation) and done flag (episode finished) as defaults.
        self.state = None
        self.done = False

    def get_observation(self):
        """
        Constructs and returns the current observation vector from the game state.
        """
        obs = []

        team = self.game_manager.current_team
        enemy = "drop_p1" if team == "drop_p2" else "drop_p2"

        # Store board occupancy: 1 if tile is occupied, else 0
        for row in self.game_manager.tile_grid:
            for tile in row:
                obs.append(1.0 if tile.occupied_by else 0.0)

        # Store the health (wounds) of all friendly units,
        # padded with 0.0 up to max_units.
        friendly_units = []
        for row in self.game_manager.tile_grid:
            for tile in row:
                unit = tile.occupied_by
                if unit and unit.team == team:
                    friendly_units.append(unit)

        for i in range(self.max_units):
            if i < len(friendly_units):
                obs.append(float(friendly_units[i].wounds))
            else:
                obs.append(0.0)

        # Store positions (x, y) of all friendly units,
        # padded with (0.0, 0.0) up to max_units.
        for i in range(self.max_units):
            if i < len(friendly_units):
                x, y = friendly_units[i].position
                obs.extend([float(x), float(y)])
            else:
                obs.extend([0.0, 0.0])

        # Collect all enemy units on the grid.
        enemy_units = []
        for row in self.game_manager.tile_grid:
            for tile in row:
                unit = tile.occupied_by
                if unit and unit.team == enemy:
                    enemy_units.append(unit)

        # Store positions (x, y) of all enemy units,
        # padded with (0.0, 0.0) up to max_units.
        for i in range(self.max_units):
            if i < len(enemy_units):
                x, y = enemy_units[i].position
                obs.extend([float(x), float(y)])
            else:
                obs.extend([0.0, 0.0])

        # Store the current turn index as a float
        obs.append(float(self.game_manager.current_turn))

        # Store the remaining APL of the selected unit,
        # only if it belongs to the active team; else 0.0.
        if self.game_manager.selected_pos:
            sr, sc = self.game_manager.selected_pos
            unit = self.game_manager.tile_grid[sr][sc].occupied_by
            if unit and unit.team == team:
                obs.append(float(unit.remaining_apl))
            else:
                obs.append(0.0)
        else:
            obs.append(0.0)

        #logger.debug(f"[OBS] {len(obs)} values for team {team}: {obs[:20]}...")
        #self.log_observation_summary(obs)

        return np.array(obs, dtype=np.float32)

    def is_objective_tile(self, pos):
        """
        Return True if this (row, col) position is an objective tile.
        """
        row, col = pos
        try:
            tile = self.game_manager.tile_grid[row][col]
            return hasattr(tile, "objective") and tile.objective  # Attribute and truth check
        except IndexError:
            return False

    def log_observation_summary(self, obs):

        """Debug helper to print obs segment summaries (for testing)."""

        #offset = 0
        #grid_cells = self.grid_rows * self.grid_cols
        #max_units = self.max_units

        #logger.debug(f"[SUMMARY] Grid occupied count: {sum(obs[offset:offset + grid_cells])}")
        #offset += grid_cells

        #logger.debug(f"[SUMMARY] Friendly wounds: {obs[offset:offset + max_units]}")
        #offset += max_units

        #logger.debug(f"[SUMMARY] Friendly positions: {obs[offset:offset + max_units * 2]}")
        #offset += max_units * 2

        #logger.debug(f"[SUMMARY] Enemy positions: {obs[offset:offset + max_units * 2]}")
        #offset += max_units * 2

        #logger.debug(f"[SUMMARY] Turn: {obs[offset]}, APL: {obs[offset + 1]}")
        pass  # Function placeholder

    def get_current_unit_position(self):
        """
        Return the (row, col) of the currently selected unit.
        If no unit is selected, return a fallback position (mainly for testing).
        """
        if self.game_manager and self.game_manager.selected_pos:
            return self.game_manager.selected_pos
        return self.current_unit_pos  # fallback value

    def take_objective(self):
        """
        Capture objective under the selected unit if not already owned.
        Returns True if a new capture occurred.
        """
        # Check if unit is on an objective tile
        current_pos = self.get_current_unit_position()
        if self.is_objective_tile(current_pos):
            row_idx, col_idx = current_pos
            tile = self.game_manager.tile_grid[row_idx][col_idx]
            team = self.game_manager.current_team

            # Capture if not already owned
            if tile.captured_by != team:
                tile.captured_by = team
                self.last_unit_state["was_on_objective"] = True  # Track that unit is on an objective

                msg = f"[REWARD] Objective captured at {current_pos} by {team}"
                logger.info(msg) # Log capture to console/file
                if hasattr(self.game_manager, "log_to_ui"):
                    self.game_manager.log_to_ui(msg)
                return True # Successful new capture

            # Already owned
            self.last_unit_state["was_on_objective"] = True
        else:
            # Not an objective tile
            self.last_unit_state["was_on_objective"] = False
        return False # No capture

    def move_off_objective(self):
        """
        Returns True if the unit just moved off an objective tile.
        """
        current_pos = self.get_current_unit_position()

        # If unit is not on an objective but was previously marked as being on one
        if not self.is_objective_tile(current_pos) and self.last_unit_state.get("was_on_objective", False):
            self.last_unit_state["was_on_objective"] = False
            return True
        return False

    def incapacitate_enemy(self):
        """
        Returns True if a new enemy unit has been incapacitated since the last step.
        """
        team = self.game_manager.current_team
        enemy = "drop_p1" if team == "drop_p2" else "drop_p2"

        # Collect names of all enemy units currently alive
        current_enemy_ids = {
            unit.name
            for row in self.game_manager.tile_grid
            for tile in row
            if (unit := tile.occupied_by) and unit.team == enemy
        }
        # Retrieve previously known enemy IDs, defaulting to current set
        prev_enemy_ids = self.last_unit_state.get("enemy_ids_last_step", current_enemy_ids)
        # Find which enemies are incapacitated this step
        newly_incapacitated = prev_enemy_ids - current_enemy_ids
        # Exclude any enemies already counted this episode
        unique_new = newly_incapacitated - self.incapacitated_this_episode

        # Save current enemy list for next step comparison
        self.last_unit_state["enemy_ids_last_step"] = current_enemy_ids
        #logger.debug(f"[DEBUG] Prev enemies: {prev_enemy_ids}")
        #logger.debug(f"[DEBUG] Current enemies: {current_enemy_ids}")

        if unique_new: # If there are fresh incapacitations
            msg = f"TAPMDP: Newly incapacitated enemy units: {unique_new}"
            logger.info(msg)
            if hasattr(self.game_manager, "log_to_ui"):
                self.game_manager.log_to_ui(msg)
            self.incapacitated_this_episode.update(unique_new) # Update internal tracking
            return True # New incapacitation detected

        return False # No new incapacitation

    def apply_action(self, action):
        """
        Apply a chosen action to the current unit.
        """
        #logger.debug(f"[DEBUG] Applying action: {action}")
        if not self.game_manager.selected_pos:
            return  # No unit selected

        # Selected row and col
        sr, sc = self.game_manager.selected_pos
        # Get selected unit
        unit = self.game_manager.tile_grid[sr][sc].occupied_by

        if action in range(8):  # Move in 8 directions
            # N, NE, E, SE, S, SW, W, NW
            directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                          (1, 0), (1, -1), (0, -1), (-1, -1)]
            dr, dc = directions[action] # chosen directon
            new_r, new_c = sr + dr, sc + dc # New coordinates
            # Check in grid
            if (0 <= new_r < len(self.game_manager.tile_grid) and
                    0 <= new_c < len(self.game_manager.tile_grid[0])):
                self.game_manager.move_unit((sr, sc), (new_r, new_c)) # Move unit

        elif action == 8:  # Shoot
            self.auto_shoot(unit)

        elif action == 9:  # Melee
            self.auto_melee(unit)

        elif action == 10:  # Fall Back
            self.game_manager.handle_fall_back()

        elif action == 11:  # Pass
            self.game_manager.pass_turn()

    def auto_select_unit(self):
        """
        Automatically selects the first available unit for the current team.
        """
        # Iterate over all tiles on the grid and select a unit from the team not gone yet
        team = self.game_manager.current_team
        for row_idx, row in enumerate(self.game_manager.tile_grid):
            for col_idx, tile in enumerate(row):
                unit = tile.occupied_by
                if (
                        unit and
                        unit.team == team and
                        getattr(unit, "activated_turn", 0) != self.game_manager.current_turn
                ):
                    selected = self.game_manager.select_unit(row_idx, col_idx)
                    if selected:
                        logger.debug(f"[AUTO_SELECT] Selected {unit.name} for team {team}")
                        return  # Stop after selecting the first valid unit
        #logger.debug(f"[AUTO_SELECT] No valid unit found to select for team {team}")

    def auto_melee(self, unit):
        """
        Automatically perform a melee attack if an enemy is adjacent.
        """
        if not unit or unit.remaining_apl <= 0:
            return  # Cannot attack if no unit or apl

        row_idx, col_idx = unit.position
        adjacent_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4 directions

        # Check all 4 adjacent tiles
        for dr, dc in adjacent_dirs:
            nr, nc = row_idx + dr, col_idx + dc
            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                tile = self.game_manager.tile_grid[nr][nc]
                target = tile.occupied_by  # Is enemy on adjacent tile?

                if target and target.team != unit.team:
                    melee_weapons = [weapon for weapon in unit.weapons if weapon.type == "Melee"]
                    if melee_weapons:
                        weapon = melee_weapons[0]
                        attack_with_weapon(unit, target, weapon, self.game_manager.dice_panel)
                        unit.remaining_apl -= 1  # Reduce APL

                        if target.incapacitated:
                            tile.occupied_by = None  # Remove incapacitated enemy from grid
                            self.game_manager.log_to_ui(f"{target.name} was incapacitated by {unit.name}!")
                            self.game_manager.update_score(unit.team, "kill")  # unified scoring & UI update

                        if unit.remaining_apl <= 0:
                            self.game_manager.finish_unit_turn(unit)  # End unit turn
                    break  # Only one attack per call

    def auto_shoot(self, unit):
        """
        Automatically perform a ranged attack if a valid target is in line-of-sight.
        """
        if not unit or unit.remaining_apl <= 0:
            return  # cannot shoot

        row_idx, col_idx = unit.position
        # Collect all enemy units on the board
        enemies = [
            (tile.occupied_by, (enemy_row_idx, enemy_col_idx))
            for enemy_row_idx, row in enumerate(self.game_manager.tile_grid)
            for enemy_col_idx, tile in enumerate(row)
            if tile.occupied_by and tile.occupied_by.team != unit.team
        ]

        for enemy, (enemy_row, enemy_col) in enemies:
            distance = self.game_manager.map.grid_distance((row_idx, col_idx), (enemy_row, enemy_col))
            # Find the first shooting weapon that is within range of the target.
            # --- Find a suitable weapon using a for loop ---
            weapon = None  # Start by assuming no weapon is found.
            for w in unit.weapons:
                # Check if the weapon matches the criteria.
                if w.type == "Shooting" and w.range_inches >= distance:
                    weapon = w  # If it matches, assign it and...
                    break  # ...stop searching immediately.

            # If a valid weapon was found and there is a clear line of sight, perform the attack.
            if weapon and self.game_manager.game_map.has_los((row_idx, col_idx), (enemy_row, enemy_col)):
                attack_with_weapon(unit, enemy, weapon, self.game_manager.dice_panel)
                unit.remaining_apl -= 1

                if enemy.incapacitated:
                    self.game_manager.tile_grid[enemy_row][enemy_col].occupied_by = None
                    self.game_manager.log_to_ui(f"{enemy.name} was incapacitated by {unit.name}!")
                    self.game_manager.update_score(unit.team, "kill")  # unified scoring & UI update

                if unit.remaining_apl <= 0:
                    self.game_manager.finish_unit_turn(unit)
                break

    def reset(self, seed=None, options=None):
        """
        Gymnasium reset() of (obs, info).
        Rebuilds the TAP game state and reinitialises environment.
        """
        # Ensure Gyms seed is set
        super().reset(seed=seed)

        # per-episode set of enemy IDs already rewarded
        self.incapacitated_this_episode = set()

        # Step 1: Reset the game state
        self.game_manager.reset_game(self.grid_rows, self.grid_cols)

        # Step 2: Load and deploy teams
        teams = Team.load_teams_from_file("team/teams.json")
        team1 = teams[0]
        team2 = teams[1]
        self.game_manager.deploy_team(team1, side="drop_p1")
        self.game_manager.deploy_team(team2, side="drop_p2")

        # Step 3: Start the game and roll initiative
        self.game_manager.check_game_start()

        # Step 4: Auto-select a valid unit for the current team
        self.auto_select_unit()

        # Step 5: Reset internal env state
        self.last_unit_state = {
            "was_on_objective": False,
            "incapacitated_enemies": set(),
            "enemy_ids_last_step": set(),
        }

        # Log positions of tiles marked as objectives
        for row_idx, row in enumerate(self.game_manager.tile_grid):
            for col_idx, tile in enumerate(row):
                if hasattr(tile, "objective") and tile.objective:
                    #logger.debug(f"[RESET] Found objective at: ({row_idx}, {col_idx})")
                    pass

        obs = self.get_observation()
        self.state = obs
        self.done = False
        return obs, {}

    def _selected_unit_spent_apl(self):
        """
        Returns True if the selected unit is missing or has spent all APL.
        Used to auto-switch selection after actions.
        """
        if not self.game_manager.selected_pos:
            return True
        sr, sc = self.game_manager.selected_pos
        unit = self.game_manager.tile_grid[sr][sc].occupied_by
        return not unit or unit.remaining_apl <= 0

    def step(self, action):
        """
         Gymnasium step(action) for (obs, reward, terminated, truncated, info).

        Hanldes:
         - Game action (movement, combat). via GameManager
         - Reward shaping for agent behaviour.
         - Event tracking for primary objectives.
         """
        reward = 0.0 # cumulative reward for this step

        self.step_count += 1
        logger.debug(f"[TRAIN] Step: {self.step_count}, Action chosen: {action}")

        # Apply the action
        self.apply_action(action)

        # Re-select unit if APL spent or team switched
        if not self.game_manager.selected_pos or self._selected_unit_spent_apl():
            self.auto_select_unit()

        # Base step penalty to prevent stalling
        reward -= 0.01

        # Motion/proximity shaping around the currently selected unit
        unit = None
        if self.game_manager.selected_pos:
            sr, sc = self.game_manager.selected_pos
            unit = self.game_manager.tile_grid[sr][sc].occupied_by

            if unit and unit.position:
                row_idx, col_idx = unit.position
                tile = self.game_manager.tile_grid[row_idx][col_idx]

                # Penalize standing in drop zone
                if tile.terrain in ("drop_p1", "drop_p2"):
                    reward -= 1.0
                    logger.info("[PENALTY] Still in drop zone. -1.0")

                # Reward leaving drop zone once with one time bonus
                if not hasattr(self, 'units_left_drop_zone'):
                    self.units_left_drop_zone = set() # Create per-env cache
                if unit.name not in self.units_left_drop_zone and tile.terrain not in ("drop_p1", "drop_p2"):
                    reward += 3.0
                    self.units_left_drop_zone.add(unit.name)
                    logger.info(f"[REWARD] {unit.name} left drop zone! +3.0")

                # Set the center of the map based on the current grid size
                # Reward shaping to center
                if (self.grid_rows, self.grid_cols) == (22, 30):
                    center_r, center_c = 10, 15  # For the large grid
                else:  # for small grid
                    center_r, center_c = 5, 7
                # Calculate distance to that hard-coded center point
                dist_to_center = self.game_manager.map.grid_distance(unit.position, (center_r, center_c))
                # If the unit is close to the center, give a small, scaled bonus
                if dist_to_center <= 5:  # Using a threshold of 5 tiles
                    proximity_bonus = (5 - dist_to_center) * 0.2  # Small bonus from 0.2 to 1.0
                    reward += proximity_bonus
                    logger.info(f"[REWARD] {unit.name} near map center. +{proximity_bonus:.1f}")

                # Discourage standing still too long per unit last position memory
                if not hasattr(self, "unit_last_positions"):
                    self.unit_last_positions = {}
                last_pos = self.unit_last_positions.get(unit.name)
                if last_pos == unit.position:
                    reward -= 0.2
                    logger.info(f"[PENALTY] {unit.name} didn't move this step. -0.2")
                self.unit_last_positions[unit.name] = unit.position

        # Action-based rewards and attempts at combat
        if action in (8, 9):  # Shoot or Melee
            reward += 0.5  # Small nudge to encourage shooting attempts
            logger.info("[REWARD] Attempted combat. +0.5")
        elif action == 11:  # Pass
            reward -= 0.5
            logger.info("[PENALTY] Passed turn. -0.5")

        # Event-based rewards with higher signal priority
        took_objective = self.take_objective() # newly captured this step
        incapacitated = self.incapacitate_enemy() # new kill

        if took_objective:
            reward += 5.0  # High value for achieving a primary objective
            logger.info("[REWARD] Objective captured! +5.0")
        if incapacitated:
            reward += 5.0  # High value for achieving the other primary objective
            logger.info("[REWARD] Enemy incapacitated! +5.0")
        if self.move_off_objective():
            reward -= 5.0  # High penalty for leaving objective
            logger.info("[PENALTY] Abandoned an objective! -5.0")

        # Recurring bonus for holding objectives
        team = self.game_manager.current_team
        held = 0
        for row in self.game_manager.tile_grid:
            for tile in row:
                if tile.objective and tile.captured_by == team:
                    held += 1

        # This bonus is the primary reason for the agent to get and hold objectives
        if held >= 2:
            bonus = held * 5.0  # Large +10 reward for holding two objectives
            reward += bonus
            logger.info(f"[REWARD] {team} holds {held} objectives! +{bonus:.1f}")
        elif held == 1:
            reward += 2.5  # Bonus for holding 1
            logger.info(f"[REWARD] {team} holds 1 objective. +2.5")

        # Small penalty when an action has no useful outcome
        if not took_objective and not incapacitated and action not in (8, 9, 11):
            reward -= 0.2
            logger.info("[PENALTY] Ineffective action (e.g. move into heavy terrain). -0.2")

        # Episode termination / truncation flags
        terminated = self.game_manager.game_over
        truncated = False

        # Build next observation and default info dict
        obs = self.get_observation()
        info = {
            "action": action,
            "objective_captured": took_objective,
            "enemy_incapacitated": incapacitated,
            "game_over": terminated
        }

        # End-of-episode outcome bonus/penalty (win/loss/draw)
        if terminated:
            p1_score = self.game_manager.score.points.get("drop_p1", 0)
            p2_score = self.game_manager.score.points.get("drop_p2", 0)
            team = self.game_manager.current_team
            # Terminal rewards are increased to be the ultimate goal
            if team == "drop_p1":
                if p1_score > p2_score:
                    reward += 20.0
                elif p1_score < p2_score:
                    reward -= 20.0
                else:
                    reward -= 2.0  # Draw penalty
            else:  # drop_p2
                if p2_score > p1_score:
                    reward += 20.0
                elif p2_score < p1_score:
                    reward -= 20.0
                else:
                    reward -= 2.0  # Draw penalty

        # Store the latest observation (for internal reference) and return the step output
        self.state = obs
        return obs, reward, terminated, truncated, info

    def get_obs(self):
        """
        Helper alias for get_observation()
        """
        return self.get_observation()

