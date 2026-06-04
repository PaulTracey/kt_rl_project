#controller/game_manager.py
# --------------------------------------------------------------
# game_manager module
#
# Central controller for the Tactical Advisor Prototype (TAP).
#
# Handles game state, turn flow, advisor integration, and UI updates.
# --------------------------------------------------------------
import random
import pygame
from team.combat import attack_with_weapon
from config import Config
from maps.map import Map
from loguru import logger
from ui.draw_grid import draw_grid
from ui.saliency_map import get_saliency_map_ppo
from ui.saliency_map import get_saliency_map


class GameManager:

    """
       Manages game state, UI updates, advisor calls, and turn logic for TAP.
    """

    def __init__(self, score, turn_label, score_label, status_panel, log_panel, team_by_side, dice_panel, game_map, calculate_dimensions_func,ui_manager=None, advisor_model=None,advisor_type=None):
        # Map and grid references
        self.game_map = game_map
        self.map = game_map
        self.tile_grid = game_map.get_grid() if game_map else None

        # UI and shared state
        self.score = score
        self.turn_label = turn_label
        self.score_label = score_label
        self.status_panel = status_panel
        self.log_panel = log_panel
        self.team_by_side = team_by_side
        self.calculate_dimensions = calculate_dimensions_func
        self.dice_panel = dice_panel
        self.ui_manager = ui_manager

        # Advisor (RL) settings
        self.advisor_model = advisor_model
        self.advisor_type = advisor_type
        self.env = None
        self.last_saliency_map = None

        # Game state
        self.current_turn = 1
        self.current_team = "drop_p1"
        self.game_over = False
        self.game_started = False
        self.selected_pos = None
        self.reachable_tiles = set()
        self.active_unit = None

    # Check both teams been have chosen and placed
    def both_teams_deployed(self):
        return self.team_by_side["drop_p1"] and self.team_by_side["drop_p2"]

    def roll_for_initiative(self):
        """
        Simple dice roll off to decide who acts first this turn.
        Updates the dice panel if present.
        """
        self.log_to_ui("Rolling for initiative...")
        p1_name = self.get_team_display_name("drop_p1")
        p2_name = self.get_team_display_name("drop_p2")

        while True:
            p1_roll = random.randint(1, 6)
            p2_roll = random.randint(1, 6)
            logger.debug(f"Initiative rolls — {p1_name}: {p1_roll}, {p2_name}: {p2_roll}")

            # Only touch UI if the dice panel exists
            if self.dice_panel:
                dice_html = (
                    f"<b>Initiative Roll</b><br>"
                    f"{p1_name}: <font color='#00FF00'>{p1_roll}</font><br>"
                    f"{p2_name}: <font color='#00FF00'>{p2_roll}</font>"  # Test with bright green
                )
                self.dice_panel.html_text = dice_html
                self.dice_panel.rebuild()

            if p1_roll != p2_roll:
                winner_side = "drop_p1" if p1_roll > p2_roll else "drop_p2"
                winner_name = self.get_team_display_name(winner_side)
                self.log_to_ui(f"<b>{winner_name} wins the initiative!</b>")
                self.current_team = winner_side
                break
            else:
                self.log_to_ui("It's a tie! Re-rolling...")



    def check_game_start(self):

        """
        Starts the game once both teams are deployed.
        Sets turn to 1 and shows it in the UI.
        """

        if self.both_teams_deployed() and not self.game_started:
            self.roll_for_initiative()
            self.game_started = True
            self.current_turn = 1
            self.turn_label.set_text(f"Turn: {self.current_turn}")
            self.log_to_ui("Both teams deployed. Game started!")

    def log_to_ui(self, message):
        """
        Appends a message to the UI log panel, keeping only the last 5 entries.
        """
        if self.log_panel:
            current_log = self.log_panel.html_text
            # Use the correct parameter name 'message'
            updated_log = current_log + "<br>" + message
            log_lines = updated_log.split("<br>")
            if len(log_lines) > 5:
                log_lines = log_lines[-5:]
            self.log_panel.html_text = "<br>".join(log_lines)
            try:
                self.log_panel.rebuild()
            except Exception:
                pass

    def get_team_display_name(self, side):
        """
        Returns the team name if known, otherwise returns the side code.
        """
        team = self.team_by_side.get(side)
        return team.name if team else side

    def set_grid_size(self, rows, cols):
        """
        Resizes the map and reloads the correct layout.
        Resets selections and updates the status panel.
        """
        self.map.__init__(rows, cols)

        if (rows, cols) == (22, 30):
            self.map.load_from_txt("maps/map1.txt")
        elif (rows, cols) == (11, 15):
            self.map.load_from_txt("maps/small_map1.txt")  # <-- ADD THIS LINE
        else:
            self.map.generate_standard_zones()

        self.tile_grid = self.map.get_grid()

        self.selected_pos = None
        self.reachable_tiles.clear()

        self.status_panel.html_text = (
            f"<b>Status Panel</b><br>Grid reset to {rows} × {cols}."
        )
        self.status_panel.rebuild()

    def is_turn_complete(self):
        """
        Checks if all units on the current team have acted this turn.
        If the team has no units left skip their turn.
        """
        team_units = [
            tile.occupied_by for row in self.tile_grid for tile in row
            if tile.occupied_by and tile.occupied_by.team == self.current_team
        ]

        # If the current team has no units left, skip their turn
        if not team_units:
            self.log_to_ui(f"{self.get_team_display_name(self.current_team)} has no remaining units. Skipping turn.")
            return False

        return all(getattr(unit, "activated_turn", 0) == self.current_turn for unit in team_units)

    def finish_unit_turn(self, unit):
        """
        Ends the selected unit's activation and marks it as done this turn
        Checks for end-of-turn or game-over and switches to the next side when needed.
        """

        # Mark as activated and clear selection
        unit.activated_turn = self.current_turn
        self.log_to_ui(f"{unit.name}'s activation is over.")
        logger.info(f"{unit.name} finished turn on Turn {self.current_turn}")
        self.selected_pos = None
        self.reachable_tiles.clear()
        self.active_unit = None

        # Check game and end
        if self.is_game_over():
            self.handle_game_over()
            return

        # If both sides have no more units to activate end the turn.
        if not self.has_unactivated_units("drop_p1") and not self.has_unactivated_units("drop_p2"):
            self.log_to_ui(f"--- End of Turn {self.current_turn} ---")
            self.score_objectives()

            # Check again if the game should end (turn limit reached)
            self.current_turn += 1
            if self.turn_label:
                self.turn_label.set_text(f"Turn: {self.current_turn}")

            # Check for game over again.
            if self.is_game_over():
                self.handle_game_over()
                return

            # Reset activation flags on all units
            for row in self.tile_grid:
                for tile in row:
                    if tile.occupied_by:
                        tile.occupied_by.activated_turn = 0

            # Roll initiative to determine who starts the new turn.
            self.roll_for_initiative()
            return

        # Otherwise switch to the next team with units remaining
        if self.has_unactivated_units("drop_p1") and self.has_unactivated_units("drop_p2"):
            self.current_team = "drop_p2" if self.current_team == "drop_p1" else "drop_p1"
        elif self.has_unactivated_units("drop_p1"):
            self.current_team = "drop_p1"
        else:  # p2_has_more
            self.current_team = "drop_p2"

        self.log_to_ui(f"Now it's {self.get_team_display_name(self.current_team)}'s activation.")

    def has_unactivated_units(self, side):
        """
        True if the side has at least one unit that has not acted this turn.
        """
        return any(
            tile.occupied_by and tile.occupied_by.team == side and
            getattr(tile.occupied_by, "activated_turn", 0) != self.current_turn
            for row in self.tile_grid for tile in row
        )


    def is_game_over(self):
        """
        Ends the game if turn limit is reached or one team has no units left.
        """
        if self.current_turn > 4:
            return True

        p1_units = [tile.occupied_by for row in self.tile_grid for tile in row
                    if tile.occupied_by and tile.occupied_by.team == "drop_p1"]
        p2_units = [tile.occupied_by for row in self.tile_grid for tile in row
                    if tile.occupied_by and tile.occupied_by.team == "drop_p2"]

        return not p1_units or not p2_units

    def handle_game_over(self):
        """
        Calculates the result and updates the UI when the game ends.
        """
        self.game_over = True
        self.log_to_ui("<b>Game Over</b>")
        logger.info("Game Over triggered.")


        # Check remaining units
        p1_units = [tile.occupied_by for row in self.tile_grid for tile in row
                    if tile.occupied_by and tile.occupied_by.team == "drop_p1"]
        p2_units = [tile.occupied_by for row in self.tile_grid for tile in row
                    if tile.occupied_by and tile.occupied_by.team == "drop_p2"]

        if not p1_units and not p2_units:
            result = "Draw!"
        elif not p1_units:
            result = f"{self.get_team_display_name('drop_p2')} Wins!"
        elif not p2_units:
            result = f"{self.get_team_display_name('drop_p1')} Wins!"
        else:
            # Score Comparison Logic
            # If both teams still have units, it's a turn limit victory. Compare scores.
            self.log_to_ui("Turn limit reached. Calculating winner by points...")
            p1_score = self.score.points.get('drop_p1', 0)
            p2_score = self.score.points.get('drop_p2', 0)

            if p1_score > p2_score:
                result = f"{self.get_team_display_name('drop_p1')} Wins on Points!"
            elif p2_score > p1_score:
                result = f"{self.get_team_display_name('drop_p2')} Wins on Points!"
            else:
                result = "Draw on Points!"
            # End: new score comparison logic

        if self.turn_label:
            self.turn_label.set_text(result)

        self.log_to_ui(result if result != "Draw!" else "It's a draw!")
        logger.info(f"Game ended on Turn {self.current_turn} — Result: {result}")

    def reset_game(self, rows, cols):
        """
        Sets a new board size and map, clears game state, and resets UI text.
        """

        # Update global config for rows and cols
        Config.GRID_ROWS = rows
        Config.GRID_COLS = cols

        # Replace the map with a fresh one of the right size
        self.map = Map(rows, cols)

        # Load a layout that matches the new size
        if (rows, cols) == (22, 30):
            self.map.load_from_txt("maps/map1.txt")
        elif (rows, cols) == (11, 15):
            self.map.load_from_txt("maps/small_map1.txt")
        else:
            self.map.generate_standard_zones()

        # Update the cached grid
        self.tile_grid = self.map.get_grid()

        # Clear game state and selections
        self.team_by_side["drop_p1"] = None
        self.team_by_side["drop_p2"] = None
        self.game_started = False
        self.game_over = False
        self.current_turn = 1
        self.selected_pos = None
        self.reachable_tiles.clear()
        self.active_unit = None
        self.score.reset()

        return self.map

    def score_objectives(self):
        """
        Awards points for holding objectives and updates the score label.
        """
        for row_idx, row in enumerate(self.tile_grid):
            for col_idx, tile in enumerate(row):
                if tile.objective and tile.occupied_by:
                    team = tile.occupied_by.team
                    self.update_score(team, "objective")
                    self.log_to_ui(f"{self.get_team_display_name(team)} scored 1 point for holding {tile.objective}.")
                    logger.debug(f"{self.get_team_display_name(team)} scored for holding objective at ({row_idx},{col_idx})")



    def deploy_team(self, team, side):
        """
        Places every unit from a team in the side's drop zone.
        Clears old units in that zone before placing new ones.
        """
        self.log_to_ui(f"Deploying team '{team.name}' to side '{side}'")
        logger.info(f"Deploying team {team.name} to {side}")

        drop_zone_tiles = []

        # Collect open tiles in the drop zone, clearing any old unit there
        for row_idx, row in enumerate(self.tile_grid):
            for col_idx, tile in enumerate(row):
                if tile.terrain == side:
                    if tile.occupied_by:
                        self.log_to_ui(f"Removed old unit '{tile.occupied_by.name}' from ({row_idx},{col_idx})")
                    tile.occupied_by = None
                    drop_zone_tiles.append((row_idx, col_idx))

        # Not enough room in the drop zone
        if len(drop_zone_tiles) < len(team.units):
            self.log_to_ui("[WARNING] Not enough drop zone tiles for all units!")
            return

        # Place units randomly within the zone
        random.shuffle(drop_zone_tiles)
        for index, (unit, (row_idx, col_idx)) in enumerate(zip(team.units, drop_zone_tiles)):
            unit.reset()
            tile = self.tile_grid[row_idx][col_idx]
            tile.occupied_by = unit
            unit.team = side
            unit.position = (row_idx, col_idx)
            unit.id = f"agent_{index}"
            self.log_to_ui(f"Placed unit '{unit.name}' at ({row_idx + 1},{col_idx + 1}) on team {unit.team}")

    # Helpers to update the training metrics panel
    def update_training_stats(self, reward, total_episodes, total_wins):
        self.training_rewards_label.set_text(f"Last Reward: {reward:.3f}")
        self.training_episodes_label.set_text(f"Episodes: {total_episodes}")
        self.training_wins_label.set_text(f"Wins: {total_wins}")


    def reset_training_stats(self):
        self.training_rewards_label.set_text("Last Reward: —")
        self.training_episodes_label.set_text("Episodes: —")
        self.training_wins_label.set_text("Wins: —")

    def move_unit(self, source_pos, destination_pos, is_fall_back=False):
        """
        Moves a unit on the grid and spends APL.
        If APL hits zero, the unit's activation ends.
        """
        source_row, source_col = source_pos
        dest_row, dest_col = destination_pos
        unit_to_move = self.tile_grid[source_row][source_col].occupied_by

        # Lock in the unit if its their first action.
        self.lock_in_active_unit(unit_to_move)

        target_tile = self.tile_grid[dest_row][dest_col]

        # Block movement into heavy terrain
        if target_tile.terrain == "heavy":
            self.log_to_ui(f"Cannot move into {target_tile.terrain} terrain.")
            return

        # If in melee, must fall back (unless this is an explicit fall back)
        enemy_units = [
            tile.occupied_by for row in self.tile_grid for tile in row
            if tile.occupied_by and tile.occupied_by.team != unit_to_move.team
        ]
        if unit_to_move.is_in_melee(enemy_units) and not is_fall_back:
            self.log_to_ui(f"{unit_to_move.name} cannot move — must Fall Back from melee!")
            return  # Block the move

        # Move unit on the map
        self.tile_grid[source_row][source_col].occupied_by = None
        target_tile.occupied_by = unit_to_move
        unit_to_move.position = (dest_row, dest_col)

        self.log_to_ui(
            f"Moved {unit_to_move.name} from ({source_row + 1},{source_col + 1}) to ({dest_row + 1},{dest_col + 1})")
        unit_to_move.remaining_apl -= 1
        self.log_to_ui(f"{unit_to_move.name} now has {unit_to_move.remaining_apl} APL left.")

        if unit_to_move.remaining_apl <= 0:
            self.finish_unit_turn(unit_to_move)

        # Clear current selection and reachable highlights
        self.selected_pos = None
        self.reachable_tiles.clear()

    def handle_fall_back(self):
        """
        Special move to leave melee. Costs 2 APL and moves up to 5 tiles if possible.
        """
        if not self.selected_pos:
            self.log_to_ui("No unit selected to fall back.")
            return

        source_row, source_col = self.selected_pos
        fallback_unit = self.tile_grid[source_row][source_col].occupied_by

        if not fallback_unit or fallback_unit.remaining_apl < 2:
            self.log_to_ui("Unit cannot fall back — requires 2 APL.")
            return

        # Check if adjacent enemies exist
        adjacent_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        adjacent_enemies = []
        for delta_row, delta_col in adjacent_directions:
            new_row = source_row + delta_row
            new_col = source_col + delta_col

            if 0 <= new_row < len(self.tile_grid) and 0 <= new_col < len(self.tile_grid[0]):
                tile = self.tile_grid[new_row][new_col]
                if tile.occupied_by and tile.occupied_by.team != fallback_unit.team:
                    adjacent_enemies.append((new_row, new_col))

        if not adjacent_enemies:
            self.log_to_ui("No adjacent enemies to fall back from.")
            return

        # Get reachable tiles up to 5 move
        reachable_tiles = self.get_reachable_tiles(source_row, source_col, 5)
        for dest_row, dest_col in reachable_tiles:
            tile = self.tile_grid[dest_row][dest_col]
            if tile.terrain != "heavy" and tile.occupied_by is None:
                self.move_unit((source_row, source_col), (dest_row, dest_col), is_fall_back=True)
                fallback_unit.remaining_apl -= 1  # second APL cost for fallback
                self.log_to_ui(f"{fallback_unit.name} fell back to ({dest_row + 1},{dest_col + 1}).")
                return

        self.log_to_ui("No valid fallback tile found.")
        logger.info(f"{fallback_unit.name} attempted fallback but found no valid tile.")

    def attack_unit(self, attacker_pos, target_pos, dice_panel):
        """
        Handles both shooting and melee attacks.
        Updates scores and removes incapacitated targets.
        """
        attacker_row, attacker_col = attacker_pos
        target_row, target_col = target_pos
        attacker = self.tile_grid[attacker_row][attacker_col].occupied_by
        target = self.tile_grid[target_row][target_col].occupied_by

        # First action locks this unit as the active unit
        self.lock_in_active_unit(attacker)

        if not attacker or not target:
            self.log_to_ui("Invalid attack: one of the tiles is empty.")
            return

        if attacker.team == target.team:
            self.log_to_ui("Cannot attack friendly units.")
            return

        distance = self.get_grid_distance((attacker_row, attacker_col), (target_row, target_col))

        # Ranged attack
        if distance > 1:
            # Cannot shoot while in melee
            if attacker.is_in_melee([
                tile.occupied_by for row in self.tile_grid for tile in row
                if tile.occupied_by and tile.occupied_by.team != attacker.team
            ]):
                self.log_to_ui(f"{attacker.name} is in melee and cannot shoot!")
                return

            ranged_weapons = [weapon for weapon in attacker.weapons if weapon.type == "Shooting"]
            if ranged_weapons:
                weapon = ranged_weapons[0]
                if distance <= weapon.range_inches and self.has_los((attacker_row, attacker_col),
                                                                    (target_row, target_col)):
                    attack_with_weapon(attacker, target, weapon, dice_panel)
                    attacker.remaining_apl -= 1
                    self.log_to_ui(f"{attacker.name} now has {attacker.remaining_apl} APL left.")

                    if target.incapacitated:
                        self.tile_grid[target_row][target_col].occupied_by = None
                        self.log_to_ui(f"{target.name} was incapacitated!")

                        self.update_score(attacker.team, "kill")

                    if attacker.remaining_apl <= 0:
                        self.finish_unit_turn(attacker)
                else:
                    self.log_to_ui(f"{attacker.name} is out of range or cannot see {target.name}.")
            else:
                self.log_to_ui(f"{attacker.name} has no ranged weapon.")

        # Melee attack
        elif distance == 1:
            melee_weapons = [weapon for weapon in attacker.weapons if weapon.type == "Melee"]
            if melee_weapons:
                weapon = melee_weapons[0]
                attack_with_weapon(attacker, target, weapon, dice_panel)
                attacker.remaining_apl -= 1
                self.log_to_ui(f"{attacker.name} now has {attacker.remaining_apl} APL left.")

                if target.incapacitated:
                    self.tile_grid[target_row][target_col].occupied_by = None
                    self.log_to_ui(f"{target.name} was incapacitated!")

                    self.update_score(attacker.team, "kill")

                if attacker.remaining_apl <= 0:
                    self.finish_unit_turn(attacker)
            else:
                self.log_to_ui(f"{attacker.name} has no melee weapon.")

        else:
            self.log_to_ui(f"{attacker.name} cannot reach {target.name}.")

    # Wrappers around Map helpers
    def get_reachable_tiles(self, row, col, move):
        return self.map.reachable_tiles(row, col, move)

    def get_grid_distance(self, start_pos, end_pos):
        return self.map.grid_distance(start_pos, end_pos)

    def has_los(self, start_pos, end_pos):
        return self.map.has_los(start_pos, end_pos)

    def pass_turn(self):
        """
        Handles passing the turn for the currently selected unit.
        Spend all APL and end its activation.
        """
        if not self.selected_pos:
            self.log_to_ui("No unit selected to pass.")
            return

        selected_row, selected_col = self.selected_pos
        selected_unit = self.tile_grid[selected_row][selected_col].occupied_by

        if (
                selected_unit and
                selected_unit.team == self.current_team and
                getattr(selected_unit, "activated_turn", 0) != self.current_turn
        ):
            selected_unit.remaining_apl = 0  # Forcing pass
            self.log_to_ui(f"{selected_unit.name} passed their turn.")
            logger.info(f"{selected_unit.name} passed their turn manually.")
            self.finish_unit_turn(selected_unit)
        else:
            self.log_to_ui("No valid unit selected to pass turn.")

    def update_status_panel(self, row, col, status_panel):
        """
        Shows info about the clicked tile or unit in the status panel.
        """

        if not status_panel:
            return

        tile = self.tile_grid[row][col]
        unit = tile.occupied_by
        terrain = tile.terrain
        objective = tile.objective or "None"

        if unit:
            html = (
                    f"<b>Status</b><br>"
                   #f"Selected at {row + 1},{col + 1}<br>"
                    f"<b>{unit.kill_team} — {unit.name}</b><br>"
                    f"APL: {unit.apl} | Move: {unit.move} | Save: {unit.save}+ | Wounds: {unit.wounds}<br>"
                    f"<u>Weapons:</u><br>" +
                    "<br>".join([
                        f"{w.name} ({w.type}) - ATK:{w.attack} HIT:{w.hit}+ "
                        f"DMG:{w.damage}/{w.crit_damage} RNG:{w.range}"
                        for w in unit.weapons
                    ]) +
                   # f"<br>Terrain: {terrain}<br>"
                    f"<br>Objective: {objective}<br>"
                    #f"AP: {unit.apl}<br>"
                   # f"Reachable: {len(self.reachable_tiles)}"
            )
        else:
            html = (
                f"<b>Status</b><br>"
                f"Clicked {row},{col}<br>"
                f"Terrain: {terrain}<br>"
                f"Objective: {objective}"
            )

        try:
           status_panel.set_text(html)
        except AttributeError:
             status_panel.html_text = html
             status_panel.rebuild()


    def lock_in_active_unit(self, unit):
        """
        First action of the turn locks this unit as active.
        Prevents switching to another unit mid-activation.
        """
        if not self.active_unit:
            self.active_unit = unit
            self.log_to_ui(f"<b>{unit.name}</b> is now activated. Finish their actions.")

    def select_unit(self, row, col):
        """
        Selects a unit for activation if it belongs to the current team and has not already acted this turn.
        """

        tile = self.tile_grid[row][col]
        unit = tile.occupied_by

        #logger.debug(f"[SELECT] Trying to select: {unit.name if unit else 'None'} at ({row}, {col})")

        if not unit:
            logger.debug("[SELECT] No unit on tile.")
            return False

        #logger.debug(f"[SELECT] Current team: {self.current_team}, Unit team: {unit.team}")

        if unit.team != self.current_team:
            #logger.debug("[SELECT] Wrong team.")
            self.log_to_ui(f"It's not {self.get_team_display_name(unit.team)}'s turn!")
            return False

        if getattr(unit, "activated_turn", 0) == self.current_turn:
            #logger.debug("[SELECT] Unit already activated this turn.")
            self.log_to_ui(f"{unit.name} has already acted this turn.")
            return False

        # Do not allow switching if the current active unit already spent APL
        if self.active_unit:
            #logger.debug(f"[SELECT] Current active unit: {self.active_unit.name} | APL: {self.active_unit.remaining_apl}/{self.active_unit.apl}")
            if self.active_unit != unit:
                if self.active_unit.remaining_apl < self.active_unit.apl:
                    logger.debug("[SELECT] Cannot switch — APL already spent.")
                    self.log_to_ui(f"You must finish {self.active_unit.name}'s activation first!")
                    return False

        # Select the unit
        self.selected_pos = (row, col)

        # Ensure remaining_apl is set up
        if not hasattr(unit, "remaining_apl") or unit.remaining_apl <= 0:
            unit.remaining_apl = unit.apl

        # Show reachable tiles based on the unit's Move
        self.reachable_tiles = self.get_reachable_tiles(row, col, unit.move)
        self.update_status_panel(row, col, self.status_panel)
        return True

    def handle_reachable_tile_click(self, dest_row, dest_col, dice_panel, status_panel):
        """
        Handles clicks on highlighted tiles, attacks if an enemy is there, moves if it's empty.
        """

        # check unit has been selected
        if not self.selected_pos:
            return

        # get selected unit position
        source_row, source_col = self.selected_pos
        # Get the unit object for the attacker and the target.
        attacker = self.tile_grid[source_row][source_col].occupied_by
        target = self.tile_grid[dest_row][dest_col].occupied_by

        # guard that unit has APL
        if attacker.remaining_apl <= 0:
            self.finish_unit_turn(attacker)
            return

        # If a hostile unit is on the destination tile, attack it
        if target and attacker.team != target.team:
            self.attack_unit((source_row, source_col), (dest_row, dest_col), dice_panel)
        # Otherwise, move to the empty tile
        else:
            self.move_unit((source_row, source_col), (dest_row, dest_col))
            self.update_status_panel(dest_row, dest_col, status_panel)

    # ----------------------------------------------
    # UI Redraw methods
    # ----------------------------------------------
    def redraw(self):
        # Only attempt redraw if both the UI manager and the window surface exist
        if hasattr(self, "ui_manager") and self.ui_manager and hasattr(self, "window_surface"):
            # Fill the whole window with black before drawing.
            self.window_surface.fill((0, 0, 0))
            # Redraw the tactical grid, units and objectives.
            draw_grid(self.window_surface, self, self.calculate_grid_top())
            # Set pygame_gui render buttons, panels, overlays.
            self.ui_manager.draw_ui(self.window_surface)
            # Push all these changes to the actual game window
            pygame.display.update()

    def calculate_grid_top(self):
        """
        Calculates the vertical offset for centering the grid and panels block.
        """
        total_block_height = (
                Config.GRID_ROWS * Config.CELL_SIZE +
                Config.LOG_PANEL_HEIGHT +
                Config.STATUS_PANEL_HEIGHT
        )
        return (Config.WINDOW_HEIGHT - total_block_height) // 2

    def get_held_objectives(self, team):
        """
        Counts how many objective markers this team currently holds.
        """
        count = 0
        for row in self.tile_grid:
            for tile in row:
                if tile.objective and tile.occupied_by and tile.occupied_by.team == team:
                    count += 1
        return count

    def update_score(self, team, reason):
        """
        Adds 1 point to a team's score for a kill or objective
        and updates the UI score label.
        """
        if reason == "kill":
            self.score.add_kill(team)
        elif reason == "objective":
            self.score.add_objective(team)

        # Always update the UI label after changing the score data
        if self.score_label:
            p1_score = self.score.points.get('drop_p1', 0)
            p2_score = self.score.points.get('drop_p2', 0)
            self.score_label.set_text(f"Score — P1: {p1_score} | P2: {p2_score}")

            # ensure the change hits the screen during RL auto actions
        try:
            self.redraw()
        except Exception:
            pass

# ===== Helper checks used by the advisor (private methods) =====

    def _on_objective(self, unit_or_pos) -> bool:
        """
        Returns True if the unit or position is on an objective tile.
        """
        if unit_or_pos is None:
            return False
        if isinstance(unit_or_pos, tuple):
            row_idx, col_idx = unit_or_pos
        else:
            if not getattr(unit_or_pos, "position", None):
                return False
            row_idx, col_idx = unit_or_pos.position
        tile = self.tile_grid[row_idx][col_idx]
        return bool(getattr(tile, "objective", None))

    def _adjacent_enemy(self, unit):
        """
        True if there is at least one enemy in the 8 surrounding tiles.
        """
        if not unit or not getattr(unit, "position", None):
            return False

        row_idx, col_idx = unit.position

        # Check all 8 surrounding tiles and skip the middle
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue
                # Get coordinates of neighboring tile.
                neighbor_row = row_idx + delta_row
                neighbor_col = col_idx + delta_col

                # Check if the neighbor tile is within the grid boundaries
                if 0 <= neighbor_row < len(self.tile_grid) and 0 <= neighbor_col < len(self.tile_grid[0]):
                    occupant = self.tile_grid[neighbor_row][neighbor_col].occupied_by
                    if occupant and occupant.team != unit.team:
                        return True  # Found an enemy
        return False  # No adjacent enemies found

    def _has_ranged_option(self, unit):
        """
        True if the unit can legally shoot an enemy (has gun, not in melee, in range, with LOS).
        """
        # check unit exists and has a valid position
        if not unit or not getattr(unit, "position", None):
            return False

        # A unit cannot shoot while engaged in melee.
        all_enemies = [tile.occupied_by for row in self.tile_grid for tile in row
                       if tile.occupied_by and tile.occupied_by.team != unit.team]
        if unit.is_in_melee(all_enemies):
            return False

        # Check if the unit has a shooting weapon.
        ranged_weapons = [weapon for weapon in getattr(unit, "weapons", []) if
                          getattr(weapon, "type", "") == "Shooting"]
        if not ranged_weapons:
            return False

        weapon = ranged_weapons[0]
        source_row, source_col = unit.position

        # Check every tile for a valid target in range and with line of sight.
        for target_row, row in enumerate(self.tile_grid):
            for target_col, tile in enumerate(row):
                potential_target = tile.occupied_by
                if potential_target and potential_target.team != unit.team:
                    distance = self.get_grid_distance((source_row, source_col), (target_row, target_col))
                    if distance <= weapon.range_inches and self.has_los((source_row, source_col),
                                                                        (target_row, target_col)):
                        return True  # Found a valid target

        return False  # No valid targets found

    #----------------------------------------------------------
    # Advisor Action Wrapper
    #----------------------------------------------------------

    def get_advisor_action(self, unit):
        """
        Gets a tactical recommendation from a trained RL model for a given unit.

        - PPO: Applies legality checks to override invalid actions.
               Provides saliency data (why), action index (what), and probability (how sure).

        - DQN: Predicts an action and provides saliency data (why), but no probability

        - IPPO: Not supported in theis prototype.


        Args:
            unit : The friendly unit that requires an action.

        Returns:
           The recommended action index (0-11) int, or None if no valid
           action could be determined.
        """

        # If no model, a model type, or a unit, do nothing.
        if not self.advisor_model or not self.advisor_type or not unit:
            return None

        try:
            # Current game state turned into a flat observation for the model
            observation = self.env.get_obs()

            # -----------------------------
            # PPO: saliency + confidence
            # -----------------------------
            if self.advisor_type == "ppo":
                action, _ = self.advisor_model.predict(observation, deterministic=False)
                self.last_advisor_adjusted = False

                # Legality check 1: attack
                # Ensure attack suggestions are actually possible.
                try:
                    # If Melee (9) but no adjacent enemy, prefer Shoot (8) if available,
                    # otherwise fall back to Pass (11).
                    if int(action) == 9 and not self._adjacent_enemy(unit):  # Melee
                        action = 8 if self._has_ranged_option(unit) else 11  # Shoot or Pass
                        self.last_advisor_adjusted = True
                    elif int(action) == 8 and not self._has_ranged_option(unit):  # Shoot
                        action = 9 if self._adjacent_enemy(unit) else 11  # Melee or Pass
                        self.last_advisor_adjusted = True
                except Exception:
                    # keep the advisor original action.
                    pass

                # Legality check 2: Pass turn is only sensible actions if on an objective else
                # otherwise replace with more useful action like shoot or move.
                try:
                    if int(action) == 11 and not self._on_objective(unit):  # Pass
                        if self._has_ranged_option(unit):
                            action = 8 # Shoot
                        elif self._adjacent_enemy(unit):
                            action = 9 # Melee
                        else:
                            # move towards a simple nearby move into an open tile.
                            move_directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
                            row_idx, col_idx = unit.position
                            rows, cols = len(self.tile_grid), len(self.tile_grid[0])
                            for idx, (delta_row, delta_col) in enumerate(move_directions):
                                new_row, new_col = row_idx + delta_row, col_idx + delta_col
                                if 0 <= new_row < rows and 0 <= new_col < cols:
                                    tile = self.tile_grid[new_row][new_col]
                                    if tile.occupied_by is None and tile.terrain != "heavy":
                                        action = idx  # 0–7 move dir
                                        break
                        self.last_advisor_adjusted = True
                except Exception:
                    # keep the advisor’s original action.
                    pass

                # Attempt to generate a saliency map to explain the AI's decision.
                try:
                    # Call the helper function to get the saliency data, the confirmed action,
                    # and the model's confidence in that action.
                    saliency_data, action_index, conf, _ = get_saliency_map_ppo(self.advisor_model, observation, action_index=int(action))
                    self.last_saliency_map = saliency_data
                    self.last_advisor_action = action_index
                except Exception as e:
                    self.log_to_ui(f"[Saliency Error] {e}")
                    self.last_saliency_map = None
                    conf = None
                # Show confidence only if the advisors choice was not  auto-adjusted
                self.last_advisor_confidence = None if self.last_advisor_adjusted else conf
                return action
            # ------------------------------
            # DQN: saliency only
            # ------------------------------
            elif self.advisor_type == "dqn":
                action, _ = self.advisor_model.predict(observation, deterministic=True)
                try:
                    saliency_data, action_index = get_saliency_map(self.advisor_model, observation,action_index=int(action))
                    self.last_saliency_map = saliency_data
                    self.last_advisor_action = action_index
                except Exception as e:
                    self.log_to_ui(f"[Saliency Error DQN] {e}")
                    self.last_saliency_map = None
                # DQN action probability not set up. Dont show confidence.
                self.last_advisor_confidence = None
                return action
        # !-----------------------------------------------------!
        # IPPO: Not implemented
        # !-----------------------------------------------------!
        except Exception as e:
            # Any unexpected failure gets logged; clear UI explainability fields.
            self.log_to_ui(f"[Advisor Error] {e}")
            self.last_saliency_map = None
            self.last_advisor_confidence = None
            return None