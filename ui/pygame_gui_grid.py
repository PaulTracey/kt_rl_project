# pygame_gui_grid.py
# --------------------------------------------------------------
# help UI module
#
# Pygame GUI for the Tactical Advisor Prototype (TAP).
#
# Used for:
# - Handling layout and UI events
# - Rendering the grid
# - Bridging user actions to the GameManager
# - Connecting to RL environments
# --------------------------------------------------------------

import pygame
import math
import pygame_gui
from maps.map import Map
from team.team import Team
from team.score import Score
from ui.help_ui import HelpUI
from controller.game_manager import GameManager
from config import Config
from sarl_tap_mdp import SARL_TAPMDP
from .draw_grid import draw_grid
from loguru import logger
from training.random_agent import run_random_agent
from .saliency_map import draw_saliency_overlay
from pathlib import Path


# -----------------------------------------------------------------------------
# Pygame / pygame_gui setup
# -----------------------------------------------------------------------------
pygame.init()  # Start pygame modules
pygame.display.set_caption("Tactical Advisor Prototype (TAP)")
ui_manager = pygame_gui.UIManager((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT), "ui/theme.json")
# Create the main UI manager which controls widgets, themes, and UI events
ui_manager.preload_fonts([{"name": "fira_code", "point_size": 14, "style": "bold"}])
# Create a clock to regulate the frame rate and measure time between updates
clock = pygame.time.Clock()
# ------------------------------------------------------------------------------
# File dialog state start folder (logs/)
# ------------------------------------------------------------------------------
file_dialog = None
START_DIR = Path("logs")
START_DIR.mkdir(exist_ok=True)
# -------------------------------------------------------------------------------
# Constants and globals
# -------------------------------------------------------------------------------
# Main game window
window_surface = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
# Track which team is on each side of the board
team_by_side = {
    "drop_p1": None,
    "drop_p2": None
}
def calculate_dimensions():
    """Work out grid width/height in pixels and vertical placement."""
    global GRID_WIDTH, GRID_HEIGHT, GRID_TOP
    # Grid size in pixels
    GRID_WIDTH  = Config.GRID_COLS * Config.CELL_SIZE
    GRID_HEIGHT = Config.GRID_ROWS * Config.CELL_SIZE
    # Total height of grid + log panel + status panel
    total_block_height = GRID_HEIGHT + Config.LOG_PANEL_HEIGHT + Config.STATUS_PANEL_HEIGHT
    # Position vertically centered
    GRID_TOP = (Config.WINDOW_HEIGHT - total_block_height) // 2
# Run once
calculate_dimensions()
# ------------------------------------------------------------------------------
# Load map & teams
# ------------------------------------------------------------------------------
# Create map object
game_map = Map(Config.GRID_ROWS, Config.GRID_COLS)
# Load the terrain
game_map.load_from_txt("maps/map1.txt")
# Load available teams
teams = Team.load_teams_from_file("team/teams.json")
# Get just the team names for dropdown
team_names = [team.name for team in teams]
# ------------------------------------------------------------------------------
# Action Mappings for Advisor
# ------------------------------------------------------------------------------
# Mapping for action indexes used by the RL agent
ACTION_NAMES = {
    0: "Move North",
    1: "Move North-East",
    2: "Move East",
    3: "Move South-East",
    4: "Move South",
    5: "Move South-West",
    6: "Move West",
    7: "Move North-West",
    8: "Shoot",
    9: "Melee Attack",
    10: "Fall Back",
    11: "Pass Turn"
}
# ----------------------------------------------------------------------------
# UI Panels & Controls
# ----------------------------------------------------------------------------
# Left-side panel for game play options
left_panel = pygame_gui.elements.UIPanel(
    relative_rect=pygame.Rect(
        10, 10,
        Config.LEFT_PANEL_WIDTH - 20,
        Config.WINDOW_HEIGHT - 20
    ),
    manager=ui_manager,
    starting_layer_height=1
)

status_panel = pygame_gui.elements.UITextBox(
    html_text="<b>Status Panel</b><br>Unit Info, HP, Actions.",
    relative_rect=pygame.Rect(
        Config.LEFT_PADDING,
        GRID_TOP + GRID_HEIGHT + Config.LOG_PANEL_HEIGHT + 10,
        GRID_WIDTH,
        Config.STATUS_PANEL_HEIGHT
    ),
    manager=ui_manager
)

advisor_panel = pygame_gui.elements.UITextBox(
    html_text="<b>Advisor Panel</b><br>Select a unit to receive tactical advice.",
    relative_rect=pygame.Rect(
        Config.LEFT_PADDING,
        GRID_TOP + GRID_HEIGHT + 10,
        GRID_WIDTH,
        150
    ),
    manager=ui_manager
)
# Right-side panel for training
training_panel = pygame_gui.elements.UIPanel(
    relative_rect=pygame.Rect(
        Config.LEFT_PADDING + GRID_WIDTH + 10,
        10,
        Config.PANEL_WIDTH - 20,
        Config.WINDOW_HEIGHT - 20
    ),
    manager=ui_manager,
    starting_layer_height=1
)
# Hidden toggle button (training panel always shown for now)
toggle_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(
        (10, training_panel.relative_rect.height - 40),
        (Config.PANEL_WIDTH - 40, 30)
    ),
    container=training_panel,
    manager=ui_manager,
    text='Show Training Panel'
)
# disable toggle button for now
toggle_button.hide()
# Grid size switch buttons
full_size_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((10, 10), (250, 30)),
    text="Grid: 22 x 30 (Start)",
    container=training_panel,
    manager=ui_manager
)
half_size_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((10, 50), (250, 30)),
    text="Grid: 11 x 15 (Start)",
    container=training_panel,
    manager=ui_manager
)

# Team dropdown options ---
team_options = ["Select team"] + team_names

team1_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((10, 10), (160, 20)),
    text="Team 1:",
    container=left_panel,
    manager=ui_manager
)

team1_dropdown = pygame_gui.elements.UIDropDownMenu(
    options_list=team_options,
    starting_option="Select team",
    relative_rect=pygame.Rect((10, 30), (Config.LEFT_PANEL_WIDTH - 40, 30)),
    container=left_panel,
    manager=ui_manager
)

team2_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((10, 70), (160, 20)),
    text="Team 2:",
    container=left_panel,
    manager=ui_manager
)

team2_dropdown = pygame_gui.elements.UIDropDownMenu(
    options_list=team_options,
    starting_option="Select team",
    relative_rect=pygame.Rect((10, 90), (Config.LEFT_PANEL_WIDTH - 40, 30)),
    container=left_panel,
    manager=ui_manager
)

left_log_panel = pygame_gui.elements.UITextBox(
    html_text="<b>Log</b><br>Waiting for events...",
    relative_rect=pygame.Rect(
        (10, 350),
        (Config.LEFT_PANEL_WIDTH - 40, 350)
    ),
    container=left_panel,
    manager=ui_manager
)

algorithm_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((10, 90), (160, 20)),
    text="Algorithm:",
    container=training_panel,
    manager=ui_manager
)

algorithm_dropdown = pygame_gui.elements.UIDropDownMenu(
    options_list=["Select Algorithm", "Random",  "DQN", "PPO"],
    starting_option="Select Algorithm",
    relative_rect=pygame.Rect((10, 110), (Config.PANEL_WIDTH - 40, 30)),
    container=training_panel,
    manager=ui_manager
)

timesteps_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((10, 150), (160, 20)),
    text="Timesteps:",
    container=training_panel,
    manager=ui_manager
)

timesteps_input = pygame_gui.elements.UITextEntryLine(
    relative_rect=pygame.Rect((10, 170), (Config.PANEL_WIDTH - 40, 30)),
    container=training_panel,
    manager=ui_manager
)
timesteps_input.set_text("100000")

training_status_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((10, 210), (Config.PANEL_WIDTH - 40, 30)),
    text="",
    container=training_panel,
    manager=ui_manager
)

# Metrics labels
reward_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((10, 270), (Config.PANEL_WIDTH - 40, 30)),
    text="Last Reward: -",
    container=training_panel,
    manager=ui_manager
)

episode_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((10, 310), (Config.PANEL_WIDTH - 40, 30)),
    text="Episodes: -",
    container=training_panel,
    manager=ui_manager
)

win_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((10, 350), (Config.PANEL_WIDTH - 38, 30)),
    text="Wins: -",
    container=training_panel,
    manager=ui_manager
)

train_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((10, 470), (Config.PANEL_WIDTH - 40, 30)),
    text="Train",
    container=training_panel,
    manager=ui_manager
)

# Load Model button
load_model_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((10, 510), (Config.PANEL_WIDTH - 40, 30)),
    text="Load Model",
    container=training_panel,
    manager=ui_manager
)

loaded_model_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((10, 550), (Config.PANEL_WIDTH - 40, 30)),
    text="Loaded: -",
    container=training_panel,
    manager=ui_manager
)

clear_model_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((10, 590), (Config.PANEL_WIDTH - 40, 30)),
    text="Clear Model",
    container=training_panel,
    manager=ui_manager
)


# Existing selection list
disable_ui_checkbox = pygame_gui.elements.UISelectionList(
    relative_rect=pygame.Rect((10, 210), (Config.PANEL_WIDTH - 40, 30)),
    item_list=["Disable UI during training"],
    manager=ui_manager,
    container=training_panel
)

dice_panel = pygame_gui.elements.UITextBox(
    html_text="<b>Dice:</b><br>No rolls yet.",
    relative_rect=pygame.Rect((10, 130), (Config.LEFT_PANEL_WIDTH - 40, 200)),
    container=left_panel,
    manager=ui_manager
)

turn_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((10, 800), (Config.LEFT_PANEL_WIDTH - 40, 100)),
    text="Deploying Phase ...",
    container=left_panel,
    manager=ui_manager,
    object_id="#turn_label"
)

score_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((10, 880), (Config.LEFT_PANEL_WIDTH - 40, 40)),
    text="Score - P1: 0 | P2: 0",
    container=left_panel,
    manager=ui_manager,
    object_id="#score_label"
)

pass_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((10, 740), (Config.LEFT_PANEL_WIDTH - 40, 30)),
    text="Pass Turn (p)",
    container=left_panel,
    manager=ui_manager
)
# help overlay in place# help overlay in place but not complete
help_ui = HelpUI(ui_manager, left_panel, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
# Connect UI elements into the game manager
score = Score()
game_manager = GameManager(
    score=score,
    turn_label=turn_label,
    score_label=score_label,
    status_panel=status_panel,
    log_panel=left_log_panel,
    dice_panel=dice_panel,
    team_by_side=team_by_side,
    game_map=game_map,
    calculate_dimensions_func=calculate_dimensions,
    ui_manager=ui_manager

)
game_manager.window_surface = window_surface

fall_back_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((10, 700), (Config.LEFT_PANEL_WIDTH - 40, 30)),
    text="Fall Back",
    container=left_panel,
    manager=ui_manager
)

# RL environment (default is 22x30 grid; can be reset to 11x15 via button)
env = SARL_TAPMDP(game_manager, grid_size=(22, 30))
game_manager.env = env

# Link training metrics to UI labels
game_manager.training_rewards_label = reward_label
game_manager.training_episodes_label = episode_label
game_manager.training_wins_label = win_label

# -----------------------------------------------------------------------------
# Interactive state
# ----------------------------------------------------------------------------
checkbox_checked = True # used only by the (hidden) toggle button
# Draw the starting grid before the game loop begins
draw_grid(window_surface, game_manager, GRID_TOP)

# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
running = True # Game keeps running until set to False
while running:
    # Keep game running at about 60 frames per second
    # time_delta = seconds passed since the last frame
    time_delta = clock.tick(60) / 1000.0

    # Go through all user and system events
    for event in pygame.event.get():
        # Send events to the help overlay
        help_ui.process_event(event)
        # If player clicks the window close button stop the game
        if event.type == pygame.QUIT:
            running = False

        # Keyboard
        elif event.type == pygame.KEYDOWN:

            # ON 'p' key pass the selected units turn
            if event.key == pygame.K_p:
                if game_manager.selected_pos: # check if a unit is selected
                    source_row, source_col = game_manager.selected_pos
                    unit = game_map.grid[source_row][source_col].occupied_by

                    # Only allow pass if it is the unit's turn and it has not acted yet
                    if unit and unit.team == game_manager.current_team and getattr(unit, "activated_turn", 0) != game_manager.current_turn:
                        unit.remaining_apl = 0  # use up their apl
                        game_manager.log_to_ui(f"{unit.name} passed their turn (via keyboard).")
                        logger.debug(f"Turn passed by: {unit.name}")
                        game_manager.finish_unit_turn(unit)
                    else:
                        game_manager.log_to_ui("Selected unit cannot pass - already acted or invalid.")
                else:
                    game_manager.log_to_ui("No unit selected to pass.")

            # 'r' key takes a random action (used for testing env.step manually)
            elif event.key == pygame.K_r:
                action = env.action_space.sample() # pick a random action from RL agent's action space
                obs, reward, done, info = env.step(action) # step environment with that action
                logger.debug(f"Manual env.step() -> action: {action}, reward: {reward}, done: {done}")

        # Mouse (grid)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # If a file dialog is open, let pygame_gui handle it and skip grid logic
            if file_dialog is not None and file_dialog.alive():
                ui_manager.process_events(event)
                continue

            # Get mouse position and convert it into grid row/column
            mouse_x, mouse_y = event.pos
            col = (mouse_x - Config.LEFT_PADDING) // Config.CELL_SIZE
            row = (mouse_y - GRID_TOP)  // Config.CELL_SIZE

            # Handle clicks inside the grid area
            if 0 <= row < Config.GRID_ROWS and 0 <= col < Config.GRID_COLS:
                # If game hasn’t started yet block grid clicks
                if not game_manager.game_started:
                    game_manager.log_to_ui("The game hasn't started. Please deploy both teams.")
                    continue
                # If game is already over, ignore clicks
                if game_manager.game_over:
                    continue
                # Find the clicked tile and what is on it
                tile = game_map.grid[row][col]
                unit = tile.occupied_by

                terrain = tile.terrain
                objective = tile.objective or "None"
                logger.debug(f"Clicked on grid cell: ({row}, {col}) - Terrain: {terrain}, Objective: {objective}")

                # If a unit is on the tile
                if unit:
                    # Clicked a friendly unit to select it
                    if unit.team == game_manager.current_team:
                        if not game_manager.select_unit(row, col):
                            continue

                        # Show in console which unit was picked and ask advisor for suggestion
                        logger.debug(f"Unit {unit.name} selected. Calling advisor...")
                        advisor_action = game_manager.get_advisor_action(unit)

                        # Advisor returns an action index and converts to readable text
                        if advisor_action is not None:
                            try:
                                action_index = int(advisor_action)
                                action_text = ACTION_NAMES.get(action_index, f"Unknown ({action_index})")

                                # Advisor also provides confidence score (only PPO right now).
                                confidence = getattr(game_manager, "last_advisor_confidence", None)
                                if confidence is None:
                                    conf_str = "N/A"
                                else:
                                    conf_str = f"{math.ceil(confidence * 100)}%"

                            except Exception as e:
                                # If decoding fails, show an error instead
                                action_text = f"[Error decoding action] {e}"

                            # Update advisor panel with suggestion and confidence
                            advisor_panel.set_text(
                                f"<b>Advisor Panel</b><br>"
                                f"Recommended action: {action_text}<br>"
                                f"Confidence: {conf_str}"
                            )
                        else:
                            # Advisor gave no suggestion
                            advisor_panel.set_text(
                               f"<b>Advisor Panel</b><br>No suggestion available."
                            )

                    # Enemy unit clicked. Attempt attack with the selected unit
                    else:
                        if game_manager.selected_pos:
                            source_row, source_col = game_manager.selected_pos
                            attacker = game_map.grid[source_row][source_col].occupied_by
                            if attacker and attacker.team == game_manager.current_team:
                                logger.debug(f"Attack initiated: {attacker.name} is attacking {unit.name}")
                                game_manager.attack_unit((source_row, source_col), (row, col), dice_panel)

                    # Always refresh the status panel with info on the clicked tile
                    game_manager.update_status_panel(row, col, status_panel)

                # If empty tile that is reachable.  Move the selected unit.
                elif game_manager.selected_pos and (row, col) in game_manager.reachable_tiles:
                    logger.debug(f"Unit moved to ({row}, {col})")
                    game_manager.handle_reachable_tile_click(row, col, dice_panel, status_panel)

        # ---------------------------------------------------------------------
        # Team dropdowns
        # ---------------------------------------------------------------------
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == team1_dropdown:
                if event.text != "Select team":
                    selected_team = next(team for team in teams if team.name == event.text)
                    game_manager.deploy_team(selected_team, side="drop_p1")
                    team_by_side["drop_p1"] = selected_team
                    game_manager.check_game_start()

            elif event.ui_element == team2_dropdown:
                if event.text != "Select team":
                    selected_team = next(team for team in teams if team.name == event.text)
                    game_manager.deploy_team(selected_team, side="drop_p2")
                    team_by_side["drop_p2"] = selected_team
                    game_manager.check_game_start()

        # ---------------------------------------------------------------------
        # Other GUI buttons
        # ---------------------------------------------------------------------
        elif event.type == pygame_gui.UI_BUTTON_PRESSED:
            # Toggle training panel visibility (currently not used)
            if event.ui_element == toggle_button:
                checkbox_checked = not checkbox_checked
                training_panel.visible = checkbox_checked
                full_size_button.visible = checkbox_checked
                half_size_button.visible = checkbox_checked
                team1_dropdown.visible = checkbox_checked
                team2_dropdown.visible = checkbox_checked
                team1_label.visible = checkbox_checked
                team2_label.visible = checkbox_checked

            # Fall Back button call fallback action via GameManager
            elif event.ui_element == fall_back_button:
                game_manager.handle_fall_back()

            # Reset game to full-size grid (22x30)
            elif event.ui_element == full_size_button:
                game_map = game_manager.reset_game(22, 30)
                calculate_dimensions()

                turn_label.set_text("Deploying Phase ...")
                score_label.set_text("Score — P1: 0 | P2: 0")
                left_log_panel.html_text = "<b>Log</b><br>Waiting for events..."
                left_log_panel.rebuild()
                status_panel.html_text = "<b>Status Panel</b><br>Grid reset to 22 × 30."
                status_panel.rebuild()

                env = SARL_TAPMDP(game_manager, grid_size=(22, 30))
                game_manager.env = env

                turn_label.set_text("Deploying Phase ...")
                score_label.set_text("Score — P1: 0 | P2: 0")
                left_log_panel.html_text = "<b>Log</b><br>Waiting for events..."
                left_log_panel.rebuild()
                status_panel.html_text = "<b>Status Panel</b><br>Grid reset to 22 × 30."
                status_panel.rebuild()

            # Open file dialog (logs/)
            elif event.ui_element == load_model_button:
                from pygame_gui.windows import UIFileDialog
                allowed_suffixes = ['.zip']  # SB3 default save format
                try:
                    file_dialog = UIFileDialog(
                        rect=pygame.Rect(160, 100, 600, 400),
                        manager=ui_manager,
                        window_title="Load Pretrained Model",
                        initial_file_path=str(START_DIR),
                        allow_picking_directories=False,
                        allow_existing_files_only=True,
                        allow_multi_select=False,
                        allowed_suffixes=allowed_suffixes
                    )
                except TypeError:
                    # Fallback for older pygame_gui versions
                    file_dialog = UIFileDialog(
                        rect=pygame.Rect(160, 100, 600, 400),
                        manager=ui_manager,
                        window_title="Load Pretrained Model",
                        initial_file_path=str(START_DIR)
                    )
            # Reset game to half-size grid (11x15)
            elif event.ui_element == half_size_button:
                game_map = game_manager.reset_game(11, 15)
                calculate_dimensions()

                # Re-create dropdowns for a clean UI reset
                team1_dropdown.kill()
                team1_dropdown = pygame_gui.elements.UIDropDownMenu(
                    options_list=team_options,
                    starting_option="Select team",
                    relative_rect=pygame.Rect((10, 30), (Config.LEFT_PANEL_WIDTH - 40, 30)),
                    container=left_panel,
                    manager=ui_manager
                )
                team2_dropdown.kill()
                team2_dropdown = pygame_gui.elements.UIDropDownMenu(
                    options_list=team_options,
                    starting_option="Select team",
                    relative_rect=pygame.Rect((10, 90), (Config.LEFT_PANEL_WIDTH - 40, 30)),
                    container=left_panel,
                    manager=ui_manager
                )
                env = SARL_TAPMDP(game_manager, grid_size=(11, 15))
                game_manager.env = env

            # Pass Turn button triggers pass turn action
            elif event.ui_element == pass_button:
                game_manager.pass_turn()

            # Clear the currently loaded advisor model
            elif event.ui_element == clear_model_button:
                game_manager.advisor_model = None
                game_manager.advisor_type = None
                loaded_model_label.set_text("Loaded: -")
                training_status_label.set_text("Model cleared")
                game_manager.log_to_ui("Advisor model cleared.")


            # Train button. Start training loop
            elif event.ui_element == train_button:
                # When starting training
                training_status_label.set_text("Training...")

                # Force immediate UI update so "Training..." appears
                ui_manager.update(time_delta)
                window_surface.fill((0, 0, 0))
                draw_grid(window_surface, game_manager, GRID_TOP)
                status_panel.rebuild()
                ui_manager.draw_ui(window_surface)
                pygame.display.update()

                # Draw saliency heatmap if available (for PPO/DQN only)
                if game_manager.advisor_type in ["ppo", "dqn","marl_ppo"]:
                    draw_saliency_overlay(
                        surface=window_surface,
                        saliency=game_manager.last_saliency_map,
                        grid_rows=Config.GRID_ROWS,
                        grid_cols=Config.GRID_COLS,
                        cell_size=Config.CELL_SIZE,
                        top_offset=GRID_TOP,
                        left_padding=Config.LEFT_PADDING
                    )

                # get selected algorithm
                algorithm_name = algorithm_dropdown.selected_option
                game_manager.reset_training_stats()

                try:
                    num_timesteps = int(timesteps_input.get_text())
                except ValueError:
                    game_manager.log_to_ui("Please enter a valid number of timesteps.")
                    continue

                # Import the training function
                if algorithm_name == "PPO":
                    from training.train_ppo import train
                elif algorithm_name == "DQN":
                    from training.train_dqn import train
                elif algorithm_name == "MARL-PPO":
                    from training.train_marl_ippo import train
                elif algorithm_name == "Random":
                    run_random_agent(env, timesteps=num_timesteps)
                    training_status_label.set_text("Random agent complete!")
                    game_manager.log_to_ui("Random agent completed evaluation.")
                    continue
                else:
                    game_manager.log_to_ui("Unsupported or no algorithm selected.")
                    continue

                # Decide if UI should be disabled during training
                disable_ui = disable_ui_checkbox.get_single_selection() == "Disable UI during training"
                # Train the chosen algorithm
                if algorithm_name == "MARL-PPO":
                    model = train(timesteps=num_timesteps)  # MARL env is created internally
                else:
                    model = train(env, timesteps=num_timesteps, live_ui_mode=not disable_ui)

                training_status_label.set_text("Training complete!")
                game_manager.log_to_ui(f"{algorithm_name} training complete!")

                # Store model in GameManager so advisor can use it
                game_manager.advisor_model = model
                game_manager.advisor_type = algorithm_name.lower().replace("-", "_")

        # ---------------------------------------------------------------------
        # File dialog (loading models)
        # ---------------------------------------------------------------------
        # File dialog events (siblings of UI_BUTTON_PRESSED)
        elif event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
            if file_dialog is not None and event.ui_element == file_dialog:
                model_path = event.text  # full file path
                try:
                    algorithm_name = algorithm_dropdown.selected_option
                    advisor_type = None
                    model = None

                    if algorithm_name == "PPO" or ("ppo" in model_path.lower() and algorithm_name == "Select Algorithm"):
                        from stable_baselines3 import PPO
                        model = PPO.load(model_path)
                        advisor_type = "ppo"

                    elif algorithm_name == "DQN" or ("dqn" in model_path.lower() and algorithm_name == "Select Algorithm"):
                        from stable_baselines3 import DQN
                        model = DQN.load(model_path)
                        advisor_type = "dqn"

                    elif algorithm_name == "MARL-PPO" or "marl" in model_path.lower():
                        game_manager.log_to_ui("MARL-PPO load not wired here (RLlib restore differs).")
                        training_status_label.set_text("Load cancelled.")
                        model = None

                    else:
                        game_manager.log_to_ui(
                            "Pick PPO/DQN in the dropdown or include it in the filename (*ppo*.zip / *dqn*.zip)."
                        )
                        training_status_label.set_text("Select algorithm first.")
                        model = None

                    # If load worked, save the model into GameManager so the advisor can use it
                    if model is not None and advisor_type is not None:
                        game_manager.advisor_model = model
                        game_manager.advisor_type = advisor_type
                        loaded_model_label.set_text(f"Loaded: {advisor_type.upper()}")
                        training_status_label.set_text(f"{advisor_type.upper()} model loaded!")
                        game_manager.log_to_ui(f"Loaded model from: {model_path}")

                except Exception as e:
                    training_status_label.set_text("Load failed.")
                    game_manager.log_to_ui(f"Error loading model: {e}")

        # Close file dialog if user cancels
        elif event.type == pygame_gui.UI_WINDOW_CLOSE:
            if file_dialog is not None and event.ui_element == file_dialog:
                file_dialog = None

        # Let pygame_gui handle all other UI events
        ui_manager.process_events(event)

    # -------------------------------------------------------------------------
    # Redraw loop (every frame)
    # -------------------------------------------------------------------------
    ui_manager.update(time_delta)
    window_surface.fill((0,0,0))
    # Calls draw grid
    draw_grid(window_surface, game_manager, GRID_TOP)

    # If advisor model has saliency map show saliency overlay
    if game_manager.advisor_type in ["ppo", "dqn", "marl_ppo"] and game_manager.last_saliency_map is not None:
        draw_saliency_overlay(
            surface=window_surface,
            saliency=game_manager.last_saliency_map,
            grid_rows=Config.GRID_ROWS,
            grid_cols=Config.GRID_COLS,
            cell_size=Config.CELL_SIZE,
            top_offset=GRID_TOP,
            left_padding=Config.LEFT_PADDING
        )

    # Draw all UI elements
    ui_manager.draw_ui(window_surface)

    # Check if game ended
    if not game_manager.game_over and game_manager.both_teams_deployed() and game_manager.is_game_over():
        game_manager.handle_game_over()

    # Refresh display
    pygame.display.update()

# Quit pygame cleanly
pygame.quit()
