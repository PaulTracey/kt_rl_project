# config.py
# --------------------------------------------------------------
# configuration module
#
# Contains global constants and settings for TAP.
#
# Used for:
# - Window and grid dimensions
# - Game and UI configuration defaults
# - Random seed for reproducibility
# --------------------------------------------------------------
import pygame

class Config:
    # -------------------------------------------------------------------------
    # Window and Grid Dimensions
    # -------------------------------------------------------------------------
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 960

    GRID_ROWS = 22
    GRID_COLS = 30
    CELL_SIZE = 24

    # Padding and Layout
    LEFT_PANEL_WIDTH = 250
    LEFT_PADDING = LEFT_PANEL_WIDTH + 10
    PANEL_WIDTH = 300
    LOG_PANEL_HEIGHT = 150
    STATUS_PANEL_HEIGHT = 250
    TOGGLE_BUTTON_HEIGHT = 40
    # Colors (RGB)
    TERRAIN_COLORS = {
        "drop_p1": (100, 160, 255),
        "drop_p2": (255, 150, 200),
        "killzone": (180, 140, 255),
        "light": (150, 150, 150),
        "heavy": (0, 0, 0),
        "obj_1": (255, 255, 100),
        "obj_2": (255, 200, 50),
        "obj_3": (255, 150, 0),
    }

    # -------------------------------------------------------------------------
    # Fonts
    # -------------------------------------------------------------------------
    @staticmethod
    def get_default_font(size=16, bold=False):
        return pygame.font.SysFont("arial", size, bold=bold)

    # -------------------------------------------------------------------------
    # Team Colors
    # -------------------------------------------------------------------------
    TEAM_COLORS = {
        "drop_p1": {
            "circle": (255, 255, 255),
            "text": (0, 0, 0),
        },
        "drop_p2": {
            "circle": (255, 255, 0),
            "text": (0, 0, 0),
        },
        "default": {
            "circle": (200, 200, 200),
            "text": (0, 0, 0),
        }
    }

    # -------------------------------------------------------------------------
    # Game Settings
    # -------------------------------------------------------------------------
    DEFAULT_EPISODES = 1000
    FPS_LIMIT = 60

    # -------------------------------------------------------------------------
    # UI Labels and Dropdown Defaults
    # -------------------------------------------------------------------------
    TEAM_OPTIONS_LABEL = "Select team"
    ALGORITHM_OPTIONS = ["Select Algorithm", "PPO", "DQN", "A3C"]

    #Global Seed
    SEED = 42