# draw_grid.py
# --------------------------------------------------------------
# grid draw module
#
# Provides functions to draw the tactical grid for the
# Tactical Advisor Prototype (TAP).
#
# ---------------------------------------------------------------
import pygame
from config import Config



def draw_grid(surface, game_manager, grid_top):
    """
        Draws the full grid on the Pygame surface.
    """

    # Font for row/column labels and unit letters
    font = pygame.font.SysFont("arial", 16, bold=True)
    # Column labels (numbers across the top)
    for c in range(Config.GRID_COLS):
        label = font.render(str(c +1), True, (200, 200, 200))
        x = Config.LEFT_PADDING + c * Config.CELL_SIZE + Config.CELL_SIZE // 2
        surface.blit(label, label.get_rect(center=(x, grid_top - 10)))

    # Draw row labels (numbers down the left)
    for r in range(Config.GRID_ROWS):
        label = font.render(str(r +1), True, (200, 200, 200))
        y = grid_top + r * Config.CELL_SIZE + Config.CELL_SIZE // 2
        surface.blit(label, label.get_rect(center=(Config.LEFT_PADDING - 10, y)))

    # Draw each cell and any content inside it
    for r, row in enumerate(game_manager.tile_grid):
        for c, tile in enumerate(row):
            rect = pygame.Rect(
                Config.LEFT_PADDING + c * Config.CELL_SIZE,
                grid_top    + r * Config.CELL_SIZE,
                Config.CELL_SIZE, Config.CELL_SIZE
            )
            # Background, Objectives use "killzone" color, otherwise terrain color
            bg = Config.TERRAIN_COLORS["killzone"] if tile.has_objective() \
                 else Config.TERRAIN_COLORS.get(tile.terrain, (40,40,40))
            pygame.draw.rect(surface, bg, rect)

            # Highlight the selected cell with a yellow border
            if game_manager.selected_pos == (r, c):
                pygame.draw.rect(surface, (255,255,0), rect, 3)

            # Draw an objective marker a red circle with number.
            if tile.objective:
                cx, cy = rect.center
                pygame.draw.circle(surface, (255,0,0), (cx,cy), Config.CELL_SIZE//2 - 2)
                num = tile.objective[-1]
                ts = font.render(num, True, (255,255,255))
                surface.blit(ts, ts.get_rect(center=(cx,cy)))

            # reachable highlight of green overlay
            if (r, c) in game_manager.reachable_tiles:
                hl = pygame.Surface((Config.CELL_SIZE, Config.CELL_SIZE), pygame.SRCALPHA)
                hl.fill((0,255,0,80))
                surface.blit(hl, rect.topleft)

            # Draw unit of in this tile
            if tile.occupied_by:
                cx, cy = rect.center
                unit = tile.occupied_by

                # Choose colour based on team
                if unit.team == "drop_p1":
                    circle_color = (255, 255, 255)  # white
                    text_color = (0, 0, 0)  # black text for contrast
                elif unit.team == "drop_p2":
                    circle_color = (255, 255, 0)  # yellow
                    text_color = (0, 0, 0)  # black text for contrast
                else:
                    circle_color = (200, 200, 200)  # grey fallback
                    text_color = (0, 0, 0)

                # Draw a circle for the unit and put the first letter of its name on top
                pygame.draw.circle(surface, circle_color, (cx, cy), Config.CELL_SIZE // 3)
                letter = unit.name[0].upper()
                unit_font = pygame.font.SysFont("arial", 14, bold=True)
                text_surf = unit_font.render(letter, True, text_color)
                text_rect = text_surf.get_rect(center=(cx, cy))
                surface.blit(text_surf, text_rect)

                # Draw s grey overlay if unit already activated in turn
                if getattr(unit, "activated_turn", 0) == game_manager.current_turn:
                    overlay = pygame.Surface((Config.CELL_SIZE, Config.CELL_SIZE), pygame.SRCALPHA)
                    overlay.fill((30, 30, 30, 80))  # Grey with alpha
                    surface.blit(overlay, rect.topleft)

                # Light orange overlay to indicate active unit
                if unit == game_manager.active_unit:
                    overlay = pygame.Surface((Config.CELL_SIZE, Config.CELL_SIZE), pygame.SRCALPHA)
                    overlay.fill((255, 165, 0, 100))  # Light orange with alpha
                    surface.blit(overlay, rect.topleft)

            # grid lines over everything
            pygame.draw.rect(surface, (80,80,80), rect, 1)