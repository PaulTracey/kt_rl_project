# maps/map.py
#--------------------------------------------------------------------------------------
#
# map module
#
# Core battlefield map representation for the Tactical Advisor Prototype (TAP)
#
# This module defines the `Map` class, which models the Kill Team battlefield as a
# 2D grid of `Tile` objects. Each tile represents terrain, drop zones, or objectives.
#--------------------------------------------------------------------------------------

from collections import deque
from maps.tile import Tile

class Map:
    def __init__(self, rows=22, cols=30):
        """
        Initializes the map as a 2D grid of Tile objects.
        """

        # Store the number of rows and columns
        self.rows, self.cols = rows, cols

        # Create a 2D grid of Tile objects
        self.grid = [[Tile() for _ in range(cols)] for _ in range(rows)]

        # Legend to convert characters from the map file to map items on the grid
        self.legend = {
            ".": "killzone",  # open terrain on the killzone floor
            "#": "heavy",     # heavy cover to block movement and sight
            "^": "light",     # Not implemented
            "S": "drop_p1",   # drop area one
            "s": "drop_p2",   # drop area two
            "1": "obj_1",     # 1 of 3 objective markers
            "2": "obj_2",     # 2 of 3 objective markers
            "3": "obj_3"      # 3 of 3 objective markers
        }
        # Set of terrain types that block line of sight, currently only heavy.
        self.blocking_terrain = {"heavy"}

    def load_from_txt(self, file_path):
        """
        Loads the map layout from a text file.
        """

        with open(file_path, 'r') as file:
            # Read non-empty lines and remove trailing newlines
            lines = [line.rstrip('\n') for line in file if line.strip()]

        # Update map size to that given in file contents
        self.rows, self.cols = len(lines), len(lines[0])
        self.grid = []

        # loop through file to get all characters to build grid representation
        for row_idx, line in enumerate(lines):
            row = []
            for col_idx, char in enumerate(line):
                # Default to "killzone" if unknown
                terrain = self.legend.get(char, "killzone")
                tile = Tile(terrain=terrain)
                # If the character is '1','2','3', set the objective
                if char in ("1", "2", "3"):
                    tile.objective = f"obj_{char}"
                row.append(tile)
            self.grid.append(row)

    def generate_standard_zones(self):
        """
        Generates a standard map layout with drop zones and objectives.
        """
        # First, reset all tiles on the grid to default.
        for row_idx in range(self.rows):
            for col_idx in range(self.cols):
                tile = self.grid[row_idx][col_idx]
                tile.terrain = "killzone"
                tile.objective = None

        # Set the drop zones for player 1 (left side) and player 2 (right side).
        for row_idx in range(self.rows):
            # Player 1 drop zone (first 3 columns)
            for col_idx in range(3):
                self.grid[row_idx][col_idx].terrain = "drop_p1"
            # Player 2 drop zone (last 3 columns)
            for col_idx in range(self.cols - 3, self.cols):
                self.grid[row_idx][col_idx].terrain = "drop_p2"

        # Set fixed objective marker positions.
        self.grid[2][4].objective = "obj_1"
        self.grid[5][7].objective = "obj_2"
        self.grid[8][10].objective = "obj_3"

    def reset(self):
        """
        Resets the dynamic state of every tile on the grid.
        """
        for row in self.grid:
            for tile in row:
                tile.reset()

    def reachable_tiles(self, start_row, start_col, max_movement):
        """
        Calculates all reachable tiles from a starting point using Breadth First Search.
        """

        # Keep track of visited tiles so no loop back
        visited = {(start_row, start_col)}
        # Breadth First Search queue stores: (row, col, steps_used)
        queue = deque([(start_row, start_col, 0)])
        # Final set of all reachable coordinates (excluding start)
        reachable = set()

        # BFS loop
        while queue:
            current_row, current_col, steps = queue.popleft()

            # Explore all 8 neighbor directions (N, S, E, W, and diagonals)
            for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                next_row, next_col = current_row + delta_row, current_col + delta_col

                # Check bounds
                if 0 <= next_row < self.rows and 0 <= next_col < self.cols:
                    tile = self.grid[next_row][next_col]
                    # Skip tiles that are impassable or block movement.
                    if tile.movement_cost() == -1:
                        continue

                    # Block diagonal "corner cutting"
                    if abs(delta_row) == 1 and abs(delta_col) == 1:
                        if (self.grid[current_row][next_col].movement_cost() == -1 or
                                self.grid[next_row][current_col].movement_cost() == -1):
                            continue
                    # If we have not visited this tile and have APL left
                    if (next_row, next_col) not in visited and steps + 1 <= max_movement:
                        visited.add((next_row, next_col))
                        reachable.add((next_row, next_col))
                        queue.append((next_row, next_col, steps + 1))
        # Do not count the starting tile as “reachable”
        reachable.discard((start_row, start_col))
        # Return (row, col) tuples for all reachable tiles
        return reachable

    def get_grid(self):
        """
        Returns the 2D list of Tile objects.
        """
        return self.grid

    def grid_distance(self, start_pos, end_pos):
        """
        Calculates the Chebyshev distance between two points on the grid.
        """
        start_row, start_col = start_pos
        end_row, end_col = end_pos
        return max(abs(start_row - end_row), abs(start_col - end_col))

    def has_los(self, start_pos, end_pos):
        """
        Checks for a clear line of sight between two points using Bresenham's algorithm.
        """
        # Walk along the line from start to end using Bresenham’s algorithm.
        # If encounter blocking terrain, return False.
        # If reach the end without hitting an obstacle, return True.

        # Get the # Starting and ending coordinates.
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        # Line setup (Bresenham’s algorithm)
        delta_x = abs(end_col - start_col)  # difference in columns
        delta_y = -abs(end_row - start_row)  # negative difference in rows
        step_x = 1 if start_col < end_col else -1  # direction along x-axis
        step_y = 1 if start_row < end_row else -1  # direction along y-axis
        error = delta_x + delta_y  # decision variable

        # Current position starts at the origin tile
        current_x, current_y = start_col, start_row

        # Step along the line until the destination tile is reached
        while (current_x, current_y) != (end_col, end_row):
            double_error = 2 * error
            if double_error >= delta_y:
                error += delta_y
                current_x += step_x  # move sideways.
            if double_error <= delta_x:
                error += delta_x
                current_y += step_y  # move vertically.

            # Check the tile just entered, but not the final destination
            if (current_x, current_y) != (end_col, end_row):
                # Get the tile we just stepped onto from our game grid.
                tile = self.grid[current_y][current_x]
                if tile.terrain in self.blocking_terrain:
                    return False  # line of sight blocked

        # Line completed with no blocking terrain
        return True

