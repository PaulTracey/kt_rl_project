# maps/tile.py
# --------------------------------------------------------------
# Tile module
#
# Defines the Tile class, representing a single cell on the map.
# A tile stores terrain type, objectives, occupancy, and capture
# state.
# --------------------------------------------------------------

class Tile:
    """
    Represents a single map cell in the killzone.
    """

    __slots__ = (
        'terrain',      # 'killzone', 'light', 'heavy', 'drop_p1', 'drop_p2'.
        'objective',    # None or 'obj_1', 'obj_2', 'obj_3'
        'occupied_by',  # None or a unit object/ID
        'captured_by'   # None or 'player_1'/'player_2'
    )

    def __init__(self, terrain='killzone', objective=None):
        """Initializes a Tile."""

        self.terrain      = terrain
        self.objective    = objective
        self.occupied_by  = None
        self.captured_by  = None

    def movement_cost(self) -> int:
        """
        Returns Action Point cost for moving into this tile. Returns -1 if impassable.
        """
        if self.terrain == 'heavy':
            return -1  # Impassible
        return 1

    def has_objective(self) -> bool:
        """
        True if this tile holds an objective.
        """
        return self.objective is not None

    def reset(self):
        """
        Reset per-turn state (clears occupancy and capture).
        """
        self.occupied_by = None
        self.captured_by = None

