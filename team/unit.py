# team/unit.py
# --------------------------------------------------------------------
#
# Defines the Unit class for TAP.
#
# A Unit represents an individual operative on the battlefield.
# It has personal attributes (APL, Move, Save, Wounds), a set of
# weapons, and dynamic state such as position, wounds, and activation.
# ---------------------------------------------------------------------
from .weapon import Weapon

class Unit:
    """
    Initialize a Unit with attributes and weapons.

    Args:
        name: Display name of the unit (e.g. "Sniper", "Leader").
        personal_attributes: Dict containing APL, move, save, and wounds values.
        weapons: List of dicts defining the unit's weapons.
        kill_team: Name of the team this unit belongs to.
    """

    def __init__(self, name: str, personal_attributes: dict, weapons: list[dict], kill_team: str = "Unknown"):
        self.name = name  # Role name (e.g. "Sniper", "Leader")
        self.kill_team = kill_team
        self.apl = personal_attributes["apl"]
        self.move = personal_attributes["move"]
        self.save = personal_attributes["save"]
        self.wounds = personal_attributes["wounds"]
        self.weapons = [Weapon(**w) for w in weapons]

        # Dynamic state, reset each game
        self.team = None  # "drop_p1" or "drop_p2"
        self.current_wounds = self.wounds
        self.incapacitated = False
        self.activated_turn = 0  # Turn number when last activated
        # self.has_acted = False
        self.remaining_apl = self.apl
        self.position = None  # (row, col) tuple for board location

    def is_in_melee(self, enemy_units: list) -> bool:
        """
        Check if this unit is in melee range (within 1 inch) of any enemy unit.
        Melee range is defined as being within 1 grid square in both
        row and column distance (Chebyshev distance <= 1).
        """
        if self.position is None:
            return False

        unit_row, unit_col = self.position
        for enemy in enemy_units:
            if enemy.incapacitated or enemy.position is None:
                continue
            enemy_row, enemy_col = enemy.position
            delta_row = abs(unit_row - enemy_row)
            delta_col = abs(unit_col - enemy_col)
            if delta_row <= 1 and delta_col <= 1:
                return True
        return False

    def __repr__(self):
        """
        Debug-friendly representation of the unit.
        """
        return f"<{self.kill_team} {self.name} APL={self.apl} Move={self.move}>"

    def reset(self):
        """
        Resets the unit's dynamic state for the start of a new game.
        """
        self.team = None
        self.current_wounds = self.wounds
        self.incapacitated = False
        self.activated_turn = 0
        self.remaining_apl = self.apl
        self.position = None

