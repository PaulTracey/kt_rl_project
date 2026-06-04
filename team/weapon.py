# team/weapon.py
# --------------------------------------------------------------
# weapon module
#
# Defines the Weapon class for the Tactical Advisor Prototype (TAP).
#
# A Unit can have a melee and ranged weapon.
# Each weapon has stats for number of attacks, hit threshold,
# damage, critical damage, and range.
# --------------------------------------------------------------
class Weapon:
    def __init__(self, name, type, attack, hit, damage, crit_damage, range="Melee"):
        """
        Initialize a Weapon.

        Args:
            name: Weapon name.
            type: Weapon type, ranged or melee.
            attack: Number of dice rolled when attacking.
            hit: Minimum dice roll needed to score a hit.
            damage: Damage inflicted on a normal hit.
            crit_damage: (not used yet) Damage inflicted on a critical hit (e.g., roll of 6).
            range: Weapon range (string, e.g. '12"' for shooting or melee (1")).
        """

        self.name = name
        self.type = type
        self.attack = attack
        self.hit = hit
        self.damage = damage
        self.crit_damage = crit_damage
        self.range = range

    def __repr__(self):
        """
        Debug-friendly representation of the weapon.
        """
        return (
            f"<{self.type} Weapon {self.name} ATK:{self.attack} HIT:{self.hit}+ "
            f"DMG:{self.damage}/{self.crit_damage} RNG:{self.range}>"
        )

    @property
    def range_inches(self):
        """
        Convert the weapon’s range into grid units (integers).

        Returns:
            int: range in grid units as an integer. Melee is always 1.
        """
        if self.range == "Melee":
            return 1
        try:
            return int(self.range.replace('"', ''))
        except (ValueError, AttributeError):
            return 0


