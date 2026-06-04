# team/combat.py
# --------------------------------------------------------------
# combat module
#
# Core combat mechanics for the Tactical Advisor Prototype (TAP).
#
# Provides helper functions for resolving attacks, handling dice rolls,
# applying damage, and managing incapacitation.
# --------------------------------------------------------------
from team.dice import Dice

def get_weapon(unit, weapon_type="Shooting"):
    """
    Returns the first weapon of the given type from a unit's weapon list.
    Falls back to the first weapon if none match.
    """
    for weapon in unit.weapons:
        if weapon.type == weapon_type:
            return weapon
    return unit.weapons[0] if unit.weapons else None


def attack_with_weapon(attacker, target, weapon, dice_panel=None):
    """
      Resolve a single attack action with the given weapon.
      Process:
        1. Attacker rolls to hit.
        2. Defender rolls to save.
        3. Compute net hits and apply damage.
        4. Update target wounds and incapacitation state.
        5. Log results to dice panel if available.
      """
    # Create a Dice roller instance
    dice = Dice()
    print(f"[DEBUG] Using weapon: {weapon}")

    # 1. Attacker rolls to hit
    attack_rolls = dice.roll(weapon.attack)
    hits = dice.count_hits(attack_rolls, weapon.hit)

    # 2. Defender rolls to save
    # for these simplified rules always 3.
    # can be changed when modifiers are introduced
    defense_rolls = dice.roll(3)
    saves = dice.count_hits(defense_rolls, target.save)

    # 3. Net hits and damage
    net_hits = max(hits - saves, 0)
    damage = net_hits * weapon.damage
    target.current_wounds -= damage

    # 4. Incapacitation logic
    if target.current_wounds <= 0:
        target.incapacitated = True
        target.current_wounds = 0

    # 5. Log/dice panel output
    log = (
        f"<b>Dice:</b><br>"
        f"{attacker.name} attacks {target.name} with {weapon.name}<br>"
        f"Attack Rolls: {attack_rolls} → {hits} hits<br>"
        f"Defense Rolls: {defense_rolls} → {saves} saves<br>"
        f"{net_hits} net hits → {damage} damage<br>"
        f"{target.name} now has {target.current_wounds} wounds"
    )
    # Display in dice panel if provided, otherwise print to console
    if dice_panel:
        dice_panel.html_text = log
        dice_panel.rebuild()
    else:
        print(log)
