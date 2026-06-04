# team/dice.py
# --------------------------------------------------------------
# dice module
#
# Class for simulating dice rolls in the Tactical Advisor Prototype (TAP).
#
# Provides methods for rolling standard dice and counting hits against
# a given threshold, used in attack and defense resolution.
# --------------------------------------------------------------
import random

class Dice:
    def __init__(self, sides=6):
        """
        Create a dice roller. Default is a standard six-sided die (D6).
        """
        self.sides = sides

    def roll(self, num=1):
        """
        Rolls num dice and returns a list of results.
        """
        return [random.randint(1, self.sides) for _ in range(num)]

    def count_hits(self, rolls, hit_threshold):
        """
        Counts how many dice equal or exceed the given hit threshold.
        """
        return sum(1 for roll in rolls if roll >= hit_threshold)
