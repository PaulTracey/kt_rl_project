# --------------------------------------------------------------
# score module
#
# Tracks player/team points in the Tactical Advisor Prototype (TAP).
#
# Provides methods to award points for kills and objectives,
# retrieve scores, and reset the scoreboard.
# --------------------------------------------------------------
class Score:
    def __init__(self):
        """
        Initialize the score tracker.
        Each side starts with 0 points.
        """
        self.points = {
            "drop_p1": 0,
            "drop_p2": 0
        }

    def add_kill(self, team):
        """
        Award 1 point to the given team for incapacitating an enemy unit.
        """
        self.points[team] += 1

    def add_objective(self, team):
        """
        Award 1 point to the given team for holding an objective.
        """
        self.points[team] += 1

    def get_score(self, team):
        """
        Return the current score for the given team.
        Defaults to 0 if the team key does not exist.
        """
        return self.points.get(team, 0)

    def reset(self):
        """
        Reset all team scores to 0.
        """
        for team in self.points:
            self.points[team] = 0

