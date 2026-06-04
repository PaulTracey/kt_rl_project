# team/team.py
# --------------------------------------------------------------
# team module
#
# Defines the Team class for TAP.
#
# A Team is a collection of Unit objects that can be loaded
# from a JSON file and deployed into the killzone.
# --------------------------------------------------------------
import json
from .unit import Unit

class Team:
    def __init__(self, name: str, units: list[Unit]):
        """
        Initialize a Team with a name and a list of Unit objects.
        """
        self.name = name
        self.units = units

    @classmethod
    def load_teams_from_file(cls, file_path: str) -> list["Team"]:
        """
        Reads a JSON file containing a list of team-definitions.
        """
        with open(file_path, "r") as file:
            data = json.load(file)

        teams = []
        for team_data in data:
            units = []
            # Get the team name
            team_name_from_json = team_data["team_name"]
            # Build Unit objects for each entry
            for unit_data in team_data["units"]:
                unit = Unit(
                    name=unit_data["name"],
                    personal_attributes=unit_data["personal_attributes"],
                    weapons=unit_data["weapons"],
                    kill_team=team_name_from_json
                )
                units.append(unit)
            # Create and add the Team
            teams.append(Team(name=team_name_from_json, units=units))
        return teams



