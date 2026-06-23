from dataclasses import dataclass


@dataclass(slots=True)
class PlayerTeam:
    player_id: int
    bbox: tuple[int, int, int, int]
    cluster_id: int
    team_id: int
