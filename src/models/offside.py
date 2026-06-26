from dataclasses import dataclass

from src.models.vanishing_point import Point


@dataclass(slots=True)
class PlayerProjectedPoint:
    player_id: int
    team_id: int
    keypoint_index: int
    keypoint_name: str
    original_point: Point
    ground_center: Point
    projected_point: Point
    advance_score: float


@dataclass(slots=True)
class OffsideResult:
    projected_players: list[PlayerProjectedPoint]
    offside_attackers: list[PlayerProjectedPoint]
    reference_defender: PlayerProjectedPoint | None
    attacking_team_id: int
    defending_team_id: int
    attacking_side: str
