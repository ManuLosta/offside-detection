import json
import logging
from pathlib import Path

from src.models.detection import PlayerDetection
from src.models.offside import OffsideResult, PlayerProjectedPoint
from src.models.pose import Keypoint
from src.models.pose import PlayerPose
from src.models.team import PlayerTeam
from src.models.vanishing_point import VanishingPointResult

logger = logging.getLogger(__name__)


def load_detections(path: Path) -> list[PlayerDetection]:
    logger.info("Loading detections from: %s", path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    detections: list[PlayerDetection] = []
    for player in payload.get("players", []):
        bbox = tuple(player["bbox"])
        detections.append(
            PlayerDetection(
                bbox=bbox,  # type: ignore[arg-type]
                confidence=float(player["confidence"]),
            )
        )
    return detections


def save_detections(
    path: Path,
    image_path: Path,
    image_shape: tuple[int, int, int],
    detections: list[PlayerDetection],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width, _ = image_shape
    payload = {
        "image": image_path.name,
        "width": width,
        "height": height,
        "players": [
            {
                "bbox": list(detection.bbox),
                "confidence": detection.confidence,
            }
            for detection in detections
        ],
    }

    logger.info("Saving detections to: %s", path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_poses(
    path: Path,
    image_path: Path,
    poses: list[PlayerPose],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "image": image_path.name,
        "players": [
            {
                "player_id": pose.player_id,
                "bbox": list(pose.bbox),
                "keypoints": [
                    {
                        "x": round(kp.x, 2),
                        "y": round(kp.y, 2),
                        "confidence": round(kp.confidence, 4),
                    }
                    for kp in pose.keypoints
                ],
            }
            for pose in poses
        ],
    }

    logger.info("Saving poses to: %s", path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_poses(path: Path) -> list[PlayerPose]:
    logger.info("Loading poses from: %s", path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    poses: list[PlayerPose] = []
    for player in payload.get("players", []):
        poses.append(
            PlayerPose(
                player_id=int(player["player_id"]),
                bbox=tuple(player["bbox"]),  # type: ignore[arg-type]
                keypoints=[
                    Keypoint(
                        x=float(keypoint["x"]),
                        y=float(keypoint["y"]),
                        confidence=float(keypoint["confidence"]),
                    )
                    for keypoint in player.get("keypoints", [])
                ],
            )
        )
    return poses


def save_teams(
    path: Path,
    teams: list[PlayerTeam],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "players": [
            {
                "player_id": team.player_id,
                "bbox": list(team.bbox),
                "cluster_id": team.cluster_id,
                "team_id": team.team_id,
            }
            for team in teams
        ]
    }

    logger.info("Saving team assignments to: %s", path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_vanishing_point(path: Path, result: VanishingPointResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "horizontal": {
            "x": (
                None
                if result.horizontal_point is None
                else round(result.horizontal_point[0], 2)
            ),
            "y": (
                None
                if result.horizontal_point is None
                else round(result.horizontal_point[1], 2)
            ),
            "filtered_lines": result.horizontal_filtered_lines_count,
            "inlier_lines": len(result.horizontal_inlier_lines),
        },
        "vertical": {
            "x": (
                None
                if result.vertical_point is None
                else round(result.vertical_point[0], 2)
            ),
            "y": (
                None
                if result.vertical_point is None
                else round(result.vertical_point[1], 2)
            ),
            "filtered_lines": result.vertical_filtered_lines_count,
            "inlier_lines": len(result.vertical_inlier_lines),
        },
        "detected_lines": result.detected_lines_count,
    }

    logger.info("Saving vanishing point to: %s", path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_vanishing_point(path: Path) -> VanishingPointResult:
    logger.info("Loading vanishing point from: %s", path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    horizontal = payload.get("horizontal", {})
    vertical = payload.get("vertical", {})

    horizontal_point = _point_from_payload(horizontal)
    vertical_point = _point_from_payload(vertical)

    return VanishingPointResult(
        horizontal_point=horizontal_point,
        vertical_point=vertical_point,
        detected_lines_count=int(payload.get("detected_lines", 0)),
        horizontal_filtered_lines_count=int(horizontal.get("filtered_lines", 0)),
        vertical_filtered_lines_count=int(vertical.get("filtered_lines", 0)),
    )


def save_offside_result(path: Path, result: OffsideResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "attacking_side": result.attacking_side,
        "attacking_team_id": result.attacking_team_id,
        "defending_team_id": result.defending_team_id,
        "reference_defender": (
            None
            if result.reference_defender is None
            else _projected_player_to_payload(result.reference_defender)
        ),
        "offside_player_ids": [
            player.player_id for player in result.offside_attackers
        ],
        "players": [
            _projected_player_to_payload(player)
            for player in result.projected_players
        ],
    }

    logger.info("Saving offside result to: %s", path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_teams(path: Path) -> list[PlayerTeam]:
    logger.info("Loading team assignments from: %s", path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    teams: list[PlayerTeam] = []
    for player in payload.get("players", []):
        teams.append(
            PlayerTeam(
                player_id=int(player["player_id"]),
                bbox=tuple(player["bbox"]),  # type: ignore[arg-type]
                cluster_id=int(player["cluster_id"]),
                team_id=int(player["team_id"]),
            )
        )
    return teams


def _point_from_payload(payload: dict) -> tuple[float, float] | None:
    x = payload.get("x")
    y = payload.get("y")
    if x is None or y is None:
        return None
    return float(x), float(y)


def _projected_player_to_payload(player: PlayerProjectedPoint) -> dict:
    return {
        "player_id": player.player_id,
        "team_id": player.team_id,
        "keypoint_index": player.keypoint_index,
        "keypoint_name": player.keypoint_name,
        "original_point": _rounded_point(player.original_point),
        "ground_center": _rounded_point(player.ground_center),
        "projected_point": _rounded_point(player.projected_point),
        "advance_score": round(player.advance_score, 6),
    }


def _rounded_point(point: tuple[float, float]) -> list[float]:
    return [round(point[0], 2), round(point[1], 2)]
