import json
import logging
from pathlib import Path

from src.models.detection import PlayerDetection
from src.models.pose import PlayerPose
from src.models.team import PlayerTeam

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
