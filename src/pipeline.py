"""Pipeline orchestration for the offside-detection project.

The pipeline is split into three swappable stages driven by the
:class:`~src.detection.base.Detector`, :class:`~src.team.base.TeamClassifier`
and :class:`~src.pose.base.PoseEstimator` protocols. Each stage loads upstream
results from disk when available and recomputes them in memory otherwise, so
any subset of stages can be run independently.
"""

import logging
from pathlib import Path

import numpy as np

from src.detection.base import Detector
from src.detection.yolo_detector import YOLODetector
from src.models.detection import PlayerDetection
from src.models.pose import PlayerPose
from src.models.team import PlayerTeam
from src.pose.base import PoseEstimator
from src.pose.mediapipe_pose_estimator import MediaPipePoseEstimator
from src.team.base import TeamClassifier
from src.team.kmeans_classifier import KMeansTeamClassifier
from src.utils.image import draw_detections, load_image, save_image
from src.utils.json_io import (
    load_detections,
    load_teams,
    save_detections,
    save_poses,
    save_teams,
)
from src.visualize.draw_pose import draw_poses
from src.visualize.draw_teams import draw_team_assignments

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    stages: tuple[bool, bool, bool],
    *,
    detector: Detector | None = None,
    team_classifier: TeamClassifier | None = None,
    pose_estimator: PoseEstimator | None = None,
) -> int:
    """Run the detection / team / pose pipeline.

    ``stages`` is ``(run_detection, run_teams, run_poses)``; at least one stage
    must be requested, otherwise ``ValueError`` is raised. Engines default to
    the YOLO/KMeans/MediaPipe implementations; pass custom objects satisfying
    the protocols to swap them (e.g. ``pose_estimator=YOLOPoseEstimator()``).
    """
    run_detection, run_teams, run_poses = stages
    if not (run_detection or run_teams or run_poses):
        raise ValueError(
            "At least one stage must be enabled in `stages`; got (False, False, False)."
        )

    if not input_path.exists():
        logger.error("Input image not found: %s", input_path)
        return 1

    detections_dir = output_dir / "detections"
    teams_dir = output_dir / "teams"
    poses_dir = output_dir / "poses"

    image = load_image(input_path)
    stem = input_path.stem
    detections: list[PlayerDetection] | None = None
    teams: list[PlayerTeam] | None = None

    if run_detection:
        detections = _run_detection(
            detector or YOLODetector(), image, input_path, stem, detections_dir
        )

    if run_teams:
        classifier = team_classifier or KMeansTeamClassifier()
        detections = _resolve_detections(
            detections, image, detections_dir / f"{stem}_detections.json", detector
        )
        teams = _run_teams(classifier, image, detections, stem, teams_dir)

    if run_poses:
        estimator = pose_estimator or MediaPipePoseEstimator()
        detections = _resolve_detections(
            detections, image, detections_dir / f"{stem}_detections.json", detector
        )
        teams = _resolve_teams(
            teams, image, detections, teams_dir / f"{stem}_teams.json", team_classifier
        )
        _run_poses(
            estimator, image, detections, teams, input_path, stem, poses_dir
        )

    logger.info("Pipeline complete. Outputs saved under: %s", output_dir)
    return 0


def _run_detection(
    detector: Detector,
    image: np.ndarray,
    input_path: Path,
    stem: str,
    detections_dir: Path,
) -> list[PlayerDetection]:
    detections = detector.detect(image)
    annotated_image = draw_detections(image, detections)

    json_path = detections_dir / f"{stem}_detections.json"
    annotated_image_path = detections_dir / f"{stem}_annotated.jpg"

    save_detections(json_path, input_path, image.shape, detections)
    save_image(annotated_image_path, annotated_image)
    logger.info("Total detected players: %d", len(detections))
    return detections


def _run_teams(
    team_classifier: TeamClassifier,
    image: np.ndarray,
    detections: list[PlayerDetection],
    stem: str,
    teams_dir: Path,
) -> list[PlayerTeam]:
    teams = team_classifier.classify(image, detections)

    teams_json_path = teams_dir / f"{stem}_teams.json"
    teams_image_path = teams_dir / f"{stem}_teams.jpg"
    save_teams(teams_json_path, teams)
    teams_image = draw_team_assignments(image, teams)
    save_image(teams_image_path, teams_image)
    logger.info("Total classified players: %d", len(teams))
    return teams


def _run_poses(
    pose_estimator: PoseEstimator,
    image: np.ndarray,
    detections: list[PlayerDetection],
    teams: list[PlayerTeam],
    input_path: Path,
    stem: str,
    poses_dir: Path,
) -> list[PlayerPose]:
    classified_ids = [team.player_id for team in teams]
    filtered_detections = [detections[i] for i in classified_ids]

    poses = pose_estimator.estimate(image, filtered_detections, player_ids=classified_ids)
    pose_annotated_image = draw_poses(image, poses)

    poses_json_path = poses_dir / f"{stem}_poses.json"
    poses_image_path = poses_dir / f"{stem}_players_pose.jpg"

    save_poses(poses_json_path, input_path, poses)
    save_image(poses_image_path, pose_annotated_image)
    logger.info("Total players with valid poses: %d", len(poses))
    return poses


def _resolve_detections(
    detections: list[PlayerDetection] | None,
    image: np.ndarray,
    detections_path: Path,
    detector: Detector | None,
) -> list[PlayerDetection]:
    """Return detections from memory, or load/recompute them from disk."""
    if detections is not None:
        return detections
    if detections_path.exists():
        return load_detections(detections_path)
    logger.warning(
        "Detections not found at %s; running detection in memory for downstream stage.",
        detections_path,
    )
    return (detector or YOLODetector()).detect(image)


def _resolve_teams(
    teams: list[PlayerTeam] | None,
    image: np.ndarray,
    detections: list[PlayerDetection],
    teams_path: Path,
    team_classifier: TeamClassifier | None,
) -> list[PlayerTeam]:
    """Return teams from memory, or load/recompute them from disk."""
    if teams is not None:
        return teams
    if teams_path.exists():
        return load_teams(teams_path)
    logger.warning(
        "Teams not found at %s; running team classification in memory for downstream stage.",
        teams_path,
    )
    return (team_classifier or KMeansTeamClassifier()).classify(image, detections)
