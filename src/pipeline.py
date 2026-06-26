"""Pipeline orchestration for the offside-detection project.

The pipeline is split into swappable stages driven by detector, team
classifier, pose estimator and vanishing-point estimator protocols. Downstream
stages load upstream results from disk when available and recompute them in
memory otherwise, so any subset of stages can be run independently.
"""

import logging
from pathlib import Path

import numpy as np

from src.detection.base import Detector
from src.detection.yolo_detector import YOLODetector
from src.models.detection import PlayerDetection
from src.models.offside import OffsideResult
from src.models.pose import PlayerPose
from src.models.team import PlayerTeam
from src.models.vanishing_point import VanishingPointResult
from src.offside.calculator import OffsideCalculator
from src.perspective.base import VanishingPointEstimator
from src.perspective.field_vanishing_point import FieldVanishingPointEstimator
from src.pose.base import PoseEstimator
from src.pose.mediapipe_pose_estimator import MediaPipePoseEstimator
from src.pose.yolo_pose_estimator import YOLOPoseEstimator
from src.team.base import TeamClassifier
from src.team.kmeans_classifier import KMeansTeamClassifier
from src.utils.image import draw_detections, load_image, save_image
from src.utils.json_io import (
    load_detections,
    load_poses,
    load_teams,
    load_vanishing_point,
    save_detections,
    save_offside_result,
    save_poses,
    save_teams,
    save_vanishing_point,
)
from src.visualize.draw_offside import draw_offside_detection
from src.visualize.draw_pose import draw_poses
from src.visualize.draw_teams import draw_team_assignments
from src.visualize.draw_vanishing_point import draw_vanishing_point

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    stages: tuple[bool, bool, bool, bool] | tuple[bool, bool, bool, bool, bool],
    *,
    detector: Detector | None = None,
    team_classifier: TeamClassifier | None = None,
    pose_estimator: PoseEstimator | None = None,
    vanishing_point_estimator: VanishingPointEstimator | None = None,
    offside_calculator: OffsideCalculator | None = None,
) -> int:
    run_detection, run_teams, run_poses, run_vanishing_point, run_offside = (
        _normalize_stages(stages)
    )
    if not (
        run_detection
        or run_teams
        or run_poses
        or run_vanishing_point
        or run_offside
    ):
        raise ValueError(
            "At least one stage must be enabled in `stages`; "
            "got all False values."
        )

    if not input_path.exists():
        logger.error("Input image not found: %s", input_path)
        return 1

    detections_dir = output_dir / "detections"
    teams_dir = output_dir / "teams"
    poses_dir = output_dir / "poses"
    vanishing_point_dir = output_dir / "vanishing_point"
    offside_dir = output_dir / "offside"

    image = load_image(input_path)
    stem = input_path.stem
    detections: list[PlayerDetection] | None = None
    teams: list[PlayerTeam] | None = None
    poses: list[PlayerPose] | None = None
    vanishing_point: VanishingPointResult | None = None

    if run_vanishing_point:
        vanishing_point = _run_vanishing_point(
            vanishing_point_estimator or FieldVanishingPointEstimator(),
            image,
            stem,
            vanishing_point_dir,
        )

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
        estimator = pose_estimator or YOLOPoseEstimator()
        detections = _resolve_detections(
            detections, image, detections_dir / f"{stem}_detections.json", detector
        )
        teams = _resolve_teams(
            teams, image, detections, teams_dir / f"{stem}_teams.json", team_classifier
        )
        poses = _run_poses(
            estimator, image, detections, teams, input_path, stem, poses_dir
        )

    if run_offside:
        estimator = pose_estimator or YOLOPoseEstimator()
        detections = _resolve_detections(
            detections, image, detections_dir / f"{stem}_detections.json", detector
        )
        teams = _resolve_teams(
            teams, image, detections, teams_dir / f"{stem}_teams.json", team_classifier
        )
        poses = _resolve_poses(
            poses,
            image,
            detections,
            teams,
            poses_dir / f"{stem}_poses.json",
            estimator,
        )
        vanishing_point = _resolve_vanishing_point(
            vanishing_point,
            image,
            vanishing_point_dir / f"{stem}_vanishing_point.json",
            vanishing_point_estimator,
        )
        _run_offside(
            offside_calculator or OffsideCalculator(),
            image,
            poses,
            teams,
            vanishing_point,
            stem,
            offside_dir,
        )

    logger.info("Pipeline complete. Outputs saved under: %s", output_dir)
    return 0


def _normalize_stages(
    stages: tuple[bool, bool, bool, bool] | tuple[bool, bool, bool, bool, bool],
) -> tuple[bool, bool, bool, bool, bool]:
    if len(stages) == 4:
        run_detection, run_teams, run_poses, run_vanishing_point = stages
        return run_detection, run_teams, run_poses, run_vanishing_point, False
    return stages


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

    poses = pose_estimator.estimate(
        image, filtered_detections, player_ids=classified_ids
    )
    pose_annotated_image = draw_poses(image, poses)

    poses_json_path = poses_dir / f"{stem}_poses.json"
    poses_image_path = poses_dir / f"{stem}_players_pose.jpg"

    save_poses(poses_json_path, input_path, poses)
    save_image(poses_image_path, pose_annotated_image)
    logger.info("Total players with valid poses: %d", len(poses))
    return poses


def _run_vanishing_point(
    estimator: VanishingPointEstimator,
    image: np.ndarray,
    stem: str,
    vanishing_point_dir: Path,
) -> VanishingPointResult:
    result = estimator.estimate(image)
    annotated_image = draw_vanishing_point(image, result)

    json_path = vanishing_point_dir / f"{stem}_vanishing_point.json"
    annotated_image_path = vanishing_point_dir / f"{stem}_vanishing_point.jpg"

    save_vanishing_point(json_path, result)
    save_image(annotated_image_path, annotated_image)
    logger.info(
        "Vanishing points horizontal=%s vertical=%s; "
        "lines detected=%d horizontal filtered/inliers=%d/%d "
        "vertical filtered/inliers=%d/%d",
        result.horizontal_point,
        result.vertical_point,
        result.detected_lines_count,
        result.horizontal_filtered_lines_count,
        len(result.horizontal_inlier_lines),
        result.vertical_filtered_lines_count,
        len(result.vertical_inlier_lines),
    )
    return result


def _run_offside(
    calculator: OffsideCalculator,
    image: np.ndarray,
    poses: list[PlayerPose],
    teams: list[PlayerTeam],
    vanishing_point: VanishingPointResult,
    stem: str,
    offside_dir: Path,
) -> OffsideResult:
    result = calculator.detect(poses, teams, vanishing_point)

    if vanishing_point.vertical_point is None:
        raise ValueError("Vertical vanishing point is required to draw offside line")

    offside_json_path = offside_dir / f"{stem}_offside.json"
    offside_image_path = offside_dir / f"{stem}_offside.jpg"
    save_offside_result(offside_json_path, result)

    offside_image = draw_offside_detection(
        image,
        result,
        vanishing_point.vertical_point,
    )
    save_image(offside_image_path, offside_image)
    logger.info(
        "Offside detection complete: projected players=%d, offside attackers=%d, "
        "reference defender=%s",
        len(result.projected_players),
        len(result.offside_attackers),
        (
            None
            if result.reference_defender is None
            else result.reference_defender.player_id
        ),
    )
    return result


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


def _resolve_poses(
    poses: list[PlayerPose] | None,
    image: np.ndarray,
    detections: list[PlayerDetection],
    teams: list[PlayerTeam],
    poses_path: Path,
    pose_estimator: PoseEstimator,
) -> list[PlayerPose]:
    """Return poses from memory, or load/recompute them from disk."""
    if poses is not None:
        return poses
    if poses_path.exists():
        return load_poses(poses_path)
    logger.warning(
        "Poses not found at %s; running pose estimation in memory for offside stage.",
        poses_path,
    )
    classified_ids = [team.player_id for team in teams]
    filtered_detections = [detections[i] for i in classified_ids]
    return pose_estimator.estimate(image, filtered_detections, player_ids=classified_ids)


def _resolve_vanishing_point(
    vanishing_point: VanishingPointResult | None,
    image: np.ndarray,
    vanishing_point_path: Path,
    vanishing_point_estimator: VanishingPointEstimator | None,
) -> VanishingPointResult:
    """Return vanishing point result from memory, or load/recompute it from disk."""
    if vanishing_point is not None:
        return vanishing_point
    if vanishing_point_path.exists():
        return load_vanishing_point(vanishing_point_path)
    logger.warning(
        "Vanishing point not found at %s; running estimation in memory for offside stage.",
        vanishing_point_path,
    )
    return (vanishing_point_estimator or FieldVanishingPointEstimator()).estimate(image)
