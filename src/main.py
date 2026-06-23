import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from src.config import DEFAULT_INPUT_IMAGE, OUTPUT_DIR
from src.detection.yolo_detector import YOLODetector
from src.models.detection import PlayerDetection
from src.models.team import PlayerTeam
from src.pose.yolo_pose_estimator import YOLOPoseEstimator
from src.team.color_features import extract_features
from src.team.kmeans_classifier import classify_teams
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect football players and estimate poses in a match image."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_IMAGE,
        help="Path to the input football match image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Root directory where detection and pose outputs are saved.",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Run player detection only.",
    )
    parser.add_argument(
        "--teams",
        action="store_true",
        help="Run team classification only (requires existing detection output).",
    )
    parser.add_argument(
        "--poses",
        action="store_true",
        help="Run pose estimation only on classified players (requires existing team output).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path: Path = args.input
    output_dir: Path = args.output_dir

    if not input_path.exists():
        logger.error("Input image not found: %s", input_path)
        return 1

    if args.detect:
        run_detection = True
        run_teams = False
        run_poses = False
    elif args.teams:
        run_detection = False
        run_teams = True
        run_poses = False
    elif args.poses:
        run_detection = False
        run_teams = False
        run_poses = True
    else:
        run_detection = True
        run_teams = True
        run_poses = True

    detections_dir = output_dir / "detections"
    teams_dir = output_dir / "teams"
    poses_dir = output_dir / "poses"
    detections_dir.mkdir(parents=True, exist_ok=True)
    teams_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path)
    stem = input_path.stem
    detections: list[PlayerDetection] | None = None

    if run_detection:
        detector = YOLODetector()
        detections = detector.detect(image)
        annotated_image = draw_detections(image, detections)

        json_path = detections_dir / f"{stem}_detections.json"
        annotated_image_path = detections_dir / f"{stem}_annotated.jpg"

        save_detections(json_path, input_path, image.shape, detections)
        save_image(annotated_image_path, annotated_image)
        logger.info("Total detected players: %d", len(detections))

    teams: list[PlayerTeam] | None = None

    if run_teams:
        if detections is None:
            detections_path = detections_dir / f"{stem}_detections.json"
            if detections_path.exists():
                detections = load_detections(detections_path)
            else:
                logger.warning(
                    "Detections not found at %s; running detection in memory for team classification.",
                    detections_path,
                )
                detector = YOLODetector()
                detections = detector.detect(image)

        features_list = [extract_features(image, detection.bbox) for detection in detections]
        features = np.stack(features_list) if features_list else np.empty((0, 64))
        all_teams = classify_teams(features, detections)

        unknown_count = sum(1 for team in all_teams if team.team_id == -1)
        teams = [team for team in all_teams if team.team_id != -1]
        logger.info(
            "Filtered %d unknown player(s); keeping %d classified player(s)",
            unknown_count,
            len(teams),
        )

        teams_json_path = teams_dir / f"{stem}_teams.json"
        teams_image_path = teams_dir / f"{stem}_teams.jpg"
        save_teams(teams_json_path, teams)
        teams_image = draw_team_assignments(image, teams)
        save_image(teams_image_path, teams_image)
        logger.info("Total classified players: %d", len(teams))

    if run_poses:
        if detections is None:
            detections_path = detections_dir / f"{stem}_detections.json"
            if detections_path.exists():
                detections = load_detections(detections_path)
            else:
                logger.warning(
                    "Detections not found at %s; running detection in memory for pose estimation.",
                    detections_path,
                )
                detector = YOLODetector()
                detections = detector.detect(image)

        if teams is None:
            teams_path = teams_dir / f"{stem}_teams.json"
            if teams_path.exists():
                teams = load_teams(teams_path)
            else:
                logger.warning(
                    "Teams not found at %s; running team classification in memory for pose estimation.",
                    teams_path,
                )
                features_list = [extract_features(image, detection.bbox) for detection in detections]
                features = np.stack(features_list) if features_list else np.empty((0, 64))
                all_teams = classify_teams(features, detections)
                teams = [team for team in all_teams if team.team_id != -1]

        classified_ids = [team.player_id for team in teams]
        filtered_detections = [detections[i] for i in classified_ids]

        pose_estimator = YOLOPoseEstimator()
        poses = pose_estimator.estimate(image, filtered_detections, player_ids=classified_ids)
        pose_annotated_image = draw_poses(image, poses)

        poses_json_path = poses_dir / f"{stem}_poses.json"
        poses_image_path = poses_dir / f"{stem}_players_pose.jpg"

        save_poses(poses_json_path, input_path, poses)
        save_image(poses_image_path, pose_annotated_image)
        logger.info("Total players with valid poses: %d", len(poses))

    logger.info("Pipeline complete. Outputs saved under: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
