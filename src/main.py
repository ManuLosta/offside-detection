import argparse
import logging
import sys
from pathlib import Path

from src.config import DEFAULT_INPUT_IMAGE, OUTPUT_DIR
from src.detection.yolo_detector import YOLODetector
from src.models.detection import PlayerDetection
from src.pose.yolo_pose_estimator import YOLOPoseEstimator
from src.utils.image import draw_detections, load_image, save_image
from src.utils.json_io import load_detections, save_detections, save_poses
from src.visualize.draw_pose import draw_poses

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
        "--poses",
        action="store_true",
        help="Run pose estimation only (requires existing detection output).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path: Path = args.input
    output_dir: Path = args.output_dir

    if not input_path.exists():
        logger.error("Input image not found: %s", input_path)
        return 1

    run_detection = args.detect or not args.poses
    run_poses = args.poses or not args.detect

    detections_dir = output_dir / "detections"
    poses_dir = output_dir / "poses"
    detections_dir.mkdir(parents=True, exist_ok=True)
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

    if run_poses:
        pose_estimator = YOLOPoseEstimator()

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

        poses = pose_estimator.estimate(image, detections)
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
