import argparse
import logging
import sys
from pathlib import Path

from src.config import DEFAULT_INPUT_IMAGE, OUTPUT_DIR
from src.detection.yolo_detector import YOLODetector
from src.utils.image import draw_detections, load_image, save_image
from src.utils.json_io import save_detections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect football players in a match image using YOLO11s."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_IMAGE,
        help="Path to the input football match image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where JSON and annotated image are saved.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path: Path = args.input
    output_dir: Path = args.output_dir

    if not input_path.exists():
        logger.error("Input image not found: %s", input_path)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector()

    image = load_image(input_path)
    detections = detector.detect(image)
    annotated_image = draw_detections(image, detections)

    stem = input_path.stem
    json_path = output_dir / f"{stem}_detections.json"
    annotated_image_path = output_dir / f"{stem}_annotated.jpg"

    save_detections(json_path, input_path, image.shape, detections)
    save_image(annotated_image_path, annotated_image)

    logger.info("Pipeline complete. Outputs saved to: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
