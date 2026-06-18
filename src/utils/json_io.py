import json
import logging
from pathlib import Path

from src.models.detection import PlayerDetection

logger = logging.getLogger(__name__)


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
