import logging
from pathlib import Path

import cv2
import numpy as np

from src.models.detection import PlayerDetection

logger = logging.getLogger(__name__)


def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    logger.info("Loading image: %s", path)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")

    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving annotated image: %s", path)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise RuntimeError(f"Failed to save image: {path}")


def draw_detections(
    image: np.ndarray,
    detections: list[PlayerDetection],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        label = f"player: {detection.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness,
        )
    return annotated
