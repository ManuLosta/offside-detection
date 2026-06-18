import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.config import CONFIDENCE_THRESHOLD, DEVICE, MODEL_NAME, PERSON_CLASS_ID
from src.models.detection import PlayerDetection

logger = logging.getLogger(__name__)


class YOLODetector:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        device: str = DEVICE,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        logger.info("Loading YOLO model: %s", model_name)
        self.model = YOLO(model_name)
        logger.info("YOLO model loaded")

    def detect(self, image: np.ndarray | Path | str) -> list[PlayerDetection]:
        if isinstance(image, (str, Path)):
            source = str(image)
            logger.info("Running inference on: %s", source)
        else:
            source = image
            logger.info("Running inference on image array")

        results = self.model(
            source,
            conf=self.confidence_threshold,
            classes=[PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )[0]

        detections: list[PlayerDetection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().item())
            detections.append(
                PlayerDetection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=round(confidence, 4),
                )
            )

        logger.info("Detected %d player(s)", len(detections))
        return detections
