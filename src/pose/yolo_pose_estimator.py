import logging

import numpy as np
from ultralytics import YOLO

from src.config import (
    DEVICE,
    POSE_CONFIDENCE_THRESHOLD,
    POSE_CROP_PADDING,
    POSE_IMAGE_SIZE,
    YOLO_POSE_MODEL,
)
from src.models.detection import PlayerDetection
from src.models.pose import Keypoint, PlayerPose

logger = logging.getLogger(__name__)


class YOLOPoseEstimator:
    def __init__(
        self,
        model_name: str = YOLO_POSE_MODEL,
        img_size: int = POSE_IMAGE_SIZE,
        confidence_threshold: float = POSE_CONFIDENCE_THRESHOLD,
        crop_padding: int = POSE_CROP_PADDING,
        device: str = DEVICE,
    ) -> None:
        self.model_name = model_name
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.crop_padding = crop_padding
        self.device = device

        logger.info("Loading YOLO pose model: %s", model_name)
        self.model = YOLO(model_name)
        logger.info("YOLO pose model loaded")

    def estimate(
        self,
        image: np.ndarray,
        detections: list[PlayerDetection],
        player_ids: list[int] | None = None,
    ) -> list[PlayerPose]:
        if image.ndim != 3:
            raise ValueError(f"Expected 3-channel BGR image, got ndim={image.ndim}")

        if player_ids is None:
            player_ids = list(range(len(detections)))
        if len(player_ids) != len(detections):
            raise ValueError(
                f"player_ids count ({len(player_ids)}) does not match detection count ({len(detections)})"
            )

        image_height, image_width = image.shape[:2]
        poses: list[PlayerPose] = []

        for player_id, detection in zip(player_ids, detections):
            x1, y1, x2, y2 = detection.bbox

            crop_x1 = max(0, x1 - self.crop_padding)
            crop_y1 = max(0, y1 - self.crop_padding)
            crop_x2 = min(image_width, x2 + self.crop_padding)
            crop_y2 = min(image_height, y2 + self.crop_padding)

            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                logger.warning(
                    "Empty crop for player_id=%d bbox=%s; skipping pose estimation",
                    player_id,
                    detection.bbox,
                )
                continue

            results = self.model(
                crop,
                imgsz=self.img_size,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )[0]

            if results.keypoints is None or len(results.keypoints) == 0:
                logger.warning(
                    "No pose detected for player_id=%d bbox=%s",
                    player_id,
                    detection.bbox,
                )
                continue

            pose_index = self._select_pose_index(results, crop_x1, crop_y1, crop_x2, crop_y2)
            keypoints = self._extract_keypoints(results, pose_index, crop_x1, crop_y1)

            poses.append(
                PlayerPose(
                    player_id=player_id,
                    bbox=detection.bbox,
                    keypoints=keypoints,
                )
            )

        logger.info("Estimated poses for %d/%d player(s)", len(poses), len(detections))
        return poses

    def _select_pose_index(
        self,
        results,
        crop_x1: int,
        crop_y1: int,
        crop_x2: int,
        crop_y2: int,
    ) -> int:
        num_poses = len(results.keypoints)
        if num_poses == 1:
            return 0

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        crop_cx = crop_w / 2
        crop_cy = crop_h / 2

        boxes = results.boxes.xyxy.cpu().numpy()
        centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))
        distances = np.linalg.norm(centers - np.array([crop_cx, crop_cy]), axis=1)
        return int(np.argmin(distances))

    def _extract_keypoints(
        self,
        results,
        pose_index: int,
        offset_x: int,
        offset_y: int,
    ) -> list[Keypoint]:
        keypoints_data = results.keypoints.data[pose_index].cpu().numpy()

        extracted: list[Keypoint] = []
        for x, y, conf in keypoints_data:
            extracted.append(
                Keypoint(
                    x=float(x + offset_x),
                    y=float(y + offset_y),
                    confidence=float(conf),
                )
            )
        return extracted
