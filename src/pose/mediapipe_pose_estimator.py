import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.config import (
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_PRESENCE_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    MEDIAPIPE_POSE_MODEL,
    POSE_CROP_PADDING,
    PROJECT_ROOT,
)
from src.models.detection import PlayerDetection
from src.models.pose import Keypoint, PlayerPose

logger = logging.getLogger(__name__)

# MediaPipe BlazePose emits 33 landmarks; the project's visualization, JSON
# output and skeleton expect the 17-keypoint COCO layout. This table maps each
# COCO index to its BlazePose counterpart.
COCO_FROM_BLAZE: tuple[int, ...] = (
    0,   # COCO 0  nose           <- Blaze 0  nose
    2,   # COCO 1  left_eye       <- Blaze 2  left_eye
    5,   # COCO 2  right_eye      <- Blaze 5  right_eye
    7,   # COCO 3  left_ear       <- Blaze 7  left_ear
    8,   # COCO 4  right_ear      <- Blaze 8  right_ear
    11,  # COCO 5  left_shoulder  <- Blaze 11 left_shoulder
    12,  # COCO 6  right_shoulder <- Blaze 12 right_shoulder
    13,  # COCO 7  left_elbow     <- Blaze 13 left_elbow
    14,  # COCO 8  right_elbow    <- Blaze 14 right_elbow
    15,  # COCO 9  left_wrist     <- Blaze 15 left_wrist
    16,  # COCO 10 right_wrist    <- Blaze 16 right_wrist
    23,  # COCO 11 left_hip       <- Blaze 23 left_hip
    24,  # COCO 12 right_hip      <- Blaze 24 right_hip
    25,  # COCO 13 left_knee      <- Blaze 25 left_knee
    26,  # COCO 14 right_knee     <- Blaze 26 right_knee
    27,  # COCO 15 left_ankle     <- Blaze 27 left_ankle
    28,  # COCO 16 right_ankle    <- Blaze 28 right_ankle
)


class MediaPipePoseEstimator:
    """Pose estimator backed by MediaPipe's PoseLandmarker (Tasks API).

    Produces 17-keypoint COCO-layout :class:`PlayerPose` results by running the
    BlazePose landmarker on a padded crop of each detected player and mapping
    the 33 BlazePose landmarks down to the COCO subset. Satisfies the
    :class:`~src.pose.base.PoseEstimator` protocol.
    """

    def __init__(
        self,
        model_path: str = MEDIAPIPE_POSE_MODEL,
        crop_padding: int = POSE_CROP_PADDING,
        min_detection_confidence: float = MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_presence_confidence: float = MEDIAPIPE_MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence: float = MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    ) -> None:
        resolved = Path(model_path)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / model_path
        if not resolved.exists():
            raise FileNotFoundError(
                f"MediaPipe pose model not found: {resolved}. "
                "Download pose_landmarker_full.task from the MediaPipe model "
                "registry and place it at the project root."
            )

        self.model_path = str(resolved)
        self.crop_padding = crop_padding
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as base_options_lib

        logger.info("Loading MediaPipe pose model: %s", self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options_lib.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        logger.info("MediaPipe pose model loaded")

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

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            result = self._landmarker.detect(mp_image)

            if not result.pose_landmarks:
                logger.warning(
                    "No pose detected for player_id=%d bbox=%s",
                    player_id,
                    detection.bbox,
                )
                continue

            landmarks = result.pose_landmarks[0]
            keypoints = self._extract_keypoints(
                landmarks, crop_x1, crop_y1, crop_x2, crop_y2
            )

            poses.append(
                PlayerPose(
                    player_id=player_id,
                    bbox=detection.bbox,
                    keypoints=keypoints,
                )
            )

        logger.info("Estimated poses for %d/%d player(s)", len(poses), len(detections))
        return poses

    def _extract_keypoints(
        self,
        landmarks,
        crop_x1: int,
        crop_y1: int,
        crop_x2: int,
        crop_y2: int,
    ) -> list[Keypoint]:
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        extracted: list[Keypoint] = []
        for blaze_idx in COCO_FROM_BLAZE:
            lm = landmarks[blaze_idx]
            visibility = lm.visibility if lm.visibility is not None else 0.0
            extracted.append(
                Keypoint(
                    x=float(lm.x * crop_w + crop_x1),
                    y=float(lm.y * crop_h + crop_y1),
                    confidence=float(visibility),
                )
            )
        return extracted

    def close(self) -> None:
        landmarker = getattr(self, "_landmarker", None)
        if landmarker is not None:
            landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "MediaPipePoseEstimator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def __del__(self) -> None:
        self.close()
