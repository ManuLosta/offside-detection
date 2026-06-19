import logging

import cv2
import numpy as np

from src.models.pose import PlayerPose

logger = logging.getLogger(__name__)

# 0-indexed COCO pose skeleton edges.
SKELETON: list[tuple[int, int]] = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]


def draw_poses(
    image: np.ndarray,
    poses: list[PlayerPose],
    bbox_color: tuple[int, int, int] = (0, 255, 0),
    keypoint_color: tuple[int, int, int] = (0, 0, 255),
    skeleton_color: tuple[int, int, int] = (255, 0, 0),
    bbox_thickness: int = 2,
    keypoint_radius: int = 4,
    skeleton_thickness: int = 2,
) -> np.ndarray:
    annotated = image.copy()

    for pose in poses:
        x1, y1, x2, y2 = pose.bbox
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            bbox_color,
            bbox_thickness,
        )
        cv2.putText(
            annotated,
            f"player_id: {pose.player_id}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            bbox_color,
            bbox_thickness,
        )

        for start_idx, end_idx in SKELETON:
            if start_idx >= len(pose.keypoints) or end_idx >= len(pose.keypoints):
                continue

            start_kp = pose.keypoints[start_idx]
            end_kp = pose.keypoints[end_idx]
            if start_kp.confidence <= 0 or end_kp.confidence <= 0:
                continue

            pt_start = (int(round(start_kp.x)), int(round(start_kp.y)))
            pt_end = (int(round(end_kp.x)), int(round(end_kp.y)))
            cv2.line(
                annotated,
                pt_start,
                pt_end,
                skeleton_color,
                skeleton_thickness,
            )

        for kp in pose.keypoints:
            if kp.confidence <= 0:
                continue

            pt = (int(round(kp.x)), int(round(kp.y)))
            cv2.circle(annotated, pt, keypoint_radius, keypoint_color, -1)

    logger.info("Drew poses for %d player(s)", len(poses))
    return annotated
