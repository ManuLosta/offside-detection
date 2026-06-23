import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

GRASS_LOWER_HSV = (35, 40, 40)
GRASS_UPPER_HSV = (90, 255, 255)

HUE_BINS = 32
SATURATION_BINS = 32
HUE_RANGE = (0, 180)
SATURATION_RANGE = (0, 256)

FEATURE_DIM = HUE_BINS + SATURATION_BINS


def extract_features(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Extract a normalized HSV histogram feature vector from a crop.

    Grass pixels are masked out before histogram computation. If the crop is
    empty or contains only grass, a zero vector is returned.
    """
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        logger.warning("Empty crop for bbox %s; returning zero feature vector", bbox)
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, GRASS_LOWER_HSV, GRASS_UPPER_HSV)
    non_grass_mask = cv2.bitwise_not(grass_mask)

    if cv2.countNonZero(non_grass_mask) == 0:
        logger.warning(
            "Crop for bbox %s contains only grass; returning zero feature vector", bbox
        )
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    hue_hist = cv2.calcHist([hsv], [0], non_grass_mask, [HUE_BINS], HUE_RANGE)
    saturation_hist = cv2.calcHist(
        [hsv], [1], non_grass_mask, [SATURATION_BINS], SATURATION_RANGE
    )

    cv2.normalize(hue_hist, hue_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    cv2.normalize(saturation_hist, saturation_hist, alpha=1.0, norm_type=cv2.NORM_L1)

    features = np.concatenate([hue_hist.flatten(), saturation_hist.flatten()])
    return features.astype(np.float32)
