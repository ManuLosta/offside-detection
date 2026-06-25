from typing import Protocol

import numpy as np

from src.models.vanishing_point import VanishingPointResult


class VanishingPointEstimator(Protocol):
    def estimate(self, image: np.ndarray) -> VanishingPointResult:
        """Estimate the camera vanishing point from a football frame."""
