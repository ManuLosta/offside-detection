from typing import Protocol

import numpy as np

from src.models.vanishing_point import VanishingPointResult


class VanishingPointEstimator(Protocol):
    def estimate(self, image: np.ndarray) -> VanishingPointResult:
        """Estimate field vanishing points from a football frame."""
