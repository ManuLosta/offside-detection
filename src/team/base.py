from typing import Protocol, runtime_checkable

import numpy as np

from src.models.detection import PlayerDetection
from src.models.team import PlayerTeam


@runtime_checkable
class TeamClassifier(Protocol):
    """Abstract team classifier.

    Implementations assign a ``team_id`` (``0`` or ``1`` for the two teams,
    ``-1`` for unknown/referee) to each detection. Swap the classification
    engine by providing a custom implementation to
    :func:`src.pipeline.run_pipeline`.
    """

    def classify(
        self,
        image: np.ndarray,
        detections: list[PlayerDetection],
    ) -> list[PlayerTeam]:
        ...
