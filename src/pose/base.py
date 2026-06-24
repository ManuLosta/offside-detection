from typing import Protocol, runtime_checkable

import numpy as np

from src.models.detection import PlayerDetection
from src.models.pose import PlayerPose


@runtime_checkable
class PoseEstimator(Protocol):
    """Abstract pose estimator.

    Implementations produce per-player :class:`PlayerPose` results. Swap the
    pose engine by providing a custom implementation to
    :func:`src.pipeline.run_pipeline`.
    """

    def estimate(
        self,
        image: np.ndarray,
        detections: list[PlayerDetection],
        player_ids: list[int] | None = None,
    ) -> list[PlayerPose]:
        ...
