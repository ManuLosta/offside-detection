from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from src.models.detection import PlayerDetection


@runtime_checkable
class Detector(Protocol):
    """Abstract player detector.

    Implementations produce a list of :class:`PlayerDetection` bounding boxes
    for a given input image. Swap the detection engine by providing a custom
    implementation to :func:`src.pipeline.run_pipeline`.
    """

    def detect(self, image: np.ndarray | Path | str) -> list[PlayerDetection]:
        ...
