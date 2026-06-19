from dataclasses import dataclass


@dataclass(slots=True)
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass(slots=True)
class PlayerPose:
    player_id: int
    bbox: tuple[int, int, int, int]
    keypoints: list[Keypoint]
