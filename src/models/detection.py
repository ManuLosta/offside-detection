from dataclasses import dataclass


@dataclass(slots=True)
class PlayerDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
