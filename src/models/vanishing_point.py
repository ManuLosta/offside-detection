from dataclasses import dataclass


LineSegment = tuple[int, int, int, int]
Point = tuple[float, float]


@dataclass(slots=True)
class VanishingPointResult:
    point: Point | None
    inlier_lines: list[LineSegment]
    detected_lines_count: int
    filtered_lines_count: int
