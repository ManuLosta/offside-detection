from dataclasses import dataclass


LineSegment = tuple[int, int, int, int]
Point = tuple[float, float]


@dataclass(init=False, slots=True)
class VanishingPointResult:
    horizontal_point: Point | None
    vertical_point: Point | None
    horizontal_inlier_lines: list[LineSegment]
    vertical_inlier_lines: list[LineSegment]
    detected_lines_count: int
    horizontal_filtered_lines_count: int
    vertical_filtered_lines_count: int

    def __init__(
        self,
        *,
        horizontal_point: Point | None = None,
        vertical_point: Point | None = None,
        horizontal_inlier_lines: list[LineSegment] | None = None,
        vertical_inlier_lines: list[LineSegment] | None = None,
        detected_lines_count: int = 0,
        horizontal_filtered_lines_count: int = 0,
        vertical_filtered_lines_count: int = 0,
        point: Point | None = None,
        inlier_lines: list[LineSegment] | None = None,
        filtered_lines_count: int | None = None,
    ) -> None:
        # Backward-compatible aliases for callers that still pass/read one point.
        if vertical_point is None and point is not None:
            vertical_point = point
        if vertical_inlier_lines is None and inlier_lines is not None:
            vertical_inlier_lines = inlier_lines
        if filtered_lines_count is not None and vertical_filtered_lines_count == 0:
            vertical_filtered_lines_count = filtered_lines_count

        self.horizontal_point = horizontal_point
        self.vertical_point = vertical_point
        self.horizontal_inlier_lines = horizontal_inlier_lines or []
        self.vertical_inlier_lines = vertical_inlier_lines or []
        self.detected_lines_count = detected_lines_count
        self.horizontal_filtered_lines_count = horizontal_filtered_lines_count
        self.vertical_filtered_lines_count = vertical_filtered_lines_count

    @property
    def point(self) -> Point | None:
        return self.vertical_point

    @property
    def inlier_lines(self) -> list[LineSegment]:
        return self.vertical_inlier_lines

    @property
    def filtered_lines_count(self) -> int:
        return self.horizontal_filtered_lines_count + self.vertical_filtered_lines_count
