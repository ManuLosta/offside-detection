import random

import cv2
import numpy as np

from src.models.vanishing_point import LineSegment, Point, VanishingPointResult
from src.perspective.base import VanishingPointEstimator


class FieldVanishingPointEstimator(VanishingPointEstimator):
    def __init__(
        self,
        *,
        lower_green: tuple[int, int, int] = (35, 40, 40),
        upper_green: tuple[int, int, int] = (90, 255, 255),
        min_line_length: int = 100,
        hough_threshold: int = 80,
        max_line_gap: int = 40,
        horizontal_angle_range: tuple[float, float] = (0.0, 12.0),
        vertical_angle_range: tuple[float, float] = (12.0, 89.0),
        angle_range: tuple[float, float] | None = None,
        ransac_iterations: int = 3000,
        ransac_threshold: float = 25.0,
        random_seed: int = 7,
    ) -> None:
        self.lower_green = np.array(lower_green, dtype=np.uint8)
        self.upper_green = np.array(upper_green, dtype=np.uint8)
        self.min_line_length = min_line_length
        self.hough_threshold = hough_threshold
        self.max_line_gap = max_line_gap
        if angle_range is not None:
            vertical_angle_range = angle_range
        self.horizontal_angle_range = horizontal_angle_range
        self.vertical_angle_range = vertical_angle_range
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold = ransac_threshold
        self.random_seed = random_seed

    def estimate(self, image: np.ndarray) -> VanishingPointResult:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        field_mask = self._segment_field(hsv)
        saturation = cv2.bitwise_and(hsv[:, :, 1], hsv[:, :, 1], mask=field_mask)
        edges = self._detect_edges(saturation)
        lines = self._detect_lines(edges)
        horizontal_lines, vertical_lines = self._split_horizontal_vertical_lines(lines)
        horizontal_point, horizontal_inlier_lines = self._ransac(
            horizontal_lines,
            random_seed=self.random_seed,
        )
        vertical_point, vertical_inlier_lines = self._ransac(
            vertical_lines,
            random_seed=self.random_seed + 1,
        )

        return VanishingPointResult(
            horizontal_point=horizontal_point,
            vertical_point=vertical_point,
            horizontal_inlier_lines=horizontal_inlier_lines,
            vertical_inlier_lines=vertical_inlier_lines,
            detected_lines_count=len(lines),
            horizontal_filtered_lines_count=len(horizontal_lines),
            vertical_filtered_lines_count=len(vertical_lines),
        )

    def _segment_field(self, hsv: np.ndarray) -> np.ndarray:
        field_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        kernel = np.ones((9, 9), np.uint8)
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel)

    def _detect_edges(self, saturation: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(saturation, (5, 5), 0)
        return cv2.Canny(blur, 50, 150)

    def _detect_lines(self, edges: np.ndarray) -> list[LineSegment]:
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        if raw_lines is None:
            return []
        return [tuple(int(v) for v in line[0]) for line in raw_lines]

    def _split_horizontal_vertical_lines(
        self, lines: list[LineSegment]
    ) -> tuple[list[LineSegment], list[LineSegment]]:
        horizontal_min, horizontal_max = self.horizontal_angle_range
        vertical_min, vertical_max = self.vertical_angle_range
        horizontal_lines: list[LineSegment] = []
        vertical_lines: list[LineSegment] = []

        for line in lines:
            angle = _normalized_abs_angle(line)
            length = _line_length(line)
            if length <= self.min_line_length:
                continue
            if horizontal_min <= angle < horizontal_max:
                horizontal_lines.append(line)
            elif vertical_min <= angle <= vertical_max:
                vertical_lines.append(line)

        return horizontal_lines, vertical_lines

    def _ransac(
        self, lines: list[LineSegment], *, random_seed: int
    ) -> tuple[Point | None, list[LineSegment]]:
        if len(lines) < 2:
            return None, []

        rng = random.Random(random_seed)
        best_point: Point | None = None
        best_lines: list[LineSegment] = []

        for _ in range(self.ransac_iterations):
            line_1, line_2 = rng.sample(lines, 2)
            point = _intersection(line_1, line_2)
            if point is None:
                continue

            inliers = [
                line
                for line in lines
                if _point_line_distance(point, line) < self.ransac_threshold
            ]
            if len(inliers) > len(best_lines):
                best_lines = inliers
                best_point = point

        if best_point is None or len(best_lines) < 2:
            return best_point, best_lines

        refined_point = _least_squares_intersection(best_lines)
        return refined_point or best_point, best_lines


def _line_angle(line: LineSegment) -> float:
    x1, y1, x2, y2 = line
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def _normalized_abs_angle(line: LineSegment) -> float:
    angle = abs(_line_angle(line))
    if angle > 90.0:
        angle = 180.0 - angle
    return float(angle)


def _line_length(line: LineSegment) -> float:
    x1, y1, x2, y2 = line
    return float(np.hypot(x2 - x1, y2 - y1))


def _intersection(line_1: LineSegment, line_2: LineSegment) -> Point | None:
    x1, y1, x2, y2 = line_1
    x3, y3, x4, y4 = line_2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None

    determinant_1 = x1 * y2 - y1 * x2
    determinant_2 = x3 * y4 - y3 * x4
    px = (determinant_1 * (x3 - x4) - (x1 - x2) * determinant_2) / denominator
    py = (determinant_1 * (y3 - y4) - (y1 - y2) * determinant_2) / denominator
    return (float(px), float(py))


def _point_line_distance(point: Point, line: LineSegment) -> float:
    px, py = point
    x1, y1, x2, y2 = line
    denominator = np.hypot(y2 - y1, x2 - x1)
    if denominator < 1e-6:
        return float("inf")
    numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    return float(numerator / denominator)


def _least_squares_intersection(lines: list[LineSegment]) -> Point | None:
    coefficients = []
    constants = []
    for x1, y1, x2, y2 in lines:
        a = y1 - y2
        b = x2 - x1
        norm = np.hypot(a, b)
        if norm < 1e-6:
            continue
        coefficients.append([a / norm, b / norm])
        constants.append((a * x1 + b * y1) / norm)

    if len(coefficients) < 2:
        return None

    solution, *_ = np.linalg.lstsq(
        np.array(coefficients, dtype=np.float64),
        np.array(constants, dtype=np.float64),
        rcond=None,
    )
    return (float(solution[0]), float(solution[1]))
