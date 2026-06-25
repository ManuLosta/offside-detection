import cv2
import numpy as np

from src.models.vanishing_point import LineSegment, Point, VanishingPointResult


def draw_vanishing_point(
    image: np.ndarray,
    result: VanishingPointResult,
    *,
    horizontal_line_color: tuple[int, int, int] = (255, 255, 0),
    vertical_line_color: tuple[int, int, int] = (255, 0, 255),
    horizontal_point_color: tuple[int, int, int] = (255, 255, 0),
    vertical_point_color: tuple[int, int, int] = (255, 0, 255),
    line_color: tuple[int, int, int] | None = None,
    point_color: tuple[int, int, int] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    if line_color is not None:
        horizontal_line_color = line_color
        vertical_line_color = line_color
    if point_color is not None:
        horizontal_point_color = point_color
        vertical_point_color = point_color

    annotated = image.copy()
    for line in result.horizontal_inlier_lines:
        _draw_clipped_line(annotated, line, horizontal_line_color, thickness)

    for line in result.vertical_inlier_lines:
        _draw_clipped_line(annotated, line, vertical_line_color, thickness)

    if result.horizontal_point is not None:
        _draw_point(annotated, result.horizontal_point, "VP-H", horizontal_point_color)

    if result.vertical_point is not None:
        _draw_point(annotated, result.vertical_point, "VP-V", vertical_point_color)

    return annotated


def _draw_clipped_line(
    image: np.ndarray,
    line: LineSegment,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    norm = np.hypot(dx, dy)
    if norm < 1e-6:
        return

    scale = max(width, height) * 4
    extended_1 = (int(round(x1 - dx / norm * scale)), int(round(y1 - dy / norm * scale)))
    extended_2 = (int(round(x1 + dx / norm * scale)), int(round(y1 + dy / norm * scale)))
    clipped = cv2.clipLine((0, 0, width, height), extended_1, extended_2)
    if clipped[0]:
        cv2.line(image, clipped[1], clipped[2], color, thickness)


def _draw_point(
    image: np.ndarray,
    point: Point,
    label: str,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    x, y = (int(round(point[0])), int(round(point[1])))
    if 0 <= x < width and 0 <= y < height:
        cv2.circle(image, (x, y), 12, color, -1)
        label_position = (min(x + 15, width - 1), max(y - 15, 0))
        cv2.putText(
            image,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        return

    border_point = _nearest_border_point(point, width, height)
    cv2.circle(image, border_point, 10, color, -1)
    cv2.arrowedLine(
        image,
        (width // 2, height // 2),
        border_point,
        color,
        2,
        tipLength=0.08,
    )
    cv2.putText(
        image,
        f"{label} ({point[0]:.0f}, {point[1]:.0f})",
        (min(border_point[0] + 12, width - 180), max(border_point[1] - 12, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def _nearest_border_point(point: Point, width: int, height: int) -> tuple[int, int]:
    x, y = point
    center_x = width / 2
    center_y = height / 2
    dx = x - center_x
    dy = y - center_y
    candidates: list[tuple[float, float]] = []

    if abs(dx) > 1e-6:
        candidates.extend(
            [
                ((0 - center_x) / dx, 0.0),
                ((width - 1 - center_x) / dx, float(width - 1)),
            ]
        )

    if abs(dy) > 1e-6:
        candidates.extend(
            [
                ((0 - center_y) / dy, 0.0),
                ((height - 1 - center_y) / dy, float(height - 1)),
            ]
        )

    border_candidates = []
    for t, fixed_value in candidates:
        if t <= 0:
            continue
        candidate_x = center_x + dx * t
        candidate_y = center_y + dy * t
        if 0 <= candidate_x <= width - 1 and 0 <= candidate_y <= height - 1:
            if fixed_value in (0.0, float(width - 1)):
                border_candidates.append((candidate_x, candidate_y))
            else:
                border_candidates.append((candidate_x, candidate_y))

    if not border_candidates:
        return (int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1)))

    nearest = min(
        border_candidates,
        key=lambda p: np.hypot(p[0] - center_x, p[1] - center_y),
    )
    return (int(round(nearest[0])), int(round(nearest[1])))
