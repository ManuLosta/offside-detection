import cv2
import numpy as np

from src.models.offside import OffsideResult
from src.models.vanishing_point import Point

DEFENDER_COLOR = (255, 0, 0)


def draw_offside_detection(
    image: np.ndarray,
    result: OffsideResult,
    vertical_vanishing_point: Point,
) -> np.ndarray:
    annotated = image.copy()
    offside_ids = {player.player_id for player in result.offside_attackers}

    for player in result.projected_players:
        original_x, original_y = player.original_point
        ground_x, ground_y = player.ground_center
        projected_x, projected_y = player.projected_point

        if player.player_id in offside_ids:
            color = (0, 0, 255)
            label = "OFFSIDE"
        elif player.team_id == result.attacking_team_id:
            color = (0, 255, 0)
            label = "ATT"
        elif player.team_id == result.defending_team_id:
            color = DEFENDER_COLOR
            label = "DEF"
        else:
            color = (180, 180, 180)
            label = "OTHER"

        cv2.circle(
            annotated,
            (int(round(original_x)), int(round(original_y))),
            4,
            color,
            -1,
        )
        cv2.circle(
            annotated,
            (int(round(ground_x)), int(round(ground_y))),
            5,
            (255, 255, 0),
            -1,
        )
        cv2.circle(
            annotated,
            (int(round(projected_x)), int(round(projected_y))),
            8,
            color,
            -1,
        )
        cv2.line(
            annotated,
            (int(round(original_x)), int(round(original_y))),
            (int(round(projected_x)), int(round(projected_y))),
            color,
            2,
        )
        cv2.putText(
            annotated,
            f"id:{player.player_id} {label}",
            (int(round(projected_x)) + 6, int(round(projected_y)) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    if result.reference_defender is not None:
        _draw_offside_line(
            annotated,
            result.reference_defender.projected_point,
            vertical_vanishing_point,
        )
        _draw_reference_defender(annotated, result.reference_defender.projected_point)

    return annotated


def _draw_reference_defender(image: np.ndarray, point: Point) -> None:
    point_x, point_y = point
    center = (int(round(point_x)), int(round(point_y)))
    cv2.circle(image, center, 14, (255, 255, 255), 3)
    cv2.putText(
        image,
        "REF DEF",
        (center[0] + 8, center[1] + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def _draw_offside_line(
    image: np.ndarray,
    point: Point,
    vertical_vanishing_point: Point,
) -> None:
    line_points = _line_points_inside_image_from_point_and_vp(
        point,
        vertical_vanishing_point,
        image.shape,
    )
    if line_points is None:
        return

    point_1, point_2 = line_points
    cv2.line(image, point_1, point_2, (0, 0, 255), 3)
    cv2.putText(
        image,
        "OFFSIDE LINE",
        point_1,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        3,
    )


def _line_points_inside_image_from_point_and_vp(
    point: Point,
    vanishing_point: Point,
    image_shape: tuple[int, ...],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    height, width = image_shape[:2]
    x1, y1 = point
    x2, y2 = vanishing_point
    dx = x2 - x1
    dy = y2 - y1
    candidates = []

    if abs(dx) > 1e-6:
        for x in [0, width]:
            t = (x - x1) / dx
            y = y1 + t * dy
            if 0 <= y <= height:
                candidates.append((int(round(x)), int(round(y))))

    if abs(dy) > 1e-6:
        for y in [0, height]:
            t = (y - y1) / dy
            x = x1 + t * dx
            if 0 <= x <= width:
                candidates.append((int(round(x)), int(round(y))))

    if len(candidates) < 2:
        return None

    return candidates[0], candidates[1]
