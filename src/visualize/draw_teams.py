import logging

import cv2
import numpy as np

from src.models.team import PlayerTeam

logger = logging.getLogger(__name__)

TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 0, 0),  # blue
    1: (0, 0, 255),  # red
    -1: (0, 255, 255),  # yellow
}


def draw_team_assignments(
    image: np.ndarray,
    teams: list[PlayerTeam],
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and team labels for classified players."""
    annotated = image.copy()

    for team in teams:
        x1, y1, x2, y2 = team.bbox
        color = TEAM_COLORS.get(team.team_id, TEAM_COLORS[-1])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"id:{team.player_id} team:{team.team_id}"
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness,
        )

    logger.info("Drew team assignments for %d player(s)", len(teams))
    return annotated
