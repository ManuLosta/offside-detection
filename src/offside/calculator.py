import math
from dataclasses import replace

import numpy as np

from src.models.offside import OffsideResult, PlayerProjectedPoint
from src.models.pose import PlayerPose
from src.models.team import PlayerTeam
from src.models.vanishing_point import Point, VanishingPointResult

KEYPOINT_CONFIDENCE_THRESHOLD = 0.20

PLAYABLE_KEYPOINTS: dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


class OffsideCalculator:
    def __init__(
        self,
        *,
        attacking_team_id: int = 0,
        defending_team_id: int = 1,
        attacking_side: str = "right",
    ) -> None:
        if attacking_side not in {"left", "right"}:
            raise ValueError("attacking_side must be 'left' or 'right'")

        self.attacking_team_id = attacking_team_id
        self.defending_team_id = defending_team_id
        self.attacking_side = attacking_side

    def detect(
        self,
        poses: list[PlayerPose],
        teams: list[PlayerTeam],
        vanishing_point: VanishingPointResult,
    ) -> OffsideResult:
        if vanishing_point.horizontal_point is None:
            raise ValueError("Horizontal vanishing point is required for projection")
        if vanishing_point.vertical_point is None:
            raise ValueError("Vertical vanishing point is required for offside line")

        projected_players = _get_all_projected_players(
            poses,
            teams,
            vanishing_point.horizontal_point,
            self.attacking_side,
        )
        projected_players = _add_angle_scores(
            projected_players,
            vanishing_point.vertical_point,
            self.attacking_side,
        )

        reference_defender = _get_reference_defender(
            projected_players,
            self.defending_team_id,
        )
        if reference_defender is None:
            return OffsideResult(
                projected_players=projected_players,
                offside_attackers=[],
                reference_defender=None,
                attacking_team_id=self.attacking_team_id,
                defending_team_id=self.defending_team_id,
                attacking_side=self.attacking_side,
            )

        attackers = [
            p for p in projected_players if p.team_id == self.attacking_team_id
        ]
        offside_attackers = [
            attacker
            for attacker in attackers
            if attacker.advance_score > reference_defender.advance_score
        ]

        return OffsideResult(
            projected_players=projected_players,
            offside_attackers=offside_attackers,
            reference_defender=reference_defender,
            attacking_team_id=self.attacking_team_id,
            defending_team_id=self.defending_team_id,
            attacking_side=self.attacking_side,
        )


def _get_team_id(player_id: int, teams: list[PlayerTeam]) -> int:
    for team in teams:
        if team.player_id == player_id:
            return team.team_id
    return -1


def _get_keypoint_xy(pose: PlayerPose, idx: int) -> Point | None:
    if idx >= len(pose.keypoints):
        return None

    keypoint = pose.keypoints[idx]
    if keypoint.confidence < KEYPOINT_CONFIDENCE_THRESHOLD:
        return None

    return float(keypoint.x), float(keypoint.y)


def _estimate_ground_center_from_pose(pose: PlayerPose) -> Point | None:
    left_ankle = _get_keypoint_xy(pose, 15)
    right_ankle = _get_keypoint_xy(pose, 16)

    if left_ankle and right_ankle:
        return (
            float((left_ankle[0] + right_ankle[0]) / 2),
            float((left_ankle[1] + right_ankle[1]) / 2),
        )
    if left_ankle:
        return left_ankle
    if right_ankle:
        return right_ankle

    left_knee = _get_keypoint_xy(pose, 13)
    right_knee = _get_keypoint_xy(pose, 14)
    left_hip = _get_keypoint_xy(pose, 11)
    right_hip = _get_keypoint_xy(pose, 12)

    leg_ground_candidates = []
    if left_hip and left_knee:
        hx, hy = left_hip
        kx, ky = left_knee
        leg_ground_candidates.append((float(kx + (kx - hx)), float(ky + (ky - hy))))
    if right_hip and right_knee:
        hx, hy = right_hip
        kx, ky = right_knee
        leg_ground_candidates.append((float(kx + (kx - hx)), float(ky + (ky - hy))))

    if leg_ground_candidates:
        xs = [point[0] for point in leg_ground_candidates]
        ys = [point[1] for point in leg_ground_candidates]
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

    knees = [point for point in [left_knee, right_knee] if point is not None]
    if knees:
        xs = [point[0] for point in knees]
        ys = [point[1] for point in knees]
        knee_center = (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))

        shoulders = [
            point
            for point in [_get_keypoint_xy(pose, 5), _get_keypoint_xy(pose, 6)]
            if point is not None
        ]
        hips = [point for point in [left_hip, right_hip] if point is not None]

        if shoulders and hips:
            shoulder_y = sum(point[1] for point in shoulders) / len(shoulders)
            hip_y = sum(point[1] for point in hips) / len(hips)
            torso_height = abs(hip_y - shoulder_y)
            return knee_center[0], float(knee_center[1] + torso_height * 0.9)

        return knee_center[0], float(knee_center[1] + 40)

    hips = [point for point in [left_hip, right_hip] if point is not None]
    shoulders = [
        point
        for point in [_get_keypoint_xy(pose, 5), _get_keypoint_xy(pose, 6)]
        if point is not None
    ]

    if hips:
        hip_x = sum(point[0] for point in hips) / len(hips)
        hip_y = sum(point[1] for point in hips) / len(hips)
        if shoulders:
            shoulder_y = sum(point[1] for point in shoulders) / len(shoulders)
            torso_height = abs(hip_y - shoulder_y)
            return float(hip_x), float(hip_y + torso_height * 1.8)
        return float(hip_x), float(hip_y + 80)

    visible_points = []
    for idx in PLAYABLE_KEYPOINTS:
        point = _get_keypoint_xy(pose, idx)
        if point is not None:
            visible_points.append(point)

    if not visible_points:
        return None

    lowest_points = sorted(visible_points, key=lambda point: point[1], reverse=True)[:2]
    xs = [point[0] for point in lowest_points]
    ys = [point[1] for point in lowest_points]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def _project_keypoint_vertical_to_ground_line(
    keypoint_point: Point,
    ground_center: Point,
    horizontal_vanishing_point: Point,
) -> Point:
    keypoint_x, _ = keypoint_point
    ground_x, ground_y = ground_center
    vanishing_x, vanishing_y = horizontal_vanishing_point

    dx = vanishing_x - ground_x
    if abs(dx) < 1e-6:
        return float(ground_x), float(ground_y)

    slope = (vanishing_y - ground_y) / dx
    intercept = ground_y - slope * ground_x
    projected_x = keypoint_x
    projected_y = slope * projected_x + intercept

    return float(projected_x), float(projected_y)


def _advance_score_from_attacking_side(
    projected_point: Point,
    attacking_side: str,
) -> float:
    point_x, _ = projected_point
    return float(point_x if attacking_side == "right" else -point_x)


def _find_farthest_projected_playable_point(
    pose: PlayerPose,
    teams: list[PlayerTeam],
    horizontal_vanishing_point: Point,
    attacking_side: str,
) -> PlayerProjectedPoint | None:
    team_id = _get_team_id(pose.player_id, teams)
    ground_center = _estimate_ground_center_from_pose(pose)
    if ground_center is None:
        return None

    candidates = []
    for idx, name in PLAYABLE_KEYPOINTS.items():
        keypoint_point = _get_keypoint_xy(pose, idx)
        if keypoint_point is None:
            continue

        projected_point = _project_keypoint_vertical_to_ground_line(
            keypoint_point,
            ground_center,
            horizontal_vanishing_point,
        )
        score = _advance_score_from_attacking_side(projected_point, attacking_side)
        candidates.append(
            PlayerProjectedPoint(
                player_id=pose.player_id,
                team_id=team_id,
                keypoint_index=idx,
                keypoint_name=name,
                original_point=keypoint_point,
                ground_center=ground_center,
                projected_point=projected_point,
                advance_score=score,
            )
        )

    if not candidates:
        return None

    return max(candidates, key=lambda candidate: candidate.advance_score)


def _get_all_projected_players(
    poses: list[PlayerPose],
    teams: list[PlayerTeam],
    horizontal_vanishing_point: Point,
    attacking_side: str,
) -> list[PlayerProjectedPoint]:
    projected_players = []
    for pose in poses:
        projected = _find_farthest_projected_playable_point(
            pose,
            teams,
            horizontal_vanishing_point,
            attacking_side,
        )
        if projected is not None:
            projected_players.append(projected)
    return projected_players


def _angle_from_vp(point: Point, vanishing_point: Point) -> float:
    point_x, point_y = point
    vanishing_x, vanishing_y = vanishing_point
    return math.atan2(point_y - vanishing_y, point_x - vanishing_x)


def _add_angle_scores(
    projected_players: list[PlayerProjectedPoint],
    vertical_vanishing_point: Point,
    attacking_side: str,
) -> list[PlayerProjectedPoint]:
    if not projected_players:
        return []

    raw_angles = [
        _angle_from_vp(player.projected_point, vertical_vanishing_point)
        for player in projected_players
    ]
    unwrapped_angles = np.unwrap(np.array(raw_angles, dtype=np.float64))
    projected_xs = np.array(
        [player.projected_point[0] for player in projected_players],
        dtype=np.float64,
    )

    if len(projected_players) >= 2:
        corr = np.corrcoef(projected_xs, unwrapped_angles)[0, 1]
        direction_sign = 1 if np.isnan(corr) or corr >= 0 else -1
    else:
        direction_sign = 1

    right_scores = direction_sign * unwrapped_angles
    if attacking_side == "left":
        right_scores = -right_scores

    return [
        replace(player, advance_score=float(score))
        for player, score in zip(projected_players, right_scores)
    ]


def _get_reference_defender(
    projected_players: list[PlayerProjectedPoint],
    defending_team_id: int,
) -> PlayerProjectedPoint | None:
    defenders = [
        player
        for player in projected_players
        if player.team_id == defending_team_id
    ]
    if not defenders:
        return None
    return max(defenders, key=lambda player: player.advance_score)
