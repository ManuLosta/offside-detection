import logging

import numpy as np
from sklearn.cluster import KMeans

from src.models.detection import PlayerDetection
from src.models.team import PlayerTeam

logger = logging.getLogger(__name__)

N_CLUSTERS = 3
RANDOM_STATE = 42


def classify_teams(
    features: np.ndarray,
    detections: list[PlayerDetection],
) -> list[PlayerTeam]:
    """Cluster player colour features and assign team IDs.

    The two largest KMeans clusters become teams ``0`` and ``1``; the remaining
    cluster (referee, goalkeeper, outliers) is labelled ``-1``.
    """
    n_samples = len(detections)
    if n_samples == 0:
        return []

    if features.shape[0] != n_samples:
        raise ValueError(
            f"Feature count ({features.shape[0]}) does not match detection count ({n_samples})"
        )

    if n_samples < N_CLUSTERS:
        logger.warning(
            "Only %d detections available, need at least %d for KMeans; assigning all to unknown team",
            n_samples,
            N_CLUSTERS,
        )
        return [
            PlayerTeam(
                player_id=player_id,
                bbox=detection.bbox,
                cluster_id=-1,
                team_id=-1,
            )
            for player_id, detection in enumerate(detections)
        ]

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    cluster_ids = kmeans.fit_predict(features)

    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)
    cluster_sizes = {int(cid): int(count) for cid, count in zip(unique_clusters, counts)}

    # Largest cluster -> team 0, second largest -> team 1, rest -> -1.
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda item: item[1], reverse=True)
    team_mapping: dict[int, int] = {}
    for rank, (cluster_id, _) in enumerate(sorted_clusters):
        team_mapping[cluster_id] = rank if rank < 2 else -1

    teams: list[PlayerTeam] = []
    for player_id, detection in enumerate(detections):
        cluster_id = int(cluster_ids[player_id])
        teams.append(
            PlayerTeam(
                player_id=player_id,
                bbox=detection.bbox,
                cluster_id=cluster_id,
                team_id=team_mapping[cluster_id],
            )
        )

    team_counts: dict[int, int] = {}
    for team in teams:
        team_counts[team.team_id] = team_counts.get(team.team_id, 0) + 1

    logger.info("Cluster sizes: %s", cluster_sizes)
    logger.info("Team counts: %s", team_counts)

    return teams
