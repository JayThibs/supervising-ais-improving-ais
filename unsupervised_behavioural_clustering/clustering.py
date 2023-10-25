from typing import List
import logging

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def kmeans_cluster(embeddings: np.ndarray, num_clusters: int = 10) -> List[int]:
    """Apply K-Means clustering to embeddings.

    Args:
        embeddings: Array of embedding vectors.
        num_clusters: Number of clusters to generate.

    Returns:
        List of cluster labels per embedding.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    try:
        clusters = kmeans.fit_predict(embeddings)
    except ValueError as e:
        logger.error(f"Clustering failed: {e}")
        raise

    return clusters


def test_kmeans_cluster():
    """Test clustering on sample data."""
    sample_data = np.random.rand(100, 64)
    clusters = kmeans_cluster(sample_data, num_clusters=5)
    assert len(clusters) == 100
    assert set(clusters) == set(range(5))
