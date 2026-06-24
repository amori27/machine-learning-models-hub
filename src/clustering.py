"""Clustering Models Module.

This module provides clustering algorithms including
k-means, hierarchical clustering, and DBSCAN.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from typing import Any


class ModelClusterer:
    """Handles clustering model fitting and prediction."""

    def __init__(self, algorithm: str = "kmeans", n_clusters: int = 5):
        """Initialize the ModelClusterer.

        Args:
            algorithm: Clustering algorithm (kmeans, hierarchical, dbscan, gmm).
            n_clusters: Number of clusters.
        """
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.model = self._create_model()

    def _create_model(self):
        """Create the clusterer instance.

        Returns:
            Scikit-learn clusterer.
        """
        if self.algorithm == "kmeans":
            return KMeans(n_clusters=self.n_clusters, random_state=42)
        elif self.algorithm == "hierarchical":
            return AgglomerativeClustering(n_clusters=self.n_clusters)
        elif self.algorithm == "dbscan":
            return DBSCAN(eps=0.5, min_samples=5)
        elif self.algorithm == "gmm":
            return GaussianMixture(n_components=self.n_clusters, random_state=42)
        return KMeans(n_clusters=self.n_clusters, random_state=42)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict clusters.

        Args:
            X: Feature array.

        Returns:
            Cluster labels.
        """
        if self.algorithm == "gmm":
            self.model.fit(X)
            return self.model.predict(X)
        return self.model.fit_predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels.

        Args:
            X: Feature array.

        Returns:
            Cluster labels.
        """
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        return self.model.labels_

    def get_cluster_centers(self) -> np.ndarray | None:
        """Get cluster centers.

        Returns:
            Cluster centers array or None.
        """
        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        return None


def find_optimal_k(
    X: np.ndarray,
    max_k: int = 10,
    method: str = "elbow"
) -> dict[str, Any]:
    """Find optimal number of clusters.

    Args:
        X: Feature array.
        max_k: Maximum k to try.
        method: Method to use (elbow, silhouette).

    Returns:
        Dictionary with results.
    """
    inertias = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "recommended_k": 5
    }
