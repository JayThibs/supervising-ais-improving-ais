import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Protocol, TypeVar
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap
from termcolor import colored

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ClusteringResult(Protocol):
    """Protocol for clustering algorithm results."""
    labels_: np.ndarray


class ClusteringAlgorithm:
    """Base class for clustering algorithms."""
    
    def __init__(self, **kwargs):
        """Initialize the clustering algorithm with parameters."""
        self.params = kwargs
        self.model = None
        
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the clustering algorithm to the data.
        
        Args:
            data: Input data for clustering
            
        Returns:
            Array of cluster labels
        """
        raise NotImplementedError("Subclasses must implement fit method")
        
    def get_model(self) -> Any:
        """Get the underlying model."""
        return self.model


class KMeansAlgorithm(ClusteringAlgorithm):
    """K-means clustering algorithm."""
    
    def __init__(self, n_clusters: int = 8, random_state: int = 42, **kwargs):
        """
        Initialize K-means clustering.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for KMeans
        """
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.random_state = random_state
        
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit K-means clustering to the data.
        
        Args:
            data: Input data for clustering
            
        Returns:
            Array of cluster labels
        """
        params = {
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            **self.params
        }
        
        self.model = KMeans(**params)
        self.model.fit(data)
        
        return self.model.labels_


class SpectralClusteringAlgorithm(ClusteringAlgorithm):
    """Spectral clustering algorithm."""
    
    def __init__(self, n_clusters: int = 8, affinity: str = 'rbf', random_state: int = 42, **kwargs):
        """
        Initialize spectral clustering.
        
        Args:
            n_clusters: Number of clusters
            affinity: Affinity type ('rbf', 'nearest_neighbors', etc.)
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for SpectralClustering
        """
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.random_state = random_state
        
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit spectral clustering to the data.
        
        Args:
            data: Input data for clustering
            
        Returns:
            Array of cluster labels
        """
        params = {
            'n_clusters': self.n_clusters,
            'affinity': self.affinity,
            'random_state': self.random_state,
            **self.params
        }
        
        try:
            self.model = SpectralClustering(**params)
            self.model.fit(data)
            return self.model.labels_
        except Exception as e:
            logger.error(colored(f"Error in spectral clustering: {str(e)}", "red"))
            logger.warning(colored("Using fallback K-means clustering", "yellow"))
            
            kmeans = KMeansAlgorithm(n_clusters=self.n_clusters, random_state=self.random_state)
            labels = kmeans.fit(data)
            self.model = kmeans.get_model()
            
            return labels


class AgglomerativeClusteringAlgorithm(ClusteringAlgorithm):
    """Agglomerative clustering algorithm."""
    
    def __init__(self, n_clusters: int = 8, linkage: str = 'ward', **kwargs):
        """
        Initialize agglomerative clustering.
        
        Args:
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            **kwargs: Additional parameters for AgglomerativeClustering
        """
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.linkage = linkage
        
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit agglomerative clustering to the data.
        
        Args:
            data: Input data for clustering
            
        Returns:
            Array of cluster labels
        """
        params = {
            'n_clusters': self.n_clusters,
            'linkage': self.linkage,
            **self.params
        }
        
        self.model = AgglomerativeClustering(**params)
        self.model.fit(data)
        
        return self.model.labels_


class DBSCANAlgorithm(ClusteringAlgorithm):
    """DBSCAN clustering algorithm."""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs):
        """
        Initialize DBSCAN clustering.
        
        Args:
            eps: Maximum distance between samples for them to be considered neighbors
            min_samples: Minimum number of samples in a neighborhood for a point to be a core point
            **kwargs: Additional parameters for DBSCAN
        """
        super().__init__(**kwargs)
        self.eps = eps
        self.min_samples = min_samples
        
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit DBSCAN clustering to the data.
        
        Args:
            data: Input data for clustering
            
        Returns:
            Array of cluster labels
        """
        params = {
            'eps': self.eps,
            'min_samples': self.min_samples,
            **self.params
        }
        
        self.model = DBSCAN(**params)
        self.model.fit(data)
        
        return self.model.labels_


class GaussianMixtureAlgorithm(ClusteringAlgorithm):
    """Gaussian Mixture Model clustering algorithm."""
    
    def __init__(self, n_components: int = 8, random_state: int = 42, **kwargs):
        """
        Initialize Gaussian Mixture Model clustering.
        
        Args:
            n_components: Number of mixture components
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for GaussianMixture
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.random_state = random_state
        
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit Gaussian Mixture Model to the data.
        
        Args:
            data: Input data for clustering
            
        Returns:
            Array of cluster labels
        """
        params = {
            'n_components': self.n_components,
            'random_state': self.random_state,
            **self.params
        }
        
        self.model = GaussianMixture(**params)
        self.model.fit(data)
        
        return self.model.predict(data)


class UMAPClusteringAlgorithm(ClusteringAlgorithm):
    """UMAP dimensionality reduction followed by clustering."""
    
    def __init__(self, 
                n_clusters: int = 8, 
                n_components: int = 2, 
                n_neighbors: int = 15, 
                min_dist: float = 0.1,
                random_state: int = 42, 
                clustering_algorithm: str = 'kmeans',
                **kwargs):
        """
        Initialize UMAP + clustering.
        
        Args:
            n_clusters: Number of clusters
            n_components: Number of dimensions for UMAP reduction
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            random_state: Random state for reproducibility
            clustering_algorithm: Algorithm to use after UMAP ('kmeans', 'spectral', 'agglomerative', 'dbscan', 'gmm')
            **kwargs: Additional parameters for UMAP and clustering
        """
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.clustering_algorithm = clustering_algorithm
        
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit UMAP + clustering to the data.
        
        Args:
            data: Input data for clustering
            
        Returns:
            Array of cluster labels
        """
        umap_params = {
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'random_state': self.random_state
        }
        
        try:
            reducer = umap.UMAP(**umap_params)
            reduced_data = reducer.fit_transform(data)
            
            if self.clustering_algorithm == 'kmeans':
                clustering_algo = KMeansAlgorithm(n_clusters=self.n_clusters, random_state=self.random_state)
            elif self.clustering_algorithm == 'spectral':
                clustering_algo = SpectralClusteringAlgorithm(n_clusters=self.n_clusters, random_state=self.random_state)
            elif self.clustering_algorithm == 'agglomerative':
                clustering_algo = AgglomerativeClusteringAlgorithm(n_clusters=self.n_clusters)
            elif self.clustering_algorithm == 'dbscan':
                clustering_algo = DBSCANAlgorithm()
            elif self.clustering_algorithm == 'gmm':
                clustering_algo = GaussianMixtureAlgorithm(n_components=self.n_clusters, random_state=self.random_state)
            else:
                logger.warning(colored(f"Unknown clustering algorithm: {self.clustering_algorithm}. Using K-means.", "yellow"))
                clustering_algo = KMeansAlgorithm(n_clusters=self.n_clusters, random_state=self.random_state)
                
            labels = clustering_algo.fit(reduced_data)
            self.model = {
                'umap': reducer,
                'clustering': clustering_algo.get_model()
            }
            
            return labels
            
        except Exception as e:
            logger.error(colored(f"Error in UMAP clustering: {str(e)}", "red"))
            logger.warning(colored("Using fallback K-means clustering on original data", "yellow"))
            
            kmeans = KMeansAlgorithm(n_clusters=self.n_clusters, random_state=self.random_state)
            labels = kmeans.fit(data)
            self.model = kmeans.get_model()
            
            return labels


class ClusteringFactory:
    """Factory for creating clustering algorithms."""
    
    @staticmethod
    def create_algorithm(algorithm_name: str, **kwargs) -> ClusteringAlgorithm:
        """
        Create a clustering algorithm.
        
        Args:
            algorithm_name: Name of the algorithm ('kmeans', 'spectral', 'agglomerative', 'dbscan', 'gmm', 'umap')
            **kwargs: Parameters for the algorithm
            
        Returns:
            Clustering algorithm instance
        """
        if algorithm_name.lower() == 'kmeans':
            return KMeansAlgorithm(**kwargs)
        elif algorithm_name.lower() == 'spectral':
            return SpectralClusteringAlgorithm(**kwargs)
        elif algorithm_name.lower() == 'agglomerative':
            return AgglomerativeClusteringAlgorithm(**kwargs)
        elif algorithm_name.lower() == 'dbscan':
            return DBSCANAlgorithm(**kwargs)
        elif algorithm_name.lower() == 'gmm':
            return GaussianMixtureAlgorithm(**kwargs)
        elif algorithm_name.lower() == 'umap':
            return UMAPClusteringAlgorithm(**kwargs)
        else:
            logger.warning(colored(f"Unknown algorithm: {algorithm_name}. Using K-means.", "yellow"))
            return KMeansAlgorithm(**kwargs)


def evaluate_clustering(data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluate clustering quality using various metrics.
    
    Args:
        data: Input data
        labels: Cluster labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    if len(np.unique(labels)) <= 1:
        logger.warning(colored("Cannot evaluate clustering with only one cluster", "yellow"))
        return metrics
    
    try:
        metrics['silhouette'] = silhouette_score(data, labels)
    except Exception as e:
        logger.warning(colored(f"Error calculating silhouette score: {str(e)}", "yellow"))
    
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(data, labels)
    except Exception as e:
        logger.warning(colored(f"Error calculating Calinski-Harabasz score: {str(e)}", "yellow"))
    
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(data, labels)
    except Exception as e:
        logger.warning(colored(f"Error calculating Davies-Bouldin score: {str(e)}", "yellow"))
    
    return metrics


def find_optimal_clusters(data: np.ndarray, 
                         algorithm: str = 'kmeans', 
                         min_clusters: int = 2, 
                         max_clusters: int = 20,
                         random_state: int = 42,
                         **kwargs) -> Tuple[int, np.ndarray, Dict[str, List[float]]]:
    """
    Find the optimal number of clusters using the elbow method.
    
    Args:
        data: Input data
        algorithm: Clustering algorithm to use
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
        random_state: Random state for reproducibility
        **kwargs: Additional parameters for the clustering algorithm
        
    Returns:
        Tuple of (optimal number of clusters, labels for optimal clustering, metrics for each number of clusters)
    """
    metrics = {
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': []
    }
    
    best_score = -float('inf')
    best_n_clusters = min_clusters
    best_labels = None
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        logger.info(colored(f"Trying {n_clusters} clusters...", "cyan"))
        
        clustering_algo = ClusteringFactory.create_algorithm(
            algorithm, 
            n_clusters=n_clusters, 
            random_state=random_state,
            **kwargs
        )
        
        labels = clustering_algo.fit(data)
        
        cluster_metrics = evaluate_clustering(data, labels)
        
        for metric_name, metric_value in cluster_metrics.items():
            if metric_name in metrics:
                metrics[metric_name].append(metric_value)
            
        if 'silhouette' in cluster_metrics and cluster_metrics['silhouette'] > best_score:
            best_score = cluster_metrics['silhouette']
            best_n_clusters = n_clusters
            best_labels = labels
    
    if best_labels is None:
        logger.warning(colored("Could not find optimal clustering. Using default.", "yellow"))
        clustering_algo = ClusteringFactory.create_algorithm(
            algorithm, 
            n_clusters=min_clusters, 
            random_state=random_state,
            **kwargs
        )
        best_labels = clustering_algo.fit(data)
        best_n_clusters = min_clusters
    
    logger.info(colored(f"Optimal number of clusters: {best_n_clusters}", "green"))
    
    return best_n_clusters, best_labels, metrics
