import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.behavioural_clustering.utils.clustering_algorithms import (
    ClusteringAlgorithm, KMeansAlgorithm, SpectralClusteringAlgorithm,
    AgglomerativeClusteringAlgorithm, DBSCANAlgorithm, GaussianMixtureAlgorithm,
    UMAPClusteringAlgorithm, ClusteringFactory, evaluate_clustering,
    find_optimal_clusters
)


class TestClusteringAlgorithm(unittest.TestCase):
    """Tests for the base ClusteringAlgorithm class."""
    
    def test_init(self):
        """Test initialization."""
        params = {'param1': 'value1', 'param2': 'value2'}
        algorithm = ClusteringAlgorithm(**params)
        self.assertEqual(algorithm.params, params)
        self.assertIsNone(algorithm.model)
        
    def test_fit_not_implemented(self):
        """Test that fit raises NotImplementedError."""
        algorithm = ClusteringAlgorithm()
        with self.assertRaises(NotImplementedError):
            algorithm.fit(np.array([[1, 2], [3, 4]]))
            
    def test_get_model(self):
        """Test get_model method."""
        algorithm = ClusteringAlgorithm()
        algorithm.model = "test_model"
        self.assertEqual(algorithm.get_model(), "test_model")


class TestKMeansAlgorithm(unittest.TestCase):
    """Tests for the KMeansAlgorithm class."""
    
    def test_init(self):
        """Test initialization."""
        algorithm = KMeansAlgorithm(n_clusters=5, random_state=42)
        self.assertEqual(algorithm.n_clusters, 5)
        self.assertEqual(algorithm.random_state, 42)
        
    @patch('src.behavioural_clustering.utils.clustering_algorithms.KMeans')
    def test_fit(self, mock_kmeans):
        """Test fit method."""
        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.labels_ = np.array([0, 1, 0, 1])
        mock_kmeans.return_value = mock_kmeans_instance
        
        data = np.array([[1, 2], [3, 4], [1, 3], [4, 2]])
        
        algorithm = KMeansAlgorithm(n_clusters=2, random_state=42)
        labels = algorithm.fit(data)
        
        mock_kmeans.assert_called_once_with(n_clusters=2, random_state=42)
        
        mock_kmeans_instance.fit.assert_called_once_with(data)
        
        np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1]))
        
        self.assertEqual(algorithm.model, mock_kmeans_instance)


class TestSpectralClusteringAlgorithm(unittest.TestCase):
    """Tests for the SpectralClusteringAlgorithm class."""
    
    def test_init(self):
        """Test initialization."""
        algorithm = SpectralClusteringAlgorithm(n_clusters=5, affinity='nearest_neighbors', random_state=42)
        self.assertEqual(algorithm.n_clusters, 5)
        self.assertEqual(algorithm.affinity, 'nearest_neighbors')
        self.assertEqual(algorithm.random_state, 42)
        
    @patch('src.behavioural_clustering.utils.clustering_algorithms.SpectralClustering')
    def test_fit(self, mock_spectral):
        """Test fit method."""
        mock_spectral_instance = MagicMock()
        mock_spectral_instance.labels_ = np.array([0, 1, 0, 1])
        mock_spectral.return_value = mock_spectral_instance
        
        data = np.array([[1, 2], [3, 4], [1, 3], [4, 2]])
        
        algorithm = SpectralClusteringAlgorithm(n_clusters=2, affinity='rbf', random_state=42)
        labels = algorithm.fit(data)
        
        mock_spectral.assert_called_once_with(n_clusters=2, affinity='rbf', random_state=42)
        
        mock_spectral_instance.fit.assert_called_once_with(data)
        
        np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1]))
        
        self.assertEqual(algorithm.model, mock_spectral_instance)
        
    @patch('src.behavioural_clustering.utils.clustering_algorithms.SpectralClustering')
    @patch('src.behavioural_clustering.utils.clustering_algorithms.KMeansAlgorithm')
    def test_fit_with_error(self, mock_kmeans_algo, mock_spectral):
        """Test fit method with error."""
        mock_spectral.side_effect = ValueError("Test error")
        
        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.fit.return_value = np.array([0, 1, 0, 1])
        mock_kmeans_instance.get_model.return_value = "kmeans_model"
        mock_kmeans_algo.return_value = mock_kmeans_instance
        
        data = np.array([[1, 2], [3, 4], [1, 3], [4, 2]])
        
        algorithm = SpectralClusteringAlgorithm(n_clusters=2, affinity='rbf', random_state=42)
        labels = algorithm.fit(data)
        
        mock_kmeans_algo.assert_called_once_with(n_clusters=2, random_state=42)
        
        mock_kmeans_instance.fit.assert_called_once_with(data)
        
        np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1]))
        
        self.assertEqual(algorithm.model, "kmeans_model")


class TestAgglomerativeClusteringAlgorithm(unittest.TestCase):
    """Tests for the AgglomerativeClusteringAlgorithm class."""
    
    def test_init(self):
        """Test initialization."""
        algorithm = AgglomerativeClusteringAlgorithm(n_clusters=5, linkage='complete')
        self.assertEqual(algorithm.n_clusters, 5)
        self.assertEqual(algorithm.linkage, 'complete')
        
    @patch('src.behavioural_clustering.utils.clustering_algorithms.AgglomerativeClustering')
    def test_fit(self, mock_agglomerative):
        """Test fit method."""
        mock_agglomerative_instance = MagicMock()
        mock_agglomerative_instance.labels_ = np.array([0, 1, 0, 1])
        mock_agglomerative.return_value = mock_agglomerative_instance
        
        data = np.array([[1, 2], [3, 4], [1, 3], [4, 2]])
        
        algorithm = AgglomerativeClusteringAlgorithm(n_clusters=2, linkage='ward')
        labels = algorithm.fit(data)
        
        mock_agglomerative.assert_called_once_with(n_clusters=2, linkage='ward')
        
        mock_agglomerative_instance.fit.assert_called_once_with(data)
        
        np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1]))
        
        self.assertEqual(algorithm.model, mock_agglomerative_instance)


class TestDBSCANAlgorithm(unittest.TestCase):
    """Tests for the DBSCANAlgorithm class."""
    
    def test_init(self):
        """Test initialization."""
        algorithm = DBSCANAlgorithm(eps=0.7, min_samples=10)
        self.assertEqual(algorithm.eps, 0.7)
        self.assertEqual(algorithm.min_samples, 10)
        
    @patch('src.behavioural_clustering.utils.clustering_algorithms.DBSCAN')
    def test_fit(self, mock_dbscan):
        """Test fit method."""
        mock_dbscan_instance = MagicMock()
        mock_dbscan_instance.labels_ = np.array([0, 1, 0, 1])
        mock_dbscan.return_value = mock_dbscan_instance
        
        data = np.array([[1, 2], [3, 4], [1, 3], [4, 2]])
        
        algorithm = DBSCANAlgorithm(eps=0.5, min_samples=2)
        labels = algorithm.fit(data)
        
        mock_dbscan.assert_called_once_with(eps=0.5, min_samples=2)
        
        mock_dbscan_instance.fit.assert_called_once_with(data)
        
        np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1]))
        
        self.assertEqual(algorithm.model, mock_dbscan_instance)


class TestGaussianMixtureAlgorithm(unittest.TestCase):
    """Tests for the GaussianMixtureAlgorithm class."""
    
    def test_init(self):
        """Test initialization."""
        algorithm = GaussianMixtureAlgorithm(n_components=5, random_state=42)
        self.assertEqual(algorithm.n_components, 5)
        self.assertEqual(algorithm.random_state, 42)
        
    @patch('src.behavioural_clustering.utils.clustering_algorithms.GaussianMixture')
    def test_fit(self, mock_gmm):
        """Test fit method."""
        mock_gmm_instance = MagicMock()
        mock_gmm_instance.predict.return_value = np.array([0, 1, 0, 1])
        mock_gmm.return_value = mock_gmm_instance
        
        data = np.array([[1, 2], [3, 4], [1, 3], [4, 2]])
        
        algorithm = GaussianMixtureAlgorithm(n_components=2, random_state=42)
        labels = algorithm.fit(data)
        
        mock_gmm.assert_called_once_with(n_components=2, random_state=42)
        
        mock_gmm_instance.fit.assert_called_once_with(data)
        
        mock_gmm_instance.predict.assert_called_once_with(data)
        
        np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1]))
        
        self.assertEqual(algorithm.model, mock_gmm_instance)


class TestUMAPClusteringAlgorithm(unittest.TestCase):
    """Tests for the UMAPClusteringAlgorithm class."""
    
    def test_init(self):
        """Test initialization."""
        algorithm = UMAPClusteringAlgorithm(
            n_clusters=5, 
            n_components=3, 
            n_neighbors=20, 
            min_dist=0.2, 
            random_state=42, 
            clustering_algorithm='spectral'
        )
        self.assertEqual(algorithm.n_clusters, 5)
        self.assertEqual(algorithm.n_components, 3)
        self.assertEqual(algorithm.n_neighbors, 20)
        self.assertEqual(algorithm.min_dist, 0.2)
        self.assertEqual(algorithm.random_state, 42)
        self.assertEqual(algorithm.clustering_algorithm, 'spectral')
        
    @patch('src.behavioural_clustering.utils.clustering_algorithms.umap.UMAP')
    @patch('src.behavioural_clustering.utils.clustering_algorithms.KMeansAlgorithm')
    def test_fit(self, mock_kmeans_algo, mock_umap):
        """Test fit method."""
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.1, 0.3], [0.4, 0.2]])
        mock_umap.return_value = mock_umap_instance
        
        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.fit.return_value = np.array([0, 1, 0, 1])
        mock_kmeans_instance.get_model.return_value = "kmeans_model"
        mock_kmeans_algo.return_value = mock_kmeans_instance
        
        data = np.array([[1, 2], [3, 4], [1, 3], [4, 2]])
        
        algorithm = UMAPClusteringAlgorithm(
            n_clusters=2, 
            n_components=2, 
            n_neighbors=15, 
            min_dist=0.1, 
            random_state=42, 
            clustering_algorithm='kmeans'
        )
        labels = algorithm.fit(data)
        
        mock_umap.assert_called_once_with(
            n_components=2, 
            n_neighbors=15, 
            min_dist=0.1, 
            random_state=42
        )
        
        mock_umap_instance.fit_transform.assert_called_once_with(data)
        
        mock_kmeans_algo.assert_called_once_with(n_clusters=2, random_state=42)
        
        mock_kmeans_instance.fit.assert_called_once_with(
            np.array([[0.1, 0.2], [0.3, 0.4], [0.1, 0.3], [0.4, 0.2]])
        )
        
        np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1]))
        
        self.assertEqual(algorithm.model['umap'], mock_umap_instance)
        self.assertEqual(algorithm.model['clustering'], "kmeans_model")
        
    @patch('src.behavioural_clustering.utils.clustering_algorithms.umap.UMAP')
    @patch('src.behavioural_clustering.utils.clustering_algorithms.KMeansAlgorithm')
    def test_fit_with_error(self, mock_kmeans_algo, mock_umap):
        """Test fit method with error."""
        mock_umap.side_effect = ValueError("Test error")
        
        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.fit.return_value = np.array([0, 1, 0, 1])
        mock_kmeans_instance.get_model.return_value = "kmeans_model"
        mock_kmeans_algo.return_value = mock_kmeans_instance
        
        data = np.array([[1, 2], [3, 4], [1, 3], [4, 2]])
        
        algorithm = UMAPClusteringAlgorithm(
            n_clusters=2, 
            n_components=2, 
            n_neighbors=15, 
            min_dist=0.1, 
            random_state=42, 
            clustering_algorithm='kmeans'
        )
        labels = algorithm.fit(data)
        
        mock_kmeans_algo.assert_called_once_with(n_clusters=2, random_state=42)
        
        mock_kmeans_instance.fit.assert_called_once_with(data)
        
        np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1]))
        
        self.assertEqual(algorithm.model, "kmeans_model")


class TestClusteringFactory(unittest.TestCase):
    """Tests for the ClusteringFactory class."""
    
    def test_create_algorithm_kmeans(self):
        """Test creating a KMeansAlgorithm."""
        algorithm = ClusteringFactory.create_algorithm('kmeans', n_clusters=5, random_state=42)
        self.assertIsInstance(algorithm, KMeansAlgorithm)
        self.assertEqual(algorithm.n_clusters, 5)
        self.assertEqual(algorithm.random_state, 42)
        
    def test_create_algorithm_spectral(self):
        """Test creating a SpectralClusteringAlgorithm."""
        algorithm = ClusteringFactory.create_algorithm('spectral', n_clusters=5, affinity='nearest_neighbors', random_state=42)
        self.assertIsInstance(algorithm, SpectralClusteringAlgorithm)
        self.assertEqual(algorithm.n_clusters, 5)
        self.assertEqual(algorithm.affinity, 'nearest_neighbors')
        self.assertEqual(algorithm.random_state, 42)
        
    def test_create_algorithm_agglomerative(self):
        """Test creating an AgglomerativeClusteringAlgorithm."""
        algorithm = ClusteringFactory.create_algorithm('agglomerative', n_clusters=5, linkage='complete')
        self.assertIsInstance(algorithm, AgglomerativeClusteringAlgorithm)
        self.assertEqual(algorithm.n_clusters, 5)
        self.assertEqual(algorithm.linkage, 'complete')
        
    def test_create_algorithm_dbscan(self):
        """Test creating a DBSCANAlgorithm."""
        algorithm = ClusteringFactory.create_algorithm('dbscan', eps=0.7, min_samples=10)
        self.assertIsInstance(algorithm, DBSCANAlgorithm)
        self.assertEqual(algorithm.eps, 0.7)
        self.assertEqual(algorithm.min_samples, 10)
        
    def test_create_algorithm_gmm(self):
        """Test creating a GaussianMixtureAlgorithm."""
        algorithm = ClusteringFactory.create_algorithm('gmm', n_components=5, random_state=42)
        self.assertIsInstance(algorithm, GaussianMixtureAlgorithm)
        self.assertEqual(algorithm.n_components, 5)
        self.assertEqual(algorithm.random_state, 42)
        
    def test_create_algorithm_umap(self):
        """Test creating a UMAPClusteringAlgorithm."""
        algorithm = ClusteringFactory.create_algorithm('umap', n_clusters=5, n_components=3, random_state=42)
        self.assertIsInstance(algorithm, UMAPClusteringAlgorithm)
        self.assertEqual(algorithm.n_clusters, 5)
        self.assertEqual(algorithm.n_components, 3)
        self.assertEqual(algorithm.random_state, 42)
        
    def test_create_algorithm_unknown(self):
        """Test creating an unknown algorithm."""
        algorithm = ClusteringFactory.create_algorithm('unknown', n_clusters=5, random_state=42)
        self.assertIsInstance(algorithm, KMeansAlgorithm)
        self.assertEqual(algorithm.n_clusters, 5)
        self.assertEqual(algorithm.random_state, 42)


class TestEvaluateClustering(unittest.TestCase):
    """Tests for the evaluate_clustering function."""
    
    @patch('src.behavioural_clustering.utils.clustering_algorithms.silhouette_score')
    @patch('src.behavioural_clustering.utils.clustering_algorithms.calinski_harabasz_score')
    @patch('src.behavioural_clustering.utils.clustering_algorithms.davies_bouldin_score')
    def test_evaluate_clustering(self, mock_db, mock_ch, mock_silhouette):
        """Test evaluate_clustering function."""
        mock_silhouette.return_value = 0.8
        mock_ch.return_value = 100.0
        mock_db.return_value = 0.2
        
        data = np.array([[1, 2], [3, 4], [1, 3], [4, 2]])
        labels = np.array([0, 1, 0, 1])
        
        metrics = evaluate_clustering(data, labels)
        
        mock_silhouette.assert_called_once_with(data, labels)
        mock_ch.assert_called_once_with(data, labels)
        mock_db.assert_called_once_with(data, labels)
        
        self.assertEqual(metrics['silhouette'], 0.8)
        self.assertEqual(metrics['calinski_harabasz'], 100.0)
        self.assertEqual(metrics['davies_bouldin'], 0.2)
        
    def test_evaluate_clustering_single_cluster(self):
        """Test evaluate_clustering function with a single cluster."""
        data = np.array([[1, 2], [3, 4], [1, 3], [4, 2]])
        labels = np.array([0, 0, 0, 0])
        
        metrics = evaluate_clustering(data, labels)
        
        self.assertEqual(metrics, {})


class TestFindOptimalClusters(unittest.TestCase):
    """Tests for the find_optimal_clusters function."""
    
    @patch('src.behavioural_clustering.utils.clustering_algorithms.ClusteringFactory')
    @patch('src.behavioural_clustering.utils.clustering_algorithms.evaluate_clustering')
    def test_find_optimal_clusters(self, mock_evaluate, mock_factory):
        """Test find_optimal_clusters function."""
        mock_algorithm = MagicMock()
        mock_algorithm.fit.side_effect = [
            np.array([0, 0, 1, 1]),  # 2 clusters
            np.array([0, 1, 2, 0])   # 3 clusters
        ]
        mock_factory.create_algorithm.return_value = mock_algorithm
        
        mock_evaluate.side_effect = [
            {'silhouette': 0.6, 'calinski_harabasz': 80.0, 'davies_bouldin': 0.3},  # 2 clusters
            {'silhouette': 0.8, 'calinski_harabasz': 100.0, 'davies_bouldin': 0.2}  # 3 clusters
        ]
        
        data = np.array([[1, 2], [3, 4], [1, 3], [4, 2]])
        
        n_clusters, labels, metrics = find_optimal_clusters(
            data, 
            algorithm='kmeans', 
            min_clusters=2, 
            max_clusters=3, 
            random_state=42
        )
        
        self.assertEqual(mock_factory.create_algorithm.call_count, 2)
        mock_factory.create_algorithm.assert_any_call('kmeans', n_clusters=2, random_state=42)
        mock_factory.create_algorithm.assert_any_call('kmeans', n_clusters=3, random_state=42)
        
        self.assertEqual(mock_algorithm.fit.call_count, 2)
        mock_algorithm.fit.assert_any_call(data)
        
        self.assertEqual(mock_evaluate.call_count, 2)
        mock_evaluate.assert_any_call(data, np.array([0, 0, 1, 1]))
        mock_evaluate.assert_any_call(data, np.array([0, 1, 2, 0]))
        
        self.assertEqual(n_clusters, 3)
        
        np.testing.assert_array_equal(labels, np.array([0, 1, 2, 0]))
        
        self.assertEqual(len(metrics), 3)
        self.assertEqual(len(metrics['silhouette']), 2)
        self.assertEqual(metrics['silhouette'][0], 0.6)
        self.assertEqual(metrics['silhouette'][1], 0.8)
        self.assertEqual(len(metrics['calinski_harabasz']), 2)
        self.assertEqual(metrics['calinski_harabasz'][0], 80.0)
        self.assertEqual(metrics['calinski_harabasz'][1], 100.0)
        self.assertEqual(len(metrics['davies_bouldin']), 2)
        self.assertEqual(metrics['davies_bouldin'][0], 0.3)
        self.assertEqual(metrics['davies_bouldin'][1], 0.2)


if __name__ == '__main__':
    unittest.main()
