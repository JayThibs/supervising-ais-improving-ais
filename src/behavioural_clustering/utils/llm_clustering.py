"""
Modern LLM-based clustering algorithms for behavioral analysis.

This module implements state-of-the-art clustering methods that leverage
large language models for more interpretable and domain-adaptive clustering:

1. k-LLMmeans: Uses LLM-generated summaries as centroids for interpretable clustering
   (Based on: https://arxiv.org/abs/2502.09667)

2. SPILL: Domain-Adaptive Intent Clustering using Selection and Pooling with LLMs
   (Based on: https://arxiv.org/abs/2503.15351)
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Protocol, TypeVar, Callable
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import umap
from termcolor import colored
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from behavioural_clustering.models.model_factory import initialize_model
from behavioural_clustering.utils.clustering_algorithms import ClusteringAlgorithm, evaluate_clustering
from behavioural_clustering.utils.embedding_utils import embed_texts

logger = logging.getLogger(__name__)

T = TypeVar('T')


class EmbeddingSettings:
    """Settings for embedding generation."""
    
    def __init__(self, 
                embedding_model: str = "text-embedding-3-small",
                batch_size: int = 10,
                max_retries: int = 3,
                initial_sleep_time: float = 1.0):
        """
        Initialize embedding settings.
        
        Args:
            embedding_model: Name of the embedding model
            batch_size: Batch size for embedding requests
            max_retries: Maximum number of retries for failed requests
            initial_sleep_time: Initial sleep time for exponential backoff
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.initial_sleep_time = initial_sleep_time


class LLMClusteringAlgorithm(ClusteringAlgorithm):
    """Base class for LLM-based clustering algorithms."""
    
    def __init__(self, 
                model_family: str = "anthropic", 
                model_name: str = "claude-3-haiku-20240307",
                system_message: str = "",
                temperature: float = 0.1,
                max_tokens: int = 300,
                embedding_settings: Optional[EmbeddingSettings] = None,
                **kwargs):
        """
        Initialize the LLM-based clustering algorithm.
        
        Args:
            model_family: LLM model family (e.g., "anthropic", "openai")
            model_name: Specific model to use
            system_message: System message for the LLM
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
            embedding_settings: Settings for embedding generation
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.model_family = model_family
        self.model_name = model_name
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.embedding_settings = embedding_settings or EmbeddingSettings()
        self.llm = None
        self.centroids = None
        self.centroid_summaries = None
        
    def _initialize_llm(self):
        """Initialize the LLM if not already initialized."""
        if self.llm is None:
            model_info = {
                "model_family": self.model_family,
                "model_name": self.model_name,
                "system_message": self.system_message
            }
            self.llm = initialize_model(
                model_info, 
                temperature=self.temperature, 
                max_tokens=self.max_tokens
            )
            
    def _generate_with_llm(self, prompt: str) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt for the LLM
            
        Returns:
            Generated text
        """
        self._initialize_llm()
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(colored(f"Error generating with LLM: {str(e)}", "red"))
            return ""
            
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        try:
            return np.array(embed_texts(texts, self.embedding_settings))
        except Exception as e:
            logger.error(colored(f"Error getting embeddings: {str(e)}", "red"))
            return np.zeros((len(texts), 1536))  # OpenAI embeddings are 1536-dimensional


class KLLMmeansAlgorithm(LLMClusteringAlgorithm):
    """
    k-LLMmeans clustering algorithm.
    
    This algorithm uses LLM-generated summaries as centroids for clustering,
    providing more interpretable results than traditional k-means.
    
    Based on: https://arxiv.org/abs/2502.09667
    """
    
    def __init__(self, 
                n_clusters: int = 8, 
                random_state: int = 42,
                max_iterations: int = 10,
                convergence_threshold: float = 0.01,
                batch_size: int = 10,
                parallel_summarization: bool = True,
                max_workers: int = 4,
                model_family: str = "anthropic", 
                model_name: str = "claude-3-haiku-20240307",
                **kwargs):
        """
        Initialize k-LLMmeans clustering.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random state for reproducibility
            max_iterations: Maximum number of iterations
            convergence_threshold: Threshold for convergence
            batch_size: Batch size for processing
            parallel_summarization: Whether to parallelize summarization
            max_workers: Maximum number of parallel workers
            model_family: LLM model family
            model_name: Specific model to use
            **kwargs: Additional parameters
        """
        super().__init__(
            model_family=model_family,
            model_name=model_name,
            **kwargs
        )
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.batch_size = batch_size
        self.parallel_summarization = parallel_summarization
        self.max_workers = max_workers
        
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        
    def _generate_centroid_summary(self, texts: List[str], cluster_id: int) -> str:
        """
        Generate a summary for a cluster centroid using an LLM.
        
        Args:
            texts: List of texts in the cluster
            cluster_id: Cluster identifier
            
        Returns:
            Summary text representing the centroid
        """
        if len(texts) > self.batch_size:
            sampled_texts = random.sample(texts, self.batch_size)
        else:
            sampled_texts = texts
            
        prompt = f"""You are analyzing a cluster of text responses from language models. 
Your task is to create a concise summary (1-2 sentences) that captures the central theme or concept of this cluster.

Here are {len(sampled_texts)} examples from Cluster {cluster_id}:

{chr(10).join([f"- {text}" for text in sampled_texts])}

Based on these examples, provide a concise summary that represents the central theme of this cluster.
Your summary should be specific enough to distinguish this cluster from others.
"""
        
        summary = self._generate_with_llm(prompt)
        
        summary = summary.strip()
        if not summary:
            summary = f"Cluster {cluster_id}"
            
        return summary
        
    def _generate_all_centroid_summaries(self, 
                                        data: np.ndarray, 
                                        texts: List[str], 
                                        labels: np.ndarray) -> List[str]:
        """
        Generate summaries for all centroids.
        
        Args:
            data: Embedding data
            texts: Original texts
            labels: Cluster labels
            
        Returns:
            List of centroid summaries
        """
        summaries = [""] * self.n_clusters
        
        cluster_texts = {i: [] for i in range(self.n_clusters)}
        for i, label in enumerate(labels):
            cluster_texts[label].append(texts[i])
            
        if self.parallel_summarization and self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_cluster = {
                    executor.submit(self._generate_centroid_summary, cluster_texts[i], i): i 
                    for i in range(self.n_clusters)
                }
                
                for future in tqdm(as_completed(future_to_cluster), total=self.n_clusters, desc="Generating summaries"):
                    cluster_id = future_to_cluster[future]
                    try:
                        summary = future.result()
                        summaries[cluster_id] = summary
                    except Exception as e:
                        logger.error(colored(f"Error generating summary for cluster {cluster_id}: {str(e)}", "red"))
                        summaries[cluster_id] = f"Cluster {cluster_id}"
        else:
            for i in tqdm(range(self.n_clusters), desc="Generating summaries"):
                summaries[i] = self._generate_centroid_summary(cluster_texts[i], i)
                
        return summaries
        
    def _compute_centroid_embeddings(self, summaries: List[str]) -> np.ndarray:
        """
        Compute embeddings for centroid summaries.
        
        Args:
            summaries: List of centroid summaries
            
        Returns:
            Array of centroid embeddings
        """
        try:
            embeddings = self._get_embeddings(summaries)
            
            if hasattr(self, 'data_dim') and embeddings.shape[1] != self.data_dim:
                logger.warning(colored(f"Embedding dimension mismatch: {embeddings.shape[1]} vs {self.data_dim}", "yellow"))
                logger.warning(colored("Falling back to standard k-means centroids", "yellow"))
                return self.kmeans.cluster_centers_
                
            return embeddings
        except Exception as e:
            logger.error(colored(f"Error computing centroid embeddings: {str(e)}", "red"))
            
            if hasattr(self.kmeans, "cluster_centers_"):
                return self.kmeans.cluster_centers_
                
            return np.zeros((self.n_clusters, self.data_dim))
        
    def fit(self, data: np.ndarray, texts: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit k-LLMmeans clustering to the data.
        
        Args:
            data: Input embedding data for clustering
            texts: Original texts corresponding to the embeddings
            
        Returns:
            Array of cluster labels
        """
        if texts is None or len(texts) == 0:
            logger.warning(colored("No texts provided for k-LLMmeans. Falling back to standard k-means.", "yellow"))
            self.model = self.kmeans
            self.model.fit(data)
            return self.model.labels_
            
        if len(texts) != len(data):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of data points ({len(data)})")
            
        self.data_dim = data.shape[1]
        
        logger.info(colored("Initializing with standard k-means...", "cyan"))
        self.kmeans.fit(data)
        current_labels = self.kmeans.labels_
        
        centroids = self.kmeans.cluster_centers_.copy()
        
        for iteration in range(self.max_iterations):
            logger.info(colored(f"k-LLMmeans iteration {iteration+1}/{self.max_iterations}", "cyan"))
            
            centroid_summaries = self._generate_all_centroid_summaries(data, texts, current_labels)
            
            new_centroids = self._compute_centroid_embeddings(centroid_summaries)
            
            distances = pairwise_distances(data, new_centroids)
            new_labels = np.argmin(distances, axis=1)
            
            changes = np.sum(new_labels != current_labels)
            change_ratio = changes / len(data)
            logger.info(colored(f"  Changed assignments: {changes}/{len(data)} ({change_ratio:.2%})", "cyan"))
            
            if change_ratio < self.convergence_threshold:
                logger.info(colored(f"Converged after {iteration+1} iterations", "green"))
                break
                
            current_labels = new_labels
            centroids = new_centroids
            
        self.centroids = centroids
        self.centroid_summaries = centroid_summaries
        self.model = self.kmeans  # For compatibility
        
        return current_labels


class SPILLAlgorithm(LLMClusteringAlgorithm):
    """
    SPILL: Domain-Adaptive Intent Clustering algorithm.
    
    This algorithm uses Selection and Pooling with LLMs for domain-adaptive
    intent clustering without fine-tuning.
    
    Based on: https://arxiv.org/abs/2503.15351
    """
    
    def __init__(self, 
                n_clusters: int = 8, 
                random_state: int = 42,
                seed_selection_method: str = "kmeans",
                candidate_pool_size: int = 20,
                selection_threshold: float = 0.7,
                max_iterations: int = 3,
                model_family: str = "anthropic", 
                model_name: str = "claude-3-haiku-20240307",
                **kwargs):
        """
        Initialize SPILL clustering.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random state for reproducibility
            seed_selection_method: Method for selecting initial seeds ('kmeans', 'random')
            candidate_pool_size: Number of candidates to consider for each seed
            selection_threshold: Threshold for selecting candidates
            max_iterations: Maximum number of refinement iterations
            model_family: LLM model family
            model_name: Specific model to use
            **kwargs: Additional parameters
        """
        super().__init__(
            model_family=model_family,
            model_name=model_name,
            **kwargs
        )
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.seed_selection_method = seed_selection_method
        self.candidate_pool_size = candidate_pool_size
        self.selection_threshold = selection_threshold
        self.max_iterations = max_iterations
        
        if seed_selection_method == "kmeans":
            self.kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10
            )
        
        self.seeds = None
        self.seed_texts = None
        self.seed_embeddings = None
        self.intent_descriptions = None
        
    def _select_initial_seeds(self, data: np.ndarray, texts: List[str]) -> Tuple[List[int], List[str]]:
        """
        Select initial seed examples.
        
        Args:
            data: Input embedding data
            texts: Original texts
            
        Returns:
            Tuple of (seed indices, seed texts)
        """
        if self.seed_selection_method == "kmeans":
            self.kmeans.fit(data)
            centroids = self.kmeans.cluster_centers_
            
            seed_indices = []
            for i in range(self.n_clusters):
                distances = pairwise_distances(data, centroids[i].reshape(1, -1))
                closest_idx = int(np.argmin(distances))
                seed_indices.append(closest_idx)
                
        elif self.seed_selection_method == "random":
            np.random.seed(self.random_state)
            seed_indices = [int(idx) for idx in np.random.choice(len(data), self.n_clusters, replace=False)]
            
        else:
            raise ValueError(f"Unknown seed selection method: {self.seed_selection_method}")
            
        seed_texts = [texts[idx] for idx in seed_indices]
        
        return seed_indices, seed_texts
        
    def _generate_intent_description(self, text: str, intent_id: int) -> str:
        """
        Generate an intent description for a seed example.
        
        Args:
            text: Seed text
            intent_id: Intent identifier
            
        Returns:
            Intent description
        """
        prompt = f"""You are analyzing a text to identify its underlying intent or purpose.
Your task is to create a concise description (1-2 sentences) that captures the intent behind this text.

Here is the text:
"{text}"

Based on this text, provide a concise description of the intent or purpose behind it.
Your description should be specific enough to distinguish this intent from others.
"""
        
        description = self._generate_with_llm(prompt)
        
        description = description.strip()
        if not description:
            description = f"Intent {intent_id}"
            
        return description
        
    def _select_candidates(self, 
                          data: np.ndarray, 
                          texts: List[str], 
                          seed_idx: int, 
                          seed_text: str, 
                          intent_description: str) -> List[int]:
        """
        Select candidates that share the same intent as the seed.
        
        Args:
            data: Input embedding data
            texts: Original texts
            seed_idx: Index of the seed example
            seed_text: Text of the seed example
            intent_description: Description of the intent
            
        Returns:
            List of candidate indices
        """
        seed_embedding = data[seed_idx].reshape(1, -1)
        distances = pairwise_distances(data, seed_embedding)
        
        nearest_indices = np.argsort(distances.flatten())[1:self.candidate_pool_size+1]
        candidate_texts = [texts[idx] for idx in nearest_indices]
        
        selected_indices = []
        
        for i, (idx, text) in enumerate(zip(nearest_indices, candidate_texts)):
            prompt = f"""You are analyzing two texts to determine if they share the same underlying intent or purpose.

Text 1 (Seed): "{seed_text}"
Intent description: {intent_description}

Text 2 (Candidate): "{text}"

Do these two texts share the same underlying intent? Answer with a number between 0 and 1, where:
- 0 means completely different intents
- 1 means exactly the same intent
"""
            
            response = self._generate_with_llm(prompt)
            
            try:
                score = float(response.strip())
                
                if score >= self.selection_threshold:
                    selected_indices.append(idx)
            except ValueError:
                logger.warning(colored(f"Could not parse score from response: {response}", "yellow"))
                
        return selected_indices
        
    def _refine_seed_embedding(self, data: np.ndarray, selected_indices: List[int]) -> np.ndarray:
        """
        Refine the seed embedding by pooling selected candidates.
        
        Args:
            data: Input embedding data
            selected_indices: Indices of selected candidates
            
        Returns:
            Refined seed embedding
        """
        if not selected_indices:
            return np.zeros((data.shape[1],))
            
        selected_embeddings = data[selected_indices]
        refined_embedding = np.mean(selected_embeddings, axis=0)
        
        return refined_embedding
        
    def fit(self, data: np.ndarray, texts: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit SPILL clustering to the data.
        
        Args:
            data: Input embedding data for clustering
            texts: Original texts corresponding to the embeddings
            
        Returns:
            Array of cluster labels
        """
        if texts is None or len(texts) == 0:
            logger.warning(colored("No texts provided for SPILL. Falling back to standard k-means.", "yellow"))
            if self.seed_selection_method == "kmeans":
                self.model = self.kmeans
                self.model.fit(data)
                return self.model.labels_
            else:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
                self.model = kmeans
                self.model.fit(data)
                return self.model.labels_
            
        if len(texts) != len(data):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of data points ({len(data)})")
            
        logger.info(colored("Selecting initial seeds...", "cyan"))
        seed_indices, seed_texts = self._select_initial_seeds(data, texts)
        self.seed_texts = seed_texts
        
        logger.info(colored("Generating intent descriptions...", "cyan"))
        intent_descriptions = []
        for i, text in enumerate(seed_texts):
            description = self._generate_intent_description(text, i)
            intent_descriptions.append(description)
        self.intent_descriptions = intent_descriptions
        
        seed_embeddings = data[seed_indices].copy()
        
        for iteration in range(self.max_iterations):
            logger.info(colored(f"SPILL iteration {iteration+1}/{self.max_iterations}", "cyan"))
            
            new_seed_embeddings = []
            
            for i, (seed_idx, seed_text, intent_desc) in enumerate(zip(seed_indices, seed_texts, intent_descriptions)):
                logger.info(colored(f"  Processing intent {i+1}/{self.n_clusters}: {intent_desc[:50]}...", "cyan"))
                
                selected_indices = self._select_candidates(
                    data, texts, seed_idx, seed_text, intent_desc
                )
                
                logger.info(colored(f"    Selected {len(selected_indices)} candidates", "cyan"))
                
                refined_embedding = self._refine_seed_embedding(data, selected_indices)
                
                if refined_embedding is not None:
                    new_seed_embeddings.append(refined_embedding)
                else:
                    new_seed_embeddings.append(seed_embeddings[i])
                    
            seed_embeddings = np.array(new_seed_embeddings)
            
        self.seed_embeddings = seed_embeddings
        distances = pairwise_distances(data, seed_embeddings)
        labels = np.argmin(distances, axis=1)
        
        self.model = type('DummyModel', (), {'cluster_centers_': seed_embeddings, 'labels_': labels})
        
        return labels


def update_clustering_factory():
    """
    Update the ClusteringFactory to include LLM-based clustering algorithms.
    
    This function monkey-patches the ClusteringFactory.create_algorithm method
    to include k-LLMmeans and SPILL algorithms.
    """
    from behavioural_clustering.utils.clustering_algorithms import ClusteringFactory
    
    original_create_algorithm = ClusteringFactory.create_algorithm
    
    @staticmethod
    def new_create_algorithm(algorithm_name: str, **kwargs) -> ClusteringAlgorithm:
        """
        Create a clustering algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            **kwargs: Parameters for the algorithm
            
        Returns:
            Clustering algorithm instance
        """
        if algorithm_name.lower() in ['kllmmeans', 'k-llmmeans']:
            return KLLMmeansAlgorithm(**kwargs)
        elif algorithm_name.lower() in ['spill']:
            return SPILLAlgorithm(**kwargs)
        else:
            return original_create_algorithm(algorithm_name, **kwargs)
            
    ClusteringFactory.create_algorithm = new_create_algorithm
