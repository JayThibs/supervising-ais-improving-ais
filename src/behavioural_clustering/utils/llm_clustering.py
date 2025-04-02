import numpy as np
from typing import List, Optional, Union, Dict, Tuple
import logging
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random
from behavioural_clustering.models.model_factory import initialize_model

logger = logging.getLogger(__name__)

class KLLMmeansAlgorithm:
    """
    k-LLMmeans is a clustering algorithm that uses LLM-generated summaries as centroids
    for more interpretable clustering. Based on the paper "k-LLMmeans: Clustering with
    Large Language Models" (https://arxiv.org/abs/2502.09667).
    
    The algorithm works as follows:
    1. Initialization: Start with random cluster assignments or use k-means++ initialization
    2. Centroid Generation: For each cluster, generate a summary of the texts in the cluster using an LLM
    3. Embedding Generation: Convert the summaries to embeddings using the same embedding model
    4. Assignment: Assign each text to the closest centroid based on embedding similarity
    5. Iteration: Repeat steps 2-4 until convergence or maximum iterations
    """
    
    def __init__(
        self,
        n_clusters: int = 5,
        llm: Optional[object] = None,
        llm_info: Optional[Dict] = None,
        max_iterations: int = 5,
        random_state: Optional[int] = None,
        fallback_to_kmeans: bool = True
    ):
        """
        Initialize the k-LLMmeans algorithm.
        
        Args:
            n_clusters: Number of clusters to generate
            llm: Pre-initialized LLM instance
            llm_info: Dictionary with LLM configuration if llm is not provided
            max_iterations: Maximum number of iterations
            random_state: Random seed for reproducibility
            fallback_to_kmeans: Whether to fall back to standard k-means if LLM fails
        """
        self.n_clusters = n_clusters
        self.llm = llm
        self.llm_info = llm_info
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.fallback_to_kmeans = fallback_to_kmeans
        self.labels_ = np.array([])
        self.cluster_centers_ = np.array([])
        self.summaries_ = []
        
        if self.llm is None and self.llm_info is not None:
            self.llm = initialize_model(self.llm_info)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
    
    def fit(self, embeddings: np.ndarray, texts: List[str]) -> np.ndarray:
        """
        Fit the k-LLMmeans algorithm to the data.
        
        Args:
            embeddings: Array of embeddings for each text
            texts: List of original texts
            
        Returns:
            Array of cluster labels
        """
        if len(texts) < self.n_clusters:
            logger.warning(f"Number of texts ({len(texts)}) is less than n_clusters ({self.n_clusters}). Reducing n_clusters.")
            self.n_clusters = len(texts)
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.labels_ = kmeans.fit_predict(embeddings)
        
        initial_centroids = kmeans.cluster_centers_
        
        for iteration in range(self.max_iterations):
            logger.info(f"k-LLMmeans iteration {iteration+1}/{self.max_iterations}")
            
            try:
                summaries = self._generate_cluster_summaries(texts)
                
                summary_embeddings = self._embed_summaries(summaries, embeddings)
                
                new_labels = self._assign_clusters(embeddings, summary_embeddings)
                
                if np.array_equal(self.labels_, new_labels):
                    logger.info(f"k-LLMmeans converged after {iteration+1} iterations")
                    break
                
                self.labels_ = new_labels
                self.cluster_centers_ = summary_embeddings
                self.summaries_ = summaries
                
            except Exception as e:
                logger.error(f"Error in k-LLMmeans iteration: {e}")
                if self.fallback_to_kmeans:
                    logger.info("Falling back to standard k-means")
                    self.cluster_centers_ = initial_centroids
                    self.labels_ = self._assign_clusters(embeddings, initial_centroids)
                break
        
        return self.labels_
    
    def _generate_cluster_summaries(self, texts: List[str]) -> List[str]:
        """
        Generate summaries for each cluster using the LLM.
        
        Args:
            texts: List of original texts
            
        Returns:
            List of summaries, one for each cluster
        """
        if self.llm is None:
            raise ValueError("LLM not initialized. Provide either llm or llm_info.")
        
        summaries = []
        for cluster_id in range(self.n_clusters):
            cluster_texts = [texts[i] for i in range(len(texts)) if self.labels_[i] == cluster_id]
            
            if not cluster_texts:
                summaries.append("Empty cluster")
                continue
            
            if len(cluster_texts) > 10:
                sampled_texts = random.sample(cluster_texts, 10)
            else:
                sampled_texts = cluster_texts
            
            texts_list = "\n".join([f"- {text}" for text in sampled_texts])
            prompt = f"""
            I have a cluster of texts that share similar characteristics. Please provide a concise summary (1-2 sentences) 
            that captures the common theme or intent across these texts. Focus on the key patterns, topics, or behaviors 
            that unify them.

            Texts:
            {texts_list}
            
            Summary:
            """
            
            try:
                response = self.llm.generate(prompt)
                summary = response.strip()
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error generating summary for cluster {cluster_id}: {e}")
                summary = " ".join(sampled_texts[:3])[:100] + "..."
                summaries.append(summary)
        
        return summaries
    
    def _embed_summaries(self, summaries: List[str], original_embeddings: np.ndarray) -> np.ndarray:
        """
        Convert summaries to embeddings.
        
        This is a placeholder - in a real implementation, you would use the same
        embedding model that was used for the original texts.
        
        Args:
            summaries: List of summaries
            original_embeddings: Original embeddings for reference
            
        Returns:
            Array of summary embeddings
        """
        summary_embeddings = np.zeros((self.n_clusters, original_embeddings.shape[1]))
        
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(self.labels_ == cluster_id)[0]
            if len(cluster_indices) > 0:
                summary_embeddings[cluster_id] = np.mean(original_embeddings[cluster_indices], axis=0)
            else:
                summary_embeddings[cluster_id] = np.random.randn(original_embeddings.shape[1])
                summary_embeddings[cluster_id] /= np.linalg.norm(summary_embeddings[cluster_id])
        
        return summary_embeddings
    
    def _assign_clusters(self, embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each text to the closest centroid.
        
        Args:
            embeddings: Array of text embeddings
            centroids: Array of centroid embeddings
            
        Returns:
            Array of cluster labels
        """
        similarities = cosine_similarity(embeddings, centroids)
        return np.argmax(similarities, axis=1)


class SPILLAlgorithm:
    """
    SPILL (Selection and Pooling with LLMs) is a domain-adaptive intent clustering algorithm
    that uses LLMs to determine intent similarity between texts. Based on the paper
    "SPILL: Domain-Adaptive Intent Clustering based on Selection and Pooling with Large Language Models"
    (https://arxiv.org/abs/2503.15351).
    
    The algorithm works as follows:
    1. Seed Selection: Select initial seed texts using diversity sampling
    2. Intent Description: Generate intent descriptions for each seed using an LLM
    3. Candidate Selection: For each intent, select texts with similar intent using LLM-based similarity scoring
    4. Refinement: Refine the intent descriptions based on selected texts
    5. Iteration: Repeat steps 3-4 until convergence or maximum iterations
    """
    
    def __init__(
        self,
        n_clusters: int = 5,
        llm: Optional[object] = None,
        llm_info: Optional[Dict] = None,
        max_iterations: int = 3,
        selection_threshold: float = 0.5,
        random_state: Optional[int] = None
    ):
        """
        Initialize the SPILL algorithm.
        
        Args:
            n_clusters: Number of clusters to generate
            llm: Pre-initialized LLM instance
            llm_info: Dictionary with LLM configuration if llm is not provided
            max_iterations: Maximum number of iterations
            selection_threshold: Threshold for selecting texts with similar intent
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.llm = llm
        self.llm_info = llm_info
        self.max_iterations = max_iterations
        self.selection_threshold = selection_threshold
        self.random_state = random_state
        self.labels_ = np.array([])
        self.intent_descriptions_ = []
        
        if self.llm is None and self.llm_info is not None:
            self.llm = initialize_model(self.llm_info)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
    
    def fit(self, embeddings: np.ndarray, texts: List[str]) -> np.ndarray:
        """
        Fit the SPILL algorithm to the data.
        
        Args:
            embeddings: Array of embeddings for each text
            texts: List of original texts
            
        Returns:
            Array of cluster labels
        """
        if len(texts) < self.n_clusters:
            logger.warning(f"Number of texts ({len(texts)}) is less than n_clusters ({self.n_clusters}). Reducing n_clusters.")
            self.n_clusters = len(texts)
        
        self.labels_ = np.full(len(texts), -1)
        
        seed_indices = self._select_diverse_seeds(embeddings)
        seed_texts = [texts[i] for i in seed_indices]
        
        self.intent_descriptions_ = self._generate_intent_descriptions(seed_texts)
        
        for iteration in range(self.max_iterations):
            logger.info(f"SPILL iteration {iteration+1}/{self.max_iterations}")
            
            cluster_assignments = self._select_candidates(texts)
            
            for cluster_id, text_indices in enumerate(cluster_assignments):
                for idx in text_indices:
                    self.labels_[idx] = cluster_id
            
            unassigned = np.where(self.labels_ == -1)[0]
            if len(unassigned) > 0:
                logger.info(f"{len(unassigned)} texts remain unassigned")
                
                if iteration == self.max_iterations - 1:
                    self._assign_unassigned(embeddings, unassigned)
            
            if iteration < self.max_iterations - 1:
                self._refine_intent_descriptions(texts)
        
        return self.labels_
    
    def _select_diverse_seeds(self, embeddings: np.ndarray) -> List[int]:
        """
        Select diverse seed texts using embeddings.
        
        Args:
            embeddings: Array of text embeddings
            
        Returns:
            Indices of selected seed texts
        """
        n_samples = embeddings.shape[0]
        indices = np.arange(n_samples)
        
        seed_indices = [random.choice(indices)]
        
        for _ in range(1, self.n_clusters):
            min_distances = np.min([
                np.sum((embeddings - embeddings[idx])**2, axis=1)
                for idx in seed_indices
            ], axis=0)
            
            probabilities = min_distances / np.sum(min_distances)
            next_seed = np.random.choice(indices, p=probabilities)
            seed_indices.append(next_seed)
        
        return seed_indices
    
    def _generate_intent_descriptions(self, seed_texts: List[str]) -> List[str]:
        """
        Generate intent descriptions for seed texts using the LLM.
        
        Args:
            seed_texts: List of seed texts
            
        Returns:
            List of intent descriptions
        """
        if self.llm is None:
            raise ValueError("LLM not initialized. Provide either llm or llm_info.")
        
        intent_descriptions = []
        for text in seed_texts:
            prompt = f"""
            Please identify the underlying intent or purpose of the following text. 
            Provide a concise description (1-2 sentences) that captures what the person 
            is trying to accomplish or express.

            Text: "{text}"
            
            Intent description:
            """
            
            try:
                response = self.llm.generate(prompt)
                intent_description = response.strip()
                intent_descriptions.append(intent_description)
            except Exception as e:
                logger.error(f"Error generating intent description: {e}")
                intent_descriptions.append(f"Intent related to: {text[:50]}...")
        
        return intent_descriptions
    
    def _select_candidates(self, texts: List[str]) -> List[List[int]]:
        """
        Select candidate texts for each intent using LLM-based similarity scoring.
        
        Args:
            texts: List of original texts
            
        Returns:
            List of lists of text indices for each cluster
        """
        if self.llm is None:
            raise ValueError("LLM not initialized. Provide either llm or llm_info.")
        
        cluster_assignments = [[] for _ in range(self.n_clusters)]
        
        batch_size = 10
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            for text_idx, text in zip(batch_indices, batch_texts):
                if self.labels_[text_idx] != -1:
                    cluster_assignments[self.labels_[text_idx]].append(text_idx)
                    continue
                
                best_score = -1
                best_cluster = -1
                
                for cluster_id, intent_desc in enumerate(self.intent_descriptions_):
                    prompt = f"""
                    I want to determine if the following text matches a specific intent.
                    
                    Intent description: "{intent_desc}"
                    
                    Text to evaluate: "{text}"
                    
                    On a scale from 0 to 1, where 0 means completely unrelated and 1 means perfectly matching the intent,
                    rate how well the text matches the intent. Provide only the numerical score.
                    
                    Score:
                    """
                    
                    try:
                        response = self.llm.generate(prompt)
                        score_text = response.strip()
                        try:
                            score = float(score_text)
                            score = max(0, min(1, score))  # Clamp to [0, 1]
                        except ValueError:
                            score = 0.1
                        
                        if score > best_score:
                            best_score = score
                            best_cluster = cluster_id
                    except Exception as e:
                        logger.error(f"Error in intent matching: {e}")
                
                if best_score >= self.selection_threshold:
                    cluster_assignments[best_cluster].append(text_idx)
                    self.labels_[text_idx] = best_cluster
        
        return cluster_assignments
    
    def _refine_intent_descriptions(self, texts: List[str]) -> None:
        """
        Refine intent descriptions based on selected texts.
        
        Args:
            texts: List of original texts
        """
        if self.llm is None:
            raise ValueError("LLM not initialized. Provide either llm or llm_info.")
        
        new_intent_descriptions = []
        
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(self.labels_ == cluster_id)[0]
            cluster_texts = [texts[i] for i in cluster_indices]
            
            if not cluster_texts:
                new_intent_descriptions.append(self.intent_descriptions_[cluster_id])
                continue
            
            if len(cluster_texts) > 5:
                sampled_texts = random.sample(cluster_texts, 5)
            else:
                sampled_texts = cluster_texts
            
            texts_list = "\n".join([f"- {text}" for text in sampled_texts])
            prompt = f"""
            I have a group of texts that share a similar intent. The current intent description is:
            "{self.intent_descriptions_[cluster_id]}"
            
            Here are some examples of texts with this intent:
            {texts_list}
            
            Please refine the intent description to better capture what these texts have in common.
            Provide a concise description (1-2 sentences).
            
            Refined intent description:
            """
            
            try:
                response = self.llm.generate(prompt)
                refined_description = response.strip()
                new_intent_descriptions.append(refined_description)
            except Exception as e:
                logger.error(f"Error refining intent description: {e}")
                new_intent_descriptions.append(self.intent_descriptions_[cluster_id])
        
        self.intent_descriptions_ = new_intent_descriptions
    
    def _assign_unassigned(self, embeddings: np.ndarray, unassigned_indices: np.ndarray) -> None:
        """
        Assign unassigned texts to the nearest cluster based on embeddings.
        
        Args:
            embeddings: Array of text embeddings
            unassigned_indices: Indices of unassigned texts
        """
        centroids = []
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(self.labels_ == cluster_id)[0]
            if len(cluster_indices) > 0:
                centroids.append(np.mean(embeddings[cluster_indices], axis=0))
            else:
                random_embedding = np.random.randn(embeddings.shape[1])
                centroids.append(random_embedding / np.linalg.norm(random_embedding))
        
        centroids = np.array(centroids)
        
        for idx in unassigned_indices:
            similarities = cosine_similarity(embeddings[idx].reshape(1, -1), centroids)[0]
            self.labels_[idx] = np.argmax(similarities)


def update_clustering_factory(algorithm_map: Dict) -> Dict:
    """
    Update the clustering factory with LLM-based clustering algorithms.
    
    Args:
        algorithm_map: Existing algorithm map from the Clustering class
        
    Returns:
        Updated algorithm map
    """
    updated_map = algorithm_map.copy()
    updated_map["k-LLMmeans"] = KLLMmeansAlgorithm
    updated_map["SPILL"] = SPILLAlgorithm
    
    return updated_map
