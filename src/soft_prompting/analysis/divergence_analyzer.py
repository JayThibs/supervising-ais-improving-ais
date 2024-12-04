from typing import Dict, List, Optional, Tuple, Union
import json
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from collections import defaultdict
from datetime import datetime

from ..metrics.divergence_metrics import DivergenceMetrics

logger = logging.getLogger(__name__)

class DivergenceAnalyzer:
    """Analyze and visualize behavioral divergences between models."""
    
    def __init__(
        self,
        metrics: DivergenceMetrics,
        output_dir: Union[str, Path]
    ):
        """Initialize analyzer with metrics and output directory."""
        # Convert output_dir to Path object for path operations
        self.output_dir = Path(str(output_dir))  # Convert to string first in case it's already a Path
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics
    
    def analyze_training_progress(self, training_history: List[Dict]) -> Dict:
        """Analyze training convergence and early stopping effectiveness."""
        try:
            # Handle case where training_history is a list of examples
            if isinstance(training_history, list) and all(isinstance(x, dict) for x in training_history):
                # If the first item has a training_history key, use that
                if "training_history" in training_history[0]:
                    history = training_history[0]["training_history"]
                else:
                    # Otherwise, assume the list itself is the history
                    history = training_history
            else:
                history = training_history

            # Extract metrics from history
            epochs = []
            losses = []
            divergences = []
            
            for entry in history:
                if isinstance(entry, dict):
                    epochs.append(entry.get("epoch", 0))
                    losses.append(entry.get("loss", 0.0))
                    divergences.append(entry.get("kl_divergence", 0.0))
            
            if not epochs:  # If no valid data was found
                return {
                    "training_length": 0,
                    "final_loss": 0.0,
                    "best_divergence": 0.0,
                    "convergence_epoch": 0,
                    "early_stopping_effective": False
                }

            return {
                "training_length": len(epochs),
                "final_loss": losses[-1] if losses else 0.0,
                "best_divergence": max(divergences) if divergences else 0.0,
                "convergence_epoch": epochs[divergences.index(max(divergences))] if divergences else 0,
                "early_stopping_effective": len(epochs) < max(epochs) if epochs else False
            }
            
        except Exception as e:
            print(f"Error in analyze_training_progress: {str(e)}")
            print(f"Training history type: {type(training_history)}")
            print(f"Training history structure: {training_history[:2] if isinstance(training_history, list) else training_history}")
            return {
                "training_length": 0,
                "final_loss": 0.0,
                "best_divergence": 0.0,
                "convergence_epoch": 0,
                "early_stopping_effective": False
            }
    
    def analyze_divergence_patterns(
        self,
        dataset: List[Dict],
        min_divergence: float = 0.1,
        intervention_type: Optional[str] = None
    ) -> Dict:
        """Analyze patterns in high-divergence examples."""
        try:
            # Ensure we have valid data
            if not dataset:
                return {
                    "num_examples": 0,
                    "mean_divergence": 0.0,
                    "high_divergence_examples": [],
                    "token_patterns": {},
                    "semantic_patterns": {
                        "mean_semantic_diff": 0.0,
                        "std_semantic_diff": 0.0,
                        "top_semantic_clusters": []
                    }
                }

            # Filter for high-divergence examples with proper error handling
            high_div_examples = []
            for ex in dataset:
                try:
                    if ex.get("metrics", {}).get("kl_divergence", 0.0) > min_divergence:
                        high_div_examples.append(ex)
                except (KeyError, TypeError):
                    continue

            # Calculate basic statistics safely
            divergences = np.array([
                ex.get("metrics", {}).get("kl_divergence", 0.0) 
                for ex in dataset
            ])
            
            # Handle token-level disagreements safely
            token_patterns = defaultdict(int)
            for ex in high_div_examples:
                disagreements = ex.get("metrics", {}).get("disagreement_positions", {})
                for pos, count in disagreements.items():
                    token_patterns[pos] += count

            # Calculate semantic differences with proper error handling
            semantic_diffs = []
            for ex in high_div_examples:
                try:
                    if "model1_probs" in ex.get("metrics", {}) and "model2_probs" in ex.get("metrics", {}):
                        m1_probs = np.array(ex["metrics"]["model1_probs"])
                        m2_probs = np.array(ex["metrics"]["model2_probs"])
                        if m1_probs.size > 0 and m2_probs.size > 0:
                            semantic_diffs.append(np.mean(np.abs(m1_probs - m2_probs)))
                except (ValueError, TypeError):
                    continue

            semantic_diffs = np.array(semantic_diffs)

            return {
                "num_examples": len(dataset),
                "mean_divergence": float(np.mean(divergences)) if divergences.size > 0 else 0.0,
                "high_divergence_examples": len(high_div_examples),
                "token_patterns": dict(sorted(
                    token_patterns.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]),
                "semantic_patterns": {
                    "mean_semantic_diff": float(np.mean(semantic_diffs)) if semantic_diffs.size > 0 else 0.0,
                    "std_semantic_diff": float(np.std(semantic_diffs)) if semantic_diffs.size > 0 else 0.0,
                    "top_semantic_clusters": []  # Add clustering if needed
                }
            }
        except Exception as e:
            logger.warning(f"Error in analyze_divergence_patterns: {e}")
            return {
                "error": str(e),
                "num_examples": len(dataset) if dataset else 0,
                "mean_divergence": 0.0,
                "high_divergence_examples": 0,
                "token_patterns": {},
                "semantic_patterns": {
                    "mean_semantic_diff": 0.0,
                    "std_semantic_diff": 0.0,
                    "top_semantic_clusters": []
                }
            }
    
    def cluster_divergent_behaviors(
        self,
        dataset: List[Dict],
        n_clusters: int = 5
    ) -> Dict:
        """Cluster examples by divergence patterns."""
        # Extract features for clustering
        features = []
        for ex in dataset:
            feature_vec = [
                ex["metrics"]["kl_divergence"],
                ex["metrics"]["token_disagreement_rate"],
                ex["metrics"]["vocab_jaccard_similarity"]
            ]
            features.append(feature_vec)
            
        features = np.array(features)
        
        # Normalize features
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Analyze clusters
        cluster_stats = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            cluster_stats[int(cluster_id)].append(dataset[i])
            
        cluster_summaries = {}
        for cluster_id, examples in cluster_stats.items():
            cluster_summaries[cluster_id] = {
                "size": len(examples),
                "mean_divergence": np.mean([
                    ex["metrics"]["kl_divergence"] for ex in examples
                ]),
                "example_prompts": [ex["prompt"] for ex in examples[:3]]
            }
            
        return cluster_summaries
    
    def visualize_divergences(self, dataset: List[Dict], method: str = "tsne"):
        """Visualize divergence patterns."""
        try:
            if not dataset:
                logger.warning("Empty dataset provided for visualization")
                return

            # Extract features safely
            features = []
            metrics = []
            texts = []

            for ex in dataset:
                try:
                    metrics_dict = ex.get("metrics", {})
                    feature_vector = [
                        metrics_dict.get("kl_divergence", 0.0),
                        metrics_dict.get("token_disagreement_rate", 0.0),
                        # Add more features as needed
                    ]
                    
                    if all(v is not None for v in feature_vector):
                        features.append(feature_vector)
                        metrics.append(metrics_dict.get("kl_divergence", 0.0))
                        texts.append(ex.get("input_text", "")[:50] + "...")
                except (KeyError, TypeError):
                    continue

            if not features:
                logger.warning("No valid features found for visualization")
                return

            features = np.array(features)
            
            # Only normalize if we have data
            if features.size > 0:
                # Avoid division by zero in normalization
                std = features.std(axis=0)
                std[std == 0] = 1
                features = (features - features.mean(axis=0)) / std

                # Create visualization
                if method == "tsne" and len(features) > 1:
                    reducer = TSNE(n_components=2, random_state=42)
                    embedding = reducer.fit_transform(features)

                    plt.figure(figsize=(12, 8))
                    scatter = plt.scatter(
                        embedding[:, 0],
                        embedding[:, 1],
                        c=metrics,
                        cmap='viridis',
                        alpha=0.6
                    )
                    plt.colorbar(scatter, label='KL Divergence')
                    plt.title("Divergence Pattern Visualization")
                    plt.savefig(str(self.plots_dir / "divergence_patterns.png"))
                    plt.close()

        except Exception as e:
            logger.warning(f"Could not create divergence visualization: {e}")
    
    def generate_report(
        self,
        dataset: List[Dict],
        output_file: Optional[str] = None
    ) -> Dict:
        """Generate comprehensive analysis report with optimized data handling."""
        try:
            # Create report with minimal data copying
            report = {
                "divergence_patterns": self.analyze_divergence_patterns(dataset),
                "training_progress": self.analyze_training_progress(dataset),
                "timestamp": datetime.now().isoformat(),
                "dataset_size": len(dataset),
                "summary": {
                    "mean_divergence": float(np.mean([
                        ex.get("metrics", {}).get("kl_divergence", 0.0) 
                        for ex in dataset
                    ])),
                    "num_examples": len(dataset)
                }
            }
            
            if output_file:
                # Use optimized serialization
                from ..utils.serialization import serialize_for_json
                serializable_report = serialize_for_json(report, max_prob_length=100)
                
                with open(str(self.output_dir / output_file), 'w') as f:
                    json.dump(serializable_report, f)
            
            return report
            
        except Exception as e:
            logger.error(f"Error in generate_report: {str(e)}")
            raise
    
    def visualize_training_convergence(self, dataset: List[Dict]):
        """Visualize training convergence and early stopping."""
        plt.figure(figsize=(12, 6))
        
        for ex in dataset[:5]:  # Plot first 5 examples
            history = ex["training_history"]
            epochs = [h["epoch"] for h in history]
            divergences = [h["kl_divergence"] for h in history]
            plt.plot(epochs, divergences, alpha=0.7, label=f"Run {ex.get('id', 'unknown')}")
            
        plt.xlabel("Epoch")
        plt.ylabel("KL Divergence")
        plt.title("Training Convergence Analysis")
        plt.legend()
        # Convert path to string for saving
        plt.savefig(str(self.plots_dir / "training_convergence.png"))
        plt.close()
