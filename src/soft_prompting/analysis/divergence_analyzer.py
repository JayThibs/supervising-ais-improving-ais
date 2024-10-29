from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from collections import defaultdict

from ..metrics.divergence_metrics import DivergenceMetrics

logger = logging.getLogger(__name__)

class DivergenceAnalyzer:
    """Analyze and visualize behavioral divergences between models."""
    
    def __init__(
        self,
        metrics: DivergenceMetrics,
        output_dir: Path
    ):
        self.metrics = metrics
        self.output_dir = output_dir
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_training_progress(self, training_history: List[Dict]) -> Dict:
        """Analyze training convergence and early stopping effectiveness."""
        epochs = [h["epoch"] for h in training_history]
        losses = [h["loss"] for h in training_history]
        divergences = [h["kl_divergence"] for h in training_history]
        
        return {
            "training_length": len(epochs),
            "final_loss": losses[-1],
            "best_divergence": max(divergences),
            "convergence_epoch": epochs[divergences.index(max(divergences))],
            "early_stopping_effective": len(epochs) < max(epochs)
        }
    
    def analyze_divergence_patterns(
        self,
        dataset: List[Dict],
        min_divergence: float = 0.1,
        intervention_type: Optional[str] = None
    ) -> Dict:
        """
        Analyze patterns in high-divergence examples.
        Optimized for detecting effects of interventions like unlearning or sandbagging.
        
        Args:
            dataset: List of examples with metrics
            min_divergence: Minimum divergence threshold
            intervention_type: Type of intervention being analyzed (e.g. "unlearning", "sandbagging")
        """
        # Filter for high-divergence examples
        high_div_examples = [
            ex for ex in dataset 
            if ex["metrics"]["kl_divergence"] > min_divergence
        ]
        
        # Collect token-level disagreements
        token_patterns = defaultdict(int)
        for ex in high_div_examples:
            for pos, count in ex["metrics"]["disagreement_positions"].items():
                token_patterns[pos] += count
                
        # Sort by frequency
        common_patterns = dict(
            sorted(
                token_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
        )
        
        # Analyze semantic patterns
        semantic_diffs = np.array([
            ex["metrics"]["vocab_jaccard_similarity"]
            for ex in high_div_examples
        ])
        
        # Add analysis of early stopping impact
        if "training_history" in dataset[0]:
            training_analysis = self.analyze_training_progress(
                [ex["training_history"] for ex in dataset]
            )
            return {
                **{
                    "num_high_divergence": len(high_div_examples),
                    "common_disagreements": common_patterns,
                    "mean_semantic_diff": float(semantic_diffs.mean()),
                    "std_semantic_diff": float(semantic_diffs.std())
                },
                "training_analysis": training_analysis
            }
            
        return {
            "num_high_divergence": len(high_div_examples),
            "common_disagreements": common_patterns,
            "mean_semantic_diff": float(semantic_diffs.mean()),
            "std_semantic_diff": float(semantic_diffs.std())
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
    
    def visualize_divergences(
        self,
        dataset: List[Dict],
        method: str = "tsne"
    ):
        """Create visualization of divergence patterns."""
        features = []
        metrics = []
        texts = []
        
        for ex in dataset:
            features.append([
                ex["metrics"]["kl_divergence"],
                ex["metrics"]["token_disagreement_rate"],
                ex["metrics"]["vocab_jaccard_similarity"]
            ])
            metrics.append(ex["metrics"]["kl_divergence"])
            texts.append(ex["prompt"][:50] + "...")
            
        features = np.array(features)
        
        # Normalize features
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        
        # Dimensionality reduction
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
            embedding = reducer.fit_transform(features)
        
        # Create plot
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
        
        # Save plot
        plt.savefig(self.plots_dir / "divergence_patterns.png")
        plt.close()
        
    def generate_report(
        self,
        dataset: List[Dict],
        output_file: Optional[str] = None
    ) -> Dict:
        """Generate comprehensive analysis report."""
        report = {}
        
        # Overall statistics
        report["overall_stats"] = {
            "num_examples": len(dataset),
            "mean_divergence": np.mean([
                ex["metrics"]["kl_divergence"] for ex in dataset
            ]),
            "max_divergence": max(
                ex["metrics"]["kl_divergence"] for ex in dataset
            ),
            "early_stopping_stats": {
                "total_runs": len(dataset),
                "early_stopped_runs": sum(
                    1 for ex in dataset 
                    if ex.get("early_stopped", False)
                )
            }
        }
        
        # Pattern analysis
        report["divergence_patterns"] = self.analyze_divergence_patterns(dataset)
        
        # Clustering analysis
        report["behavior_clusters"] = self.cluster_divergent_behaviors(dataset)
        
        # Create visualizations
        self.visualize_divergences(dataset)
        
        # Add training convergence visualization
        if "training_history" in dataset[0]:
            self.visualize_training_convergence(dataset)
            
        if output_file:
            import json
            with open(self.output_dir / output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
        
    def visualize_training_convergence(self, dataset: List[Dict]):
        """Visualize training convergence and early stopping."""
        plt.figure(figsize=(12, 6))
        
        for ex in dataset[:5]:  # Plot first 5 examples
            history = ex["training_history"]
            epochs = [h["epoch"] for h in history]
            divergences = [h["kl_divergence"] for h in history]
            plt.plot(epochs, divergences, alpha=0.7, label=f"Run {ex['id']}")
            
        plt.xlabel("Epoch")
        plt.ylabel("KL Divergence")
        plt.title("Training Convergence Analysis")
        plt.legend()
        plt.savefig(self.plots_dir / "training_convergence.png")
        plt.close()
