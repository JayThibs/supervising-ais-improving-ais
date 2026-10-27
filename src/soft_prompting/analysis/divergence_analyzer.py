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
        output_file: Optional[str] = None,
        intervention_type: Optional[str] = None
    ) -> Dict:
        """Generate comprehensive analysis report."""
        try:
            # Create report with more robust error handling
            report = {
                "divergence_patterns": self.analyze_divergence_patterns(dataset),
                "training_progress": self.analyze_training_progress(dataset),
                "timestamp": datetime.now().isoformat(),
                "dataset_size": len(dataset),
                "summary": {
                    "mean_divergence": np.mean([ex.get("metrics", {}).get("kl_divergence", 0.0) for ex in dataset]),
                    "num_examples": len(dataset)
                }
            }
            
            # Create visualizations
            try:
                self.visualize_divergences(dataset)
            except Exception as e:
                print(f"Warning: Could not create divergence visualization: {e}")
                
            try:
                if dataset and "training_history" in dataset[0]:
                    self.visualize_training_convergence(dataset)
            except Exception as e:
                print(f"Warning: Could not create training convergence visualization: {e}")
                
            if output_file:
                # Ensure the output path is handled correctly
                output_path = self.output_dir / output_file
                
                try:
                    # Convert Path objects to strings before serialization
                    def convert_paths(obj):
                        if isinstance(obj, Path):
                            return str(obj)
                        elif isinstance(obj, dict):
                            return {k: convert_paths(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_paths(i) for i in obj]
                        return obj
                    
                    serializable_report = convert_paths(report)
                    
                    # Convert output_path to string for file operations
                    with open(str(output_path), 'w') as f:
                        json.dump(serializable_report, f, indent=2)
                        
                except Exception as e:
                    print(f"Error saving report: {str(e)}")
                    print(f"Report structure: {report.keys()}")
                    raise
                
            return report
            
        except Exception as e:
            print(f"Error in generate_report: {str(e)}")
            print(f"Input dataset type: {type(dataset)}")
            if dataset:
                print(f"First example structure: {dataset[0].keys() if isinstance(dataset[0], dict) else 'not a dict'}")
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
