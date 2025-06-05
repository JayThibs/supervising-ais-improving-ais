"""
Differential Circuit Analyzer

This module compares computational circuits between two models (typically base and intervention)
to identify meaningful differences that might correspond to behavioral changes.

Key Concepts:
    - Circuit Alignment: Matching corresponding features between models
    - Difference Metrics: Quantifying how much circuits have changed
    - Pattern Detection: Identifying systematic changes across prompts
    - Importance Weighting: Focusing on differences that matter

Technical Background:
    When a model is fine-tuned or otherwise modified, the changes manifest as:
    1. New feature activations (concepts the model now recognizes)
    2. Suppressed features (concepts the model now ignores)
    3. Rerouted information flow (different computational paths)
    4. Strengthened/weakened connections (changed importance weights)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.metrics.pairwise import cosine_similarity

from circuit_tracer import Graph
from .circuit_tracer_wrapper import CircuitAwareModel


@dataclass
class FeatureDifference:
    """
    Represents a difference in feature activation between models.
    
    Attributes:
        layer: Which layer the feature belongs to
        position: Token position where feature activated
        feature_idx: Index of the feature in the transcoder
        base_activation: Activation strength in base model (0 if not active)
        intervention_activation: Activation strength in intervention model
        interpretation: Human-readable interpretation of what this feature represents
        contexts: Example contexts where this feature activates
    """
    layer: int
    position: int
    feature_idx: int
    base_activation: float
    intervention_activation: float
    interpretation: Optional[str] = None
    contexts: List[str] = field(default_factory=list)
    
    @property
    def activation_delta(self) -> float:
        """Absolute change in activation strength."""
        return abs(self.intervention_activation - self.base_activation)
    
    @property
    def relative_change(self) -> float:
        """Relative change in activation (handles zero base activation)."""
        if self.base_activation == 0:
            return float('inf') if self.intervention_activation > 0 else 0
        return abs(self.intervention_activation - self.base_activation) / self.base_activation
    
    def is_new_feature(self) -> bool:
        """Check if this feature is newly active in intervention model."""
        return self.base_activation == 0 and self.intervention_activation > 0
    
    def is_suppressed_feature(self) -> bool:
        """Check if this feature was suppressed by the intervention."""
        return self.base_activation > 0 and self.intervention_activation == 0


@dataclass
class ConnectionDifference:
    """
    Represents a difference in connections between nodes.
    
    Attributes:
        source: Source node identifier
        target: Target node identifier  
        base_weight: Connection weight in base model
        intervention_weight: Connection weight in intervention model
        source_type: Type of source node (feature, error, embedding, logit)
        target_type: Type of target node
    """
    source: str
    target: str
    base_weight: float
    intervention_weight: float
    source_type: str
    target_type: str
    
    @property
    def weight_change(self) -> float:
        """Absolute change in connection weight."""
        return abs(self.intervention_weight - self.base_weight)
        
    def is_new_connection(self) -> bool:
        """Check if this connection only exists in intervention model."""
        return abs(self.base_weight) < 1e-6 and abs(self.intervention_weight) >= 1e-6
        
    def is_removed_connection(self) -> bool:
        """Check if this connection was removed by intervention."""
        return abs(self.base_weight) >= 1e-6 and abs(self.intervention_weight) < 1e-6


@dataclass 
class CircuitDifference:
    """
    Complete representation of circuit differences between models for a given input.
    
    This is the main output of circuit comparison, containing all types of
    differences found between the models.
    """
    prompt: str
    feature_differences: List[FeatureDifference]
    connection_differences: List[ConnectionDifference]
    statistics: Dict[str, float]
    base_graph: Graph
    intervention_graph: Graph
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the circuit differences."""
        return {
            'n_feature_differences': len(self.feature_differences),
            'n_new_features': sum(1 for f in self.feature_differences if f.is_new_feature()),
            'n_suppressed_features': sum(1 for f in self.feature_differences if f.is_suppressed_feature()),
            'n_connection_differences': len(self.connection_differences),
            'n_new_connections': sum(1 for c in self.connection_differences if c.is_new_connection()),
            'n_removed_connections': sum(1 for c in self.connection_differences if c.is_removed_connection()),
            'avg_activation_delta': np.mean([f.activation_delta for f in self.feature_differences]) if self.feature_differences else 0,
            'avg_weight_change': np.mean([c.weight_change for c in self.connection_differences]) if self.connection_differences else 0,
            'prompt_preview': self.prompt[:50] + '...' if len(self.prompt) > 50 else self.prompt
        }


class DifferentialCircuitAnalyzer:
    """
    Analyzes differences in computational circuits between two models.
    
    This class implements sophisticated comparison algorithms to identify
    meaningful differences in how two models process the same input.
    The key insight is that behavioral differences often correspond to
    specific patterns of circuit changes.
    
    Example:
        >>> base_config = CircuitTracingConfig("gpt2", "gpt2-transcoders")
        >>> int_config = CircuitTracingConfig("gpt2-finetuned", "gpt2-transcoders")
        >>> base_model = CircuitAwareModel(base_config)
        >>> finetuned_model = CircuitAwareModel(int_config)
        >>> analyzer = DifferentialCircuitAnalyzer(base_model, finetuned_model)
        >>> 
        >>> differences = analyzer.analyze_prompt("The AI should")
        >>> print(f"Found {len(differences.feature_differences)} feature changes")
    """
    
    def __init__(self, base_model: CircuitAwareModel, intervention_model: CircuitAwareModel):
        """
        Initialize the differential analyzer.
        
        Args:
            base_model: The original/reference model
            intervention_model: The modified model (finetuned, edited, etc.)
            
        Raises:
            ValueError: If models have incompatible architectures
        """
        self.base_model = base_model
        self.intervention_model = intervention_model
        
        # Validate model compatibility
        base_info = base_model.get_model_info()
        int_info = intervention_model.get_model_info()
        
        if base_info['n_layers'] != int_info['n_layers']:
            raise ValueError(f"Models have different layer counts: {base_info['n_layers']} vs {int_info['n_layers']}")
            
        if base_info['d_model'] != int_info['d_model']:
            raise ValueError(f"Models have different hidden dimensions: {base_info['d_model']} vs {int_info['d_model']}")
            
        print(f"Initialized differential analyzer for {base_info['n_layers']}-layer models")
        
    def analyze_prompt(self, prompt: str, detailed: bool = True) -> CircuitDifference:
        """
        Analyze circuit differences for a single prompt.
        
        This is the main entry point for circuit comparison. It extracts
        circuits from both models and performs comprehensive comparison.
        
        Args:
            prompt: Input text to analyze
            detailed: Whether to include detailed feature interpretations
            
        Returns:
            CircuitDifference object containing all found differences
            
        Algorithm:
            1. Extract circuits (attribution graphs) from both models
            2. Align features between models
            3. Compute activation differences
            4. Analyze connection changes
            5. Compute summary statistics
        """
        print(f"Analyzing circuit differences for: '{prompt[:50]}...'")
        
        # Step 1: Extract circuits from both models (returns Graph objects)
        base_graph = self.base_model.extract_circuit(prompt, return_graph_object=True)
        int_graph = self.intervention_model.extract_circuit(prompt, return_graph_object=True)
        
        # Step 2: Compare feature activations
        feature_diffs = self._compare_features(base_graph, int_graph)
        
        # Step 3: Compare connections
        connection_diffs = self._compare_connections(base_graph, int_graph)
        
        # Step 4: Compute statistics
        statistics = self._compute_statistics(base_graph, int_graph, feature_diffs, connection_diffs)
        
        return CircuitDifference(
            prompt=prompt,
            feature_differences=feature_diffs,
            connection_differences=connection_diffs,
            statistics=statistics,
            base_graph=base_graph,
            intervention_graph=int_graph
        )
        
    def _compare_features(self, base_graph: Graph, int_graph: Graph) -> List[FeatureDifference]:
        """
        Compare feature activations between models using Graph objects.
        
        This identifies which features changed their activation patterns,
        which is often indicative of behavioral changes.
        
        Args:
            base_graph: Graph object from base model
            int_graph: Graph object from intervention model
            
        Returns:
            List of FeatureDifference objects
            
        Technical Note:
            Feature alignment assumes transcoders are shared between models,
            so features with the same index represent the same concept.
        """
        differences = []
        
        # Extract feature info from graphs
        # active_features is tensor of shape (n_active_features, 3): (layer, pos, feature_idx)
        base_features = {}  # (layer, pos, feature_idx) -> activation_value
        int_features = {}
        
        # Map features from base model
        for i, (layer, pos, feat_idx) in enumerate(base_graph.active_features.tolist()):
            key = (layer, pos, feat_idx)
            # Use activation values if available, otherwise use 1.0
            if hasattr(base_graph, 'activation_values') and base_graph.activation_values is not None:
                base_features[key] = base_graph.activation_values[i].item()
            else:
                base_features[key] = 1.0
                
        # Map features from intervention model  
        for i, (layer, pos, feat_idx) in enumerate(int_graph.active_features.tolist()):
            key = (layer, pos, feat_idx)
            if hasattr(int_graph, 'activation_values') and int_graph.activation_values is not None:
                int_features[key] = int_graph.activation_values[i].item()
            else:
                int_features[key] = 1.0
        
        # Find all features that are active in either model
        all_feature_keys = set(base_features.keys()) | set(int_features.keys())
        
        for layer, pos, feat_idx in sorted(all_feature_keys):
            base_activation = base_features.get((layer, pos, feat_idx), 0.0)
            int_activation = int_features.get((layer, pos, feat_idx), 0.0)
            
            # Only record if there's a meaningful difference
            if abs(base_activation - int_activation) > 0.01:  # Threshold for noise
                diff = FeatureDifference(
                    layer=layer,
                    position=pos,
                    feature_idx=feat_idx,
                    base_activation=base_activation,
                    intervention_activation=int_activation
                )
                differences.append(diff)
                    
        # Sort by importance of change
        differences.sort(key=lambda d: d.activation_delta, reverse=True)
        
        return differences
        
    def _compare_connections(self, base_graph: Graph, int_graph: Graph) -> List[ConnectionDifference]:
        """
        Compare connections (edges) between graphs.
        
        This identifies new, removed, or changed connections, which indicate
        changes in how information flows through the models.
        
        Args:
            base_graph: Graph object from base model
            int_graph: Graph object from intervention model
            
        Returns:
            List of ConnectionDifference objects
        """
        differences = []
        
        # Get adjacency matrices
        base_adj = base_graph.adjacency_matrix.cpu().numpy()
        int_adj = int_graph.adjacency_matrix.cpu().numpy()
        
        # Create node labels for both graphs
        base_labels = self._create_node_labels(base_graph)
        int_labels = self._create_node_labels(int_graph)
        
        # We need to align nodes between graphs
        # This is complex because active features might differ
        # For now, we'll compare connections between nodes that exist in both graphs
        
        # Find common node positions (simplified approach)
        # In practice, we'd need more sophisticated alignment
        
        # Compare adjacency matrices where both exist
        min_rows = min(base_adj.shape[0], int_adj.shape[0])
        min_cols = min(base_adj.shape[1], int_adj.shape[1])
        
        for i in range(min_rows):
            for j in range(min_cols):
                base_weight = base_adj[i, j]
                int_weight = int_adj[i, j]
                
                # Check if there's a meaningful difference
                if abs(base_weight - int_weight) > 0.01:
                    # Get node labels
                    source_label = base_labels.get(j, f"node_{j}")
                    target_label = base_labels.get(i, f"node_{i}")
                    
                    # Determine node types
                    source_type = self._get_node_type(j, base_graph)
                    target_type = self._get_node_type(i, base_graph)
                    
                    diff = ConnectionDifference(
                        source=source_label,
                        target=target_label,
                        base_weight=base_weight,
                        intervention_weight=int_weight,
                        source_type=source_type,
                        target_type=target_type
                    )
                    differences.append(diff)
                    
        # Sort by magnitude of change
        differences.sort(key=lambda d: d.weight_change, reverse=True)
        
        # Return top differences to avoid overwhelming output
        return differences[:100]  # Top 100 most changed connections
        
    def _create_node_labels(self, graph: Graph) -> Dict[int, str]:
        """
        Create human-readable labels for nodes in the graph.
        
        Args:
            graph: Graph object
            
        Returns:
            Dictionary mapping node index to label
        """
        labels = {}
        node_idx = 0
        
        # Feature nodes (first in adjacency matrix)
        for layer, pos, feat_idx in graph.active_features.tolist():
            labels[node_idx] = f"F_L{layer}_P{pos}_{feat_idx}"
            node_idx += 1
            
        # Error nodes
        n_layers = self.base_model.n_layers
        n_pos = graph.n_pos
        
        for layer in range(n_layers):
            for pos in range(n_pos):
                labels[node_idx] = f"E_L{layer}_P{pos}"
                node_idx += 1
                
        # Token embedding nodes
        for pos in range(n_pos):
            labels[node_idx] = f"Tok_{pos}"
            node_idx += 1
            
        # Logit nodes
        for i in range(len(graph.logit_tokens)):
            labels[node_idx] = f"Logit_{i}"
            node_idx += 1
            
        return labels
        
    def _get_node_type(self, node_idx: int, graph: Graph) -> str:
        """
        Determine the type of a node based on its index.
        
        Args:
            node_idx: Index of the node in adjacency matrix
            graph: Graph object
            
        Returns:
            Node type: "feature", "error", "embedding", or "logit"
        """
        n_features = len(graph.active_features)
        n_layers = self.base_model.n_layers
        n_pos = graph.n_pos
        n_errors = n_layers * n_pos
        n_embeds = n_pos
        
        if node_idx < n_features:
            return "feature"
        elif node_idx < n_features + n_errors:
            return "error"
        elif node_idx < n_features + n_errors + n_embeds:
            return "embedding"
        else:
            return "logit"
            
    def _compute_statistics(self,
                          base_graph: Graph,
                          int_graph: Graph,
                          feature_diffs: List[FeatureDifference],
                          connection_diffs: List[ConnectionDifference]) -> Dict[str, float]:
        """
        Compute summary statistics about the circuit differences.
        
        These statistics help quantify the overall magnitude of change
        and can be used to prioritize which prompts to investigate further.
        
        Args:
            base_graph: Graph object from base model
            int_graph: Graph object from intervention model
            feature_diffs: List of feature differences
            connection_diffs: List of connection differences
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Graph structure statistics
        stats['base_n_features'] = len(base_graph.active_features)
        stats['int_n_features'] = len(int_graph.active_features)
        stats['base_n_nodes'] = base_graph.adjacency_matrix.shape[0]
        stats['int_n_nodes'] = int_graph.adjacency_matrix.shape[0]
        
        # Feature overlap statistics
        base_features = set(tuple(f) for f in base_graph.active_features.tolist())
        int_features = set(tuple(f) for f in int_graph.active_features.tolist())
        stats['feature_jaccard'] = len(base_features & int_features) / len(base_features | int_features) if base_features | int_features else 1.0
        
        # Feature change statistics
        if feature_diffs:
            stats['mean_activation_delta'] = np.mean([f.activation_delta for f in feature_diffs])
            stats['max_activation_delta'] = max(f.activation_delta for f in feature_diffs)
            stats['n_new_features'] = sum(1 for f in feature_diffs if f.is_new_feature())
            stats['n_suppressed_features'] = sum(1 for f in feature_diffs if f.is_suppressed_feature())
        else:
            stats['mean_activation_delta'] = 0
            stats['max_activation_delta'] = 0
            stats['n_new_features'] = 0
            stats['n_suppressed_features'] = 0
            
        # Connection change statistics
        if connection_diffs:
            stats['mean_weight_change'] = np.mean([c.weight_change for c in connection_diffs])
            stats['max_weight_change'] = max(c.weight_change for c in connection_diffs)
            stats['n_new_connections'] = sum(1 for c in connection_diffs if c.is_new_connection())
            stats['n_removed_connections'] = sum(1 for c in connection_diffs if c.is_removed_connection())
        else:
            stats['mean_weight_change'] = 0
            stats['max_weight_change'] = 0
            stats['n_new_connections'] = 0
            stats['n_removed_connections'] = 0
            
        # Overall change magnitude
        stats['total_change_score'] = (
            stats['mean_activation_delta'] * 0.4 +
            stats['mean_weight_change'] * 0.4 +
            (1 - stats['feature_jaccard']) * 0.2
        )
        
        return stats
        
    def analyze_prompt_batch(self, prompts: List[str], n_workers: int = 4) -> List[CircuitDifference]:
        """
        Analyze multiple prompts in parallel for efficiency.
        
        Args:
            prompts: List of prompts to analyze
            n_workers: Number of parallel workers
            
        Returns:
            List of CircuitDifference objects
        """
        # TODO: Implement parallel processing using multiprocessing
        # For now, process sequentially
        differences = []
        
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}")
            try:
                diff = self.analyze_prompt(prompt)
                differences.append(diff)
            except Exception as e:
                print(f"Error processing prompt '{prompt[:30]}...': {e}")
                continue
                
        return differences
        
    def find_systematic_changes(self, differences: List[CircuitDifference]) -> Dict[str, Any]:
        """
        Identify systematic patterns across multiple circuit differences.
        
        This is crucial for distinguishing random variations from meaningful
        intervention effects. Systematic changes that appear across multiple
        prompts are more likely to represent true behavioral modifications.
        
        Args:
            differences: List of circuit differences from multiple prompts
            
        Returns:
            Dictionary containing:
                - consistent_features: Features that consistently change
                - feature_clusters: Groups of features that change together
                - dominant_patterns: Most common types of changes
        """
        # Track feature changes across prompts
        feature_change_counts = defaultdict(int)
        feature_change_directions = defaultdict(list)
        feature_change_magnitudes = defaultdict(list)
        
        for diff in differences:
            for feat_diff in diff.feature_differences:
                key = (feat_diff.layer, feat_diff.feature_idx)
                feature_change_counts[key] += 1
                
                # Track whether activation increased or decreased
                if feat_diff.base_activation < feat_diff.intervention_activation:
                    feature_change_directions[key].append(1)
                else:
                    feature_change_directions[key].append(-1)
                    
                # Track magnitude of changes
                feature_change_magnitudes[key].append(feat_diff.activation_delta)
                    
        # Find consistently changing features
        total_prompts = len(differences)
        consistent_features = {}
        
        for key, count in feature_change_counts.items():
            if count >= 0.3 * total_prompts:  # Changes in at least 30% of prompts
                avg_direction = np.mean(feature_change_directions[key])
                avg_magnitude = np.mean(feature_change_magnitudes[key])
                
                consistent_features[key] = {
                    'layer': key[0],
                    'feature_idx': key[1],
                    'count': count,
                    'frequency': count / total_prompts,
                    'avg_direction': avg_direction,
                    'avg_magnitude': avg_magnitude,
                    'consistency': abs(avg_direction)  # How consistent the direction is
                }
        
        # Sort by combination of frequency and consistency
        consistent_features = dict(sorted(
            consistent_features.items(), 
            key=lambda x: x[1]['frequency'] * x[1]['consistency'], 
            reverse=True
        ))
        
        # Find feature clusters (features that tend to change together)
        # This is a simplified co-occurrence analysis
        feature_cooccurrence = defaultdict(int)
        
        for diff in differences:
            active_features = [(f.layer, f.feature_idx) for f in diff.feature_differences]
            for i, feat1 in enumerate(active_features):
                for feat2 in active_features[i+1:]:
                    if feat1 != feat2:
                        pair = tuple(sorted([feat1, feat2]))
                        feature_cooccurrence[pair] += 1
                        
        # Find frequently co-occurring features
        feature_clusters = []
        min_cooccurrence = 0.2 * total_prompts
        
        for (feat1, feat2), count in feature_cooccurrence.items():
            if count >= min_cooccurrence:
                feature_clusters.append({
                    'features': [feat1, feat2],
                    'cooccurrence_count': count,
                    'cooccurrence_rate': count / total_prompts
                })
                
        # Sort clusters by co-occurrence rate
        feature_clusters.sort(key=lambda x: x['cooccurrence_rate'], reverse=True)
        
        return {
            'consistent_features': consistent_features,
            'n_systematic_features': len(consistent_features),
            'total_unique_changes': len(feature_change_counts),
            'feature_clusters': feature_clusters[:10],  # Top 10 clusters
            'summary_stats': {
                'avg_features_changed_per_prompt': np.mean([len(d.feature_differences) for d in differences]),
                'avg_connections_changed_per_prompt': np.mean([len(d.connection_differences) for d in differences])
            }
        }