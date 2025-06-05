"""
Pattern Recognizer Module

This module identifies systematic patterns in circuit changes that are likely
to correspond to meaningful behavioral differences. It groups similar changes
and extracts high-level patterns that can be converted into hypotheses.

Key Pattern Types:
    1. Feature Suppression: Previously active features become inactive
    2. Feature Emergence: New features become active
    3. Path Rerouting: Information takes different routes through the model
    4. Strength Modulation: Connections become stronger or weaker
    5. Co-occurrence: Multiple features change together systematically

Algorithm:
    1. Cluster similar circuit changes across prompts
    2. Identify statistically significant patterns
    3. Extract pattern characteristics
    4. Map patterns to behavioral predictions
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import torch

from ..circuit_analysis import CircuitDifference, FeatureDifference, ConnectionDifference


@dataclass
class CircuitPattern:
    """
    Represents a systematic pattern of circuit changes.
    
    Attributes:
        pattern_id: Unique identifier for this pattern
        pattern_type: Type of pattern (suppression, emergence, rerouting, etc.)
        affected_features: Features involved in this pattern
        affected_connections: Connections involved in this pattern
        occurrence_count: How many times this pattern appears
        prompts: Example prompts where this pattern occurs
        confidence: Statistical confidence in this pattern
        description: Human-readable description
    """
    pattern_id: str
    pattern_type: str
    affected_features: List[Tuple[int, int]]  # (layer, feature_idx)
    affected_connections: List[Tuple[str, str]]  # (source, target)
    occurrence_count: int
    prompts: List[str] = field(default_factory=list)
    confidence: float = 0.0
    description: str = ""
    
    def get_summary(self) -> Dict[str, Any]:
        """Get pattern summary."""
        return {
            'pattern_id': self.pattern_id,
            'type': self.pattern_type,
            'n_features': len(self.affected_features),
            'n_connections': len(self.affected_connections),
            'occurrences': self.occurrence_count,
            'confidence': self.confidence,
            'description': self.description
        }


class PatternRecognizer:
    """
    Identifies systematic patterns in circuit changes across multiple prompts.
    
    This class implements several pattern detection algorithms:
    1. **Feature clustering**: Groups features that change together
    2. **Change direction analysis**: Identifies consistent increase/decrease patterns
    3. **Co-occurrence mining**: Finds features that always change together
    4. **Temporal analysis**: Detects patterns in how changes propagate through layers
    
    The key insight is that random changes are unlikely to form patterns,
    while intervention effects create systematic, repeated patterns.
    
    Example:
        >>> recognizer = PatternRecognizer(min_pattern_support=0.3)
        >>> 
        >>> # Analyze circuit differences from multiple prompts
        >>> differences = [analyzer.analyze_prompt(p) for p in prompts]
        >>> patterns = recognizer.find_patterns(differences)
        >>> 
        >>> for pattern in patterns:
        >>>     print(f"{pattern.pattern_type}: {pattern.description}")
    """
    
    def __init__(self,
                 min_pattern_support: float = 0.3,
                 clustering_method: str = "dbscan",
                 feature_similarity_threshold: float = 0.8):
        """
        Initialize the pattern recognizer.
        
        Args:
            min_pattern_support: Minimum fraction of prompts where pattern must appear
            clustering_method: Method for clustering similar changes ("dbscan" or "hierarchical")
            feature_similarity_threshold: Threshold for considering features similar
        """
        self.min_pattern_support = min_pattern_support
        self.clustering_method = clustering_method
        self.feature_similarity_threshold = feature_similarity_threshold
        
        # Pattern templates for common behavioral changes
        self.pattern_templates = self._initialize_pattern_templates()
        
    def _initialize_pattern_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize templates for common pattern types.
        
        These templates help map circuit patterns to behavioral predictions.
        """
        return {
            'safety_enhancement': {
                'indicators': ['suppression of harmful features', 'emergence of refusal features'],
                'behavioral_prediction': 'Model will be more cautious and refuse harmful requests'
            },
            'capability_reduction': {
                'indicators': ['suppression across multiple layers', 'reduced connectivity'],
                'behavioral_prediction': 'Model will show degraded performance on specific tasks'
            },
            'style_shift': {
                'indicators': ['changed features in middle layers', 'rerouted paths'],
                'behavioral_prediction': 'Model will exhibit different writing or reasoning style'
            },
            'knowledge_update': {
                'indicators': ['new features in early layers', 'changed factual associations'],
                'behavioral_prediction': 'Model will have updated factual knowledge'
            },
            'behavioral_constraint': {
                'indicators': ['suppressed features in output layers', 'new inhibitory connections'],
                'behavioral_prediction': 'Model will avoid certain types of outputs'
            }
        }
        
    def find_patterns(self, 
                     differences: List[CircuitDifference],
                     verbose: bool = True) -> List[CircuitPattern]:
        """
        Find systematic patterns across multiple circuit differences.
        
        Args:
            differences: List of circuit differences from different prompts
            verbose: Whether to print progress information
            
        Returns:
            List of identified patterns sorted by confidence
            
        Algorithm:
            1. Extract feature changes from all differences
            2. Cluster similar changes
            3. Identify statistically significant clusters
            4. Map clusters to pattern types
            5. Generate pattern descriptions
        """
        if verbose:
            print(f"Analyzing {len(differences)} circuit differences for patterns...")
            
        patterns = []
        
        # Extract all feature changes
        all_feature_changes = self._extract_all_feature_changes(differences)
        
        # Find feature co-occurrence patterns
        cooccurrence_patterns = self._find_cooccurrence_patterns(all_feature_changes)
        patterns.extend(cooccurrence_patterns)
        
        # Find directional patterns (consistent increase/decrease)
        directional_patterns = self._find_directional_patterns(differences)
        patterns.extend(directional_patterns)
        
        # Find layer-wise patterns
        layer_patterns = self._find_layer_patterns(differences)
        patterns.extend(layer_patterns)
        
        # Find connection patterns
        connection_patterns = self._find_connection_patterns(differences)
        patterns.extend(connection_patterns)
        
        # Deduplicate and sort by confidence
        patterns = self._deduplicate_patterns(patterns)
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        if verbose:
            print(f"Found {len(patterns)} significant patterns")
            
        return patterns
        
    def _extract_all_feature_changes(self, 
                                   differences: List[CircuitDifference]) -> Dict[str, List[FeatureDifference]]:
        """
        Extract all feature changes organized by prompt.
        
        Args:
            differences: List of circuit differences
            
        Returns:
            Dictionary mapping prompt to list of feature changes
        """
        feature_changes = {}
        
        for diff in differences:
            feature_changes[diff.prompt] = diff.feature_differences
            
        return feature_changes
        
    def _find_cooccurrence_patterns(self,
                                  feature_changes: Dict[str, List[FeatureDifference]]) -> List[CircuitPattern]:
        """
        Find features that consistently change together.
        
        Args:
            feature_changes: Feature changes by prompt
            
        Returns:
            List of co-occurrence patterns
        """
        patterns = []
        
        # Build co-occurrence matrix
        cooccurrence_counts = defaultdict(int)
        feature_prompts = defaultdict(set)
        
        for prompt, changes in feature_changes.items():
            features = [(c.layer, c.feature_idx) for c in changes]
            
            # Count pairwise co-occurrences
            for i, feat1 in enumerate(features):
                feature_prompts[feat1].add(prompt)
                
                for feat2 in features[i+1:]:
                    pair = tuple(sorted([feat1, feat2]))
                    cooccurrence_counts[pair] += 1
                    
        # Find significant co-occurrences
        total_prompts = len(feature_changes)
        min_support = int(self.min_pattern_support * total_prompts)
        
        for (feat1, feat2), count in cooccurrence_counts.items():
            if count >= min_support:
                # Check if co-occurrence is more than chance
                feat1_support = len(feature_prompts[feat1])
                feat2_support = len(feature_prompts[feat2])
                expected_cooccurrence = (feat1_support * feat2_support) / total_prompts
                
                if count > expected_cooccurrence * 1.5:  # 50% more than expected
                    pattern = CircuitPattern(
                        pattern_id=f"cooc_{feat1[0]}_{feat1[1]}_{feat2[0]}_{feat2[1]}",
                        pattern_type="feature_cooccurrence",
                        affected_features=[feat1, feat2],
                        affected_connections=[],
                        occurrence_count=count,
                        confidence=count / min(feat1_support, feat2_support),
                        description=f"Features L{feat1[0]}_F{feat1[1]} and L{feat2[0]}_F{feat2[1]} change together"
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    def _find_directional_patterns(self,
                                 differences: List[CircuitDifference]) -> List[CircuitPattern]:
        """
        Find features that consistently increase or decrease.
        
        Args:
            differences: List of circuit differences
            
        Returns:
            List of directional patterns
        """
        patterns = []
        
        # Track direction of change for each feature
        feature_directions = defaultdict(list)
        
        for diff in differences:
            for feat_diff in diff.feature_differences:
                key = (feat_diff.layer, feat_diff.feature_idx)
                
                # Determine direction (-1, 0, 1)
                if feat_diff.is_new_feature():
                    direction = 1
                elif feat_diff.is_suppressed_feature():
                    direction = -1
                else:
                    direction = np.sign(feat_diff.intervention_activation - 
                                      feat_diff.base_activation)
                    
                feature_directions[key].append((direction, diff.prompt))
                
        # Find features with consistent direction
        min_occurrences = int(self.min_pattern_support * len(differences))
        
        for (layer, feat_idx), directions in feature_directions.items():
            if len(directions) >= min_occurrences:
                # Check consistency of direction
                direction_values = [d[0] for d in directions]
                mean_direction = np.mean(direction_values)
                
                if abs(mean_direction) > 0.8:  # 80% consistency
                    pattern_type = "consistent_increase" if mean_direction > 0 else "consistent_decrease"
                    
                    pattern = CircuitPattern(
                        pattern_id=f"dir_{layer}_{feat_idx}",
                        pattern_type=pattern_type,
                        affected_features=[(layer, feat_idx)],
                        affected_connections=[],
                        occurrence_count=len(directions),
                        prompts=[d[1] for d in directions[:5]],  # Sample prompts
                        confidence=abs(mean_direction),
                        description=f"Feature L{layer}_F{feat_idx} consistently {'increases' if mean_direction > 0 else 'decreases'}"
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    def _find_layer_patterns(self,
                           differences: List[CircuitDifference]) -> List[CircuitPattern]:
        """
        Find patterns that affect entire layers or layer ranges.
        
        Args:
            differences: List of circuit differences
            
        Returns:
            List of layer-wise patterns
        """
        patterns = []
        
        # Count changes by layer
        layer_change_counts = defaultdict(lambda: defaultdict(int))
        
        for diff in differences:
            layers_affected = defaultdict(int)
            
            for feat_diff in diff.feature_differences:
                layers_affected[feat_diff.layer] += 1
                
            for layer, count in layers_affected.items():
                layer_change_counts[layer][diff.prompt] = count
                
        # Find layers with systematic changes
        total_prompts = len(differences)
        
        for layer, prompt_counts in layer_change_counts.items():
            prompts_affected = len(prompt_counts)
            
            if prompts_affected >= self.min_pattern_support * total_prompts:
                avg_changes = np.mean(list(prompt_counts.values()))
                
                if avg_changes > 5:  # Significant number of changes
                    pattern = CircuitPattern(
                        pattern_id=f"layer_{layer}",
                        pattern_type="layer_wide_change",
                        affected_features=[(layer, -1)],  # -1 indicates whole layer
                        affected_connections=[],
                        occurrence_count=prompts_affected,
                        prompts=list(prompt_counts.keys())[:5],
                        confidence=prompts_affected / total_prompts,
                        description=f"Layer {layer} shows systematic changes (avg {avg_changes:.1f} features per prompt)"
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    def _find_connection_patterns(self,
                                differences: List[CircuitDifference]) -> List[CircuitPattern]:
        """
        Find patterns in connection changes.
        
        Args:
            differences: List of circuit differences
            
        Returns:
            List of connection patterns
        """
        patterns = []
        
        # Track connection changes
        connection_changes = defaultdict(lambda: {'new': 0, 'removed': 0, 'changed': 0})
        
        for diff in differences:
            for conn_diff in diff.connection_differences[:50]:  # Limit to top changes
                key = (conn_diff.source_type, conn_diff.target_type)
                
                if conn_diff.is_new_connection():
                    connection_changes[key]['new'] += 1
                elif conn_diff.is_removed_connection():
                    connection_changes[key]['removed'] += 1
                else:
                    connection_changes[key]['changed'] += 1
                    
        # Find significant connection patterns
        total_prompts = len(differences)
        min_occurrences = int(self.min_pattern_support * total_prompts)
        
        for (source_type, target_type), counts in connection_changes.items():
            total_changes = sum(counts.values())
            
            if total_changes >= min_occurrences:
                # Determine dominant change type
                dominant_type = max(counts.items(), key=lambda x: x[1])
                
                pattern = CircuitPattern(
                    pattern_id=f"conn_{source_type}_{target_type}",
                    pattern_type=f"connection_{dominant_type[0]}",
                    affected_features=[],
                    affected_connections=[(source_type, target_type)],
                    occurrence_count=total_changes,
                    confidence=dominant_type[1] / total_changes,
                    description=f"{dominant_type[0].capitalize()} connections from {source_type} to {target_type}"
                )
                patterns.append(pattern)
                
        return patterns
        
    def _deduplicate_patterns(self, patterns: List[CircuitPattern]) -> List[CircuitPattern]:
        """
        Remove duplicate or highly overlapping patterns.
        
        Args:
            patterns: List of patterns
            
        Returns:
            Deduplicated list
        """
        if not patterns:
            return patterns
            
        # Sort by confidence first
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        unique_patterns = []
        seen_features = set()
        
        for pattern in patterns:
            # Check if pattern's features overlap significantly with seen patterns
            pattern_features = set(pattern.affected_features)
            
            if not pattern_features or len(pattern_features & seen_features) < len(pattern_features) * 0.5:
                unique_patterns.append(pattern)
                seen_features.update(pattern_features)
                
        return unique_patterns
        
    def map_to_behavioral_template(self, pattern: CircuitPattern) -> Optional[str]:
        """
        Map a circuit pattern to a behavioral template.
        
        Args:
            pattern: Circuit pattern
            
        Returns:
            Template name or None if no match
        """
        # Check each template
        for template_name, template in self.pattern_templates.items():
            # Simple matching based on pattern type
            # In practice, this would be more sophisticated
            
            if pattern.pattern_type == "consistent_decrease" and pattern.confidence > 0.8:
                if any(feat[0] > 10 for feat in pattern.affected_features):  # Later layers
                    return 'capability_reduction'
                    
            elif pattern.pattern_type == "feature_cooccurrence" and len(pattern.affected_features) > 5:
                return 'style_shift'
                
            elif pattern.pattern_type == "consistent_increase":
                if any(feat[0] < 5 for feat in pattern.affected_features):  # Early layers
                    return 'knowledge_update'
                    
        return None