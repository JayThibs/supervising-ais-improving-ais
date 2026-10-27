"""
Circuit Tracer Wrapper

This module provides a unified interface for models that supports both standard generation
and circuit tracing capabilities. It wraps around Anthropic's circuit-tracer library
to enable mechanistic analysis of model behavior.

Key Concepts:
    - Transcoders: Sparse autoencoders that decompose MLP activations into interpretable features
    - ReplacementModel: A model where MLPs are replaced with transcoder-based decompositions
    - Attribution Graphs: Directed graphs showing causal influence between model components
    - Graph Structure: Nodes are [active_features, error_nodes, embed_nodes, logit_nodes]

Technical Details:
    - Transcoders typically use 4-8x expansion factor (e.g., 8192 MLP dim → 65536 features)
    - Only ~50-200 features are active per token (sparse activation)
    - Attribution requires both forward and backward passes through the model
    - Memory usage scales with number of active features and sequence length
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import networkx as nx

from transformers import AutoModelForCausalLM, AutoTokenizer
from circuit_tracer import ReplacementModel, Graph, attribute


@dataclass
class CircuitTracingConfig:
    """
    Configuration for circuit tracing operations.
    
    Attributes:
        model_name (str): HuggingFace model name or path (e.g., "google/gemma-2-2b")
        transcoder_set (str): Either a predefined set name (e.g., "gemma-2-2b") or 
            path to a config file. Predefined sets include:
            - "gemma-2-2b", "gemma-2-9b" (Gemma models)
            - "llama-3-8b" (Llama models)
        max_n_logits (int): Maximum number of logit nodes to include in graph.
            Higher values capture more output options but increase computation.
        desired_logit_prob (float): Keep logits until cumulative probability 
            reaches this threshold. Controls completeness vs efficiency.
        batch_size (int): Batch size for attribution computation. Limited by GPU memory.
        max_feature_nodes (Optional[int]): Maximum features to include per prompt.
            None means no limit (can cause memory issues).
        offload (str): Memory management strategy: "cpu", "disk", or None.
            Use "cpu" or "disk" for large models to prevent OOM.
        device (str): Device for computation ('cuda' or 'cpu').
        cache_dir (str): Directory to cache computed circuits.
        verbose (bool): Whether to show progress during attribution.
    """
    model_name: str
    transcoder_set: str
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95
    batch_size: int = 256
    max_feature_nodes: Optional[int] = 10000
    offload: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: str = "circuit_cache"
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.desired_logit_prob <= 0 or self.desired_logit_prob > 1:
            raise ValueError("desired_logit_prob must be in (0, 1]")
        
        if self.max_n_logits < 1:
            raise ValueError("max_n_logits must be >= 1")
            
        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


class CircuitAwareModel:
    """
    A wrapper that combines standard language model capabilities with circuit tracing.
    
    This class provides a unified interface for:
    1. Standard text generation (like a normal LM)
    2. Circuit extraction and analysis via attribution graphs
    3. Feature interpretation and comparison
    
    The key innovation is using a ReplacementModel that decomposes computations
    into interpretable features while maintaining the same outputs.
    
    Example:
        >>> config = CircuitTracingConfig(
        ...     model_name="google/gemma-2-2b",
        ...     transcoder_set="gemma-2-2b"
        ... )
        >>> model = CircuitAwareModel(config)
        >>> 
        >>> # Use as normal model
        >>> output = model.generate("Hello world")
        >>> 
        >>> # Extract circuits
        >>> circuit = model.extract_circuit("The cat sat on the")
        >>> print(f"Found {len(circuit['active_features'])} active features")
    """
    
    def __init__(self, config: CircuitTracingConfig):
        """
        Initialize a circuit-aware model.
        
        Args:
            config: Configuration for circuit tracing
            
        Raises:
            ValueError: If model/transcoder combination is not supported
            RuntimeError: If models cannot be loaded
        """
        self.config = config
        
        print(f"Loading circuit-aware model {config.model_name}...")
        
        # Load the replacement model with transcoders
        try:
            self.replacement_model = ReplacementModel.from_pretrained(
                config.model_name,
                config.transcoder_set,
                device=torch.device(config.device),
                dtype=torch.bfloat16 if config.device == "cuda" else torch.float32
            )
            print(f"Loaded replacement model with transcoder set: {config.transcoder_set}")
        except Exception as e:
            raise RuntimeError(f"Failed to load replacement model: {e}")
            
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Extract model info
        self.model_config = self.replacement_model.cfg
        self.n_layers = self.model_config.n_layers
        self.d_model = self.model_config.d_model
        self.d_transcoder = self.replacement_model.d_transcoder
        
        print(f"Model initialized: {self.n_layers} layers, "
              f"{self.d_model} hidden dim, {self.d_transcoder} transcoder features")
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the replacement model (standard generation).
        
        Note: The replacement model maintains the same generation capabilities
        as the original model, just with additional introspection features.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation arguments (max_length, temperature, etc.)
            
        Returns:
            Generated text (only the new tokens, not including prompt)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.replacement_model.device)
        
        # Default generation parameters
        gen_kwargs = {
            "max_length": 100,
            "temperature": 1.0,
            "do_sample": True,
            "top_p": 0.9,
            **kwargs
        }
        
        with torch.no_grad():
            outputs = self.replacement_model.generate(
                inputs.input_ids,
                **gen_kwargs
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):]  # Return only the generated part
        
    def extract_circuit(self, 
                       prompt: Union[str, List[int], torch.Tensor],
                       return_graph_object: bool = False) -> Dict[str, Any]:
        """
        Extract the computational circuit (attribution graph) for a given prompt.
        
        This is the core mechanistic analysis function. It traces how information
        flows through the model to produce the next token prediction, identifying
        which features and computational paths are most important.
        
        Args:
            prompt: Input text, token IDs, or tensor to analyze
            return_graph_object: If True, return the raw Graph object.
                                If False, return processed dictionary.
            
        Returns:
            If return_graph_object=True: Raw circuit_tracer.Graph object
            If return_graph_object=False: Dictionary containing:
                - 'graph': The Graph object
                - 'active_features': Dict mapping layer -> list of (pos, feature_idx) tuples
                - 'feature_importances': Dict mapping feature names to importance scores
                - 'adjacency_matrix': The raw adjacency matrix
                - 'metadata': Additional information about the circuit
                
        Technical Details:
            The Graph object contains nodes in this order:
            1. Active features (transcoder features that fired)
            2. Error nodes (residual not captured by transcoders)
            3. Token embedding nodes
            4. Logit nodes (next token predictions)
            
            The adjacency matrix shows direct effects between all nodes.
        """
        print(f"Extracting circuit for prompt: '{prompt[:50] if isinstance(prompt, str) else '...'}'")
        
        # Run attribution to get the Graph object
        graph = attribute(
            prompt=prompt,
            model=self.replacement_model,
            max_n_logits=self.config.max_n_logits,
            desired_logit_prob=self.config.desired_logit_prob,
            batch_size=self.config.batch_size,
            max_feature_nodes=self.config.max_feature_nodes,
            offload=self.config.offload,
            verbose=self.config.verbose
        )
        
        if return_graph_object:
            return graph
            
        # Process the graph into a more convenient format
        processed = self._process_graph(graph)
        processed['graph'] = graph
        
        return processed
        
    def _process_graph(self, graph: Graph) -> Dict[str, Any]:
        """
        Process a Graph object into a more convenient dictionary format.
        
        Args:
            graph: Circuit tracer Graph object
            
        Returns:
            Dictionary with processed circuit information
        """
        # Extract active features by layer
        active_features_by_layer = {}
        feature_importances = {}
        
        # graph.active_features is tensor of shape (n_active_features, 3)
        # Each row is (layer, pos, feature_idx)
        for layer, pos, feature_idx in graph.active_features.tolist():
            if layer not in active_features_by_layer:
                active_features_by_layer[layer] = []
            active_features_by_layer[layer].append((pos, feature_idx))
            
            # Compute importance from adjacency matrix
            # Features are the first nodes in the adjacency matrix
            feature_node_idx = len([(l, p, f) for l, p, f in graph.active_features.tolist() 
                                  if l < layer or (l == layer and (p < pos or (p == pos and f < feature_idx)))])
            
            # Sum absolute values of outgoing connections
            if feature_node_idx < graph.adjacency_matrix.shape[1]:
                importance = torch.abs(graph.adjacency_matrix[:, feature_node_idx]).sum().item()
                feature_name = f"L{layer}_P{pos}_F{feature_idx}"
                feature_importances[feature_name] = importance
        
        # Normalize importances
        total_importance = sum(feature_importances.values())
        if total_importance > 0:
            feature_importances = {k: v/total_importance for k, v in feature_importances.items()}
            
        # Extract metadata
        metadata = {
            'prompt': graph.input_string,
            'n_positions': graph.n_pos,
            'n_active_features': len(graph.active_features),
            'n_logits': len(graph.logit_tokens),
            'top_logits': [(tok, prob.item()) for tok, prob in 
                          zip(graph.logit_tokens.tolist(), graph.logit_probabilities.tolist())]
        }
        
        return {
            'active_features': active_features_by_layer,
            'feature_importances': feature_importances,
            'adjacency_matrix': graph.adjacency_matrix,
            'metadata': metadata
        }
        
    def compare_with(self, other_model: 'CircuitAwareModel', prompt: str) -> Dict[str, Any]:
        """
        Compare circuits between this model and another model on the same prompt.
        
        This is a convenience method that extracts circuits from both models
        and provides a basic comparison. For more sophisticated analysis,
        use DifferentialCircuitAnalyzer.
        
        Args:
            other_model: Another CircuitAwareModel to compare with
            prompt: Input prompt to analyze
            
        Returns:
            Dictionary containing comparison results including:
                - Feature overlap statistics
                - Unique features to each model
                - Similarity metrics
        """
        # Extract circuits from both models
        self_circuit = self.extract_circuit(prompt)
        other_circuit = other_model.extract_circuit(prompt)
        
        # Extract feature sets
        self_features = set()
        other_features = set()
        
        for layer, features in self_circuit['active_features'].items():
            for pos, feat_idx in features:
                self_features.add((layer, pos, feat_idx))
                
        for layer, features in other_circuit['active_features'].items():
            for pos, feat_idx in features:
                other_features.add((layer, pos, feat_idx))
        
        # Compute overlap statistics
        shared_features = self_features & other_features
        unique_to_self = self_features - other_features
        unique_to_other = other_features - self_features
        
        # Jaccard similarity
        jaccard = len(shared_features) / len(self_features | other_features) if self_features | other_features else 1.0
        
        return {
            'n_features_self': len(self_features),
            'n_features_other': len(other_features),
            'n_shared': len(shared_features),
            'n_unique_to_self': len(unique_to_self),
            'n_unique_to_other': len(unique_to_other),
            'jaccard_similarity': jaccard,
            'unique_to_self': list(unique_to_self)[:10],  # Top 10 for brevity
            'unique_to_other': list(unique_to_other)[:10],
            'self_metadata': self_circuit['metadata'],
            'other_metadata': other_circuit['metadata']
        }
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model configuration.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_name': self.config.model_name,
            'transcoder_set': self.config.transcoder_set,
            'n_layers': self.n_layers,
            'd_model': self.d_model,
            'd_transcoder': self.d_transcoder,
            'device': str(self.replacement_model.device),
            'dtype': str(self.replacement_model.dtype)
        }
        
    def to_networkx_graph(self, graph: Graph) -> nx.DiGraph:
        """
        Convert a circuit_tracer Graph to a NetworkX directed graph.
        
        This is useful for applying graph algorithms or custom visualizations.
        
        Args:
            graph: Circuit tracer Graph object
            
        Returns:
            NetworkX DiGraph with nodes labeled by type and edges weighted by attribution
        """
        G = nx.DiGraph()
        
        n_features = len(graph.active_features)
        n_pos = graph.n_pos
        n_layers = self.n_layers
        n_logits = len(graph.logit_tokens)
        
        # Add nodes with labels
        node_idx = 0
        
        # Feature nodes
        for layer, pos, feat_idx in graph.active_features.tolist():
            G.add_node(node_idx, 
                      label=f"F_L{layer}_P{pos}_{feat_idx}",
                      type="feature",
                      layer=layer,
                      position=pos,
                      feature_idx=feat_idx)
            node_idx += 1
            
        # Error nodes
        for layer in range(n_layers):
            for pos in range(n_pos):
                G.add_node(node_idx,
                          label=f"E_L{layer}_P{pos}",
                          type="error",
                          layer=layer,
                          position=pos)
                node_idx += 1
                
        # Token embedding nodes
        for pos in range(n_pos):
            token = graph.input_tokens[pos] if hasattr(graph, 'input_tokens') else pos
            G.add_node(node_idx,
                      label=f"Tok_{pos}",
                      type="embedding",
                      position=pos,
                      token=token)
            node_idx += 1
            
        # Logit nodes
        for i, (token, prob) in enumerate(zip(graph.logit_tokens.tolist(), 
                                              graph.logit_probabilities.tolist())):
            G.add_node(node_idx,
                      label=f"Logit_{i}",
                      type="logit",
                      token=token,
                      probability=prob)
            node_idx += 1
            
        # Add edges from adjacency matrix
        # Note: adjacency matrix is (targets, sources) where rows are targets
        adj_matrix = graph.adjacency_matrix.cpu().numpy()
        
        for target_idx in range(adj_matrix.shape[0]):
            for source_idx in range(adj_matrix.shape[1]):
                weight = adj_matrix[target_idx, source_idx]
                if abs(weight) > 1e-6:  # Threshold for meaningful connections
                    G.add_edge(source_idx, target_idx, weight=weight)
                    
        return G