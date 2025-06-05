"""
Circuit Analysis Module

This module provides tools for analyzing and comparing the internal computational circuits
of language models. It integrates with Anthropic's circuit-tracer to extract attribution
graphs and identify how information flows through models.

Core Concepts:
    - Attribution Graphs: DAGs showing how information flows from inputs to outputs
    - Transcoders: Sparse autoencoders that decompose model activations into interpretable features
    - Circuit Comparison: Identifying differences in computational paths between models
    - Feature Interpretation: Understanding what each transcoder feature represents

Main Components:
    - CircuitAwareModel: Wrapper that adds circuit tracing to standard models
    - DifferentialCircuitAnalyzer: Compares circuits between base and intervention models
    - FeatureInterpreter: Provides automated interpretation of transcoder features
    - CircuitCache: Efficient caching system for attribution graphs
"""

from .circuit_tracer_wrapper import CircuitAwareModel, CircuitTracingConfig
from .differential_analyzer import DifferentialCircuitAnalyzer, CircuitDifference
from .feature_interpreter import FeatureInterpreter, FeatureInterpretation
from .circuit_cache import CircuitCache

__all__ = [
    "CircuitAwareModel",
    "CircuitTracingConfig",
    "DifferentialCircuitAnalyzer",
    "CircuitDifference",
    "FeatureInterpreter", 
    "FeatureInterpretation",
    "CircuitCache"
]