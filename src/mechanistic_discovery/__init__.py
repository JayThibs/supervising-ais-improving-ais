"""
Mechanistic Discovery Module

This module implements a novel approach to detecting behavioral differences in language models
by combining mechanistic interpretability with targeted behavioral validation. Instead of 
exhaustively testing millions of samples, we use internal circuit analysis to guide our
search toward the most likely areas of behavioral change.

Core Components:
    - Circuit Analysis: Extract and compare computational circuits between models
    - Hypothesis Generation: Convert circuit differences into testable behavioral hypotheses
    - Behavioral Validation: Efficiently test hypotheses with statistical rigor
    - Integration: Orchestrate the full pipeline with performance optimization

Key Features:
    - 50-100x reduction in required samples
    - Maintains statistical rigor (FDR control at 5%)
    - Works with 8B parameter models on single GPU
    - Integrates with existing auto-interventions pipeline

Example:
    >>> from mechanistic_discovery import MechanisticBehavioralAnalyzer
    >>> analyzer = MechanisticBehavioralAnalyzer(
    ...     base_model_path="path/to/base",
    ...     intervention_model_path="path/to/intervention"
    ... )
    >>> findings = analyzer.analyze_model_differences(budget=1000)
    >>> print(f"Found {len(findings)} behavioral differences")

Author: Mechanistic Discovery Team
Version: 1.0.0
"""

from .integration.mechanistic_pipeline import MechanisticBehavioralAnalyzer
from .circuit_analysis import DifferentialCircuitAnalyzer, CircuitAwareModel
from .hypothesis_generation import HypothesisGenerator, ImportanceScorer
from .behavioral_validation import TargetedValidator, ActiveExplorer

__version__ = "1.0.0"

__all__ = [
    "MechanisticBehavioralAnalyzer",
    "DifferentialCircuitAnalyzer", 
    "CircuitAwareModel",
    "HypothesisGenerator",
    "ImportanceScorer",
    "TargetedValidator",
    "ActiveExplorer"
]