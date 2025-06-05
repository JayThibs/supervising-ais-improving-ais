"""
Hypothesis Generation Module

This module converts circuit differences into testable behavioral hypotheses.
It uses pattern recognition and feature interpretation to predict how circuit
changes will manifest as observable behavioral differences.

Core Components:
    - PatternRecognizer: Identifies systematic patterns in circuit changes
    - HypothesisGenerator: Converts patterns into testable predictions
    - ImportanceScorer: Prioritizes hypotheses for testing

Key Concepts:
    - Circuit-Behavior Mapping: How internal changes predict external behavior
    - Pattern Clustering: Grouping similar circuit changes
    - Hypothesis Templates: Common patterns of behavioral change
    - Confidence Estimation: Predicting which hypotheses are most likely true
"""

from .pattern_recognizer import PatternRecognizer, CircuitPattern
from .hypothesis_generator import HypothesisGenerator, BehavioralHypothesis
from .importance_scorer import ImportanceScorer

__all__ = [
    "PatternRecognizer",
    "CircuitPattern",
    "HypothesisGenerator", 
    "BehavioralHypothesis",
    "ImportanceScorer"
]