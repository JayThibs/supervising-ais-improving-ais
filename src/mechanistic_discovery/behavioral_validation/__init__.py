"""
Behavioral Validation Module

This module provides tools for validating behavioral hypotheses generated from
circuit analysis. It tests whether mechanistic differences translate into
observable behavioral changes.

Core Components:
    - TestGenerator: Creates targeted test cases for hypotheses
    - BehavioralValidator: Runs tests and analyzes results
    - StatisticalValidator: Ensures statistical rigor with FDR control
"""

from .test_generator import TestGenerator, TestCase
from .behavioral_validator import BehavioralValidator, ValidationResult
from .statistical_validator import StatisticalValidator

__all__ = [
    'TestGenerator',
    'TestCase',
    'BehavioralValidator', 
    'ValidationResult',
    'StatisticalValidator'
]