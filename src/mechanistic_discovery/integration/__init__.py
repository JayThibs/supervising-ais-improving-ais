"""
Integration Module

This module provides the main pipeline that integrates all components of the
mechanistic discovery system. It orchestrates the flow from circuit analysis
through hypothesis generation to behavioral validation.

Core Components:
    - MechanisticBehavioralAnalyzer: Main analyzer class
    - AnalysisConfig: Configuration for the analysis pipeline
    - AnalysisReport: Comprehensive results report
"""

from .analyzer import MechanisticBehavioralAnalyzer, AnalysisConfig, AnalysisReport
from .active_explorer import ActiveCircuitExplorer

__all__ = [
    'MechanisticBehavioralAnalyzer',
    'AnalysisConfig', 
    'AnalysisReport',
    'ActiveCircuitExplorer'
]