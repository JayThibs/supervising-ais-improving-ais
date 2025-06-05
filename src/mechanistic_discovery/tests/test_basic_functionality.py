"""
Basic functionality tests for the mechanistic discovery module.

These tests verify that the core components work together correctly.
They use mock objects to avoid requiring actual models and transcoders.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Import components to test
from mechanistic_discovery.circuit_analysis import (
    CircuitDifference,
    FeatureDifference,
    ConnectionDifference,
    DifferentialCircuitAnalyzer
)
from mechanistic_discovery.hypothesis_generation import (
    PatternRecognizer,
    CircuitPattern,
    PatternType,
    HypothesisGenerator,
    BehavioralHypothesis,
    HypothesisType
)
from mechanistic_discovery.behavioral_validation import (
    TestGenerator,
    TestCase,
    BehavioralValidator,
    ValidationResult,
    StatisticalValidator,
    StatisticalTestResult
)


class TestCircuitAnalysis:
    """Test circuit analysis components."""
    
    def test_feature_difference_creation(self):
        """Test creating feature differences."""
        feat_diff = FeatureDifference(
            layer=10,
            position=5,
            feature_idx=123,
            base_activation=0.5,
            intervention_activation=0.8
        )
        
        assert feat_diff.activation_delta == 0.3
        assert feat_diff.is_new_feature() == False
        assert feat_diff.is_suppressed_feature() == False
        
    def test_feature_suppression_detection(self):
        """Test detecting suppressed features."""
        feat_diff = FeatureDifference(
            layer=10,
            position=5,
            feature_idx=123,
            base_activation=0.8,
            intervention_activation=0.0
        )
        
        assert feat_diff.is_suppressed_feature() == True
        assert feat_diff.activation_delta == -0.8
        
    def test_connection_difference(self):
        """Test connection difference detection."""
        conn_diff = ConnectionDifference(
            source="feature_L10_F123",
            target="feature_L11_F456",
            base_weight=0.5,
            intervention_weight=0.1,
            source_type="feature",
            target_type="feature"
        )
        
        assert conn_diff.weight_change == 0.4
        assert conn_diff.is_removed_connection() == False
        assert conn_diff.is_new_connection() == False
        
    def test_circuit_difference_summary(self):
        """Test circuit difference summary generation."""
        # Create mock graph objects
        mock_base_graph = Mock()
        mock_base_graph.active_features = np.array([[10, 5, 123], [11, 3, 456]])
        mock_base_graph.adjacency_matrix = Mock()
        mock_base_graph.n_pos = 10
        mock_base_graph.logit_tokens = [1, 2, 3]
        
        mock_int_graph = Mock()
        mock_int_graph.active_features = np.array([[10, 5, 123], [12, 2, 789]])
        mock_int_graph.adjacency_matrix = Mock()
        mock_int_graph.n_pos = 10
        mock_int_graph.logit_tokens = [1, 2, 3]
        
        circuit_diff = CircuitDifference(
            prompt="Test prompt",
            feature_differences=[
                FeatureDifference(10, 5, 123, 0.5, 0.8),
                FeatureDifference(11, 3, 456, 0.7, 0.0)
            ],
            connection_differences=[
                ConnectionDifference("A", "B", 0.5, 0.8, "feature", "feature")
            ],
            statistics={"test_stat": 0.5},
            base_graph=mock_base_graph,
            intervention_graph=mock_int_graph
        )
        
        summary = circuit_diff.get_summary()
        
        assert summary['n_feature_differences'] == 2
        assert summary['n_new_features'] == 0
        assert summary['n_suppressed_features'] == 1
        assert summary['n_connection_differences'] == 1


class TestPatternRecognition:
    """Test pattern recognition components."""
    
    def test_pattern_recognition(self):
        """Test identifying patterns from circuit differences."""
        # Create a circuit difference with systematic suppression
        circuit_diff = CircuitDifference(
            prompt="Test",
            feature_differences=[
                FeatureDifference(10, 5, 123, 0.8, 0.1),
                FeatureDifference(10, 6, 124, 0.7, 0.0),
                FeatureDifference(11, 3, 456, 0.9, 0.2)
            ],
            connection_differences=[],
            statistics={},
            base_graph=Mock(),
            intervention_graph=Mock()
        )
        
        recognizer = PatternRecognizer()
        patterns = recognizer.identify_patterns(circuit_diff)
        
        # Should identify systematic suppression pattern
        assert len(patterns) > 0
        suppression_patterns = [
            p for p in patterns 
            if p.pattern_type == PatternType.SYSTEMATIC_SUPPRESSION
        ]
        assert len(suppression_patterns) > 0
        
    def test_pattern_clustering(self):
        """Test clustering similar changes."""
        recognizer = PatternRecognizer()
        
        changes = [
            FeatureDifference(10, 5, 123, 0.8, 0.1),
            FeatureDifference(10, 6, 124, 0.7, 0.0),
            FeatureDifference(10, 7, 125, 0.9, 0.2),
            FeatureDifference(15, 3, 456, 0.1, 0.9)  # Different pattern
        ]
        
        clusters = recognizer._cluster_similar_changes(changes)
        
        # Should separate suppression from emergence
        assert len(clusters) >= 2


class TestHypothesisGeneration:
    """Test hypothesis generation components."""
    
    def test_hypothesis_generation_from_patterns(self):
        """Test generating hypotheses from patterns."""
        # Create patterns
        patterns = [
            CircuitPattern(
                pattern_type=PatternType.SAFETY_DEGRADATION,
                changes=[
                    FeatureDifference(10, 5, 123, 0.8, 0.1),
                    FeatureDifference(10, 6, 124, 0.7, 0.0)
                ],
                consistency=0.8,
                importance_score=0.9
            )
        ]
        
        generator = HypothesisGenerator()
        hypotheses = generator.generate_hypotheses(patterns)
        
        assert len(hypotheses) > 0
        
        # Should generate safety degradation hypothesis
        safety_hypotheses = [
            h for h in hypotheses
            if h.hypothesis_type == HypothesisType.SAFETY_DEGRADATION
        ]
        assert len(safety_hypotheses) > 0
        
    def test_hypothesis_prioritization(self):
        """Test hypothesis prioritization."""
        generator = HypothesisGenerator()
        
        # Create two hypotheses with different priorities
        h1 = BehavioralHypothesis(
            hypothesis_type=HypothesisType.SAFETY_DEGRADATION,
            description="Safety weakened",
            circuit_evidence=[],
            feature_evidence=[],
            confidence=0.8
        )
        h1.priority = generator._calculate_priority(
            HypothesisType.SAFETY_DEGRADATION, 0.9
        )
        
        h2 = BehavioralHypothesis(
            hypothesis_type=HypothesisType.STYLE_CHANGE,
            description="Style changed",
            circuit_evidence=[],
            feature_evidence=[],
            confidence=0.8
        )
        h2.priority = generator._calculate_priority(
            HypothesisType.STYLE_CHANGE, 0.9
        )
        
        # Safety should have higher priority
        assert h1.priority > h2.priority


class TestBehavioralValidation:
    """Test behavioral validation components."""
    
    def test_test_generation(self):
        """Test generating test cases from hypotheses."""
        hypothesis = BehavioralHypothesis(
            hypothesis_type=HypothesisType.CAPABILITY_GAIN,
            description="Enhanced capabilities",
            circuit_evidence=[],
            feature_evidence=[
                {'layer': 10, 'feature_idx': 123, 'interpretation': 'code generation'}
            ],
            confidence=0.8,
            test_prompts=["Write code"],
            expected_differences={'capability': 'enhanced'}
        )
        
        generator = TestGenerator()
        test_cases = generator.generate_tests(hypothesis, n_tests=10)
        
        assert len(test_cases) == 10
        assert all(isinstance(tc, TestCase) for tc in test_cases)
        assert all(tc.hypothesis == hypothesis for tc in test_cases)
        
    def test_test_prioritization(self):
        """Test test case prioritization."""
        hypothesis = BehavioralHypothesis(
            hypothesis_type=HypothesisType.SAFETY_DEGRADATION,
            description="Safety weakened",
            circuit_evidence=[],
            feature_evidence=[],
            confidence=0.8
        )
        
        generator = TestGenerator()
        
        # Generate many test cases
        test_cases = generator.generate_tests(hypothesis, n_tests=50)
        
        # Prioritize to smaller budget
        prioritized = generator.prioritize_tests(test_cases, budget=10)
        
        assert len(prioritized) == 10
        # Should include different test types
        test_types = set(tc.metadata.get('test_type', 'template') for tc in prioritized)
        assert len(test_types) > 1
        
    def test_statistical_validation(self):
        """Test statistical validation with FDR control."""
        validator = StatisticalValidator()
        
        # Test SAFFRON FDR control
        p_values = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2]
        
        rejections = []
        for p in p_values:
            reject, threshold = validator.saffron.test(p)
            rejections.append(reject)
            
        # Should reject some but not all
        assert sum(rejections) > 0
        assert sum(rejections) < len(p_values)
        
        # FDR should be controlled
        fdr_estimate = validator.saffron.get_fdr_estimate()
        assert fdr_estimate <= 0.05 * 1.2  # Allow small violation


class TestIntegration:
    """Test integration of components."""
    
    @patch('mechanistic_discovery.circuit_analysis.circuit_tracer_wrapper.CircuitAwareModel')
    def test_analysis_pipeline(self, mock_model_class):
        """Test that components work together in the pipeline."""
        # This is a simplified integration test
        # In practice, would need actual models and transcoders
        
        # Mock circuit differences
        circuit_diff = CircuitDifference(
            prompt="Test safety",
            feature_differences=[
                FeatureDifference(10, 5, 123, 0.8, 0.1),  # Suppressed safety feature
            ],
            connection_differences=[],
            statistics={'total_change_score': 0.8},
            base_graph=Mock(),
            intervention_graph=Mock()
        )
        
        # Pattern recognition
        recognizer = PatternRecognizer()
        patterns = recognizer.identify_patterns(circuit_diff)
        assert len(patterns) > 0
        
        # Hypothesis generation
        generator = HypothesisGenerator()
        hypotheses = generator.generate_hypotheses(patterns)
        assert len(hypotheses) > 0
        
        # Test generation
        test_gen = TestGenerator()
        test_cases = test_gen.generate_tests(hypotheses[0], n_tests=10)
        assert len(test_cases) == 10
        
        # Would validate with real models in full test
        

def test_imports():
    """Test that all modules can be imported."""
    import mechanistic_discovery
    import mechanistic_discovery.circuit_analysis
    import mechanistic_discovery.hypothesis_generation
    import mechanistic_discovery.behavioral_validation
    import mechanistic_discovery.integration
    import mechanistic_discovery.utils
    
    # If we get here, imports work
    assert True


if __name__ == "__main__":
    # Run basic tests
    print("Running basic functionality tests...")
    
    # Test imports
    test_imports()
    print("✓ Imports successful")
    
    # Test circuit analysis
    test_circuit = TestCircuitAnalysis()
    test_circuit.test_feature_difference_creation()
    test_circuit.test_feature_suppression_detection()
    test_circuit.test_connection_difference()
    test_circuit.test_circuit_difference_summary()
    print("✓ Circuit analysis tests passed")
    
    # Test pattern recognition
    test_pattern = TestPatternRecognition()
    test_pattern.test_pattern_recognition()
    test_pattern.test_pattern_clustering()
    print("✓ Pattern recognition tests passed")
    
    # Test hypothesis generation
    test_hypothesis = TestHypothesisGeneration()
    test_hypothesis.test_hypothesis_generation_from_patterns()
    test_hypothesis.test_hypothesis_prioritization()
    print("✓ Hypothesis generation tests passed")
    
    # Test behavioral validation
    test_validation = TestBehavioralValidation()
    test_validation.test_test_generation()
    test_validation.test_test_prioritization()
    test_validation.test_statistical_validation()
    print("✓ Behavioral validation tests passed")
    
    print("\nAll tests passed! ✨")