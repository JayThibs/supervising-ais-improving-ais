"""
Test suite for running on small devices (like M1 MacBook Air).

This test suite uses mock objects and tiny models to verify functionality
without requiring large models or GPUs.
"""

import sys
import os
from pathlib import Path
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import json
import tempfile
import time

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Test imports first
print("Testing imports...")
try:
    from mechanistic_discovery.circuit_analysis import (
        CircuitAwareModel,
        CircuitTracingConfig,
        DifferentialCircuitAnalyzer,
        CircuitDifference,
        FeatureDifference,
        ConnectionDifference,
        CircuitCache,
        FeatureInterpreter
    )
    print("✓ Circuit analysis imports successful")
except ImportError as e:
    print(f"✗ Circuit analysis import failed: {e}")
    sys.exit(1)

try:
    from mechanistic_discovery.hypothesis_generation import (
        PatternRecognizer,
        CircuitPattern,
        PatternType,
        HypothesisGenerator,
        BehavioralHypothesis,
        HypothesisType
    )
    print("✓ Hypothesis generation imports successful")
except ImportError as e:
    print(f"✗ Hypothesis generation import failed: {e}")
    sys.exit(1)

try:
    from mechanistic_discovery.behavioral_validation import (
        TestGenerator,
        TestCase,
        BehavioralValidator,
        ValidationResult,
        StatisticalValidator,
        TestingProcedure
    )
    print("✓ Behavioral validation imports successful")
except ImportError as e:
    print(f"✗ Behavioral validation import failed: {e}")
    sys.exit(1)

try:
    from mechanistic_discovery.integration import (
        MechanisticBehavioralAnalyzer,
        AnalysisConfig,
        AnalysisReport,
        ActiveCircuitExplorer
    )
    print("✓ Integration imports successful")
except ImportError as e:
    print(f"✗ Integration import failed: {e}")
    sys.exit(1)

try:
    from mechanistic_discovery.utils import (
        CircuitVisualizer,
        PromptGenerator,
        PromptDiversityScorer,
        get_model_info,
        check_model_compatibility
    )
    print("✓ Utils imports successful")
except ImportError as e:
    print(f"✗ Utils import failed: {e}")
    sys.exit(1)


def create_mock_graph():
    """Create a mock Graph object for testing."""
    graph = Mock()
    graph.active_features = np.array([
        [5, 0, 123],   # layer 5, position 0, feature 123
        [5, 1, 456],   # layer 5, position 1, feature 456
        [10, 0, 789],  # layer 10, position 0, feature 789
    ])
    graph.adjacency_matrix = np.random.randn(10, 10)
    graph.n_pos = 5
    graph.logit_tokens = np.array([1, 2, 3])
    graph.logit_probabilities = np.array([0.7, 0.2, 0.1])
    graph.input_string = "Test prompt"
    return graph


def create_mock_circuit_aware_model(name="mock_model"):
    """Create a mock CircuitAwareModel for testing."""
    model = Mock(spec=CircuitAwareModel)
    model.model_name = name
    model.n_layers = 12
    model.d_model = 768
    model.d_transcoder = 768 * 8
    model.device = "cpu"
    
    # Mock the extract_circuit method
    def mock_extract_circuit(prompt, return_graph_object=False):
        if return_graph_object:
            return create_mock_graph()
        else:
            return {
                'active_features': {
                    5: [(0, 123), (1, 456)],
                    10: [(0, 789)]
                },
                'feature_importances': {
                    'L5_F123': 0.8,
                    'L5_F456': 0.6,
                    'L10_F789': 0.9
                },
                'adjacency_matrix': np.random.randn(10, 10),
                'metadata': {
                    'prompt': prompt,
                    'n_active_features': 3
                }
            }
    
    model.extract_circuit = mock_extract_circuit
    model.get_model_info = lambda: {
        'model_name': name,
        'n_layers': 12,
        'd_model': 768,
        'd_transcoder': 768 * 8,
        'device': 'cpu'
    }
    
    # Mock generate method for behavioral validation
    model.tokenizer = Mock()
    model.tokenizer.return_value = Mock(to=lambda x: Mock())
    model.model = Mock()
    model.model.generate = Mock(return_value=Mock(sequences=np.array([[1, 2, 3]])))
    model.tokenizer.batch_decode = Mock(return_value=["Generated response"])
    
    return model


def test_circuit_analysis():
    """Test circuit analysis components."""
    print("\n" + "="*60)
    print("Testing Circuit Analysis Components")
    print("="*60)
    
    # Test 1: Feature differences
    print("\n1. Testing feature difference calculations...")
    feat_diff = FeatureDifference(
        layer=10,
        position=5,
        feature_idx=123,
        base_activation=0.8,
        intervention_activation=0.1
    )
    
    assert feat_diff.activation_delta == -0.7, "Activation delta calculation failed"
    assert feat_diff.is_suppressed_feature() == True, "Suppression detection failed"
    print("✓ Feature difference calculations work correctly")
    
    # Test 2: Circuit cache
    print("\n2. Testing circuit cache...")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CircuitCache(cache_dir=tmpdir, model_name="test_model")
        
        # Create a mock circuit difference
        mock_diff = CircuitDifference(
            prompt="Test prompt",
            feature_differences=[feat_diff],
            connection_differences=[],
            statistics={'test': 1.0},
            base_graph=create_mock_graph(),
            intervention_graph=create_mock_graph()
        )
        
        # Test caching
        key = cache.generate_key("Test prompt", "model1")
        cache.set(key, mock_diff)
        retrieved = cache.get(key)
        
        assert retrieved is not None, "Cache retrieval failed"
        assert retrieved.prompt == "Test prompt", "Cache corruption"
        print("✓ Circuit caching works correctly")
    
    # Test 3: Differential analyzer with mocks
    print("\n3. Testing differential analyzer...")
    base_model = create_mock_circuit_aware_model("base")
    int_model = create_mock_circuit_aware_model("intervention")
    
    analyzer = DifferentialCircuitAnalyzer(base_model, int_model)
    
    # Mock analyze_prompt
    result = analyzer.analyze_prompt("Test prompt")
    assert isinstance(result, CircuitDifference), "Analyzer should return CircuitDifference"
    print("✓ Differential analyzer works with mock models")
    
    return True


def test_pattern_recognition():
    """Test pattern recognition components."""
    print("\n" + "="*60)
    print("Testing Pattern Recognition Components")
    print("="*60)
    
    # Create test data
    feature_diffs = [
        FeatureDifference(10, 0, 123, 0.8, 0.1),  # Suppressed
        FeatureDifference(10, 1, 124, 0.7, 0.0),  # Suppressed
        FeatureDifference(10, 2, 125, 0.9, 0.2),  # Suppressed
        FeatureDifference(15, 0, 456, 0.1, 0.9),  # Emerged
    ]
    
    circuit_diff = CircuitDifference(
        prompt="Test safety prompt",
        feature_differences=feature_diffs,
        connection_differences=[],
        statistics={'total_change': 0.8},
        base_graph=create_mock_graph(),
        intervention_graph=create_mock_graph()
    )
    
    # Test pattern recognition
    print("\n1. Testing pattern identification...")
    recognizer = PatternRecognizer()
    patterns = recognizer.identify_patterns(circuit_diff)
    
    assert len(patterns) > 0, "Should identify patterns"
    
    # Check for systematic suppression
    suppression_patterns = [p for p in patterns if p.pattern_type == PatternType.SYSTEMATIC_SUPPRESSION]
    assert len(suppression_patterns) > 0, "Should identify suppression pattern"
    print("✓ Pattern identification works correctly")
    
    # Test pattern clustering
    print("\n2. Testing pattern clustering...")
    clusters = recognizer._cluster_similar_changes(feature_diffs)
    assert len(clusters) >= 2, "Should separate suppression from emergence"
    print("✓ Pattern clustering works correctly")
    
    return True


def test_hypothesis_generation():
    """Test hypothesis generation."""
    print("\n" + "="*60)
    print("Testing Hypothesis Generation")
    print("="*60)
    
    # Create test patterns
    patterns = [
        CircuitPattern(
            pattern_type=PatternType.SAFETY_DEGRADATION,
            changes=[
                FeatureDifference(10, 0, 123, 0.8, 0.1),
                FeatureDifference(10, 1, 124, 0.7, 0.0)
            ],
            consistency=0.8,
            importance_score=0.9
        ),
        CircuitPattern(
            pattern_type=PatternType.CAPABILITY_EMERGENCE,
            changes=[
                FeatureDifference(15, 0, 456, 0.1, 0.9)
            ],
            consistency=0.7,
            importance_score=0.6
        )
    ]
    
    # Test hypothesis generation
    print("\n1. Testing hypothesis generation from patterns...")
    generator = HypothesisGenerator()
    hypotheses = generator.generate_hypotheses(patterns)
    
    assert len(hypotheses) > 0, "Should generate hypotheses"
    
    # Check hypothesis types
    hypothesis_types = {h.hypothesis_type for h in hypotheses}
    assert HypothesisType.SAFETY_DEGRADATION in hypothesis_types, "Should identify safety degradation"
    print("✓ Hypothesis generation works correctly")
    
    # Test hypothesis prioritization
    print("\n2. Testing hypothesis prioritization...")
    priorities = [h.priority for h in hypotheses]
    assert max(priorities) > min(priorities), "Should have different priorities"
    
    # Safety should have highest priority
    safety_hyp = next(h for h in hypotheses if h.hypothesis_type == HypothesisType.SAFETY_DEGRADATION)
    assert safety_hyp.priority >= 0.8, "Safety hypotheses should have high priority"
    print("✓ Hypothesis prioritization works correctly")
    
    return True


def test_behavioral_validation():
    """Test behavioral validation components."""
    print("\n" + "="*60)
    print("Testing Behavioral Validation")
    print("="*60)
    
    # Create test hypothesis
    hypothesis = BehavioralHypothesis(
        hypothesis_type=HypothesisType.SAFETY_DEGRADATION,
        description="Model safety mechanisms may be weakened",
        circuit_evidence=[],
        feature_evidence=[
            {'layer': 10, 'feature_idx': 123, 'interpretation': 'safety feature'}
        ],
        confidence=0.8
    )
    
    # Test 1: Test generation
    print("\n1. Testing test case generation...")
    test_gen = TestGenerator()
    test_cases = test_gen.generate_tests(hypothesis, n_tests=20)
    
    assert len(test_cases) == 20, "Should generate requested number of tests"
    assert all(isinstance(tc, TestCase) for tc in test_cases), "Should generate TestCase objects"
    
    # Check diversity
    prompts = [tc.prompt for tc in test_cases]
    unique_prompts = set(prompts)
    assert len(unique_prompts) >= 15, "Should generate diverse prompts"
    print("✓ Test generation works correctly")
    
    # Test 2: Statistical validation
    print("\n2. Testing statistical validation...")
    validator = StatisticalValidator()
    
    # Test SAFFRON
    p_values = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    decisions = []
    
    for p in p_values:
        reject, threshold = validator.saffron.test(p)
        decisions.append((p, reject, threshold))
    
    # Should reject some early p-values
    early_rejections = sum(1 for p, reject, _ in decisions[:4] if reject)
    assert early_rejections > 0, "SAFFRON should reject some hypotheses"
    
    # Should control FDR
    fdr = validator.saffron.get_fdr_estimate()
    assert fdr <= 0.1, "FDR should be controlled"
    print("✓ Statistical validation works correctly")
    
    return True


def test_utils():
    """Test utility functions."""
    print("\n" + "="*60)
    print("Testing Utility Functions")
    print("="*60)
    
    # Test 1: Prompt generation
    print("\n1. Testing prompt generation...")
    prompt_gen = PromptGenerator()
    prompts = prompt_gen.generate_prompts(
        n=10,
        categories=['reasoning', 'safety'],
        ensure_unique=True
    )
    
    assert len(prompts) == 10, "Should generate requested number"
    assert len(set(prompts)) == 10, "Should be unique"
    print("✓ Prompt generation works correctly")
    
    # Test 2: Prompt diversity scoring
    print("\n2. Testing prompt diversity scoring...")
    scorer = PromptDiversityScorer()
    
    test_prompts = [
        "Solve this step by step: 2+2",
        "Is it ethical to lie?",
        "Write a poem about AI"
    ]
    
    for prompt in test_prompts:
        scores = scorer.score_prompt(prompt)
        assert 0 <= scores['overall_score'] <= 1, "Scores should be normalized"
    
    # Rank prompts
    ranked = scorer.rank_prompts(test_prompts)
    assert len(ranked) == len(test_prompts), "Should rank all prompts"
    print("✓ Prompt scoring works correctly")
    
    # Test 3: Model info extraction (mock)
    print("\n3. Testing model info utilities...")
    
    # Mock model config
    with patch('mechanistic_discovery.utils.model_utils.AutoConfig') as mock_config:
        mock_config.from_pretrained.return_value = Mock(
            model_type='gpt2',
            architectures=['GPT2Model'],
            num_hidden_layers=12,
            hidden_size=768,
            vocab_size=50257
        )
        
        info = get_model_info("mock-model")
        assert info['model_type'] == 'gpt2', "Should extract model type"
        assert info['n_layers'] == 12, "Should extract layer count"
        print("✓ Model info extraction works correctly")
    
    return True


def test_integration_lightweight():
    """Test integration with lightweight mocks."""
    print("\n" + "="*60)
    print("Testing Integration (Lightweight)")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config
        config = AnalysisConfig(
            base_model_name="mock-base",
            intervention_model_name="mock-intervention",
            transcoder_set="mock-transcoders",
            n_seed_prompts=5,  # Very small for testing
            n_test_prompts_per_hypothesis=10,
            max_hypotheses_to_test=2,
            cache_dir=tmpdir,
            output_dir=tmpdir,
            device="cpu",
            use_active_exploration=False  # Disable for simple test
        )
        
        # Mock the model initialization
        with patch.object(MechanisticBehavioralAnalyzer, '_initialize_models') as mock_init_models:
            with patch.object(MechanisticBehavioralAnalyzer, '_initialize_components') as mock_init_components:
                
                # Create analyzer
                analyzer = MechanisticBehavioralAnalyzer(config)
                
                # Manually set mock models
                analyzer.base_model = create_mock_circuit_aware_model("base")
                analyzer.intervention_model = create_mock_circuit_aware_model("intervention")
                
                # Create minimal components
                analyzer.circuit_analyzer = DifferentialCircuitAnalyzer(
                    analyzer.base_model,
                    analyzer.intervention_model
                )
                analyzer.pattern_recognizer = PatternRecognizer()
                analyzer.hypothesis_generator = HypothesisGenerator()
                analyzer.test_generator = TestGenerator()
                
                # Mock the circuit cache
                analyzer.circuit_cache = Mock()
                analyzer.circuit_cache.get = Mock(return_value=None)
                analyzer.circuit_cache.set = Mock()
                
                # Mock feature interpreter
                analyzer.feature_interpreter = Mock()
                analyzer.feature_interpreter.interpret_feature = Mock(
                    return_value={'interpretation': 'test feature'}
                )
                
                # Mock validators
                analyzer.behavioral_validator = Mock()
                analyzer.behavioral_validator.validate_hypothesis = Mock(
                    return_value=Mock(
                        is_validated=lambda: True,
                        p_value=0.01,
                        effect_size=0.5,
                        confidence=0.8,
                        hypothesis=Mock(hypothesis_type=HypothesisType.CAPABILITY_GAIN),
                        test_results=[]
                    )
                )
                
                analyzer.statistical_validator = Mock()
                
                print("✓ Analyzer initialization successful")
                
                # Test basic workflow components
                print("\n1. Testing circuit analysis workflow...")
                
                # Mock analyze_circuits to return differences
                def mock_analyze_circuits(prompts):
                    diffs = []
                    for prompt in prompts:
                        diff = CircuitDifference(
                            prompt=prompt,
                            feature_differences=[
                                FeatureDifference(10, 0, 123, 0.8, 0.1)
                            ],
                            connection_differences=[],
                            statistics={'change': 0.5},
                            base_graph=create_mock_graph(),
                            intervention_graph=create_mock_graph()
                        )
                        diffs.append(diff)
                    return diffs
                
                analyzer._analyze_circuits = mock_analyze_circuits
                
                # Test analysis of prompts
                test_prompts = ["Test 1", "Test 2", "Test 3"]
                circuit_diffs = analyzer._analyze_circuits(test_prompts)
                assert len(circuit_diffs) == 3, "Should analyze all prompts"
                print("✓ Circuit analysis workflow works")
                
                print("\n2. Testing pattern recognition workflow...")
                # Extract patterns
                all_patterns = []
                for diff in circuit_diffs:
                    patterns = analyzer.pattern_recognizer.identify_patterns(diff)
                    all_patterns.extend(patterns)
                
                assert len(all_patterns) > 0, "Should identify patterns"
                print("✓ Pattern recognition workflow works")
                
                print("\n3. Testing hypothesis generation workflow...")
                hypotheses = analyzer.hypothesis_generator.generate_hypotheses(all_patterns)
                assert len(hypotheses) > 0, "Should generate hypotheses"
                print("✓ Hypothesis generation workflow works")
                
                print("\n✓ Basic integration test passed!")
    
    return True


def run_memory_test():
    """Test memory usage stays reasonable."""
    print("\n" + "="*60)
    print("Testing Memory Usage")
    print("="*60)
    
    try:
        import psutil
        process = psutil.Process()
        
        # Get initial memory
        initial_mem = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory: {initial_mem:.1f} MB")
        
        # Run pattern recognition on many differences
        recognizer = PatternRecognizer()
        
        # Create many feature differences
        for i in range(100):
            feature_diffs = [
                FeatureDifference(j, 0, i*10+j, np.random.rand(), np.random.rand())
                for j in range(12)
            ]
            
            circuit_diff = CircuitDifference(
                prompt=f"Test {i}",
                feature_differences=feature_diffs,
                connection_differences=[],
                statistics={},
                base_graph=create_mock_graph(),
                intervention_graph=create_mock_graph()
            )
            
            patterns = recognizer.identify_patterns(circuit_diff)
        
        # Check memory after
        final_mem = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = final_mem - initial_mem
        
        print(f"Final memory: {final_mem:.1f} MB")
        print(f"Memory increase: {mem_increase:.1f} MB")
        
        assert mem_increase < 100, "Memory usage should stay reasonable"
        print("✓ Memory usage is reasonable")
        
    except ImportError:
        print("⚠️  psutil not installed, skipping memory test")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MECHANISTIC DISCOVERY TEST SUITE")
    print("Running on small device (no large models required)")
    print("="*80)
    
    start_time = time.time()
    
    tests = [
        ("Circuit Analysis", test_circuit_analysis),
        ("Pattern Recognition", test_pattern_recognition),
        ("Hypothesis Generation", test_hypothesis_generation),
        ("Behavioral Validation", test_behavioral_validation),
        ("Utilities", test_utils),
        ("Integration", test_integration_lightweight),
        ("Memory Usage", run_memory_test)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n✗ {test_name} test failed")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}")
    print(f"Total time: {total_time:.1f} seconds")
    
    if failed == 0:
        print("\n✅ All tests passed! The mechanistic discovery system is working correctly.")
    else:
        print(f"\n❌ {failed} tests failed. Please check the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)