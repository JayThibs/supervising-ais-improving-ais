"""
Test with mocked dependencies to find bugs in the code structure.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import types

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

print("Testing with mocked dependencies...")

# Create mock numpy
mock_numpy = types.ModuleType('numpy')
mock_numpy.array = Mock(side_effect=lambda x: x)
mock_numpy.random = Mock()
mock_numpy.random.randn = Mock(return_value=[[0.1, 0.2], [0.3, 0.4]])
mock_numpy.mean = Mock(return_value=0.5)
mock_numpy.std = Mock(return_value=0.1)
mock_numpy.zeros = Mock(return_value=[[0, 0], [0, 0]])
mock_numpy.sqrt = Mock(side_effect=lambda x: x ** 0.5)
mock_numpy.log = Mock(side_effect=lambda x: 0.1)
mock_numpy.log2 = Mock(side_effect=lambda x: 0.1)
mock_numpy.max = Mock(return_value=1.0)
mock_numpy.min = Mock(return_value=0.0)
mock_numpy.abs = Mock(side_effect=abs)
mock_numpy.arange = Mock(side_effect=lambda n: list(range(n)))
mock_numpy.arcsin = Mock(side_effect=lambda x: x)
mock_numpy.percentile = Mock(return_value=0.5)

sys.modules['numpy'] = mock_numpy

# Mock torch
mock_torch = types.ModuleType('torch')
mock_torch.cuda = Mock()
mock_torch.cuda.is_available = Mock(return_value=False)
mock_torch.float16 = 'float16'
mock_torch.float32 = 'float32'
mock_torch.no_grad = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
mock_torch.log_softmax = Mock(return_value=Mock(item=Mock(return_value=0.1)))

sys.modules['torch'] = mock_torch

# Mock other dependencies
sys.modules['scipy'] = Mock()
sys.modules['scipy.stats'] = Mock()
sys.modules['scipy.stats'].wasserstein_distance = Mock(return_value=0.5)
sys.modules['scipy.stats'].ks_2samp = Mock(return_value=(0.1, 0.05))
sys.modules['scipy.stats'].rankdata = Mock(return_value=[1, 2, 3])
sys.modules['scipy.stats'].mannwhitneyu = Mock(return_value=(100, 0.05))
sys.modules['scipy.stats'].chi2_contingency = Mock(return_value=(5.0, 0.05, 2, [[1, 2], [3, 4]]))
sys.modules['scipy.stats'].chi2 = Mock(cdf=Mock(return_value=0.95))
sys.modules['scipy.stats'].binom_test = Mock(return_value=0.05)
sys.modules['scipy.stats'].wilcoxon = Mock(return_value=(50, 0.03))
sys.modules['scipy.stats'].norm = Mock(ppf=Mock(return_value=1.96), cdf=Mock(return_value=0.95))
sys.modules['scipy.stats'].beta = Mock()

sys.modules['sklearn'] = Mock()
sys.modules['sklearn.metrics'] = Mock()
sys.modules['sklearn.metrics.pairwise'] = Mock()
sys.modules['sklearn.metrics.pairwise'].cosine_similarity = Mock(return_value=[[0.9]])
sys.modules['sklearn.cluster'] = Mock()

sys.modules['statsmodels'] = Mock()
sys.modules['statsmodels.stats'] = Mock()
sys.modules['statsmodels.stats.multitest'] = Mock()
sys.modules['statsmodels.stats.multitest'].multipletests = Mock(return_value=(None, [0.01, 0.02], None, None))
sys.modules['statsmodels.stats.power'] = Mock()
sys.modules['statsmodels.stats.power'].TTestPower = Mock(return_value=Mock(solve_power=Mock(return_value=0.8)))

sys.modules['matplotlib'] = Mock()
sys.modules['matplotlib.pyplot'] = Mock()
sys.modules['seaborn'] = Mock()
sys.modules['networkx'] = Mock()
sys.modules['plotly'] = Mock()
sys.modules['plotly.graph_objects'] = Mock()
sys.modules['plotly.express'] = Mock()
sys.modules['plotly.subplots'] = Mock()

sys.modules['transformers'] = Mock()
sys.modules['tqdm'] = Mock()
sys.modules['tqdm'].tqdm = lambda x, **kwargs: x

sys.modules['psutil'] = Mock()
sys.modules['difflib'] = Mock()
sys.modules['difflib'].SequenceMatcher = Mock(return_value=Mock(ratio=Mock(return_value=0.8)))

# Mock circuit_tracer
mock_circuit_tracer = types.ModuleType('circuit_tracer')
mock_circuit_tracer.Graph = Mock
mock_circuit_tracer.CircuitTracer = Mock
mock_circuit_tracer.ReplacementModel = Mock

sys.modules['circuit_tracer'] = mock_circuit_tracer

# Now try importing our modules
print("\nTesting imports with mocked dependencies...")

try:
    from mechanistic_discovery.circuit_analysis import (
        FeatureDifference,
        ConnectionDifference,
        CircuitDifference
    )
    print("✓ Basic data structures import successfully")
except ImportError as e:
    print(f"✗ Failed to import basic structures: {e}")

try:
    from mechanistic_discovery.circuit_analysis import (
        CircuitAwareModel,
        CircuitTracingConfig,
        DifferentialCircuitAnalyzer,
        CircuitCache,
        FeatureInterpreter
    )
    print("✓ Circuit analysis modules import successfully")
except ImportError as e:
    print(f"✗ Failed to import circuit analysis: {e}")

try:
    from mechanistic_discovery.hypothesis_generation import (
        PatternRecognizer,
        CircuitPattern,
        PatternType,
        HypothesisGenerator,
        BehavioralHypothesis,
        HypothesisType
    )
    print("✓ Hypothesis generation modules import successfully")
except ImportError as e:
    print(f"✗ Failed to import hypothesis generation: {e}")

try:
    from mechanistic_discovery.behavioral_validation import (
        TestGenerator,
        TestCase,
        BehavioralValidator,
        ValidationResult,
        StatisticalValidator,
        TestingProcedure
    )
    print("✓ Behavioral validation modules import successfully")
except ImportError as e:
    print(f"✗ Failed to import behavioral validation: {e}")

try:
    from mechanistic_discovery.integration import (
        MechanisticBehavioralAnalyzer,
        AnalysisConfig,
        AnalysisReport,
        ActiveCircuitExplorer
    )
    print("✓ Integration modules import successfully")
except ImportError as e:
    print(f"✗ Failed to import integration: {e}")

print("\nTesting basic functionality...")

# Test FeatureDifference
try:
    feat_diff = FeatureDifference(
        layer=10,
        position=5,
        feature_idx=123,
        base_activation=0.8,
        intervention_activation=0.1
    )
    assert feat_diff.activation_delta == -0.7
    assert feat_diff.is_suppressed_feature() == True
    print("✓ FeatureDifference works correctly")
except Exception as e:
    print(f"✗ FeatureDifference test failed: {e}")

# Test ConnectionDifference
try:
    conn_diff = ConnectionDifference(
        source="feature_L10_F123",
        target="feature_L11_F456",
        base_weight=0.5,
        intervention_weight=0.1,
        source_type="feature",
        target_type="feature"
    )
    assert conn_diff.weight_change == 0.4
    print("✓ ConnectionDifference works correctly")
except Exception as e:
    print(f"✗ ConnectionDifference test failed: {e}")

# Test PatternRecognizer
try:
    recognizer = PatternRecognizer()
    
    # Create a mock circuit difference
    mock_base_graph = Mock()
    mock_base_graph.active_features = [[10, 5, 123], [11, 3, 456]]
    mock_base_graph.adjacency_matrix = Mock()
    mock_base_graph.n_pos = 10
    mock_base_graph.logit_tokens = [1, 2, 3]
    
    mock_int_graph = Mock()
    mock_int_graph.active_features = [[10, 5, 123], [12, 2, 789]]
    mock_int_graph.adjacency_matrix = Mock()
    mock_int_graph.n_pos = 10
    mock_int_graph.logit_tokens = [1, 2, 3]
    
    circuit_diff = CircuitDifference(
        prompt="Test prompt",
        feature_differences=[
            FeatureDifference(10, 5, 123, 0.8, 0.1),
            FeatureDifference(11, 3, 456, 0.7, 0.0)
        ],
        connection_differences=[],
        statistics={"test": 1.0},
        base_graph=mock_base_graph,
        intervention_graph=mock_int_graph
    )
    
    patterns = recognizer.identify_patterns(circuit_diff)
    print(f"✓ PatternRecognizer works (found {len(patterns)} patterns)")
except Exception as e:
    print(f"✗ PatternRecognizer test failed: {e}")
    import traceback
    traceback.print_exc()

# Test HypothesisGenerator
try:
    generator = HypothesisGenerator()
    
    # Create test patterns
    test_pattern = CircuitPattern(
        pattern_type=PatternType.SAFETY_DEGRADATION,
        changes=[
            FeatureDifference(10, 5, 123, 0.8, 0.1),
            FeatureDifference(10, 6, 124, 0.7, 0.0)
        ],
        consistency=0.8,
        importance_score=0.9
    )
    
    hypotheses = generator.generate_hypotheses([test_pattern])
    print(f"✓ HypothesisGenerator works (generated {len(hypotheses)} hypotheses)")
except Exception as e:
    print(f"✗ HypothesisGenerator test failed: {e}")
    import traceback
    traceback.print_exc()

# Test TestGenerator
try:
    test_gen = TestGenerator()
    
    hypothesis = BehavioralHypothesis(
        hypothesis_type=HypothesisType.SAFETY_DEGRADATION,
        description="Test hypothesis",
        circuit_evidence=[],
        feature_evidence=[],
        confidence=0.8
    )
    
    test_cases = test_gen.generate_tests(hypothesis, n_tests=5)
    print(f"✓ TestGenerator works (generated {len(test_cases)} test cases)")
except Exception as e:
    print(f"✗ TestGenerator test failed: {e}")
    import traceback
    traceback.print_exc()

print("\nChecking for structural issues...")

# Check if classes have required methods
try:
    # Check CircuitAwareModel
    config = CircuitTracingConfig(
        model_name="test",
        transcoder_set="test",
        device="cpu"
    )
    
    # We can't instantiate it without real models, but we can check the class exists
    assert CircuitAwareModel is not None
    print("✓ CircuitAwareModel class exists")
    
    # Check DifferentialCircuitAnalyzer
    assert DifferentialCircuitAnalyzer is not None
    print("✓ DifferentialCircuitAnalyzer class exists")
    
    # Check other key classes
    assert BehavioralValidator is not None
    assert StatisticalValidator is not None
    assert ActiveCircuitExplorer is not None
    assert MechanisticBehavioralAnalyzer is not None
    print("✓ All main classes exist")
    
except Exception as e:
    print(f"✗ Class structure check failed: {e}")

print("\n" + "="*60)
print("MOCK DEPENDENCY TEST COMPLETE")
print("="*60)