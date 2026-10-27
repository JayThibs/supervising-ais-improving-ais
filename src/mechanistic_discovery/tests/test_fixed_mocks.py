"""
Test with properly mocked dependencies to identify actual bugs.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import types

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

print("Testing with properly mocked dependencies...")

# Create comprehensive mock numpy
mock_numpy = types.ModuleType('numpy')
mock_numpy.array = Mock(side_effect=lambda x: x)
mock_numpy.random = Mock()
mock_numpy.random.randn = Mock(return_value=[[0.1, 0.2], [0.3, 0.4]])
mock_numpy.random.rand = Mock(return_value=0.5)
mock_numpy.random.choice = Mock(side_effect=lambda x, **kwargs: x[0] if hasattr(x, '__getitem__') else x)
mock_numpy.mean = Mock(return_value=0.5)
mock_numpy.std = Mock(return_value=0.1)
mock_numpy.zeros = Mock(return_value=[[0, 0], [0, 0]])
mock_numpy.sqrt = Mock(side_effect=lambda x: x ** 0.5 if isinstance(x, (int, float)) else x)
mock_numpy.log = Mock(side_effect=lambda x: 0.1)
mock_numpy.log2 = Mock(side_effect=lambda x: 0.1)
mock_numpy.max = Mock(return_value=1.0)
mock_numpy.min = Mock(return_value=0.0)
mock_numpy.abs = Mock(side_effect=abs)
mock_numpy.arange = Mock(side_effect=lambda n: list(range(n)))
mock_numpy.arcsin = Mock(side_effect=lambda x: x)
mock_numpy.percentile = Mock(return_value=0.5)
mock_numpy.sum = sum
mock_numpy.ndarray = list  # Simple replacement

sys.modules['numpy'] = mock_numpy
sys.modules['np'] = mock_numpy

# Mock torch properly
mock_torch = types.ModuleType('torch')
mock_torch.cuda = Mock()
mock_torch.cuda.is_available = Mock(return_value=False)
mock_torch.float16 = 'float16'
mock_torch.float32 = 'float32'
mock_torch.no_grad = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
mock_torch.log_softmax = Mock(return_value=Mock(item=Mock(return_value=0.1)))
mock_torch.Tensor = list
mock_torch.nn = Mock()
mock_torch.nn.Module = object

sys.modules['torch'] = mock_torch

# Mock transformers
mock_transformers = types.ModuleType('transformers')
mock_transformers.AutoModel = Mock()
mock_transformers.AutoTokenizer = Mock()
mock_transformers.AutoConfig = Mock()
mock_transformers.AutoModelForCausalLM = Mock()

sys.modules['transformers'] = mock_transformers

# Mock other statistical libraries
sys.modules['scipy'] = Mock()
sys.modules['scipy.stats'] = Mock()
sys.modules['sklearn'] = Mock()
sys.modules['sklearn.metrics'] = Mock()
sys.modules['sklearn.metrics.pairwise'] = Mock()
sys.modules['sklearn.cluster'] = Mock()
sys.modules['statsmodels'] = Mock()
sys.modules['statsmodels.stats'] = Mock()
sys.modules['statsmodels.stats.multitest'] = Mock()
sys.modules['statsmodels.stats.power'] = Mock()

# Mock visualization libraries
sys.modules['matplotlib'] = Mock()
sys.modules['matplotlib.pyplot'] = Mock()
sys.modules['seaborn'] = Mock()
sys.modules['networkx'] = Mock()
sys.modules['plotly'] = Mock()
sys.modules['plotly.graph_objects'] = Mock()
sys.modules['plotly.express'] = Mock()
sys.modules['plotly.subplots'] = Mock()

# Mock other utilities
sys.modules['tqdm'] = Mock()
sys.modules['tqdm'].tqdm = lambda x, **kwargs: x
sys.modules['psutil'] = Mock()

# Mock circuit_tracer with all required components
mock_circuit_tracer = types.ModuleType('circuit_tracer')

# Create Graph mock
class MockGraph:
    def __init__(self):
        self.active_features = [[10, 5, 123], [11, 3, 456]]
        self.adjacency_matrix = Mock()
        self.n_pos = 10
        self.logit_tokens = [1, 2, 3]
        self.logit_probabilities = [0.7, 0.2, 0.1]
        self.input_string = "Test prompt"

mock_circuit_tracer.Graph = MockGraph
mock_circuit_tracer.ReplacementModel = Mock()
mock_circuit_tracer.attribute = Mock(return_value=MockGraph())
mock_circuit_tracer.CircuitTracer = Mock()

sys.modules['circuit_tracer'] = mock_circuit_tracer

# Now import our modules with proper error handling
print("\nImporting mechanistic discovery modules...")

bugs_found = []

# Test basic structures
try:
    from mechanistic_discovery.circuit_analysis.circuit_tracer_wrapper import (
        FeatureDifference,
        ConnectionDifference,
        CircuitDifference,
        CircuitTracingConfig
    )
    print("✓ Basic data structures imported")
    
    # Test FeatureDifference
    feat = FeatureDifference(10, 5, 123, 0.8, 0.1)
    assert abs(feat.activation_delta - (-0.7)) < 0.001
    print("✓ FeatureDifference works correctly")
    
except Exception as e:
    bugs_found.append(f"Basic structures: {e}")
    print(f"✗ Basic structures failed: {e}")

# Test circuit analysis modules
try:
    from mechanistic_discovery.circuit_analysis import (
        CircuitAwareModel,
        DifferentialCircuitAnalyzer,
        CircuitCache,
        FeatureInterpreter
    )
    print("✓ Circuit analysis modules imported")
    
    # Test CircuitCache initialization
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CircuitCache(cache_dir=tmpdir)
        print("✓ CircuitCache initializes correctly")
        
except Exception as e:
    bugs_found.append(f"Circuit analysis: {e}")
    print(f"✗ Circuit analysis failed: {e}")
    import traceback
    traceback.print_exc()

# Test hypothesis generation
try:
    from mechanistic_discovery.hypothesis_generation import (
        PatternRecognizer,
        CircuitPattern, 
        PatternType,
        HypothesisGenerator,
        BehavioralHypothesis,
        HypothesisType
    )
    print("✓ Hypothesis generation modules imported")
    
    # Test pattern recognition
    recognizer = PatternRecognizer()
    print("✓ PatternRecognizer initializes")
    
    # Test hypothesis generator
    generator = HypothesisGenerator()
    print("✓ HypothesisGenerator initializes")
    
except Exception as e:
    bugs_found.append(f"Hypothesis generation: {e}")
    print(f"✗ Hypothesis generation failed: {e}")
    import traceback
    traceback.print_exc()

# Test behavioral validation
try:
    from mechanistic_discovery.behavioral_validation import (
        TestGenerator,
        TestCase,
        BehavioralValidator,
        ValidationResult,
        StatisticalValidator,
        TestingProcedure,
        TestResult
    )
    print("✓ Behavioral validation modules imported")
    
    # Test generators
    test_gen = TestGenerator()
    print("✓ TestGenerator initializes")
    
    # Test statistical validator
    stat_val = StatisticalValidator()
    print("✓ StatisticalValidator initializes")
    
except Exception as e:
    bugs_found.append(f"Behavioral validation: {e}")
    print(f"✗ Behavioral validation failed: {e}")
    import traceback
    traceback.print_exc()

# Test integration modules
try:
    from mechanistic_discovery.integration import (
        MechanisticBehavioralAnalyzer,
        AnalysisConfig,
        AnalysisReport,
        ActiveCircuitExplorer,
        ExplorationStrategy
    )
    print("✓ Integration modules imported")
    
    # Test config creation
    config = AnalysisConfig(
        base_model_name="test",
        intervention_model_name="test2",
        transcoder_set="test"
    )
    print("✓ AnalysisConfig works")
    
except Exception as e:
    bugs_found.append(f"Integration: {e}")
    print(f"✗ Integration failed: {e}")
    import traceback
    traceback.print_exc()

# Test utilities
try:
    from mechanistic_discovery.utils import (
        CircuitVisualizer,
        PromptGenerator,
        PromptDiversityScorer,
        get_model_info,
        check_model_compatibility
    )
    print("✓ Utility modules imported")
    
    # Test prompt generator
    prompt_gen = PromptGenerator()
    prompts = prompt_gen.generate_prompts(5)
    assert len(prompts) == 5
    print("✓ PromptGenerator works")
    
except Exception as e:
    bugs_found.append(f"Utilities: {e}")
    print(f"✗ Utilities failed: {e}")
    import traceback
    traceback.print_exc()

# Now test a simple workflow
print("\nTesting basic workflow...")

try:
    # Create mock models
    base_model = Mock()
    base_model.extract_circuit = Mock(return_value=MockGraph())
    base_model.get_model_info = Mock(return_value={
        'n_layers': 12,
        'd_model': 768,
        'd_transcoder': 768 * 8
    })
    base_model.n_layers = 12
    
    int_model = Mock()
    int_model.extract_circuit = Mock(return_value=MockGraph())
    int_model.get_model_info = Mock(return_value={
        'n_layers': 12,
        'd_model': 768,
        'd_transcoder': 768 * 8
    })
    int_model.n_layers = 12
    
    # Test differential analyzer
    analyzer = DifferentialCircuitAnalyzer(base_model, int_model)
    
    # Create a simple circuit difference
    diff = analyzer.analyze_prompt("Test prompt")
    print(f"✓ Basic analysis works (found {len(diff.feature_differences)} differences)")
    
except Exception as e:
    bugs_found.append(f"Workflow test: {e}")
    print(f"✗ Workflow test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*60)
print("BUG REPORT")
print("="*60)

if bugs_found:
    print(f"Found {len(bugs_found)} bugs:\n")
    for i, bug in enumerate(bugs_found, 1):
        print(f"{i}. {bug}")
else:
    print("✅ No bugs found! All modules load and initialize correctly.")

print("\nNote: This test only checks basic structure and initialization.")
print("Full functionality requires the actual dependencies.")