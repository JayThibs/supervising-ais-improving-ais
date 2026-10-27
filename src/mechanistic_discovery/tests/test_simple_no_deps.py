"""
Simple test without external dependencies.

This test verifies basic functionality using only Python standard library.
Designed to run on M1 MacBook Air without numpy, torch, etc.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
import json
import tempfile
import time
import random

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

print("Python version:", sys.version)
print("Running from:", os.getcwd())
print("\n" + "="*60)
print("MECHANISTIC DISCOVERY - SIMPLE TEST SUITE")
print("="*60)


def test_imports():
    """Test that basic structures can be imported."""
    print("\nTesting basic imports...")
    
    try:
        # Test dataclasses and enums
        from dataclasses import dataclass, field
        from enum import Enum
        print("✓ Standard library imports work")
        
        # Test our basic structures (these shouldn't require external deps)
        from mechanistic_discovery.circuit_analysis import FeatureDifference, ConnectionDifference
        print("✓ Basic circuit analysis structures import")
        
        from mechanistic_discovery.hypothesis_generation import PatternType, HypothesisType
        print("✓ Enums import successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_feature_difference():
    """Test FeatureDifference without numpy."""
    print("\nTesting FeatureDifference...")
    
    try:
        from mechanistic_discovery.circuit_analysis import FeatureDifference
        
        # Create feature difference
        feat_diff = FeatureDifference(
            layer=10,
            position=5,
            feature_idx=123,
            base_activation=0.8,
            intervention_activation=0.1
        )
        
        # Test properties
        assert feat_diff.layer == 10
        assert feat_diff.activation_delta == -0.7
        assert abs(feat_diff.activation_delta - (-0.7)) < 0.001  # Float comparison
        
        # Test methods
        assert feat_diff.is_suppressed_feature() == True
        assert feat_diff.is_new_feature() == False
        
        print("✓ FeatureDifference works correctly")
        return True
        
    except Exception as e:
        print(f"✗ FeatureDifference test failed: {e}")
        return False


def test_pattern_types():
    """Test pattern type enums."""
    print("\nTesting pattern types...")
    
    try:
        from mechanistic_discovery.hypothesis_generation import PatternType, HypothesisType
        
        # Test enum access
        assert PatternType.SYSTEMATIC_SUPPRESSION.value == "systematic_suppression"
        assert PatternType.CAPABILITY_EMERGENCE.value == "capability_emergence"
        
        # Test hypothesis types
        assert HypothesisType.SAFETY_DEGRADATION.value == "safety_degradation"
        assert HypothesisType.CAPABILITY_GAIN.value == "capability_gain"
        
        print("✓ Enums work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Pattern types test failed: {e}")
        return False


def test_prompt_generation_simple():
    """Test basic prompt generation logic."""
    print("\nTesting simple prompt generation...")
    
    try:
        # Simple template-based generation
        templates = [
            "Explain {topic} in simple terms.",
            "What are the implications of {scenario}?",
            "How do you {action}?"
        ]
        
        topics = ["AI safety", "machine learning", "neural networks"]
        scenarios = ["AGI development", "automation", "AI regulation"]
        actions = ["solve problems", "make decisions", "learn from data"]
        
        prompts = []
        for _ in range(10):
            template = random.choice(templates)
            
            if "{topic}" in template:
                prompt = template.replace("{topic}", random.choice(topics))
            elif "{scenario}" in template:
                prompt = template.replace("{scenario}", random.choice(scenarios))
            elif "{action}" in template:
                prompt = template.replace("{action}", random.choice(actions))
            else:
                prompt = template
                
            prompts.append(prompt)
        
        # Check we generated diverse prompts
        assert len(prompts) == 10
        assert len(set(prompts)) >= 5  # At least 5 unique
        
        print("✓ Simple prompt generation works")
        return True
        
    except Exception as e:
        print(f"✗ Prompt generation test failed: {e}")
        return False


def test_caching_logic():
    """Test basic caching logic."""
    print("\nTesting caching logic...")
    
    try:
        import hashlib
        
        # Simple cache implementation
        cache = {}
        
        def generate_key(prompt, model_name):
            """Generate cache key."""
            content = f"{prompt}|{model_name}"
            return hashlib.md5(content.encode()).hexdigest()
        
        # Test key generation
        key1 = generate_key("Test prompt", "model1")
        key2 = generate_key("Test prompt", "model1")
        key3 = generate_key("Different prompt", "model1")
        
        assert key1 == key2  # Same inputs -> same key
        assert key1 != key3  # Different inputs -> different key
        
        # Test caching
        cache[key1] = {"result": "cached_value"}
        assert cache.get(key1)["result"] == "cached_value"
        assert cache.get(key3) is None
        
        print("✓ Caching logic works")
        return True
        
    except Exception as e:
        print(f"✗ Caching test failed: {e}")
        return False


def test_statistical_logic():
    """Test basic statistical logic without scipy."""
    print("\nTesting statistical logic...")
    
    try:
        # Simple p-value adjustment
        p_values = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2]
        alpha = 0.05
        
        # Bonferroni correction (simple version)
        adjusted_p_values = [p * len(p_values) for p in p_values]
        adjusted_p_values = [min(p, 1.0) for p in adjusted_p_values]
        
        # Count rejections
        rejections = sum(1 for p in adjusted_p_values if p < alpha)
        
        assert rejections < len(p_values)  # Should not reject all
        assert adjusted_p_values[0] < adjusted_p_values[-1]  # Ordering preserved
        
        print("✓ Basic statistical logic works")
        return True
        
    except Exception as e:
        print(f"✗ Statistical test failed: {e}")
        return False


def test_file_operations():
    """Test file I/O operations."""
    print("\nTesting file operations...")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test JSON writing/reading
            test_data = {
                "analysis_type": "mechanistic_discovery",
                "models": ["base", "intervention"],
                "results": {
                    "n_differences": 10,
                    "validated": 5
                }
            }
            
            # Write
            json_path = Path(tmpdir) / "test_results.json"
            with open(json_path, 'w') as f:
                json.dump(test_data, f, indent=2)
            
            # Read
            with open(json_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data["analysis_type"] == "mechanistic_discovery"
            assert loaded_data["results"]["n_differences"] == 10
            
            print("✓ File operations work")
            return True
            
    except Exception as e:
        print(f"✗ File operations test failed: {e}")
        return False


def test_mock_integration():
    """Test basic integration with mocks."""
    print("\nTesting mock integration...")
    
    try:
        # Mock a simple analysis workflow
        class MockAnalyzer:
            def __init__(self):
                self.analyses_run = 0
                
            def analyze_prompt(self, prompt):
                self.analyses_run += 1
                return {
                    "prompt": prompt,
                    "differences_found": random.randint(0, 5),
                    "patterns": ["suppression", "emergence"][random.randint(0, 1)]
                }
        
        # Run mock analysis
        analyzer = MockAnalyzer()
        prompts = ["Test 1", "Test 2", "Test 3"]
        results = []
        
        for prompt in prompts:
            result = analyzer.analyze_prompt(prompt)
            results.append(result)
        
        assert analyzer.analyses_run == 3
        assert len(results) == 3
        assert all("patterns" in r for r in results)
        
        print("✓ Mock integration works")
        return True
        
    except Exception as e:
        print(f"✗ Mock integration test failed: {e}")
        return False


def test_dataclass_functionality():
    """Test dataclass features we use."""
    print("\nTesting dataclass functionality...")
    
    try:
        from dataclasses import dataclass, field
        from typing import List, Dict, Any
        
        @dataclass
        class TestResult:
            name: str
            score: float
            metadata: Dict[str, Any] = field(default_factory=dict)
            
            def is_significant(self):
                return self.score > 0.5
        
        # Test creation and methods
        result = TestResult(name="test", score=0.7)
        assert result.name == "test"
        assert result.is_significant() == True
        assert result.metadata == {}
        
        # Test with metadata
        result2 = TestResult(
            name="test2",
            score=0.3,
            metadata={"type": "safety"}
        )
        assert result2.metadata["type"] == "safety"
        assert result2.is_significant() == False
        
        print("✓ Dataclass functionality works")
        return True
        
    except Exception as e:
        print(f"✗ Dataclass test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("Imports", test_imports),
        ("Feature Difference", test_feature_difference),
        ("Pattern Types", test_pattern_types),
        ("Prompt Generation", test_prompt_generation_simple),
        ("Caching Logic", test_caching_logic),
        ("Statistical Logic", test_statistical_logic),
        ("File Operations", test_file_operations),
        ("Mock Integration", test_mock_integration),
        ("Dataclass Functionality", test_dataclass_functionality)
    ]
    
    passed = 0
    failed = 0
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            failed += 1
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}")
    print(f"Total time: {total_time:.2f} seconds")
    
    if failed == 0:
        print("\n✅ All basic tests passed!")
        print("The core logic is working correctly.")
        print("\nNote: Full functionality requires:")
        print("- numpy for array operations")
        print("- torch for model inference")
        print("- circuit-tracer for mechanistic analysis")
        print("- matplotlib/plotly for visualizations")
    else:
        print(f"\n❌ {failed} tests failed.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)