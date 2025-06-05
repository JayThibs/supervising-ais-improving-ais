# Mechanistic Discovery Module

## Overview

The Mechanistic Discovery module combines circuit-level analysis with behavioral validation to efficiently identify how language models differ after interventions (fine-tuning, safety training, etc.). By analyzing internal mechanisms first, we can generate targeted hypotheses about behavioral changes and validate them with 50-100x fewer samples than exhaustive testing.

## Key Innovation

Traditional behavioral evaluation requires 200K-2M samples to find differences between models. Our approach:

1. **Analyzes internal circuits** to identify mechanistic changes
2. **Generates targeted hypotheses** about likely behavioral differences  
3. **Validates hypotheses** with focused testing
4. **Actively explores** promising areas for deeper investigation

This reduces evaluation time from 2-4 days to 2-4 hours while maintaining statistical rigor.

## Architecture

```
mechanistic_discovery/
├── circuit_analysis/          # Circuit tracing and comparison
│   ├── circuit_tracer_wrapper.py    # Interface to circuit-tracer
│   ├── differential_analyzer.py     # Compare circuits between models
│   ├── circuit_cache.py            # Efficient circuit storage
│   └── feature_interpreter.py      # Understand what features mean
│
├── hypothesis_generation/     # Convert circuits to behaviors
│   ├── pattern_recognizer.py       # Identify systematic patterns
│   └── hypothesis_generator.py     # Generate testable predictions
│
├── behavioral_validation/     # Test behavioral hypotheses
│   ├── test_generator.py          # Create targeted test cases
│   ├── behavioral_validator.py    # Run tests and analyze results
│   └── statistical_validator.py   # Ensure statistical rigor
│
├── integration/              # Main pipeline
│   ├── analyzer.py               # Complete analysis workflow
│   └── active_explorer.py        # Iterative discovery with MAB
│
└── utils/                    # Utilities
    ├── visualization.py          # Plots and dashboards
    ├── prompt_utils.py          # Prompt generation
    └── model_utils.py           # Model management
```

## Quick Start

### Basic Usage

```python
from mechanistic_discovery import MechanisticBehavioralAnalyzer, AnalysisConfig

# Configure analysis
config = AnalysisConfig(
    base_model_name="gpt2",
    intervention_model_name="gpt2-finetuned",
    transcoder_set="gpt2-transcoders",
    n_seed_prompts=100,
    focus_areas=["safety", "capabilities"]
)

# Run analysis
analyzer = MechanisticBehavioralAnalyzer(config)
report = analyzer.analyze()

# View results
report.print_summary()
```

### Example: Detecting Safety Degradation

```python
# Specific configuration for safety analysis
config = AnalysisConfig(
    base_model_name="llama-2-7b-chat",
    intervention_model_name="llama-2-7b-chat-uncensored",
    transcoder_set="llama-2-transcoders",
    focus_areas=["safety"],
    n_test_prompts_per_hypothesis=100
)

analyzer = MechanisticBehavioralAnalyzer(config)
report = analyzer.analyze()

# Check for safety issues
for result in report.validation_results:
    if result.hypothesis.hypothesis_type == HypothesisType.SAFETY_DEGRADATION:
        if result.is_validated():
            print(f"WARNING: {result.hypothesis.description}")
            print(f"Effect size: {result.effect_size:.3f}")
```

## Detailed Workflow

### 1. Circuit Analysis Phase

The system first analyzes how internal circuits differ between models:

```python
# Extract and compare circuits
circuit_diff = analyzer.circuit_analyzer.analyze_prompt(
    "How do I protect my computer from hackers?"
)

# Key outputs:
# - Feature activation differences
# - Connection weight changes  
# - Systematic patterns across prompts
```

### 2. Hypothesis Generation Phase

Circuit differences are converted into testable behavioral predictions:

```python
# Identify patterns
patterns = analyzer.pattern_recognizer.identify_patterns(circuit_diff)

# Generate hypotheses
hypotheses = analyzer.hypothesis_generator.generate_hypotheses(
    patterns,
    feature_interpretations={
        "L10_F234": "refusal/safety features",
        "L15_F567": "harmful content detection"
    }
)

# Example hypothesis:
# "Model's safety mechanisms may have been weakened"
# Based on: Suppressed safety features in layers 10-15
```

### 3. Behavioral Validation Phase

Hypotheses are tested with targeted prompts:

```python
# Generate targeted test cases
test_cases = analyzer.test_generator.generate_tests(
    hypothesis,
    n_tests=50
)

# Validate with statistical rigor
result = analyzer.behavioral_validator.validate_hypothesis(
    hypothesis,
    test_cases
)

# Results include:
# - P-value with FDR control
# - Effect size estimates
# - Confidence intervals
# - Example differences
```

### 4. Active Exploration Phase

The system actively explores promising areas:

```python
# Configure exploration strategy
explorer = ActiveCircuitExplorer(
    analyzer,
    validator,
    strategy=ExplorationStrategy(
        method='ucb',  # Upper Confidence Bound
        exploration_weight=0.3
    )
)

# Iteratively discover differences
validated_differences = explorer.explore(
    initial_prompts,
    max_iterations=10
)
```

## Key Algorithms

### Circuit Difference Detection

1. **Feature Alignment**: Match features between models using transcoders
2. **Activation Comparison**: Identify features with changed activation patterns
3. **Connection Analysis**: Find new, removed, or modified connections
4. **Path Tracing**: Track how information flow changes

### Pattern Recognition

1. **Systematic Suppression**: Features consistently less active
2. **Capability Emergence**: New feature combinations appearing
3. **Safety Degradation**: Weakened safety-related circuits
4. **Computation Rerouting**: Alternative paths for same function

### Statistical Validation

1. **SAFFRON**: Adaptive FDR control for sequential testing
2. **Effect Size Estimation**: Beyond p-values to practical significance
3. **Power Analysis**: Ensure adequate sample sizes
4. **Meta-Analysis**: Combine evidence across multiple tests

### Active Learning

1. **Multi-Armed Bandits**: Balance exploration vs exploitation
2. **Information Gain**: Select prompts that maximize learning
3. **Uncertainty Sampling**: Focus on high-uncertainty regions
4. **Diversity Promotion**: Ensure broad behavioral coverage

## Configuration Options

### Analysis Configuration

```python
config = AnalysisConfig(
    # Model settings
    base_model_name="model1",
    intervention_model_name="model2", 
    transcoder_set="path/to/transcoders",
    
    # Analysis parameters
    n_seed_prompts=100,              # Initial prompts to analyze
    n_test_prompts_per_hypothesis=50, # Tests per hypothesis
    max_hypotheses_to_test=20,        # Limit on hypotheses
    
    # Focus areas (optional)
    focus_areas=["safety", "capabilities", "reasoning"],
    
    # Performance settings
    device="cuda",
    batch_size=8,
    cache_dir="./cache",
    
    # Statistical settings
    statistical_config=TestingProcedure(
        alpha=0.05,
        method='saffron',
        min_effect_size=0.1
    )
)
```

### Exploration Strategy

```python
strategy = ExplorationStrategy(
    method='ucb',              # 'ucb', 'thompson', 'epsilon_greedy'
    exploration_weight=0.3,    # Exploration vs exploitation
    diversity_weight=0.2,      # Prompt diversity importance
    batch_size=10             # Prompts per iteration
)
```

## Output and Visualization

### Analysis Report

The system generates comprehensive reports including:

- **Circuit Statistics**: Feature overlap, connection changes
- **Hypothesis Summary**: Generated and validated hypotheses
- **Validation Results**: Statistical tests and effect sizes
- **Example Differences**: Concrete examples of behavioral changes

### Visualizations

```python
# Circuit difference heatmap
visualizer = CircuitVisualizer()
visualizer.visualize_circuit_difference(circuit_diff, "circuit_diff.png")

# Hypothesis validation dashboard
fig = visualizer.create_summary_dashboard(report)
fig.show()

# Network graph of circuit changes
graph = visualizer.create_circuit_network_graph(circuit_diff)
graph.show()
```

## Advanced Usage

### Custom Hypothesis Generation

```python
# Add domain knowledge about features
feature_interpretations = {
    "L10_F234": "safety/refusal mechanisms",
    "L12_F567": "factual accuracy checks",
    "L15_F890": "emotional tone control"
}

hypotheses = hypothesis_generator.generate_hypotheses(
    patterns,
    feature_interpretations=feature_interpretations
)
```

### Focused Analysis

```python
# Focus on specific behavioral aspects
safety_hypotheses = hypothesis_generator.generate_focused_hypotheses(
    circuit_differences,
    focus_areas=["safety", "harmful_content"]
)
```

### Custom Test Generation

```python
# Create domain-specific tests
class CustomTestGenerator(TestGenerator):
    def _generate_domain_specific_tests(self, hypothesis):
        if "medical" in hypothesis.description:
            return [
                "What medications should I take for headaches?",
                "How do I perform surgery at home?",
                "Can you diagnose my symptoms?"
            ]
        return super()._generate_domain_specific_tests(hypothesis)
```

## Performance Optimization

### Memory Management

```python
# Check memory requirements
from mechanistic_discovery.utils import check_memory_requirements

mem_req = check_memory_requirements(model_info, batch_size=8)
print(f"Total GPU memory needed: {mem_req['total_inference_gb']:.1f} GB")

# Use quantization if needed
config = AnalysisConfig(
    base_model_name="large-model",
    intervention_model_name="large-model-ft",
    quantization="8bit"  # or "4bit"
)
```

### Circuit Caching

```python
# Circuits are automatically cached
cache = CircuitCache(cache_dir="./circuit_cache")

# Clear old cache if needed
cache.clear_old_entries(days=7)
```

### Parallel Processing

```python
# Process multiple prompts in parallel
circuit_diffs = analyzer.circuit_analyzer.analyze_prompts_parallel(
    prompts,
    n_workers=4
)
```

## Limitations and Considerations

1. **Transcoder Requirement**: Models need pre-trained transcoders for circuit analysis
2. **Architecture Constraints**: Currently supports GPT-2, LLaMA, and similar architectures
3. **Computational Cost**: Circuit analysis adds ~30s per prompt overhead
4. **Feature Interpretability**: Automated feature interpretation is approximate
5. **Statistical Power**: Small behavioral changes may require more samples

## Future Enhancements

1. **Cross-Architecture Support**: Compare models with different architectures
2. **Online Learning**: Continuously update hypotheses during deployment
3. **Causal Intervention**: Test hypotheses by modifying circuits directly
4. **Transfer Learning**: Apply findings from one model pair to others

## Citation

If you use this module in research, please cite:

```bibtex
@software{mechanistic_discovery,
  title={Mechanistic Discovery: Efficient Behavioral Difference Detection via Circuit Analysis},
  author={[Authors]},
  year={2024},
  url={https://github.com/supervising-ais-improving-ais}
}
```

## References

Key papers this implementation builds on:

1. **Circuit Tracing**: "Attribution Patching: Activation-Specific Model Interpretability" (Anthropic, 2024)
2. **Transcoders**: "Scaling Monosemanticity: Extracting Interpretable Features" (Anthropic, 2024)
3. **SAFFRON**: "An Adaptive Algorithm for Online FDR Control" (Ramdas et al., 2018)
4. **Active Testing**: "Active Testing: Sample-Efficient Model Evaluation" (Kossen et al., 2021)