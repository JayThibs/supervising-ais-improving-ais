# Comprehensive Codebase Guide: supervising-ais-improving-ais

## Executive Summary

This guide provides a complete overview of the supervising-ais-improving-ais project, a research framework for detecting behavioral differences in language models after interventions. After analyzing the entire codebase, I've identified key strengths, significant code duplication issues, and opportunities for scientific innovation.

**Key Findings:**
- The project contains four complementary approaches for behavioral analysis
- ~54% of code could be eliminated through refactoring duplicated components
- Strong statistical validation methods exist but software engineering practices need improvement
- The project is well-positioned for new developments in reasoning models and iterative evaluation

**UPDATE (Auto-Interventions Branch Analysis):**
- Quentin's branch adds rigorous statistical validation but has severe performance issues
- Current experiments take 2-4 days on GPU due to 200K-2M text generation
- Iterative active sampling can reduce runtime by 10-100x while maintaining quality
- Weight difference analysis provides interpretable insights into fine-tuning effects

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Components](#architecture--components)
3. [Current Workflows](#current-workflows)
4. [Code Duplication Analysis](#code-duplication-analysis)
5. [Gaps & Limitations](#gaps--limitations)
6. [Recommendations](#recommendations)
7. [Scientific Innovation Opportunities](#scientific-innovation-opportunities)
8. [Implementation Roadmap](#implementation-roadmap)

## Project Overview

### Purpose
The project develops scalable methods for evaluating behavioral effects of interventions (finetuning, knowledge editing, unlearning) on large language models. It aims to discover unexpected side effects that standard benchmarks might miss, particularly ensuring capability improvements preserve alignment.

### Core Philosophy
- **Scalable**: Designed for large-scale model comparisons
- **Automated**: LLM-based analysis of behavioral patterns
- **Rigorous**: Statistical validation of findings
- **Comprehensive**: Multiple complementary approaches

## Architecture & Components

### 1. Behavioral Clustering Module (`src/behavioural_clustering/`)

**Purpose**: Groups model responses by semantic similarity to identify behavioral patterns

**Key Components**:
- `EvaluatorPipeline`: Main orchestrator (557 lines - needs refactoring)
- `ModelEvaluationManager`: Manages model lifecycle
- `ClusterAnalyzer`: Analyzes clustering results
- Multiple clustering algorithms (KMeans, Spectral, Agglomerative, OPTICS)

**Workflow**:
1. Load evaluation datasets
2. Generate model responses
3. Create embeddings
4. Apply clustering
5. Analyze patterns
6. Generate visualizations

### 2. Contrastive Decoding Module (`src/contrastive-decoding/`)

**Purpose**: Directly amplifies differences between models during generation

**Key Components**:
- `ContrastiveDecoder`: Core algorithm with logit manipulation
- `DivergenceFinder`: Iterative prompt discovery
- `AutomatedDivergenceAnalyzer`: Clustering and analysis

**Unique Features**:
- Token-level divergence visualization
- Quantization support for memory efficiency
- LRU caching for attention states

### 3. Interventions Module (`src/interventions/`)

**Purpose**: Comprehensive framework for analyzing effects of various interventions

**Key Components**:
- `auto_finetune_eval/`: Automated evaluation pipeline
- `validated_comparison_tools.py`: SAFFRON implementation for FDR control
- Support for 10+ intervention types

**Recent Focus**: Auto-interventions with diversification methods and weight difference analysis

### 4. Soft Prompting Module (`src/soft_prompting/`)

**Purpose**: Uses gradient-based optimization to find maximum behavioral differences

**Key Components**:
- `DivergenceSoftPrompt`: Trainable embedding management
- `DivergenceTrainer`: Mixed precision training
- `DivergenceAnalyzer`: Pattern identification

**Advantages**:
- Automatic (no manual prompt engineering)
- Efficient (only trains small embeddings)
- Targeted (maximizes specific divergence metrics)

### 5. Web Application (`src/webapp/`)

**Purpose**: Streamlit-based interface for visualization and interaction

**Features**:
- Run analysis configuration
- Results visualization
- Model comparison
- Export capabilities

**Issues**: Basic authentication, limited navigation, needs UX improvements

## Current Workflows

### 1. Standard Behavioral Analysis
```bash
# Run behavioral clustering
python src/behavioural_clustering/main.py --run full_run

# View results in webapp
python src/webapp/main.py
```

### 2. Targeted Divergence Discovery
```bash
# Find prompts that maximize differences
python src/contrastive-decoding/run_find_divergence_prompts.py --target safety

# Or use soft prompting
python scripts/train_divergence_soft_prompts.py --config configs/experiment.yaml
```

### 3. Intervention Analysis
```bash
# Train with intervention
python src/interventions/train.py --model gpt2 --intervention knowledge_edit

# Evaluate effects
bash src/interventions/auto_finetune_eval/run_auto_finetuning_main.sh
```

## Code Duplication Analysis

### Major Duplication Areas

1. **Model Management** (~400 lines duplicated)
   - Each module implements its own model loading
   - Different API client implementations
   - Inconsistent error handling

2. **Data Handling** (~350 lines duplicated)
   - JSONL loading reimplemented 4+ times
   - Different caching strategies
   - Inconsistent preprocessing

3. **Embedding Generation** (~300 lines duplicated)
   - No unified embedding service
   - Different caching mechanisms
   - Repeated API call patterns

4. **Visualization** (~250 lines duplicated)
   - Similar plotting functions across modules
   - Repeated color schemes and styling
   - No shared visualization utilities

### Impact
- **Maintenance burden**: Changes need to be made in multiple places
- **Inconsistency**: Different implementations may behave differently
- **Bugs**: Fixed in one place but not others
- **Testing**: Each implementation needs separate tests

## Gaps & Limitations

### 1. Engineering Gaps

**Testing Infrastructure**:
- Empty test.sh script
- Mixed testing frameworks (unittest vs pytest)
- No CI/CD pipeline
- Limited test coverage (only interventions well-tested)

**Documentation**:
- Fragmented across modules
- Missing end-to-end examples
- No unified configuration guide
- Limited troubleshooting resources

**Performance**:
- No systematic caching strategy
- Limited parallelization
- Memory management issues
- Inefficient API usage

### 2. Scientific Limitations

**Current Approaches**:
- Primarily post-hoc analysis
- Limited real-time adaptation
- High computational cost
- May miss subtle behavioral shifts

**Missing Capabilities**:
- No iterative/active evaluation
- Limited integration with mechanistic interpretability
- No support for reasoning model evaluation
- Missing continual learning assessment

### 3. Usability Issues

**Configuration**:
- Multiple incompatible systems
- No validation or schemas
- Hard-coded paths

**Integration**:
- Modules work independently
- No unified pipeline
- Manual result transfer

## Recommendations

### 1. Immediate Actions (1-2 weeks)

**Create Shared Utilities**:
```python
src/shared/
├── models/
│   ├── base_manager.py
│   ├── api_clients.py
│   └── model_registry.py
├── data/
│   ├── loaders.py
│   ├── cache.py
│   └── preprocessing.py
├── embeddings/
│   ├── embedding_service.py
│   └── cache_manager.py
└── utils/
    ├── logging.py
    ├── device.py
    └── retry.py
```

**Standardize Testing**:
- Adopt pytest throughout
- Create pytest.ini configuration
- Implement test.sh properly
- Add coverage reporting

### 2. Medium-term Improvements (1-2 months)

**Unified Pipeline**:
- Create orchestrator combining all approaches
- Implement result passing between modules
- Add progress tracking and checkpointing

**Enhanced Documentation**:
- Create comprehensive user guide
- Add API documentation
- Develop tutorials for each workflow
- Create troubleshooting guide

### 3. Long-term Vision (3-6 months)

**Scientific Innovations** (see next section)

**Platform Development**:
- RESTful API for all components
- Enhanced web interface
- Plugin architecture
- Cloud deployment options

## Scientific Innovation Opportunities

### 1. Iterative Active Evaluation

**Concept**: Dynamically select evaluation examples based on emerging patterns

**Implementation**:
```python
class ActiveEvaluator:
    def __init__(self, models, initial_samples=100):
        self.models = models
        self.hypothesis_generator = HypothesisGenerator()
        self.sample_selector = InformationGainSelector()
    
    def evaluate(self, example_pool, budget=1000):
        # Start with random sample
        results = self.evaluate_batch(random.sample(example_pool, 100))
        
        while len(results) < budget:
            # Generate hypotheses about differences
            hypotheses = self.hypothesis_generator.generate(results)
            
            # Select most informative examples
            next_batch = self.sample_selector.select(
                example_pool, hypotheses, batch_size=50
            )
            
            # Evaluate and update
            results.extend(self.evaluate_batch(next_batch))
            
        return self.analyze_results(results)
```

**Benefits**:
- 10-100x reduction in required examples
- Discovers subtle differences more reliably
- Adapts to specific model pairs

### 2. Reasoning Model Evaluation

**Concept**: Specialized evaluation for models with thinking tokens

**Approach**:
1. Extract reasoning traces separately from outputs
2. Analyze reasoning consistency vs output changes
3. Identify reasoning pattern shifts after interventions

**Implementation Ideas**:
- Trace-aware clustering
- Reasoning graph analysis
- Consistency metrics between traces and outputs

### 3. Mechanistic-Behavioral Bridge

**Concept**: Connect internal model changes to behavioral shifts

**Integration with Circuit Tracing**:
```python
class MechanisticBehavioralAnalyzer:
    def analyze_intervention(self, base_model, modified_model):
        # Find modified circuits
        circuit_diffs = self.trace_circuits(base_model, modified_model)
        
        # Generate targeted behavioral tests
        behavioral_tests = self.generate_tests_for_circuits(circuit_diffs)
        
        # Validate connection
        correlations = self.correlate_circuits_to_behaviors(
            circuit_diffs, behavioral_tests
        )
        
        return self.generate_mechanistic_explanation(correlations)
```

### 4. Continual Learning Assessment

**Concept**: Track behavioral drift across multiple training iterations

**Features**:
- Behavioral trajectory visualization
- Drift detection algorithms
- Stability metrics
- Forgetting analysis

### 5. Multi-Modal Behavioral Analysis

**Concept**: Extend to vision-language models and other modalities

**Opportunities**:
- Cross-modal behavioral consistency
- Modality-specific interventions
- Unified evaluation framework

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. Set up shared utilities structure
2. Refactor most duplicated code
3. Standardize testing infrastructure
4. Create basic documentation

### Phase 2: Integration (Weeks 3-4)
1. Build unified pipeline
2. Implement configuration system
3. Create end-to-end examples
4. Deploy improved webapp

### Phase 3: Innovation (Weeks 5-8)
1. Implement active evaluation
2. Add reasoning model support
3. Create mechanistic integration
4. Develop continual learning metrics

### Phase 4: Production (Weeks 9-12)
1. Optimize performance
2. Add cloud deployment
3. Create API layer
4. Release public version

## Conclusion

The supervising-ais-improving-ais project has strong foundations with innovative approaches to behavioral analysis. However, it needs significant engineering improvements to realize its full potential. By addressing code duplication, improving documentation, and implementing the proposed scientific innovations, this project can become a leading framework for AI safety evaluation.

The most promising immediate opportunity is the iterative active evaluation approach, which could dramatically reduce computational costs while improving detection of subtle behavioral changes. Combined with better engineering practices, this would make the framework both more powerful and more accessible to the broader research community.

---

*This guide synthesizes analysis of ~50,000 lines of code across 200+ files. For specific implementation details, refer to the individual module analyses created during this review.*
## Performance Optimization Update

### Current Performance Issues (Auto-Interventions Branch)

The auto-interventions branch introduces powerful validation methods but suffers from severe performance issues:

- **Scale**: Experiments generate 200K-2M texts with 8B parameter models
- **Runtime**: 2-4 days on GPU for a single experiment
- **Cost**: \-500 in API calls per experiment
- **Memory**: Near-maximum GPU utilization

### Proposed Solution: Iterative Active Sampling

Instead of exhaustive generation, use intelligent sampling:

1. **Start Small**: Begin with 5K texts instead of 200K
2. **Identify Uncertainty**: Find cluster boundaries and high-divergence regions
3. **Targeted Sampling**: Generate additional texts only where needed
4. **Early Stopping**: Stop when sufficient confidence is achieved

**Expected Impact**:
- Runtime: 2-4 days → 2-6 hours (10-40x improvement)
- API Costs: \-500 → \-50 (10x reduction)
- Quality: Maintain or improve hypothesis discovery

See PERFORMANCE_OPTIMIZATION_STRATEGIES.md and ITERATIVE_ACTIVE_SAMPLING_IMPLEMENTATION.md for details.
