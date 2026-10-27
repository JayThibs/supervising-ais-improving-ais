# Context Package for Opus: Supervising AIs Improving AIs

## Project Vision
Detect unexpected behavioral changes in LLMs after interventions (fine-tuning, unlearning, knowledge editing) to ensure capability improvements don't compromise alignment or introduce harmful behaviors.

## Current Challenge
The best approach (auto-interventions) takes 2-4 days per experiment because it generates 200K-2M texts exhaustively before analysis. We need fundamentally more efficient approaches.

## What We've Tried (2+ Years)

### 1. **Behavioral Clustering** (Mature but Limited)
- Groups model outputs by semantic similarity
- Good infrastructure, customizable
- Lacks statistical rigor
- See: CONDENSED_BEHAVIORAL_CLUSTERING_SUMMARY.md

### 2. **Contrastive Decoding** (Interesting but Narrow)
- Amplifies differences during generation
- Finds extreme cases well
- Doesn't represent typical behavior
- Limited practical application

### 3. **Soft Prompting** (Automatic but Opaque)
- Gradient-based optimization for maximum divergence
- No manual prompt engineering needed
- Hard to interpret results
- Computationally intensive

### 4. **Auto-Interventions** (Rigorous but Slow)
- Generates behavioral hypotheses
- Statistical validation with SAFFRON
- Multiple validation approaches
- Takes days due to scale
- See: CONDENSED_AUTO_INTERVENTIONS_SUMMARY.md

## Key Technical Components

### Core Workflow
1. Generate texts from both models (200K-2M texts) ← **BOTTLENECK**
2. Embed and cluster texts (100-1000 clusters)
3. Find clusters that differ between models
4. Generate hypotheses about differences
5. Validate hypotheses statistically

### Key Algorithms
- **SAFFRON**: Online FDR control for multiple hypothesis testing
- **SVD Weight Analysis**: Understand internal model changes
- **Contrastive Clustering**: Compare behavioral distributions
- See: KEY_ALGORITHMS_AND_INSIGHTS.md

### Current Scale
```bash
--num_decoded_texts 200000    # Generate this many texts
--num_clusters 100            # Create this many clusters
--decoding_max_length 48      # Token length per text
--api_validation_calls 10000+ # Expensive API validations
```

## Key Insights from Research

1. **Behavioral differences are sparse** - Most outputs unchanged, differences concentrate in specific areas
2. **Clustering plateaus quickly** - 90% structure emerges with 10% of data
3. **Validation is expensive** - Generating hypotheses cheap, validating them costly
4. **Models can guide analysis** - Uncertainty signals where to look

## What We Need: Paradigm Shift Ideas

### Not Looking For
- Incremental optimizations of current approaches
- Ways to generate texts faster
- Better clustering algorithms

### Looking For
- Fundamentally different approaches to finding behavioral differences
- Ways to avoid exhaustive generation
- Methods that scale with model size, not output space
- Approaches that use models as collaborators, not just subjects

### Promising Directions to Explore

1. **Model-Guided Discovery**
   - Can models tell us where they've changed?
   - Self-reporting mechanisms
   - Uncertainty-guided sampling

2. **Mechanistic-Behavioral Bridge**
   - Connect weight changes to behavioral changes
   - Use internal structure to predict external behavior
   - Activation-based analysis

3. **Adversarial Frameworks**
   - Models trying to expose each other's differences
   - Red-team/blue-team dynamics
   - Game-theoretic formulations

4. **Meta-Learning Patterns**
   - Learn intervention → behavior mappings
   - Predict likely changes before searching
   - Transfer knowledge across interventions

5. **Reasoning Model Integration**
   - Use models like o1 to reason about differences
   - Chain-of-thought for behavioral analysis
   - Self-critique and hypothesis generation

## Constraints & Requirements

### Must Have
- Statistical rigor (control false positives)
- Scalability to large models
- Interpretable results
- Work for any intervention type

### Nice to Have
- Real-time analysis (hours not days)
- Minimal API costs
- Human-in-the-loop capability
- Mechanistic interpretability

## Questions for Opus

1. **How can we leverage the sparsity of behavioral differences to avoid exhaustive search?**

2. **What approaches from other fields (neuroscience, psychology, software testing) could apply here?**

3. **How might reasoning models fundamentally change how we approach this problem?**

4. **Can we design systems where models actively participate in discovering their own changes?**

5. **What would a completely different architecture look like that scales with model complexity, not output space?**

## Available Files for Reference
- COMPREHENSIVE_CODEBASE_GUIDE.md - Full project overview
- AUTO_INTERVENTIONS_BRANCH_ANALYSIS.md - Current performance issues
- PERFORMANCE_OPTIMIZATION_STRATEGIES.md - Proposed optimizations
- CONDENSED_AUTO_INTERVENTIONS_SUMMARY.md - Current best approach
- CONDENSED_BEHAVIORAL_CLUSTERING_SUMMARY.md - Mature infrastructure
- KEY_ALGORITHMS_AND_INSIGHTS.md - Technical details and insights

## The Ask
We need creative, fundamentally different approaches. Think beyond optimizing what exists—imagine new paradigms for detecting behavioral changes in AI systems. The current approach works but doesn't scale. What would you build from scratch knowing what we know now?