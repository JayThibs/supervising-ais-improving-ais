# Performance Optimization Strategies for Auto-Interventions

## Executive Summary
Current experiments take several days on GPU due to massive scale (200K-2M decoded texts, 100-1000 clusters). This document outlines strategies to reduce runtime by 10-100x while maintaining or improving effectiveness.

## Current Performance Profile

### Bottleneck Breakdown
1. **Text Generation (60-70%)**: Generating 200K+ texts with 8B parameter models
2. **Embedding Computation (15-20%)**: Computing embeddings for all texts
3. **Clustering (10%)**: K-means/HDBSCAN on high-dimensional embeddings
4. **API Validation (10%)**: Thousands of API calls for label validation

### Resource Usage
- GPU Memory: Near maximum with 8B models + batch processing
- Compute Time: 2-4 days for full pipeline
- API Costs: $100-500 per experiment (depending on model choice)

## Optimization Strategies

### 1. Iterative Active Sampling (10-100x speedup)
Instead of generating all 200K texts upfront:

```python
# Current approach
all_texts = batch_decode_texts(model, n=200000)  # Takes hours
embeddings = compute_embeddings(all_texts)  # More hours
clusters = kmeans(embeddings, n_clusters=100)

# Optimized approach
def iterative_active_sampling(model, target_clusters=100, max_texts=200000):
    # Start with small sample
    texts = batch_decode_texts(model, n=5000)  # 40x faster
    embeddings = compute_embeddings(texts)
    clusters = kmeans(embeddings, n_clusters=20)
    
    # Iteratively add texts where uncertainty is highest
    for iteration in range(5):
        # Find cluster boundaries with high uncertainty
        uncertain_regions = find_uncertain_boundaries(clusters)
        
        # Generate targeted samples
        new_texts = targeted_decode_texts(model, uncertain_regions, n=2000)
        texts.extend(new_texts)
        
        # Re-cluster with more data
        embeddings = compute_embeddings(texts)
        clusters = update_clusters(embeddings, clusters)
        
        if cluster_quality_sufficient(clusters):
            break
    
    return texts, clusters  # Total: ~15K texts instead of 200K
```

**Expected Impact**: 
- Reduce texts from 200K to 15-20K (10x reduction)
- Maintain 95%+ of clustering quality
- Enable real-time feedback loop

### 2. Cached Embeddings with Incremental Updates (5x speedup)
```python
# Implement embedding cache
class EmbeddingCache:
    def __init__(self, cache_dir="embeddings_cache"):
        self.cache = {}
        self.cache_dir = cache_dir
        
    def get_embeddings(self, texts, model):
        # Check cache first
        uncached_texts = []
        cached_embeddings = []
        
        for text in texts:
            text_hash = hash(text)
            if text_hash in self.cache:
                cached_embeddings.append(self.cache[text_hash])
            else:
                uncached_texts.append(text)
        
        # Compute only missing embeddings
        if uncached_texts:
            new_embeddings = model.encode(uncached_texts)
            for text, emb in zip(uncached_texts, new_embeddings):
                self.cache[hash(text)] = emb
                
        return combine_embeddings(cached_embeddings, new_embeddings)
```

### 3. Multi-Stage Filtering (3x speedup)
```python
def multi_stage_filtering(base_model, intervention_model, n_final=50000):
    # Stage 1: Quick divergence check with small samples
    quick_texts = batch_decode_texts(base_model, n=1000)
    quick_intervention = batch_decode_texts(intervention_model, n=1000)
    
    # Identify high-divergence prefixes
    divergent_prefixes = find_divergent_prefixes(quick_texts, quick_intervention)
    
    # Stage 2: Focus on divergent areas
    targeted_texts = batch_decode_texts(
        base_model, 
        n=10000, 
        prefixes=divergent_prefixes
    )
    
    # Stage 3: Full analysis only on promising clusters
    promising_clusters = preliminary_cluster(targeted_texts)
    final_texts = generate_for_clusters(promising_clusters, n=n_final)
    
    return final_texts
```

### 4. Distributed Processing (4x speedup)
```python
# Parallelize across multiple GPUs
def distributed_decode(models, n_texts, n_gpus=4):
    texts_per_gpu = n_texts // n_gpus
    
    with multiprocessing.Pool(n_gpus) as pool:
        results = pool.starmap(
            batch_decode_texts,
            [(model, texts_per_gpu, gpu_id) for gpu_id in range(n_gpus)]
        )
    
    return flatten(results)
```

### 5. Efficient Clustering (2x speedup)
```python
# Use MiniBatchKMeans for large datasets
from sklearn.cluster import MiniBatchKMeans

def efficient_clustering(embeddings, n_clusters=100):
    # MiniBatchKMeans is much faster for large datasets
    clustering = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=1024,  # Process in mini-batches
        n_init=3,  # Fewer initializations
        max_iter=100,
        reassignment_ratio=0.001  # Early stopping
    )
    
    # Fit in chunks for memory efficiency
    for i in range(0, len(embeddings), 10000):
        batch = embeddings[i:i+10000]
        clustering.partial_fit(batch)
    
    return clustering
```

### 6. Smart API Usage (10x cost reduction)
```python
def hierarchical_api_validation(hypotheses, models):
    # Use cheap model first
    cheap_results = validate_with_model(hypotheses, "gemini-1.5-flash")
    
    # Only use expensive model for borderline cases
    borderline = [h for h, score in cheap_results if 0.4 < score < 0.6]
    expensive_results = validate_with_model(borderline, "gemini-1.5-pro")
    
    return merge_results(cheap_results, expensive_results)
```

## Implementation Priority

1. **Immediate (1-2 days)**
   - Implement embedding cache
   - Reduce initial sample size to 20K
   - Use MiniBatchKMeans

2. **Short-term (1 week)**
   - Multi-stage filtering
   - Hierarchical API validation
   - Basic active sampling

3. **Medium-term (2-3 weeks)**
   - Full iterative active sampling
   - Distributed processing
   - Incremental clustering

## Expected Outcomes

### Performance Improvements
- **Runtime**: 2-4 days → 2-6 hours (10-40x improvement)
- **API Costs**: $100-500 → $10-50 (10x reduction)
- **GPU Memory**: More efficient usage, enabling larger batches

### Quality Impact
- **Clustering Quality**: Maintain 95%+ quality with 10x fewer samples
- **Hypothesis Quality**: Potentially improved through targeted sampling
- **Coverage**: Better coverage of important behavioral differences

## Validation Strategy

1. Run side-by-side comparison on small dataset
2. Measure clustering quality metrics (silhouette score, etc.)
3. Compare discovered hypotheses
4. Track runtime and resource usage

## Code Integration Points

Key files to modify:
- `auto_finetuning_interp.py`: Add iterative sampling logic
- `auto_finetuning_helpers.py`: Implement caching in batch_decode_texts
- `validated_comparison_tools.py`: Add hierarchical validation
- `run_auto_finetuning_main.sh`: Update default parameters

## Conclusion

These optimizations can reduce experiment time from days to hours while maintaining or improving quality. The iterative active sampling approach is particularly promising, as it aligns with the scientific goal of finding meaningful differences rather than exhaustive search.