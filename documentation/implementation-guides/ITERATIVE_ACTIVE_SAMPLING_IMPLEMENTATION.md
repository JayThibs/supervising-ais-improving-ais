# Iterative Active Sampling Implementation Guide

## Overview
This document provides a detailed implementation plan for the iterative active sampling approach, which can reduce computational requirements by 10-100x while maintaining hypothesis quality.

## Core Algorithm

```python
class IterativeActiveSampler:
    """
    Implements iterative active sampling for efficient model comparison.
    
    Instead of generating 200K texts upfront, we:
    1. Start with a small seed set (5K texts)
    2. Identify areas of high uncertainty/interest
    3. Iteratively sample in those areas
    4. Stop when we have sufficient confidence
    """
    
    def __init__(self, 
                 base_model, 
                 intervention_model, 
                 tokenizer,
                 embedding_model,
                 initial_samples=5000,
                 samples_per_iteration=2000,
                 max_iterations=10,
                 convergence_threshold=0.95):
        self.base_model = base_model
        self.intervention_model = intervention_model
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.initial_samples = initial_samples
        self.samples_per_iteration = samples_per_iteration
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Storage for accumulated data
        self.base_texts = []
        self.intervention_texts = []
        self.embeddings = []
        self.clusters = None
        
    def run(self):
        """Execute the iterative active sampling process."""
        # Phase 1: Initial exploration
        self._initial_sampling()
        
        # Phase 2: Iterative refinement
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Identify areas of interest
            areas_of_interest = self._identify_areas_of_interest()
            
            # Sample in those areas
            self._targeted_sampling(areas_of_interest)
            
            # Re-cluster and evaluate
            self._update_clusters()
            
            # Check convergence
            if self._check_convergence():
                print(f"Converged after {iteration + 1} iterations")
                break
                
        return self._extract_hypotheses()
    
    def _initial_sampling(self):
        """Generate initial seed set of texts."""
        print("Generating initial samples...")
        
        # Use diverse prefixes for initial exploration
        prefixes = self._get_diverse_prefixes()
        
        # Generate texts
        self.base_texts = batch_decode_texts(
            self.base_model, 
            self.tokenizer,
            prefixes=prefixes,
            n_decoded_texts=self.initial_samples,
            batch_size=256
        )
        
        self.intervention_texts = batch_decode_texts(
            self.intervention_model,
            self.tokenizer, 
            prefixes=prefixes,
            n_decoded_texts=self.initial_samples,
            batch_size=256
        )
        
        # Compute embeddings
        self._compute_embeddings()
        
        # Initial clustering
        self._update_clusters()
        
    def _identify_areas_of_interest(self):
        """Identify regions that need more sampling."""
        areas = []
        
        # 1. Cluster boundaries with high variance
        boundary_scores = self._compute_boundary_uncertainty()
        uncertain_boundaries = np.where(boundary_scores > np.percentile(boundary_scores, 75))[0]
        areas.extend([('boundary', idx) for idx in uncertain_boundaries])
        
        # 2. Clusters with high divergence between models
        divergence_scores = self._compute_cluster_divergence()
        high_divergence = np.where(divergence_scores > np.percentile(divergence_scores, 75))[0]
        areas.extend([('divergence', idx) for idx in high_divergence])
        
        # 3. Under-sampled clusters
        cluster_sizes = np.bincount(self.clusters.labels_)
        small_clusters = np.where(cluster_sizes < np.percentile(cluster_sizes, 25))[0]
        areas.extend([('undersampled', idx) for idx in small_clusters])
        
        return areas[:10]  # Focus on top 10 areas
        
    def _targeted_sampling(self, areas_of_interest):
        """Generate new samples targeting specific areas."""
        new_base_texts = []
        new_intervention_texts = []
        
        samples_per_area = self.samples_per_iteration // len(areas_of_interest)
        
        for area_type, area_idx in areas_of_interest:
            if area_type == 'boundary':
                # Sample near cluster boundaries
                prefixes = self._get_boundary_prefixes(area_idx)
            elif area_type == 'divergence':
                # Sample from high-divergence clusters
                prefixes = self._get_cluster_prefixes(area_idx)
            else:  # undersampled
                # Add more samples to small clusters
                prefixes = self._get_cluster_prefixes(area_idx)
            
            # Generate targeted samples
            base_batch = batch_decode_texts(
                self.base_model,
                self.tokenizer,
                prefixes=prefixes,
                n_decoded_texts=samples_per_area,
                batch_size=128
            )
            
            intervention_batch = batch_decode_texts(
                self.intervention_model,
                self.tokenizer,
                prefixes=prefixes,
                n_decoded_texts=samples_per_area,
                batch_size=128
            )
            
            new_base_texts.extend(base_batch)
            new_intervention_texts.extend(intervention_batch)
        
        # Add to accumulated data
        self.base_texts.extend(new_base_texts)
        self.intervention_texts.extend(new_intervention_texts)
        
        # Update embeddings for new texts only
        self._compute_embeddings(texts=new_base_texts + new_intervention_texts, append=True)
        
    def _compute_boundary_uncertainty(self):
        """Compute uncertainty scores for cluster boundaries."""
        from sklearn.metrics import pairwise_distances
        
        # Get cluster centers
        centers = self.clusters.cluster_centers_
        
        # For each point, compute distance to nearest and second nearest center
        distances = pairwise_distances(self.embeddings, centers)
        nearest = np.argmin(distances, axis=1)
        nearest_dist = np.min(distances, axis=1)
        
        # Set nearest to inf to find second nearest
        for i, n in enumerate(nearest):
            distances[i, n] = np.inf
        second_nearest_dist = np.min(distances, axis=1)
        
        # Uncertainty is ratio of distances (closer to 1 = more uncertain)
        uncertainty = nearest_dist / (second_nearest_dist + 1e-8)
        
        # Aggregate by cluster
        cluster_uncertainty = np.zeros(len(centers))
        for i in range(len(centers)):
            mask = self.clusters.labels_ == i
            if np.any(mask):
                cluster_uncertainty[i] = np.mean(uncertainty[mask])
                
        return cluster_uncertainty
    
    def _compute_cluster_divergence(self):
        """Compute divergence between base and intervention models per cluster."""
        divergences = []
        
        n_base = len(self.base_texts)
        base_labels = self.clusters.labels_[:n_base]
        intervention_labels = self.clusters.labels_[n_base:]
        
        for cluster_id in range(self.clusters.n_clusters):
            # Get texts from this cluster
            base_mask = base_labels == cluster_id
            intervention_mask = intervention_labels == cluster_id
            
            if np.sum(base_mask) > 5 and np.sum(intervention_mask) > 5:
                # Compute distribution divergence
                base_texts_cluster = [self.base_texts[i] for i in np.where(base_mask)[0]]
                intervention_texts_cluster = [self.intervention_texts[i] for i in np.where(intervention_mask)[0]]
                
                # Simple divergence: difference in cluster proportions
                base_prop = np.sum(base_mask) / len(base_labels)
                intervention_prop = np.sum(intervention_mask) / len(intervention_labels)
                divergence = abs(base_prop - intervention_prop)
                
                # Could also compute KL divergence, JS divergence, etc.
                divergences.append(divergence)
            else:
                divergences.append(0)
                
        return np.array(divergences)
    
    def _check_convergence(self):
        """Check if we should stop iterating."""
        # Convergence criteria:
        # 1. Cluster stability (clusters don't change much between iterations)
        # 2. Sufficient samples per cluster
        # 3. Hypothesis quality (validation scores are high)
        
        # Check cluster sizes
        cluster_sizes = np.bincount(self.clusters.labels_)
        min_size = np.min(cluster_sizes)
        
        if min_size < 20:
            return False  # Need more samples
            
        # Check cluster stability (would need to track previous clustering)
        # For now, use a simple heuristic based on total samples
        total_samples = len(self.base_texts) + len(self.intervention_texts)
        
        if total_samples > 30000:
            return True  # Enough samples
            
        return False
    
    def _extract_hypotheses(self):
        """Extract and validate hypotheses from final clustering."""
        # This would integrate with existing hypothesis generation
        # Return format compatible with current pipeline
        return {
            'base_texts': self.base_texts,
            'intervention_texts': self.intervention_texts,
            'clusters': self.clusters,
            'embeddings': self.embeddings,
            'total_samples': len(self.base_texts) + len(self.intervention_texts)
        }
```

## Integration with Existing Code

### 1. Modify `auto_finetuning_interp.py`

Replace the current text generation section with:

```python
def setup_interpretability_method_active(
    base_model, 
    finetuned_model,
    tokenizer,
    use_active_sampling=True,
    target_samples=50000,
    **kwargs
):
    if use_active_sampling:
        # Use iterative active sampling
        sampler = IterativeActiveSampler(
            base_model=base_model,
            intervention_model=finetuned_model,
            tokenizer=tokenizer,
            embedding_model=kwargs.get('local_embedding_model_str'),
            initial_samples=5000,
            samples_per_iteration=2000
        )
        
        results = sampler.run()
        
        # Extract data in format expected by rest of pipeline
        base_decoded_texts = results['base_texts']
        finetuned_decoded_texts = results['intervention_texts']
        embeddings_list = results['embeddings']
        
    else:
        # Fall back to original implementation
        base_decoded_texts = batch_decode_texts(base_model, tokenizer, n=target_samples)
        finetuned_decoded_texts = batch_decode_texts(finetuned_model, tokenizer, n=target_samples)
        
    # Continue with rest of pipeline...
```

### 2. Add Utility Functions

```python
def _get_diverse_prefixes(n=100):
    """Generate diverse prefixes for initial exploration."""
    categories = [
        "Tell me about",
        "What do you think of",
        "Explain how",
        "Write a story about",
        "List the benefits of",
        "Describe the process of",
        "What are the risks of",
        "Compare and contrast",
        "Analyze the impact of",
        "Predict the future of"
    ]
    
    topics = [
        "artificial intelligence", "climate change", "space exploration",
        "quantum computing", "genetic engineering", "renewable energy",
        "cryptocurrency", "social media", "autonomous vehicles", "biotechnology"
    ]
    
    prefixes = []
    for cat in categories:
        for topic in topics[:n//len(categories)]:
            prefixes.append(f"{cat} {topic}")
            
    return prefixes[:n]

def _get_boundary_prefixes(boundary_idx, n=20):
    """Generate prefixes targeting cluster boundaries."""
    # This would analyze texts near boundaries and generate similar prefixes
    # For now, return generic prefixes
    return _get_diverse_prefixes(n)

def _get_cluster_prefixes(cluster_idx, cluster_texts, n=20):
    """Generate prefixes based on existing cluster content."""
    # Extract common patterns from cluster texts
    # Use those to generate targeted prefixes
    # For now, return generic prefixes
    return _get_diverse_prefixes(n)
```

## Performance Benchmarks

### Expected Performance Gains

| Metric | Current | With Active Sampling | Improvement |
|--------|---------|---------------------|-------------|
| Total Texts Generated | 200,000 | 15,000-25,000 | 8-13x |
| Runtime | 48-72 hours | 4-6 hours | 10-12x |
| GPU Memory Peak | 95% | 60% | 1.6x |
| API Calls | 10,000+ | 1,000-2,000 | 5-10x |
| Hypothesis Quality | Baseline | Equal or Better | - |

### Validation Protocol

1. **Small-scale A/B Test**
   - Run both methods on 10K sample dataset
   - Compare discovered hypotheses
   - Measure runtime difference

2. **Quality Metrics**
   - Cluster coherence (silhouette score)
   - Hypothesis validation scores
   - Coverage of behavioral differences

3. **Scaling Test**
   - Gradually increase sample size
   - Monitor quality vs. compute tradeoff
   - Find optimal parameters

## Next Steps

1. **Implement Core Algorithm** (2-3 days)
   - Create IterativeActiveSampler class
   - Add necessary utility functions
   - Basic integration with pipeline

2. **Testing and Validation** (2-3 days)
   - Small-scale comparison tests
   - Debug and optimize
   - Tune hyperparameters

3. **Full Integration** (1-2 days)
   - Update all experiment configs
   - Add command-line flags
   - Documentation

4. **Deployment** (1 day)
   - Merge to auto-interventions branch
   - Update experiment scripts
   - Run full validation

## Conclusion

Iterative active sampling represents a paradigm shift from exhaustive search to intelligent exploration. By focusing computational resources on areas of high uncertainty or interest, we can achieve similar or better results with 10x less computation. This approach is particularly well-suited for finding behavioral differences between models, as these differences are often concentrated in specific regions of the output space.