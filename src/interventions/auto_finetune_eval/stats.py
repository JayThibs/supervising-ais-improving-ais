import numpy as np

def benjamini_hochberg_correction(p_values, alpha=0.05):
    """
    Performs Benjamini-Hochberg correction for multiple hypothesis testing.
    
    Args:
        p_values: List or array of p-values to correct (values in [0, 1])
        alpha: Desired false discovery rate (default: 0.05)
    
    Returns:
        n_significant: Number of significant p-values after correction
        significance_threshold: The largest p-value that is still deemed significant (p_(k))
        significant_mask: Boolean array (same length/order as p_values) where True marks
                          p-values that are significant after BH correction.
    """
    p_values = np.array(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return 0, 0.0, np.array([], dtype=bool)
    
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    # BH thresholds: p_(i) <= (i/m) * alpha, i is 1-indexed
    bh_thresholds = (np.arange(1, n + 1) / n) * alpha
    
    # Identify the largest i satisfying the BH condition
    significant_sorted_mask = sorted_p_values <= bh_thresholds
    
    if np.any(significant_sorted_mask):
        max_significant_idx = np.where(significant_sorted_mask)[0][-1]
        n_significant = max_significant_idx + 1  # +1 because 0-indexed
        significance_threshold = sorted_p_values[max_significant_idx]
        
        # Build mask in original order: first k (in sorted order) are significant
        significant_mask = np.zeros(n, dtype=bool)
        significant_mask[sorted_indices[:n_significant]] = True
    else:
        n_significant = 0
        significance_threshold = 0.0
        significant_mask = np.zeros(n, dtype=bool)
    
    return n_significant, significance_threshold, significant_mask
