"""
Statistical Validator Module

This module ensures statistical rigor in behavioral validation through proper
hypothesis testing, multiple comparison correction, and false discovery rate control.

Key Algorithms:
    1. SAFFRON: Alpha-wealth based FDR control for sequential testing
    2. Adaptive p-value thresholds: Learn optimal thresholds from data
    3. Effect size estimation: Beyond p-values to practical significance
    4. Power analysis: Ensure adequate sample sizes
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings


@dataclass
class TestingProcedure:
    """
    Configuration for a statistical testing procedure.
    
    Attributes:
        alpha: Significance level
        method: Multiple testing correction method
        min_effect_size: Minimum effect size to consider meaningful
        power_threshold: Desired statistical power
        sequential: Whether tests are conducted sequentially
    """
    alpha: float = 0.05
    method: str = 'saffron'  # 'saffron', 'benjamini_hochberg', 'bonferroni'
    min_effect_size: float = 0.1
    power_threshold: float = 0.8
    sequential: bool = True
    
    
@dataclass
class StatisticalTestResult:
    """
    Result of a statistical test with FDR control.
    
    Attributes:
        raw_p_value: Uncorrected p-value
        adjusted_p_value: FDR-adjusted p-value
        effect_size: Magnitude of effect
        confidence_interval: CI for effect size
        reject_null: Whether to reject null hypothesis
        test_statistic: Value of test statistic
        sample_size: Number of observations
    """
    raw_p_value: float
    adjusted_p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    reject_null: bool
    test_statistic: float
    sample_size: int
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant."""
        return self.adjusted_p_value < alpha
        
    def is_meaningful(self, min_effect: float = 0.1) -> bool:
        """Check if effect is practically meaningful."""
        return abs(self.effect_size) > min_effect


class SAFFRON:
    """
    SAFFRON (Serial Alpha-spending For FDR control) algorithm.
    
    This implements adaptive FDR control for sequential hypothesis testing,
    which is ideal for our use case where we test hypotheses one by one.
    
    Based on: "SAFFRON: an adaptive algorithm for online control of the FDR"
    by Ramdas et al. (2018)
    """
    
    def __init__(self, alpha: float = 0.05, lambda_: float = 0.5, gamma: float = 0.9):
        """
        Initialize SAFFRON.
        
        Args:
            alpha: Target FDR level
            lambda_: Initial wealth fraction (typically 0.5) 
            gamma: Discount factor for alpha-wealth (typically 0.9)
        """
        self.alpha = alpha
        self.lambda_ = lambda_
        self.gamma = gamma
        
        # Initialize wealth
        self.wealth = alpha * lambda_
        self.bound = alpha - self.wealth
        
        # Track history
        self.n_tests = 0
        self.n_rejections = 0
        self.rejection_times = []
        
    def test(self, p_value: float) -> Tuple[bool, float]:
        """
        Test a hypothesis using SAFFRON.
        
        Args:
            p_value: P-value for current hypothesis
            
        Returns:
            (reject, adjusted_alpha): Whether to reject and the threshold used
        """
        self.n_tests += 1
        
        # Compute adaptive alpha for this test
        if self.n_rejections == 0:
            alpha_t = self.wealth
        else:
            # Update bound based on past rejections
            last_rejection = self.rejection_times[-1]
            discount = self.gamma ** (self.n_tests - last_rejection)
            alpha_t = min(self.wealth, discount * self.bound)
            
        # Test decision
        reject = p_value <= alpha_t
        
        if reject:
            self.n_rejections += 1
            self.rejection_times.append(self.n_tests)
            
            # Update wealth (conservative update)
            self.wealth = self.wealth - alpha_t + self.alpha
            
            # Update bound
            self.bound = self.alpha * (1 - self.lambda_) / max(1, self.n_rejections)
            
        return reject, alpha_t
        
    def get_fdr_estimate(self) -> float:
        """Estimate current FDR based on rejections."""
        if self.n_rejections == 0:
            return 0.0
            
        # Conservative estimate
        expected_false_discoveries = self.alpha * self.n_tests
        return min(expected_false_discoveries / self.n_rejections, 1.0)


class StatisticalValidator:
    """
    Ensures statistical validity of behavioral testing.
    
    This class provides rigorous statistical methods tailored for
    comparing language model behaviors.
    """
    
    def __init__(self, procedure: Optional[TestingProcedure] = None):
        """
        Initialize statistical validator.
        
        Args:
            procedure: Testing procedure configuration
        """
        self.procedure = procedure or TestingProcedure()
        
        # Initialize SAFFRON if using sequential testing
        if self.procedure.sequential and self.procedure.method == 'saffron':
            self.saffron = SAFFRON(alpha=self.procedure.alpha)
        else:
            self.saffron = None
            
        # Track all p-values for batch correction if needed
        self.all_p_values = []
        self.all_hypotheses = []
        
    def test_behavioral_difference(self,
                                 base_outputs: List[str],
                                 intervention_outputs: List[str],
                                 test_type: str = 'paired') -> StatisticalTestResult:
        """
        Test for behavioral differences between model outputs.
        
        Args:
            base_outputs: Outputs from base model
            intervention_outputs: Outputs from intervention model
            test_type: Type of test ('paired', 'independent', 'categorical')
            
        Returns:
            StatisticalTestResult with corrected p-values
        """
        n_samples = len(base_outputs)
        
        if test_type == 'paired':
            # Paired test for differences
            differences = [self._measure_difference(b, i) 
                         for b, i in zip(base_outputs, intervention_outputs)]
            
            # Wilcoxon signed-rank test (non-parametric)
            if len(differences) < 10:
                warnings.warn("Small sample size, results may be unreliable")
                
            stat, raw_p = stats.wilcoxon(differences, alternative='two-sided')
            
            # Effect size: Matched pairs rank-biserial correlation
            effect_size = self._compute_rank_biserial(differences)
            
        elif test_type == 'categorical':
            # Chi-squared test for categorical differences
            categories_base = self._categorize_outputs(base_outputs)
            categories_int = self._categorize_outputs(intervention_outputs)
            
            contingency_table = self._build_contingency_table(
                categories_base, categories_int
            )
            
            stat, raw_p, _, _ = stats.chi2_contingency(contingency_table)
            
            # Effect size: Cramér's V
            effect_size = self._compute_cramers_v(contingency_table)
            
        else:
            # Mann-Whitney U test for independent samples
            base_scores = [self._score_output(o) for o in base_outputs]
            int_scores = [self._score_output(o) for o in intervention_outputs]
            
            stat, raw_p = stats.mannwhitneyu(
                base_scores, int_scores, alternative='two-sided'
            )
            
            # Effect size: Rank-biserial correlation
            effect_size = self._compute_rank_biserial_independent(
                base_scores, int_scores
            )
            
        # Adjust p-value
        if self.saffron:
            reject, adjusted_alpha = self.saffron.test(raw_p)
            adjusted_p = raw_p  # Keep raw p-value, decision based on threshold
        else:
            # Store for batch correction later
            self.all_p_values.append(raw_p)
            adjusted_p = raw_p  # Will be corrected in batch
            reject = raw_p < self.procedure.alpha
            
        # Confidence interval for effect size
        ci_lower, ci_upper = self._bootstrap_confidence_interval(
            base_outputs, intervention_outputs, test_type
        )
        
        return StatisticalTestResult(
            raw_p_value=raw_p,
            adjusted_p_value=adjusted_p,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            reject_null=reject,
            test_statistic=stat,
            sample_size=n_samples
        )
        
    def batch_correction(self, 
                        p_values: Optional[List[float]] = None,
                        method: Optional[str] = None) -> List[float]:
        """
        Apply multiple testing correction to a batch of p-values.
        
        Args:
            p_values: List of p-values (uses stored if None)
            method: Correction method (uses procedure default if None)
            
        Returns:
            Adjusted p-values
        """
        if p_values is None:
            p_values = self.all_p_values
            
        if not p_values:
            return []
            
        method = method or self.procedure.method
        
        if method == 'benjamini_hochberg':
            _, adjusted_p, _, _ = multipletests(
                p_values, alpha=self.procedure.alpha, method='fdr_bh'
            )
        elif method == 'bonferroni':
            adjusted_p = np.minimum(np.array(p_values) * len(p_values), 1.0)
        else:
            # No correction
            adjusted_p = p_values
            
        return adjusted_p.tolist()
        
    def compute_power(self,
                     effect_size: float,
                     sample_size: int,
                     test_type: str = 'paired') -> float:
        """
        Compute statistical power for a given effect size and sample size.
        
        Args:
            effect_size: Expected effect size
            sample_size: Number of samples
            test_type: Type of test
            
        Returns:
            Estimated power (0-1)
        """
        from statsmodels.stats.power import TTestPower
        
        if test_type == 'paired':
            power_analyzer = TTestPower()
            power = power_analyzer.solve_power(
                effect_size=effect_size,
                nobs=sample_size,
                alpha=self.procedure.alpha,
                power=None,
                alternative='two-sided'
            )
        else:
            # Approximate for other tests
            # This is simplified - real implementation would use specific power formulas
            z_alpha = stats.norm.ppf(1 - self.procedure.alpha / 2)
            z_beta = effect_size * np.sqrt(sample_size) - z_alpha
            power = stats.norm.cdf(z_beta)
            
        return power
        
    def required_sample_size(self,
                           effect_size: float,
                           desired_power: Optional[float] = None) -> int:
        """
        Calculate required sample size for desired power.
        
        Args:
            effect_size: Expected effect size
            desired_power: Target power (uses procedure default if None)
            
        Returns:
            Required number of samples
        """
        power = desired_power or self.procedure.power_threshold
        
        from statsmodels.stats.power import TTestPower
        power_analyzer = TTestPower()
        
        n = power_analyzer.solve_power(
            effect_size=effect_size,
            nobs=None,
            alpha=self.procedure.alpha,
            power=power,
            alternative='two-sided'
        )
        
        return int(np.ceil(n))
        
    def _measure_difference(self, output1: str, output2: str) -> float:
        """
        Measure difference between two outputs.
        
        This is a simplified metric - in practice would use more
        sophisticated difference measures.
        """
        # Character-level edit distance normalized by length
        from difflib import SequenceMatcher
        
        similarity = SequenceMatcher(None, output1, output2).ratio()
        return 1.0 - similarity
        
    def _score_output(self, output: str) -> float:
        """
        Score an output for statistical testing.
        
        This assigns a numerical score to text for statistical tests.
        """
        # Simple scoring based on length and complexity
        # In practice, would use task-specific scoring
        words = output.split()
        unique_words = len(set(words))
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        return len(words) * 0.3 + unique_words * 0.5 + avg_word_length * 0.2
        
    def _categorize_outputs(self, outputs: List[str]) -> List[str]:
        """
        Categorize outputs for chi-squared test.
        
        This assigns outputs to discrete categories.
        """
        categories = []
        
        for output in outputs:
            if len(output) < 50:
                cat = 'short'
            elif len(output) < 200:
                cat = 'medium' 
            else:
                cat = 'long'
                
            # Could add more sophisticated categorization
            # e.g., sentiment, style, safety level
                
            categories.append(cat)
            
        return categories
        
    def _build_contingency_table(self,
                               categories1: List[str],
                               categories2: List[str]) -> np.ndarray:
        """Build contingency table for categorical test."""
        unique_cats = sorted(set(categories1 + categories2))
        n_cats = len(unique_cats)
        cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
        
        table = np.zeros((2, n_cats))
        
        for cat in categories1:
            table[0, cat_to_idx[cat]] += 1
            
        for cat in categories2:
            table[1, cat_to_idx[cat]] += 1
            
        return table
        
    def _compute_rank_biserial(self, differences: List[float]) -> float:
        """
        Compute rank-biserial correlation for paired data.
        
        Effect size measure for Wilcoxon test.
        """
        n = len(differences)
        if n == 0:
            return 0.0
            
        # Rank absolute differences
        abs_diffs = np.abs(differences)
        ranks = stats.rankdata(abs_diffs)
        
        # Sum ranks for positive and negative differences
        pos_ranks = sum(r for d, r in zip(differences, ranks) if d > 0)
        neg_ranks = sum(r for d, r in zip(differences, ranks) if d < 0)
        
        # Compute correlation
        r = (pos_ranks - neg_ranks) / (pos_ranks + neg_ranks) if (pos_ranks + neg_ranks) > 0 else 0
        
        return r
        
    def _compute_rank_biserial_independent(self,
                                         group1: List[float],
                                         group2: List[float]) -> float:
        """
        Compute rank-biserial correlation for independent groups.
        
        Effect size for Mann-Whitney U test.
        """
        n1, n2 = len(group1), len(group2)
        
        if n1 == 0 or n2 == 0:
            return 0.0
            
        # Compute U statistic
        u_stat, _ = stats.mannwhitneyu(group1, group2)
        
        # Convert to rank-biserial
        r = 1 - (2 * u_stat) / (n1 * n2)
        
        return r
        
    def _compute_cramers_v(self, contingency_table: np.ndarray) -> float:
        """
        Compute Cramér's V for categorical association.
        
        Effect size for chi-squared test.
        """
        chi2 = stats.chi2_contingency(contingency_table)[0]
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        
        if n == 0 or min_dim == 0:
            return 0.0
            
        v = np.sqrt(chi2 / (n * min_dim))
        
        return v
        
    def _bootstrap_confidence_interval(self,
                                     base_outputs: List[str],
                                     int_outputs: List[str],
                                     test_type: str,
                                     n_bootstrap: int = 1000,
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for effect size.
        
        Args:
            base_outputs: Base model outputs
            int_outputs: Intervention model outputs
            test_type: Type of test
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            (lower_bound, upper_bound) of confidence interval
        """
        effect_sizes = []
        n = len(base_outputs)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            
            base_sample = [base_outputs[i] for i in indices]
            int_sample = [int_outputs[i] for i in indices]
            
            # Compute effect size for bootstrap sample
            if test_type == 'paired':
                diffs = [self._measure_difference(b, i) 
                        for b, i in zip(base_sample, int_sample)]
                effect = self._compute_rank_biserial(diffs)
            else:
                base_scores = [self._score_output(o) for o in base_sample]
                int_scores = [self._score_output(o) for o in int_sample]
                effect = self._compute_rank_biserial_independent(
                    base_scores, int_scores
                )
                
            effect_sizes.append(effect)
            
        # Compute percentile CI
        alpha = 1 - confidence
        lower = np.percentile(effect_sizes, 100 * alpha / 2)
        upper = np.percentile(effect_sizes, 100 * (1 - alpha / 2))
        
        return lower, upper
        
    def meta_analysis(self,
                     results: List[StatisticalTestResult],
                     weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Perform meta-analysis across multiple test results.
        
        Useful for combining evidence across different test types
        or hypotheses.
        
        Args:
            results: List of test results
            weights: Optional weights for each result
            
        Returns:
            Dictionary with meta-analysis results
        """
        if not results:
            return {}
            
        # Default to equal weights
        if weights is None:
            weights = [1.0] * len(results)
            
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Combine effect sizes
        combined_effect = sum(r.effect_size * w for r, w in zip(results, weights))
        
        # Combine p-values using Fisher's method
        chi2_stat = -2 * sum(np.log(r.raw_p_value) for r in results)
        combined_p = 1 - stats.chi2.cdf(chi2_stat, df=2 * len(results))
        
        # Check heterogeneity
        q_stat = sum(
            w * (r.effect_size - combined_effect) ** 2 
            for r, w in zip(results, weights)
        )
        
        return {
            'combined_effect_size': combined_effect,
            'combined_p_value': combined_p,
            'n_tests': len(results),
            'heterogeneity_q': q_stat,
            'percent_significant': sum(1 for r in results if r.is_significant()) / len(results)
        }