"""SAFFRON: An adaptive online FDR control procedure.

Parameters
----------
alpha : float, default 0.05
    Target (m)FDR / FDR level.

lambda_param : float in (0,1), default 0.5
    SAFFRON "candidate" threshold λ.  Only p-values ≤ λ are *candidates* for
    rejection; the per-test level α_t is always ≤ λ.  

gamma_param : float or array-like, default 0.5
    Controls the spending *schedule* γ_j.  Interpretation:
      * If ``gamma_param`` is a 1d sequence of positive numbers, we normalize it
        to sum to 1 and use it directly as γ.
      * If it's a scalar in (0,1), we interpret it as the *power-law exponent*
        "c" in γ_j ∝ 1 / j^(1+c).  E.g., ``gamma_param=0.5`` ⇒ γ_j ∝ 1/j^1.5.

Initial wealth W0
-----------------
SAFFRON requires an initial wealth W0 strictly less than (1-λ)α.  
By default: ``W0 = min(0.25 * (1-λ) * alpha, 0.99 * (1-λ) * alpha)``.
You may override this by setting the ``w0`` attribute *after* construction, *before*
calling ``test_hypothesis`` for the first time, if you have a preferred value.

Complexity note
---------------
For clarity (and to keep this a faithful, easy-to-read translation of the
SAFFRON recursion in Ramdas et al., 2018), the implementation recomputes the
current test level α_t from the complete history each time you call
``test_hypothesis``. 

Reference
----------
Ramdas, Aaditya & Zrnic, Tijana & Wainwright, Martin & Jordan, Michael. (2018). SAFFRON: an adaptive algorithm for online control of the false discovery rate. 10.48550/arXiv.1802.09098. 

"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Utility: build a gamma schedule that sums to 1.
# ---------------------------------------------------------------------------

def _normalize(seq: Sequence[float]) -> List[float]:
    tot = float(sum(seq))
    if tot <= 0:
        raise ValueError("gamma sequence must have positive sum")
    return [float(x) / tot for x in seq]


def _make_power_gamma(exponent_extra: float, length: int = 100000) -> List[float]:
    """Return first ``length`` weights of γ_j ∝ j^-(1+exponent_extra).

    SAFFRON only requires that γ_j ≥ 0 and ∑_j γ_j = 1.  A slowly decaying
    power-law puts nontrivial mass on later tests, which helps in long streams.

    Parameters
    ----------
    exponent_extra : float in (0, 5] approximately.
        The schedule exponent is 1 + exponent_extra; values in [0.3, 1.0] are
        common.  Larger ⇒ faster decay (more aggressive early spending).
    length : int, default 100000
        Number of terms to precompute.  This is plenty for most applications.
    """
    if exponent_extra <= 0:
        raise ValueError("exponent_extra must be > 0")
    a = 1.0 + float(exponent_extra)
    seq = [1.0 / (j ** a) for j in range(1, length + 1)]
    return _normalize(seq)


# ---------------------------------------------------------------------------
# SAFFRON implementation
# ---------------------------------------------------------------------------

class SAFFRON:
    """Adaptive online FDR control (SAFFRON) 

    * ``test_hypothesis(p_value, hypothesis_info=None) -> (is_rejected, alpha_t)``
    * ``get_summary() -> dict``

    Internals follow the formulas in Ramdas et al. (2018) for a *fixed* λ.
    """

    # ------------------------------------------------------------------
    # Construction / parameter handling
    # ------------------------------------------------------------------
    def __init__(self, alpha: float = 0.05, lambda_param: float = 0.5, gamma_param=0.5):
        if not (0 < alpha < 1):
            raise ValueError("alpha must lie in (0,1)")
        if not (0 < lambda_param < 1):
            raise ValueError("lambda_param must lie in (0,1)")

        self.alpha = float(alpha)
        self.lambda_param = float(lambda_param)  # SAFFRON candidate threshold λ

        # Build gamma schedule
        if isinstance(gamma_param, (list, tuple)):
            if len(gamma_param) == 0:
                raise ValueError("gamma_param sequence cannot be empty")
            self.gamma_seq = _normalize(gamma_param)
        else:  # scalar ⇒ interpret as exponent-extra
            self.gamma_seq = _make_power_gamma(float(gamma_param))

        # recommended W0: modest fraction of (1-λ)α, but < (1-λ)α.
        self.w0 = min(0.25 * (1.0 - self.lambda_param) * self.alpha,
                      0.99 * (1.0 - self.lambda_param) * self.alpha)

        # state trackers --------------------------------------------------
        self.t = 0  # total #tests processed so far
        self.p_values: List[float] = []   # history of p-values
        self.alpha_history: List[float] = []  # α_t values actually used
        self.is_rejected_history: List[bool] = []
        self.hypothesis_info: List[Any] = []

        # for summary convenience ---------------------------------------
        self.rejections: List[int] = []  # times of rejections (indices 1-based)
        self.candidate_mask: List[bool] = []  # whether p_t ≤ λ (candidates)

    # ------------------------------------------------------------------
    # Core SAFFRON recursion (fixed λ)
    # ------------------------------------------------------------------
    def _alpha_t(self, t: int) -> float:
        """Compute the SAFFRON test level α_t given history up to t-1.

        We directly implement Eq. 3 in Section 2.4 of the SAFFRON paper (constant λ):

            α_t = min{ λ,
                      w0 * γ_{k0} + ((1-λ)α - w0) * γ_{k1} + Σ_{j≥2} (1-λ)α * γ_{kj} }

        where k0 = t - C_{0+}(t),
              k1 = t - τ_1 - C_{1+}(t),
              kj = t - τ_j - C_{j+}(t) for j ≥ 2.

        Here C_{j+}(t) counts *candidates* (p ≤ λ) strictly between τ_j and t.
        τ_j is the index of the j-th rejection (τ_0 = 0 by convention).

        Implementation details:
        * We recompute all C_{j+}(t) by scanning the candidate mask.
        * γ is 1-indexed in the paper; we mirror that by adding 1 when indexing
          into ``self.gamma_seq`` (Python is 0-indexed).  If k exceeds the
          precomputed length, we append zeros on the fly.
        """
        λ = self.lambda_param
        α = self.alpha

        # gather rejection times (1-based indices). τ_0 = 0 by convention.
        τ = [0] + self.rejections[:]  # copy
        num_rej = len(self.rejections)

        # Precompute prefix counts of candidates for O(1) range queries.
        # candidate_mask is 0-based; element i corresponds to time i+1.
        cand_prefix = [0]
        csum = 0
        for c in self.candidate_mask:
            csum += int(c)
            cand_prefix.append(csum)

        def count_candidates(lo_exclusive: int, hi_inclusive: int) -> int:
            # lo_exclusive, hi_inclusive are *times* (1-based).  Count c in (lo_exclusive, hi_inclusive].
            return cand_prefix[hi_inclusive] - cand_prefix[lo_exclusive]

        # helper: safe γ lookup (extend with 0s if needed)
        def γ(idx: int) -> float:  # idx is 1-based per paper
            if idx <= 0:
                return 0.0
            if idx <= len(self.gamma_seq):
                return self.gamma_seq[idx - 1]
            # beyond precomputed tail ⇒ 0 (mass negligible); alternatively could extend power-law
            return 0.0

        # compute k0 = t - C_{0+}(t)
        C0p = count_candidates(0, t - 1)  # all candidates before t
        k0 = t - C0p

        # If no rejections yet, α_t reduces to w0*γ_k0.
        if num_rej == 0:
            return min(λ, self.w0 * γ(k0))

        # compute contribution from the first rejection separately
        τ1 = τ[1]
        C1p = count_candidates(τ1, t - 1)
        k1 = t - τ1 - C1p
        total = self.w0 * γ(k0) + ((1 - λ) * α - self.w0) * γ(k1)

        # j >= 2 rejections
        for j in range(2, num_rej + 1):
            τj = τ[j]
            Cjp = count_candidates(τj, t - 1)
            kj = t - τj - Cjp
            total += (1 - λ) * α * γ(kj)

        return min(λ, total)

    # ------------------------------------------------------------------
    # Public testing interface
    # ------------------------------------------------------------------
    def test_hypothesis(self, p_value: float, hypothesis_info: Any = None) -> Tuple[bool, float]:
        """Feed one p-value into the SAFFRON procedure.

        Returns
        -------
        (is_rejected, alpha_threshold)
            *is_rejected* is True if p_value ≤ α_t.
            *alpha_threshold* is the α_t used for this test.
        """
        if not (0.0 <= p_value <= 1.0):
            raise ValueError("p_value must lie in [0,1]")

        # determine time index of this test (1-based for formulas)
        self.t += 1
        t = self.t

        # compute α_t from past
        α_t = self._alpha_t(t)

        # observe p_t; update candidate mask
        is_candidate = p_value <= self.lambda_param
        self.candidate_mask.append(is_candidate)

        # determine rejection
        is_rejected = p_value <= α_t
        if is_rejected:
            self.rejections.append(t)

        # record history
        self.p_values.append(p_value)
        self.alpha_history.append(α_t)
        self.is_rejected_history.append(is_rejected)
        self.hypothesis_info.append(hypothesis_info)

        return is_rejected, α_t

    # ------------------------------------------------------------------
    # Summary / diagnostics
    # ------------------------------------------------------------------
    def get_summary(self) -> Dict[str, Any]:
        """Return a dictionary of summary statistics."""
        tot_tests = self.t
        tot_rej = len(self.rejections)
        λ = self.lambda_param
        α = self.alpha

        # Compute running FDP-hat_λ(t) (as defined in the paper) at the final time.
        #   FDP_hat_λ(T) = [Σ α_j * 1{p_j > λ} / (1-λ)] / max{1, R(T)}.
        # This is the quantity that SAFFRON keeps conservatively below α.
        denom = max(1, tot_rej)
        num = 0.0
        one_minus_λ = 1.0 - λ
        for aj, pj in zip(self.alpha_history, self.p_values):
            if pj > λ:
                num += aj / one_minus_λ
        fdp_hat = num / denom

        # empirical FDP (observed) at the end (naïve; for diagnostics only)
        #   = V / R, but V unknown; we report R/(#tests) as 'rejection_rate'.
        rejection_rate = tot_rej / max(1, tot_tests)

        # wealth notion: in SAFFRON we do not maintain a scalar wealth recursively
        # (wealth is implicit in the γ-schedule).  For API parity we report the
        # *effective* wealth W_eff = (1-λ)α * #rejections + w0 - Σ α_t.
        spent = sum(self.alpha_history)
        W_eff = self.w0 + (1.0 - λ) * α * tot_rej - spent

        return {
            "total_tests": tot_tests,
            "total_rejections": tot_rej,
            "current_wealth": W_eff,
            "active_candidates": sum(self.candidate_mask),  # count so far
            "rejection_rate": rejection_rate,
            "fdp_hat_lambda": fdp_hat,
        }

    # ------------------------------------------------------------------
    # Convenience: reset state
    # ------------------------------------------------------------------
    def reset(self):
        """Clear all accumulated state so the instance can be reused."""
        self.t = 0
        self.p_values.clear()
        self.alpha_history.clear()
        self.is_rejected_history.clear()
        self.hypothesis_info.clear()
        self.rejections.clear()
        self.candidate_mask.clear()


# ---------------------------------------------------------------------------
# Example usage (manual test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    rng = random.Random(0)
    proc = SAFFRON(alpha=0.05, lambda_param=0.5, gamma_param=0.5)

    # simulate 100 p-values, with 10 non-nulls at 0.01*U[0,1/5]
    pvals = []
    for i in range(100):
        if i < 10:  # non-null
            pvals.append(rng.random() * 0.01)
        else:
            pvals.append(rng.random())

    for p in pvals:
        rej, thr = proc.test_hypothesis(p)
        # print(f"p={p:.3g} thr={thr:.3g} rej={rej}")

    print(proc.get_summary())
