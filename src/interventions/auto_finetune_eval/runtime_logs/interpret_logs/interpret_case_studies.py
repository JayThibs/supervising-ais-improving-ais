import os
import sys
import time
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import math
from collections import defaultdict

# Load summary functions from ../../progressive_summary.py
# so we can drive the LLM for summary generation.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from stats import benjamini_hochberg_correction
from progressive_summary import select_hypotheses_for_summary, run_progressive_summary
from interpretation_helpers import extract_hypotheses_and_scores, round_float_df_to_sig_figs



def analyze_case_study_exps(
    paths_to_results: Optional[Dict[str, Dict[str, str]]] = None,
    alpha: float = 0.05,
    make_plots: bool = True,
    do_progressive_summaries: bool = False,
    summary_opts: Optional[Dict] = None,
    out_dir_tables: str = "tables_3/case_studies",
    out_dir_figs: str = "figs_3/case_studies",
    out_dir_summaries: str = "summaries_3/case_studies",
    n_perm_alt: int = 50000,        # permutations for alt-disc p-values (validation)
    rng_seed: int = 13,
) -> Dict[str, pd.DataFrame]:
    """
    Full pipeline (Sections 1-5).

    S1-S3 (per intervention, per dataset):
      - Dataset-level stats (BH on VALIDATION permutation p-values; no baselines):
          n_significant, mean validation AUC, mean cross-val AUC, mean validation AUC-based p-val, mean validation accuracy.
      - Token usage by query type: (labeler/discriminator/summarizer) x (input/output).
      - Optional progressive summaries saved to files.

    S4 (cross-intervention):
      - One figure (PDF): 3 subplots (datasets), each overlaid histograms for interventions; dashed line at min AUC among BH-significant.
      - One table: mean/std validation AUC vs. cross-val AUC by intervention x dataset.

    S5 (alternate Discriminator models):
      - Table (3 discriminators x 3 interventions = 9 rows; 5 cols):
          # BH-significant (validation permutation p-values), mean val AUC, mean CV AUC,
          mean val AUC-based p-val, mean val accuracy; values averaged across datasets.
      - Table (intervention x dataset = 9 rows; 6 cols):
          discriminator token usage only: gemini_in/out, qwen_in/out, gpt5nano_in/out.

    Baselines are excluded everywhere. Datasets have different hypothesis counts; no alignment.
    """        

    os.makedirs(out_dir_tables, exist_ok=True)
    os.makedirs(out_dir_figs, exist_ok=True)
    if do_progressive_summaries:
        os.makedirs(out_dir_summaries, exist_ok=True)

    datasets = ["Anthropic", "TruthfulQA", "Amazon BOLD"]
    interventions = list(paths_to_results.keys())

    # Discriminator keys & matching to model-id substrings for token accounting
    disc_keys = [
        ("gemini", "gemini"),          # main discriminator (gemini_2.5-flash-lite)
        ("qwen", "qwen"),              # qwen/qwen3-next-80b-a3b-instruct
        ("gpt-5-nano", "gpt-5"),       # openai/gpt-5-nano
    ]

    # ---------------- helpers ----------------
    def safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else np.nan

    def safe_std(arr):
        return float(np.std(arr)) if len(arr) > 0 else np.nan

    def slug(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

    def apply_bh(p_values, alpha_):
        arr = np.asarray(p_values, dtype=float)
        valid = np.isfinite(arr)
        if valid.sum() == 0:
            return 0, np.nan, np.zeros_like(arr, dtype=bool)
        n_sig, thresh, mask_valid = benjamini_hochberg_correction(list(arr[valid]), alpha=alpha_)
        mask = np.zeros_like(arr, dtype=bool)
        mask[valid] = mask_valid
        return int(n_sig), float(thresh), mask

    def auc_from_scores(y_true, scores):
        """ROC AUC via Mann–Whitney U ranks (sklearn-free)."""
        y = np.asarray(y_true, dtype=float)
        s = np.asarray(scores, dtype=float)
        if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or len(y) == 0:
            return np.nan
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        if n_pos == 0 or n_neg == 0:
            return np.nan
        n = len(s)
        order = np.argsort(s)
        sorted_s = s[order]
        ranks_sorted = np.arange(1, n + 1, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
                j += 1
            if j > i:
                avg_rank = (i + 1 + j + 1) / 2.0
                ranks_sorted[i:j + 1] = avg_rank
            i = j + 1
        ranks = np.empty_like(ranks_sorted)
        ranks[order] = ranks_sorted
        R_pos = ranks[y == 1].sum()
        U = R_pos - n_pos * (n_pos + 1) / 2.0
        auc = U / (n_pos * n_neg)
        return float(auc)

    def auc_pvalue_normal_approx(auc, n_pos, n_neg):
        """Two-sided p against AUC=0.5 (Hanley & McNeil 1982)."""
        if n_pos == 0 or n_neg == 0 or not np.isfinite(auc):
            return np.nan
        A = auc
        Q1 = A / (2.0 - A) if A < 2.0 else np.nan
        Q2 = 2 * A * A / (1 + A) if (1 + A) != 0 else np.nan
        var = (A * (1 - A) + (n_pos - 1) * (Q1 - A * A) + (n_neg - 1) * (Q2 - A * A)) / (n_pos * n_neg)
        if not np.isfinite(var) or var <= 0:
            return np.nan
        se = math.sqrt(var)
        if se == 0:
            return np.nan
        z = (A - 0.5) / se
        Phi = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        p = 2.0 * (1.0 - Phi(abs(z)))
        return float(min(max(p, 0.0), 1.0))

    def permutation_pvalue_auc(y_true, scores, n_perm=200, rng=None):
        """
        Two-sided permutation test for AUC vs 0.5 (shuffling labels).
        Returns np.nan if degenerate (no pos/neg).
        """
        y = np.asarray(y_true, dtype=int)
        s = np.asarray(scores, dtype=float)
        if len(y) == 0 or len(y) != len(s):
            return np.nan
        n_pos = int(np.sum(y == 1))
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            return np.nan
        obs_auc = auc_from_scores(y, s)
        if not np.isfinite(obs_auc):
            return np.nan
        obs = abs(obs_auc - 0.5)
        if rng is None:
            rng_local = np.random.default_rng()
        else:
            rng_local = rng
        cnt = 0
        for _ in range(max(1, int(n_perm))):
            y_perm = y.copy()
            rng_local.shuffle(y_perm)
            perm_auc = auc_from_scores(y_perm, s)
            if not np.isfinite(perm_auc):
                continue
            if abs(perm_auc - 0.5) >= obs:
                cnt += 1
        p = (cnt + 1.0) / (n_perm + 1.0)
        return float(p)

    def best_threshold_accuracy(y_true, scores):
        """Max validation accuracy from a single threshold on scores."""
        y = np.asarray(y_true, dtype=int)
        s = np.asarray(scores, dtype=float)
        n = len(y)
        if n == 0 or n != len(s):
            return np.nan, None
        order = np.argsort(-s)  # descending by score
        y_sorted = y[order]
        tp_prefix = np.cumsum(y_sorted)
        total_pos = int(np.sum(y))
        total_neg = n - total_pos
        idx = np.arange(n)
        accs = (2.0 * tp_prefix + total_neg - (idx + 1)) / float(n)
        best_i = int(np.nanargmax(accs))
        return float(accs[best_i]), float(s[order[best_i]])

    def group_tokens_by_kind(h, kind=None):
        """
        Sum tokens from API logs in `h` filtered by query_types == kind if provided.
        Returns (input_sum, output_sum), and also by model when needed elsewhere.
        """
        types = h.get("query_types", []) or []
        ins = h.get("api_model_query_input_lengths", []) or []
        outs = h.get("api_model_query_output_lengths", []) or []
        mids = h.get("api_model_str_ids", []) or []
        L = min(len(types), len(ins), len(outs), len(mids))
        types, ins, outs, mids = types[:L], ins[:L], outs[:L], mids[:L]

        total_in = total_out = 0
        by_model = defaultdict(lambda: {"in": 0, "out": 0})
        for t, i, o, m in zip(types, ins, outs, mids):
            if (kind is None) or (t == kind):
                total_in += int(i)
                total_out += int(o)
                by_model[m]["in"] += int(i)
                by_model[m]["out"] += int(o)
        return total_in, total_out, by_model

    # ---------------- parse all logs ----------------
    parsed: Dict[str, Dict[str, dict]] = {interv: {} for interv in interventions}
    for interv, ds_map in paths_to_results.items():
        for ds, path in ds_map.items():
            print(f"Processing {interv} on {ds}")
            if not path or not os.path.exists(path):
                print(f"[WARN] Missing log file for {interv} / {ds}: {path}")
                parsed[interv][ds] = None
                continue
            parsed[interv][ds] = extract_hypotheses_and_scores(path)

    # ---------------- S1–S3 per-intervention outputs ----------------
    per_intervention_stats: Dict[str, pd.DataFrame] = {}
    per_intervention_tokens: Dict[str, pd.DataFrame] = {}

    # For S4 hist figure
    aucs_by_dataset_by_intervention: Dict[str, Dict[str, List[float]]] = {d: {} for d in datasets}
    min_sig_auc_by_dataset_by_intervention: Dict[str, Dict[str, float]] = {d: {} for d in datasets}

    # For S4 AUC vs CV table
    auc_vs_cv_rows: List[Dict] = []

    for interv in interventions:
        stats_rows = []
        token_rows = []

        union_validated_hyps = []
        seen_norm = set()

        for ds in datasets:
            data = parsed[interv].get(ds)
            if not data or len(data.get("hypotheses", [])) == 0:
                print(f"No data for {interv} on {ds}")
                stats_rows.append({
                    "dataset": ds,
                    "n_hypotheses": 0,
                    "n_significant": 0,
                    "mean_val_auc": np.nan,
                    "mean_cv_auc": np.nan,
                    "mean_val_auc_pval": np.nan,
                    "mean_accuracy": np.nan,
                    "bh_threshold": np.nan,
                    "min_sig_auc": np.nan,
                })
                token_rows.append({
                    "dataset": ds,
                    "disc_input_tokens": 0, "disc_output_tokens": 0,
                    "labeler_input_tokens": 0, "labeler_output_tokens": 0,
                    "summarizer_input_tokens": 0, "summarizer_output_tokens": 0,
                })
                continue

            # Align validation arrays (no baselines)
            L = min(
                len(data.get("hypotheses", [])),
                len(data.get("accuracies", [])),
                len(data.get("auc_scores", [])),
                len(data.get("permutation_p_values", [])),
            )
            if L == 0:
                print(f"No data for {interv} on {ds}")
                continue

            val_auc = np.asarray(data["auc_scores"][:L], dtype=float)
            val_acc = np.asarray(data["accuracies"][:L], dtype=float)
            perm_p = np.asarray(data["permutation_p_values"][:L], dtype=float)

            # BH on validation permutation p-values
            n_sig, bh_thresh, sig_mask = apply_bh(perm_p, alpha)
            min_sig_auc = float(np.nanmin(val_auc[sig_mask])) if n_sig > 0 else np.nan

            # For S4 hist annotations
            aucs_by_dataset_by_intervention[ds][interv] = val_auc.tolist()
            min_sig_auc_by_dataset_by_intervention[ds][interv] = min_sig_auc

            # Validation AUC-based p-values computed from labels/scores
            y_lists = data.get("discriminative_model_validation_true_labels", []) or []
            s_lists = data.get("discriminative_model_validation_discrim_scores", []) or []
            pvals_from_auc = []
            if len(y_lists) and len(s_lists):
                LL = min(len(y_lists), len(s_lists))
                for i in range(LL):
                    y_true = y_lists[i]
                    s = s_lists[i]
                    if not y_true or not s or len(y_true) != len(s):
                        continue
                    y_arr = np.asarray(y_true)
                    n_pos = int(np.sum(y_arr == 1))
                    n_neg = int(np.sum(y_arr == 0))
                    if n_pos == 0 or n_neg == 0:
                        continue
                    auc_val = auc_from_scores(y_true, s)
                    p_auc = auc_pvalue_normal_approx(auc_val, n_pos, n_neg)
                    if np.isfinite(p_auc):
                        pvals_from_auc.append(p_auc)

            # Cross-val AUCs (comparison only)
            cv_aucs = np.asarray(data.get("cross_validated_auc_scores", []) or [], dtype=float)

            stats_rows.append({
                "dataset": ds,
                "n_hypotheses": int(L),
                "n_significant": int(n_sig),
                "mean_val_auc": safe_mean(val_auc),
                "mean_cv_auc": safe_mean(cv_aucs),
                "mean_val_auc_pval": safe_mean(pvals_from_auc),
                "mean_accuracy": safe_mean(val_acc),
                "bh_threshold": float(bh_thresh) if np.isfinite(bh_thresh) else np.nan,
                "min_sig_auc": min_sig_auc,
            })

            # Token usage by query type
            d_in, d_out, _ = group_tokens_by_kind(data, kind="discriminator")
            l_in, l_out, _ = group_tokens_by_kind(data, kind="labeler")
            s_in, s_out, _ = group_tokens_by_kind(data, kind="summarizer")
            token_rows.append({
                "dataset": ds,
                "disc_input_tokens": int(d_in) / int(L),
                "disc_output_tokens": int(d_out) / int(L),
                "labeler_input_tokens": int(l_in) / int(L),
                "labeler_output_tokens": int(l_out) / int(L),
                "summarizer_input_tokens": int(s_in) / int(n_sig / 10) if n_sig > 10 else 0,
                "summarizer_output_tokens": int(s_out) / int(n_sig / 10) if n_sig > 10 else 0,
            })

            # (Optional) progressive summaries
            if do_progressive_summaries and summary_opts:
                selected_hypotheses = select_hypotheses_for_summary(
                    data=data,
                    filter_type=summary_opts.get("filter_type", "permutation"),
                    bh_alpha=summary_opts.get("bh_alpha", alpha),
                    min_accuracy=summary_opts.get("min_accuracy", None),
                    top_k=summary_opts.get("top_k", None),
                )
                # Add to union with de-duplication by normalized text
                for key in selected_hypotheses:
                    norm_key = re.sub(r"\s+", " ", key).strip().lower()
                    if norm_key not in seen_norm:
                        seen_norm.add(norm_key)
                        union_validated_hyps.append(norm_key)

            # For S4 AUC vs CV table
            auc_vs_cv_rows.append({
                "intervention": interv,
                "dataset": ds,
                "mean_val_auc": safe_mean(val_auc),
                "std_val_auc": safe_std(val_auc),
                "mean_cv_auc": safe_mean(cv_aucs),
                "std_cv_auc": safe_std(cv_aucs),
            })
        # After we've scanned all datasets for this intervention,
        # run ONE progressive summary on the UNION of validated hypotheses.
        if do_progressive_summaries and summary_opts:
            if len(union_validated_hyps) > 0:
                # If user asked for top_k, apply it ACROSS the union (stable keep-order).
                top_k = summary_opts.get("top_k", None)
                hyps_for_summary = union_validated_hyps[:top_k] if (top_k is not None) else union_validated_hyps
                print(f"[Progressive summary] Running summary on {len(hyps_for_summary)} hypotheses...")
                print(f"[Progressive summary] Hypotheses: {hyps_for_summary}")
                run_progressive_summary(
                    name=f"{interv}__ALL_DATASETS_UNION",
                    hypotheses=hyps_for_summary,
                    api_provider=summary_opts["api_provider"],
                    model_str=summary_opts["model_str"],
                    api_key_path=summary_opts.get("api_key_path"),
                    save_dir=os.path.join(out_dir_summaries, slug(interv)),
                    max_tokens=summary_opts.get("max_tokens", 2000),
                    temperature=summary_opts.get("temperature", 0.7),
                    max_thinking_tokens=summary_opts.get("max_thinking_tokens"),
                    verbose=summary_opts.get("verbose", False),
                )

        # Save S1 & S2 tables per intervention# Save S1 & S2 tables per intervention
        df_stats = pd.DataFrame(stats_rows)
        df_tokens = pd.DataFrame(token_rows)

        df_stats = round_float_df_to_sig_figs(df_stats, sig=3)
        df_tokens = round_float_df_to_sig_figs(df_tokens, sig=3)

        per_intervention_stats[interv] = df_stats
        per_intervention_tokens[interv] = df_tokens

        df_stats.to_csv(os.path.join(out_dir_tables, f"{slug(interv)}_dataset_stats.tsv"), sep="\t", index=False)
        df_tokens.to_csv(os.path.join(out_dir_tables, f"{slug(interv)}_token_usage.tsv"), sep="\t", index=False)
        print(f"[Saved] {os.path.join(out_dir_tables, f'{slug(interv)}_dataset_stats.tsv')}")
        print(f"[Saved] {os.path.join(out_dir_tables, f'{slug(interv)}_token_usage.tsv')}")

    # ---------------- S4: cross‑intervention histograms + table ----------------
    hist_path = None
    if make_plots:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for ax, ds in zip(axes, datasets):
            any_data = False
            for interv in interventions:
                vals = aucs_by_dataset_by_intervention.get(ds, {}).get(interv, [])
                if not vals:
                    continue
                any_data = True
                ax.hist(vals, bins=np.linspace(0.0, 1.0, 21), alpha=0.45, label=f"{interv} (n={len(vals)})")
                ms = min_sig_auc_by_dataset_by_intervention.get(ds, {}).get(interv, np.nan)
                if np.isfinite(ms):
                    ax.axvline(ms, linestyle="--", linewidth=1.5, label=f"{interv} min-sig={ms:.3f}")
            ax.set_title(ds)
            ax.set_xlabel("Validation AUC")
            if any_data:
                ax.legend(fontsize=8)
        axes[0].set_ylabel("Count")
        fig.suptitle("Validation AUC distributions by dataset (overlaid by intervention)")
        fig.tight_layout()
        hist_path = os.path.join(out_dir_figs, "validation_auc_histograms_by_dataset.pdf")
        plt.savefig(hist_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {hist_path}")

    auc_vs_cv_df = pd.DataFrame(auc_vs_cv_rows, columns=[
        "intervention", "dataset",
        "mean_val_auc", "std_val_auc", "mean_cv_auc", "std_cv_auc"
    ])
    auc_vs_cv_path = os.path.join(out_dir_tables, "validation_vs_crossval_auc.tsv")
    auc_vs_cv_df = round_float_df_to_sig_figs(auc_vs_cv_df, sig=3)
    auc_vs_cv_df.to_csv(auc_vs_cv_path, sep="\t", index=False)
    print(f"[Saved] {auc_vs_cv_path}")

    # ---------------- S5: alternate Discriminator analysis ----------------
    rng = np.random.default_rng(rng_seed)

    # Helper: compute validation metrics for a (labels_list, scores_list) collection
    def compute_val_metrics_from_lists(y_lists, s_lists, do_perm=True):
        aucs, accs, pvals_auc, pvals_perm = [], [], [], []
        L = min(len(y_lists or []), len(s_lists or []))
        for i in range(L):
            y_true = y_lists[i]
            s = s_lists[i]
            if not y_true or not s or len(y_true) != len(s):
                continue
            a = auc_from_scores(y_true, s)
            acc, _ = best_threshold_accuracy(y_true, s)
            y_arr = np.asarray(y_true)
            n_pos = int(np.sum(y_arr == 1))
            n_neg = int(np.sum(y_arr == 0))
            p_auc = auc_pvalue_normal_approx(a, n_pos, n_neg)
            if do_perm:
                p_perm = permutation_pvalue_auc(y_true, s, n_perm=n_perm_alt, rng=rng)
            else:
                p_perm = np.nan
            aucs.append(a)
            accs.append(acc)
            pvals_auc.append(p_auc)
            pvals_perm.append(p_perm)
        return aucs, accs, pvals_auc, pvals_perm

    # Helper: compute CV AUCs from (labels_list, scores_list) collection
    def compute_cv_aucs_from_lists(y_lists, s_lists):
        aucs = []
        L = min(len(y_lists or []), len(s_lists or []))
        for i in range(L):
            y_true = y_lists[i]
            s = s_lists[i]
            if not y_true or not s or len(y_true) != len(s):
                continue
            a = auc_from_scores(y_true, s)
            if np.isfinite(a):
                aucs.append(a)
        return aucs

    # --- (a) Summary table averaged over datasets: discriminator × intervention ---
    alt_summary_rows = []
    # --- (a.2) Detailed table: discriminator × intervention × dataset ---
    detailed_stats_rows = []

    for disc_key, _needle in disc_keys:
        for interv in interventions:
            # Aggregate across datasets
            per_ds_counts = []
            per_ds_auc_means = []
            per_ds_auc_cv_means = []
            per_ds_auc_p_means = []
            per_ds_acc_means = []

            for ds in datasets:
                data = parsed[interv].get(ds)
                if not data:
                    continue

                # Validation metrics by discriminator
                if disc_key == "gemini":
                    # main model: use precomputed arrays directly
                    L = min(
                        len(data.get("hypotheses", [])),
                        len(data.get("auc_scores", [])),
                        len(data.get("accuracies", [])),
                        len(data.get("permutation_p_values", [])),
                    )
                    if L == 0:
                        continue
                    val_auc_list = list(np.asarray(data["auc_scores"][:L], dtype=float))
                    val_acc_list = list(np.asarray(data["accuracies"][:L], dtype=float))
                    perm_p_list = list(np.asarray(data["permutation_p_values"][:L], dtype=float))
                    # For comparability, compute AUC-based p-values (normal approx) as well
                    y_lists = data.get("discriminative_model_validation_true_labels", []) or []
                    s_lists = data.get("discriminative_model_validation_discrim_scores", []) or []
                    pvals_auc = []
                    LL = min(len(y_lists), len(s_lists), L)
                    for i in range(LL):
                        y_true = y_lists[i]
                        s = s_lists[i]
                        if not y_true or not s or len(y_true) != len(s):
                            continue
                        y_arr = np.asarray(y_true)
                        n_pos = int(np.sum(y_arr == 1))
                        n_neg = int(np.sum(y_arr == 0))
                        a = auc_from_scores(y_true, s)
                        p = auc_pvalue_normal_approx(a, n_pos, n_neg)
                        if np.isfinite(p):
                            pvals_auc.append(p)
                    # Cross-val AUCs
                    cv_aucs = list(np.asarray(data.get("cross_validated_auc_scores", []) or [], dtype=float))

                else:
                    # alternate discriminator: compute from stored lists
                    y_lists = data.get("discriminative_model_validation_true_labels", []) or []
                    s_alt_lists = (data.get("discriminative_model_validation_discrim_alternate_scores", {}) or {}).get(disc_key, []) or []
                    val_auc_list, val_acc_list, pvals_auc, perm_p_list = compute_val_metrics_from_lists(
                        y_lists, s_alt_lists, do_perm=True
                    )
                    # Cross-val AUCs from cross-validation lists if present
                    y_cv = data.get("discriminative_model_cross_validation_true_labels", []) or []
                    s_cv_alt = (data.get("discriminative_model_cross_validation_discrim_alternate_scores", {}) or {}).get(disc_key, []) or []
                    cv_aucs = compute_cv_aucs_from_lists(y_cv, s_cv_alt)

                # BH significance based on VALIDATION permutation p-values for this discriminator
                n_sig, bh_thresh, sig_mask = apply_bh(perm_p_list, alpha)
                
                # Calculate statistics for this dataset
                mean_val_auc = safe_mean(val_auc_list)
                mean_cv_auc = safe_mean(cv_aucs)
                mean_val_auc_pval = safe_mean(pvals_auc)
                mean_acc = safe_mean(val_acc_list)
                min_sig_auc = float(np.nanmin(np.array(val_auc_list)[sig_mask])) if n_sig > 0 else np.nan

                per_ds_counts.append(float(n_sig))
                per_ds_auc_means.append(mean_val_auc)
                per_ds_auc_cv_means.append(mean_cv_auc)
                per_ds_auc_p_means.append(mean_val_auc_pval)
                per_ds_acc_means.append(mean_acc)
                
                # Add to detailed rows
                detailed_stats_rows.append({
                    "discriminator": disc_key,
                    "intervention": interv,
                    "dataset": ds,
                    "n_significant": int(n_sig),
                    "mean_val_auc": mean_val_auc,
                    "mean_cv_auc": mean_cv_auc,
                    "mean_val_auc_pval": mean_val_auc_pval,
                    "mean_accuracy": mean_acc,
                    "min_sig_auc": min_sig_auc,
                })

            if len(per_ds_counts) == 0:
                # no data for this combination
                alt_summary_rows.append({
                    "discriminator": disc_key,
                    "intervention": interv,
                    "avg_n_significant": np.nan,
                    "mean_val_auc": np.nan,
                    "mean_cv_auc": np.nan,
                    "mean_val_auc_pval": np.nan,
                    "mean_accuracy": np.nan,
                })
            else:
                alt_summary_rows.append({
                    "discriminator": disc_key,
                    "intervention": interv,
                    "avg_n_significant": safe_mean(per_ds_counts),   # averaged over datasets (per spec)
                    "mean_val_auc": safe_mean(per_ds_auc_means),
                    "mean_cv_auc": safe_mean(per_ds_auc_cv_means),
                    "mean_val_auc_pval": safe_mean(per_ds_auc_p_means),
                    "mean_accuracy": safe_mean(per_ds_acc_means),
                })

    alt_summary_df = pd.DataFrame(alt_summary_rows, columns=[
        "discriminator", "intervention",
        "avg_n_significant", "mean_val_auc", "mean_cv_auc", "mean_val_auc_pval", "mean_accuracy"
    ])
    alt_summary_path = os.path.join(out_dir_tables, "alternate_discriminators__summary_by_discriminator_and_intervention.tsv")
    alt_summary_df = round_float_df_to_sig_figs(alt_summary_df, sig=3)
    alt_summary_df.to_csv(alt_summary_path, sep="\t", index=False)
    print(f"[Saved] {alt_summary_path}")

    # Save detailed table
    detailed_stats_df = pd.DataFrame(detailed_stats_rows, columns=[
        "discriminator", "intervention", "dataset",
        "n_significant", "mean_val_auc", "mean_cv_auc", "mean_val_auc_pval", "mean_accuracy", "min_sig_auc"
    ])
    detailed_stats_path = os.path.join(out_dir_tables, "alternate_discriminators__detailed_stats_by_dataset.tsv")
    detailed_stats_df = round_float_df_to_sig_figs(detailed_stats_df, sig=3)
    detailed_stats_df.to_csv(detailed_stats_path, sep="\t", index=False)
    print(f"[Saved] {detailed_stats_path}")

    # --- (b) Token usage table by intervention × dataset for Discriminator models only ---
    def which_disc_bucket(model_id: str) -> Optional[str]:
        mid = (model_id or "").lower()
        if "gemini" in mid:
            return "gemini"
        if "qwen" in mid:
            return "qwen"
        if "gpt-5" in mid or "gpt5" in mid or "nano" in mid:
            return "gpt-5-nano"
        return None

    disc_token_rows = []
    for interv in interventions:
        for ds in datasets:
            data = parsed[interv].get(ds)
            if not data:
                disc_token_rows.append({
                    "intervention": interv, "dataset": ds,
                    "gemini_in": 0, "gemini_out": 0,
                    "qwen_in": 0, "qwen_out": 0,
                    "gpt5nano_in": 0, "gpt5nano_out": 0,
                })
                continue
            L = min(
                len(data.get("hypotheses", [])),
                len(data.get("auc_scores", [])),
                len(data.get("accuracies", [])),
                len(data.get("permutation_p_values", [])),
            )
            # Only discriminator queries
            _, _, by_model = group_tokens_by_kind(data, kind="discriminator")
            buckets = {"gemini": {"in": 0, "out": 0},
                       "qwen": {"in": 0, "out": 0},
                       "gpt-5-nano": {"in": 0, "out": 0}}
            for mid, tok in by_model.items():
                b = which_disc_bucket(mid)
                if b in buckets:
                    buckets[b]["in"] += int(tok["in"])
                    buckets[b]["out"] += int(tok["out"])
            disc_token_rows.append({
                "intervention": interv, "dataset": ds,
                "gemini_in": buckets["gemini"]["in"] / L,
                "gemini_out": buckets["gemini"]["out"] / L,
                "qwen_in": buckets["qwen"]["in"] / L,
                "qwen_out": buckets["qwen"]["out"] / L,
                "gpt5nano_in": buckets["gpt-5-nano"]["in"] / L,
                "gpt5nano_out": buckets["gpt-5-nano"]["out"] / L,
            })

    disc_tokens_df = pd.DataFrame(disc_token_rows, columns=[
        "intervention", "dataset",
        "gemini_in", "gemini_out",
        "qwen_in", "qwen_out",
        "gpt5nano_in", "gpt5nano_out",
    ])
    disc_tokens_path = os.path.join(out_dir_tables, "alternate_discriminators__discriminator_token_usage_by_model.tsv")
    disc_tokens_df = round_float_df_to_sig_figs(disc_tokens_df, sig=3)
    disc_tokens_df.to_csv(disc_tokens_path, sep="\t", index=False)
    print(f"[Saved] {disc_tokens_path}")
    


    # ---------------- return bundle ----------------
    out: Dict[str, pd.DataFrame] = {
        "dataset_stats_by_intervention": per_intervention_stats,         # dict[str, DataFrame]
        "token_usage_by_intervention": per_intervention_tokens,          # dict[str, DataFrame]
        "validation_vs_crossval_auc": auc_vs_cv_df,                      # DataFrame
        "alt_discriminator_summary": alt_summary_df,                     # DataFrame
        "alt_discriminator_detailed": detailed_stats_df,                 # DataFrame
        "alt_discriminator_token_usage": disc_tokens_df,                 # DataFrame
    }
    # Helpful non-tabular artifacts
    if hist_path:
        out["histograms_pdf_path"] = hist_path

    return out
