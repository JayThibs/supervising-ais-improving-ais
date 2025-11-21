import os
import sys
import time
import ast
import pandas as pd
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from transformers import BitsAndBytesConfig
from parsing_code import parse_number_list
import itertools
import math
from collections import defaultdict

# Load summary functions from ../../progressive_summary.py
# so we can drive the LLM for summary generation.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from stats import benjamini_hochberg_correction
from progressive_summary import select_hypotheses_for_summary, run_progressive_summary, score_hypotheis_diversity
from validated_comparison_tools import read_past_embeddings_or_generate_new

# read_past_embeddings_or_generate_new usage:
# def read_past_embeddings_or_generate_new(
#         decoded_strs: List[str], 
#         local_embedding_model_str: str = "intfloat/multilingual-e5-large-instruct", 
#         recompute_embeddings: bool = True, 
#         save_embeddings: bool = False, 
#         tqdm_disable: bool = True, 
#         bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_8bit=True)
#         ) -> List[List[float]]:

# Round float columns to 2 significant figures for nicer reporting
def round_float_df_to_sig_figs(df: pd.DataFrame, sig: int = 2) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=["float", "float32", "float64"]).columns
    if len(float_cols) == 0:
        return df

    def _round_val(x):
        if pd.isna(x) or not np.isfinite(x):
            return x
        try:
            return float(f"{float(x):.{sig}g}")
        except Exception:
            return x

    df[float_cols] = df[float_cols].applymap(_round_val)
    return df

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

    # ---------------- default mapping ----------------
    if paths_to_results is None:
        paths_to_results = {
            'ROME-10': {
                'TruthfulQA': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_truthfulqa_FINAL_SOTA_runtime_log.txt',
                'Anthropic': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_anthropic_FINAL_SOTA_runtime_log.txt',
                'Amazon BOLD': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_amazon_bold_FINAL_SOTA_runtime_log.txt',
            },
            'R1 distillation': {
                'Anthropic': '../intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_FINAL_SOTA_runtime_log.txt',
                'TruthfulQA': '../intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_TruthfulQA_FINAL_SOTA_runtime_log.txt',
                'Amazon BOLD': '../intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gemini_2.5-flash-lite__gpt_5_high_thinking_diversified_SCB_prompt_amazon_bold_FINAL_SOTA_runtime_log.txt',
            },
            'WIHP': {
                'Anthropic': '../intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_diversified_SCB_prompt_anthropic_FINAL_SOTA_try_2_runtime_log.txt',
                'TruthfulQA': '../intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_diversified_SCB_prompt_truthfulqa_FINAL_SOTA_runtime_log.txt',
                'Amazon BOLD': '../intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_diversified_SCB_prompt_amazon_bold_FINAL_SOTA_runtime_log.txt',
            },
        }

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
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            parsed[interv][ds] = extract_hypotheses_and_scores(lines)

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


def analyze_ablation_exps(
    alpha: float = 0.05,
    out_dir_tables: str = "tables_3/ablation",
    out_dir_figs: str = "figs_3/ablation",
    make_plots: bool = False,
    max_kmeans_k: int = 15,
    kmeans_improvement_threshold: float = 0.02,
    recompute_embeddings: bool = False,
    save_embeddings: bool = False,
    api_provider: str = "openrouter",
    model_str: str = "openai/gpt-5-2025-08-07",
    api_key_path: str = "../../../data/api_keys/openrouter_key.txt",
    discriminative_model_str: str = "google/gemini-2.5-flash-lite-preview-09-2025",
    labeler_model_str: str = "openai/gpt-5-2025-08-07",
    summarizer_model_str: str = "openai/gpt-5-2025-08-07",
    max_tokens: int = 20000,
    temperature: float = 1.0,
    max_thinking_tokens: int = None,
) -> Dict[str, pd.DataFrame]:
    """
    Analyze ablation experiments (Anthropic dataset only).

    Experiments:
      - Interventions: R1 distillation, ROME-10, WIHP
      - Diversification methods: current_diversification, narrowed_diversification, no_diversification
      - sampled_texts_per_cluster: 10, 20, 30

    For each (intervention, diversification_method, sampled_texts_per_cluster) experiment:
      - Use validation permutation p-values (BH-corrected) to count significant hypotheses.
      - Compute mean validation AUC, mean cross-validation AUC, mean validation accuracy.
      - Compute hypothesis diversity:
          diversity = 1 - mean_pairwise_jaccard_over_word_sets(hypotheses)
      - Embed hypotheses and estimate number of clusters via k-means elbow.

      - Token usage:
          - Sum input/output tokens for labeler / discriminator / summarizer query types.

    Outputs (saved under out_dir_tables):
      - ablation_stats_by_experiment.tsv               (per experiment)
      - ablation_token_usage_by_experiment.tsv        (per experiment)
      - ablation_stats_by_diversification_and_stpc.tsv
      - ablation_token_usage_by_diversification_and_stpc.tsv
      - ablation_labeler_io_by_experiment.tsv         (labeler in/out and ratio per experiment)

    Returns a dict of DataFrames with the same content.
    """

    os.makedirs(out_dir_tables, exist_ok=True)
    os.makedirs(out_dir_figs, exist_ok=True)

    # ----------- experiment paths (Anthropic only) -----------
    paths_to_results = {
        'R1 distillation': {
            'current_diversification': {
                '10_stpc': '../hparam_sweep/intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gemini_2.5-flash-lite__gpt_5_high_thinking_diversified_SCB_prompt_anthropic_10_stpc_FINAL_SOTA_runtime_log.txt',
                '20_stpc': '../intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_FINAL_SOTA_runtime_log.txt',
                '30_stpc': '../hparam_sweep/intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gemini_2.5-flash-lite__gpt_5_high_thinking_diversified_SCB_prompt_anthropic_30_stpc_FINAL_SOTA_runtime_log.txt',
            },
            'narrowed_diversification': {
                '10_stpc': '../hparam_sweep/intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gemini_2.5-flash-lite__gpt_5_high_thinking_narrowed_diversified_SCB_prompt_anthropic_10_stpc_FINAL_SOTA_runtime_log.txt',
                '20_stpc': '../hparam_sweep/intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gemini_2.5-flash-lite__gpt_5_high_thinking_narrowed_diversified_SCB_prompt_anthropic_20_stpc_FINAL_SOTA_runtime_log.txt',
                '30_stpc': '../hparam_sweep/intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gemini_2.5-flash-lite__gpt_5_high_thinking_narrowed_diversified_SCB_prompt_anthropic_30_stpc_FINAL_SOTA_runtime_log.txt',
            },
            'no_diversification': {
                '10_stpc': '../hparam_sweep/intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gemini_2.5-flash-lite__gpt_5_high_thinking_no_diversification_SCB_prompt_anthropic_10_stpc_FINAL_SOTA_runtime_log.txt',
                '20_stpc': '../hparam_sweep/intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gemini_2.5-flash-lite__gpt_5_high_thinking_no_diversification_SCB_prompt_anthropic_20_stpc_FINAL_SOTA_runtime_log.txt',
                '30_stpc': '../hparam_sweep/intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gemini_2.5-flash-lite__gpt_5_high_thinking_no_diversification_SCB_prompt_anthropic_30_stpc_FINAL_SOTA_runtime_log.txt',
            },
        },
        'ROME-10': {
            'current_diversification': {
                '10_stpc': '../hparam_sweep/intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_anthropic_10_stpc_FINAL_SOTA_runtime_log.txt',
                '20_stpc': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_anthropic_FINAL_SOTA_runtime_log.txt',
                '30_stpc': '../hparam_sweep/intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_anthropic_30_stpc_FINAL_SOTA_runtime_log.txt',
            },
            'narrowed_diversification': {
                '10_stpc': '../hparam_sweep/intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_narrowed_diversified_SCB_prompt_anthropic_10_stpc_FINAL_SOTA_runtime_log.txt',
                '20_stpc': '../hparam_sweep/intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_narrowed_diversified_SCB_prompt_anthropic_20_stpc_FINAL_SOTA_runtime_log.txt',
                '30_stpc': '../hparam_sweep/intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_narrowed_diversified_SCB_prompt_anthropic_30_stpc_FINAL_SOTA_runtime_log.txt',
            },
            'no_diversification': {
                '10_stpc': '../hparam_sweep/intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_no_diversification_SCB_prompt_anthropic_10_stpc_FINAL_SOTA_runtime_log.txt',
                '20_stpc': '../hparam_sweep/intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_no_diversification_SCB_prompt_anthropic_20_stpc_FINAL_SOTA_runtime_log.txt',
                '30_stpc': '../hparam_sweep/intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_no_diversification_SCB_prompt_anthropic_30_stpc_FINAL_SOTA_runtime_log.txt',
            },
        },
        'WIHP': {
            'current_diversification': {
                '10_stpc': '../hparam_sweep/intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_diversified_SCB_prompt_anthropic_10_stpc_FINAL_SOTA_try_2_runtime_log.txt',
                '20_stpc': '../intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_diversified_SCB_prompt_anthropic_FINAL_SOTA_try_2_runtime_log.txt',
                '30_stpc': '../hparam_sweep/intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_diversified_SCB_prompt_anthropic_30_stpc_FINAL_SOTA_runtime_log.txt',
            },
            'narrowed_diversification': {
                '10_stpc': '../hparam_sweep/intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_narrowed_diversified_SCB_prompt_anthropic_10_stpc_FINAL_SOTA_runtime_log.txt',
                '20_stpc': '../hparam_sweep/intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_narrowed_diversified_SCB_prompt_anthropic_20_stpc_FINAL_SOTA_runtime_log.txt',
                '30_stpc': '../hparam_sweep/intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_narrowed_diversified_SCB_prompt_anthropic_30_stpc_FINAL_SOTA_runtime_log.txt',
            },
            'no_diversification': {
                '10_stpc': '../hparam_sweep/intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_no_diversification_SCB_prompt_anthropic_10_stpc_FINAL_SOTA_runtime_log.txt',
                '20_stpc': '../hparam_sweep/intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_no_diversification_SCB_prompt_anthropic_20_stpc_FINAL_SOTA_runtime_log.txt',
                '30_stpc': '../hparam_sweep/intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_no_diversification_SCB_prompt_anthropic_30_stpc_FINAL_SOTA_runtime_log.txt',
            },
        },
    }

    # ----------- helpers -----------
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
            return 0, np.nan
        n_sig, thresh, _ = benjamini_hochberg_correction(list(arr[valid]), alpha=alpha_)
        return int(n_sig), float(thresh)

    def tokenize_words(s: str) -> set:
        # Simple word-level tokenization for Jaccard (lowercased, strip punctuation boundaries)
        # You can adjust to char 3-grams if desired.
        tokens = re.findall(r"\w+", s.lower())
        return set(tokens)

    def hypothesis_jaccard_diversity(hypotheses: List[str]) -> float:
        """
        1 - mean pairwise Jaccard over word-sets across hypotheses.
        Returns np.nan if < 2 hypotheses.
        """
        if len(hypotheses) < 2:
            return np.nan
        sets = [tokenize_words(h) for h in hypotheses]
        jaccs = []
        for i, j in itertools.combinations(range(len(sets)), 2):
            a, b = sets[i], sets[j]
            if not a and not b:
                continue
            inter = len(a & b)
            union = len(a | b)
            if union == 0:
                continue
            j = inter / union
            jaccs.append(j)
        if len(jaccs) == 0:
            return np.nan
        return float(1.0 - np.mean(jaccs))

    def estimate_kmeans_clusters(hypotheses: List[str]) -> float:
        """
        Embed hypotheses and estimate cluster count via k-means elbow.

        Uses read_past_embeddings_or_generate_new; k in [1, max_k].
        Chooses the first k where fractional SSE improvement drops below
        kmeans_improvement_threshold; else max_k.
        """
        if len(hypotheses) < 2:
            return np.nan

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            # If sklearn not available, just return NaN
            return np.nan

        embeddings = read_past_embeddings_or_generate_new(
            None,
            None,
            decoded_strs=hypotheses,
            recompute_embeddings=recompute_embeddings,
            save_embeddings=save_embeddings,
            tqdm_disable=True,
        )
        X = np.asarray(embeddings, dtype=float)
        n = X.shape[0]
        if n < 2:
            return np.nan

        max_k = min(max_kmeans_k, n)
        sse = []
        for k in range(1, max_k + 1):
            km = KMeans(n_clusters=k, random_state=0, n_init=10)
            km.fit(X)
            sse.append(float(km.inertia_))

        if max_k == 1:
            return 1.0

        # Simple elbow: first k where improvement ratio falls below threshold
        best_k = max_k
        for k in range(2, max_k + 1):
            prev_sse = sse[k - 2]
            cur_sse = sse[k - 1]
            if prev_sse <= 0:
                continue
            improvement = (prev_sse - cur_sse) / prev_sse
            if improvement < kmeans_improvement_threshold:
                best_k = k - 1
                break

        return float(best_k)

    def group_tokens_by_kind(h, kind=None, model_str=None, mode="mean"):
        """
        Sum tokens from API logs in `h` filtered by query_types == kind and model_str == model_str if provided.
        If mode is "mean", returns the mean of the input and output tokens.
        If mode is "sum", returns the sum of the input and output tokens.
        Returns (input_mean, output_mean) if mode is "mean", otherwise (input_sum, output_sum).
        """
        types = h.get("query_types", []) or []
        model_ids = h.get("api_model_str_ids", []) or []
        ins = h.get("api_model_query_input_lengths", []) or []
        outs = h.get("api_model_query_output_lengths", []) or []
        L = min(len(types), len(ins), len(outs), len(model_ids))
        types, ins, outs, model_ids = types[:L], ins[:L], outs[:L], model_ids[:L]

        total_in = total_out = num_valid_elements_added = 0
        for t, i, o, m in zip(types, ins, outs, model_ids):
            if ((kind is None) or (t == kind)) and ((model_str is None) or (m == model_str)):
                total_in += int(i)
                total_out += int(o)
                num_valid_elements_added += 1
        if num_valid_elements_added == 0:
            return 0, 0
        if mode == "mean":
            return total_in / num_valid_elements_added, total_out / num_valid_elements_added
        elif mode == "sum":
            return int(total_in), int(total_out)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    # ----------- main loop over experiments -----------
    stats_rows = []
    token_rows = []

    for intervention, div_dict in paths_to_results.items():
        for diversification_method, stpc_dict in div_dict.items():
            for stpc_str, log_path in stpc_dict.items():
                stpc_int = int(stpc_str.split("_")[0])
                if not os.path.exists(log_path):
                    print(f"[WARN] Missing log file for {intervention} / {diversification_method} / {stpc_str}: {log_path}")
                    stats_rows.append({
                        "intervention": intervention,
                        "diversification_method": diversification_method,
                        "sampled_texts_per_cluster": stpc_int,
                        "n_hypotheses": 0,
                        "n_significant": 0,
                        "bh_threshold": np.nan,
                        "mean_val_auc": np.nan,
                        "mean_cv_auc": np.nan,
                        "mean_accuracy": np.nan,
                        "diversity_jaccard": np.nan,
                        "diversity_llm": np.nan,
                        "estimated_n_clusters": np.nan,
                    })
                    token_rows.append({
                        "intervention": intervention,
                        "diversification_method": diversification_method,
                        "sampled_texts_per_cluster": stpc_int,
                        "disc_input_tokens": 0,
                        "disc_output_tokens": 0,
                        "labeler_input_tokens": 0,
                        "labeler_output_tokens": 0,
                        "summarizer_input_tokens": 0,
                        "summarizer_output_tokens": 0,
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                    })
                    continue

                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                h = extract_hypotheses_and_scores(lines)

                hypotheses = h.get("hypotheses", []) or []
                val_auc = h.get("auc_scores", []) or []
                val_acc = h.get("accuracies", []) or []
                perm_p = h.get("permutation_p_values", []) or []
                cv_auc = h.get("cross_validated_auc_scores", []) or []

                L = min(len(hypotheses), len(val_auc), len(val_acc), len(perm_p))
                if L == 0:
                    n_hyp = 0
                    n_sig = 0
                    bh_thresh = np.nan
                    mean_val_auc = np.nan
                    mean_cv_auc = safe_mean(cv_auc)
                    mean_acc = np.nan
                    diversity = np.nan
                    diversity_llm = np.nan
                    est_k = np.nan
                else:
                    n_hyp = L
                    val_auc = np.asarray(val_auc[:L], dtype=float)
                    val_acc = np.asarray(val_acc[:L], dtype=float)
                    perm_p = np.asarray(perm_p[:L], dtype=float)
                    # BH on validation permutation p-values
                    n_sig, bh_thresh = apply_bh(perm_p, alpha)
                    mean_val_auc = safe_mean(val_auc)
                    mean_acc = safe_mean(val_acc)
                    mean_cv_auc = safe_mean(cv_auc)

                    # Hypothesis diversity
                    diversity = hypothesis_jaccard_diversity(hypotheses[:L])

                    diversity_llm = score_hypotheis_diversity(hypotheses[:L], api_provider, model_str, api_key_path, max_tokens=max_tokens, temperature=temperature, max_thinking_tokens=max_thinking_tokens) / max(len(hypotheses[:L]), 1)
                    #diversity_llm = -100

                    # k-means elbow on embeddings
                    est_k = estimate_kmeans_clusters(hypotheses[:L])

                # Token usage: discriminator / labeler / summarizer
                disc_in, disc_out = group_tokens_by_kind(h, kind="discriminator", model_str=discriminative_model_str)
                lab_in, lab_out = group_tokens_by_kind(h, kind="labeler", model_str=labeler_model_str)
                sum_in, sum_out = group_tokens_by_kind(h, kind="summarizer", model_str=summarizer_model_str)
                total_in = disc_in + lab_in + sum_in
                total_out = disc_out + lab_out + sum_out

                stats_rows.append({
                    "intervention": intervention,
                    "diversification_method": diversification_method,
                    "sampled_texts_per_cluster": stpc_int,
                    "n_hypotheses": int(n_hyp),
                    "n_significant": int(n_sig),
                    "bh_threshold": bh_thresh,
                    "mean_val_auc": mean_val_auc,
                    "mean_cv_auc": mean_cv_auc,
                    "mean_accuracy": mean_acc,
                    "diversity_jaccard": diversity,
                    "diversity_llm": diversity_llm,
                    "estimated_n_clusters": est_k,
                })

                token_rows.append({
                    "intervention": intervention,
                    "diversification_method": diversification_method,
                    "sampled_texts_per_cluster": stpc_int,
                    "disc_input_tokens": int(disc_in),
                    "disc_output_tokens": int(disc_out),
                    "labeler_input_tokens": int(lab_in),
                    "labeler_output_tokens": int(lab_out),
                    "summarizer_input_tokens": int(sum_in),
                    "summarizer_output_tokens": int(sum_out),
                    "total_input_tokens": int(total_in),
                    "total_output_tokens": int(total_out),
                })

    df_stats = pd.DataFrame(stats_rows)
    df_tokens = pd.DataFrame(token_rows)

    # ----------- Aggregations for the two conceptual sections -----------

    # Section 1: Effects of different diversification methods
    # "across the different diversification methods, broken down by sampled_texts_per_cluster"
    agg_stats_by_div_and_stpc = (
        df_stats
        .groupby(["diversification_method", "sampled_texts_per_cluster"])
        .agg({
            "n_hypotheses": "mean",
            "n_significant": "mean",
            "mean_val_auc": "mean",
            "mean_cv_auc": "mean",
            "mean_accuracy": "mean",
            "diversity_jaccard": "mean",
            "diversity_llm": "mean",
            "estimated_n_clusters": "mean",
        })
        .reset_index()
        .rename(columns={
            "n_hypotheses": "avg_n_hypotheses",
            "n_significant": "avg_n_significant",
        })
    )

    agg_tokens_by_div_and_stpc = (
        df_tokens
        .groupby(["diversification_method", "sampled_texts_per_cluster"])
        .agg({
            "disc_input_tokens": "mean",
            "disc_output_tokens": "mean",
            "labeler_input_tokens": "mean",
            "labeler_output_tokens": "mean",
            "summarizer_input_tokens": "mean",
            "summarizer_output_tokens": "mean",
            "total_input_tokens": "mean",
            "total_output_tokens": "mean",
        })
        .reset_index()
    )

    # Section 2: Effects of different values of sampled_texts_per_cluster
    # "against the number of sampled_texts_per_cluster, broken down by diversification method"
    # This is essentially the same grouping, but conceptually used in that section.
    # The above aggregated tables already support this analysis.

    # Labeler IO relation per experiment
    df_labeler_io = df_tokens.copy()
    df_labeler_io["labeler_io_ratio"] = df_labeler_io["labeler_output_tokens"] / df_labeler_io["labeler_input_tokens"].replace(0, np.nan)

    # Optionally: simple plot of labeler_in vs labeler_out colored by diversification method
    if make_plots:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        for method in df_labeler_io["diversification_method"].unique():
            sub = df_labeler_io[df_labeler_io["diversification_method"] == method]
            plt.scatter(
                sub["labeler_input_tokens"],
                sub["labeler_output_tokens"],
                alpha=0.7,
                label=method,
            )
        plt.xlabel("Labeler input tokens")
        plt.ylabel("Labeler output tokens")
        plt.title("Labeler input vs output tokens across ablation experiments")
        plt.legend(fontsize=8)
        plot_path = os.path.join(out_dir_figs, "labeler_input_vs_output_tokens_ablation.pdf")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {plot_path}")

    # ----------- Save tables -----------
    path_stats = os.path.join(out_dir_tables, "ablation_stats_by_experiment.tsv")
    path_tokens = os.path.join(out_dir_tables, "ablation_token_usage_by_experiment.tsv")
    path_agg_stats = os.path.join(out_dir_tables, "ablation_stats_by_diversification_and_stpc.tsv")
    path_agg_tokens = os.path.join(out_dir_tables, "ablation_token_usage_by_diversification_and_stpc.tsv")
    path_labeler_io = os.path.join(out_dir_tables, "ablation_labeler_io_by_experiment.tsv")

    df_stats = round_float_df_to_sig_figs(df_stats, sig=3)
    df_tokens = round_float_df_to_sig_figs(df_tokens, sig=3)
    agg_stats_by_div_and_stpc = round_float_df_to_sig_figs(agg_stats_by_div_and_stpc, sig=3)
    agg_tokens_by_div_and_stpc = round_float_df_to_sig_figs(agg_tokens_by_div_and_stpc, sig=3)
    df_labeler_io = round_float_df_to_sig_figs(df_labeler_io, sig=3)

    df_stats.to_csv(path_stats, sep="\t", index=False)
    df_tokens.to_csv(path_tokens, sep="\t", index=False)
    agg_stats_by_div_and_stpc.to_csv(path_agg_stats, sep="\t", index=False)
    agg_tokens_by_div_and_stpc.to_csv(path_agg_tokens, sep="\t", index=False)
    df_labeler_io.to_csv(path_labeler_io, sep="\t", index=False)

    print(f"[Saved] {path_stats}")
    print(f"[Saved] {path_tokens}")
    print(f"[Saved] {path_agg_stats}")
    print(f"[Saved] {path_agg_tokens}")
    print(f"[Saved] {path_labeler_io}")

    return {
        "per_experiment_stats": df_stats,
        "per_experiment_tokens": df_tokens,
        "stats_by_diversification_and_stpc": agg_stats_by_div_and_stpc,
        "tokens_by_diversification_and_stpc": agg_tokens_by_div_and_stpc,
        "labeler_io_by_experiment": df_labeler_io,
    }



def extract_hypotheses_and_scores(lines):
    # The relevant patterns are in the format of:
    hypothesis_text_pattern = r"accuracy: \d+\.\d+, Binomial P-value: \d+\.\d+, Permutation P-value: \d+\.\d+, Label: .*"
    label_auc_score_text_pattern = r"Label AUC score: \d+\.\d+" 
    baseline_acc_auc_text_pattern = r"Baseline accuracy score: \d+\.\d+, Baseline AUC score: \d+\.\d+" 

    cross_validated_accuracy_text_pattern = r"Cross-validation accuracy for label on pair \(\d+, \d+\): \d+\.\d+" 
    cross_validated_auc_text_pattern = r"Cross-validation AUC for label: \d+\.\d+" 
    cross_validated_baseline_accuracy_text_pattern = r"Cross-validation baseline accuracy for label: \d+\.\d+" 
    cross_validated_baseline_auc_text_pattern = r"Cross-validation baseline AUC for label: \d+\.\d+" 
    cross_validated_permutation_test_text_pattern = r"Permutation p-value for label: \d+\.\d+" 
    
    generative_score_text_pattern = r"\(Generative Score: -*\d+\.\d+, P-value: \d+\.\d+\)" 


    api_model_query_response_lengths_text_pattern = r"SCORES Logging Input/Output tokens for .*: \d+ / \d+ : prompt start: .* ..."
    discriminative_model_validation_true_labels_text_pattern = r"SCORES Logging validation True labels: .*"
    # Match a bracketed list of floats (arbitrary size/precision)
    #FLOAT = r"[+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?"
    discriminative_model_validation_discrim_scores_text_pattern = r"SCORES Logging validation Scores: .*"
    discriminative_model_validation_discrim_alternate_scores_text_pattern = r"SCORES Logging validation Alternate scores: .*"


    discriminative_model_cross_validation_true_labels_text_pattern = r"SCORES Logging cross-validation True labels: .*"
    discriminative_model_cross_validation_discrim_scores_text_pattern = r"SCORES Logging cross-validation Scores: .*" 
    discriminative_model_cross_validation_discrim_alternate_scores_text_pattern = r"SCORES Logging cross-validation Alternate scores: .*"
    
    
    hypotheses = []
    accuracies = []
    permutation_p_values = []
    auc_scores = []
    baseline_accuracies = []
    baseline_auc_scores = []

    cross_validated_accuracies = []
    cross_val_permutation_test_p_values = []
    cross_validated_auc_scores = []
    cross_validated_baseline_accuracies = []
    cross_validated_baseline_auc_scores = []

    generative_scores = []

    api_model_query_input_lengths = []
    api_model_query_output_lengths = []
    api_model_str_ids = []
    query_types = []
    query_start_type_mapping = {
        "the following label": "discriminator",
        "you will be given tw": "labeler",
        "please analyze the f": "summarizer",
        "you will be given a": "summarizer"
    }

    discriminative_model_validation_true_labels = []
    discriminative_model_validation_discrim_scores = []
    discriminative_model_cross_validation_true_labels = []
    discriminative_model_cross_validation_discrim_scores = []

    # There are two alternative discriminators: qwen/qwen3-next-80b-a3b-instruct and openai/gpt-5-nano
    discriminative_model_validation_discrim_alternate_scores = {
        "qwen": [],
        "gpt-5-nano": []
    }
    discriminative_model_cross_validation_discrim_alternate_scores = {
        "qwen": [],
        "gpt-5-nano": []
    }


    # Now we will extract the hypotheses and scores
    for line in lines:
        line = line.strip()

        # Extract API model query/response lengths
        # api_model_query_response_lengths_match = re.match(api_model_query_response_lengths_text_pattern, line)
        if "Logging Input/Output tokens for" in line:
            query_length = re.search(r"Input/Output tokens for (.*): (\d+) / (\d+) : prompt start: (.*)...", line)
            if query_length:
                api_model_str_ids.append(query_length.group(1).strip())
                api_model_query_input_lengths.append(int(query_length.group(2).strip()))
                api_model_query_output_lengths.append(int(query_length.group(3).strip()))
                
                start_of_queries = query_length.group(4).strip().lower()
                query_type_match_found = False
                for key in query_start_type_mapping.keys():
                    if key in start_of_queries:
                        query_types.append(query_start_type_mapping[key])
                        query_type_match_found = True
                if not query_type_match_found:
                    query_types.append("unknown")
                    print(f"Unknown query type: {start_of_queries}")
                    print(f"Line: {line}")
            continue

        # Extract discriminative model validation true labels
        # discriminative_model_validation_true_labels_match = re.match(discriminative_model_validation_true_labels_text_pattern, line)
        #if discriminative_model_validation_true_labels_match:
        if "SCORES Logging validation True labels" in line:
            labels_str_match = re.search(r"Logging validation True labels: (.*)]", line)
            if labels_str_match:
                found = labels_str_match.group(1).strip() + ']'
                parsed_labels = ast.literal_eval(found)
                discriminative_model_validation_true_labels.append(parsed_labels)

            continue

        # Extract discriminative model validation discrimin scores
        # discriminative_model_validation_discrim_scores_match = re.match(discriminative_model_validation_discrim_scores_text_pattern, line)
        if "SCORES Logging validation Scores:" in line:
            scores_str_match = re.search(r"Logging validation Scores: (.*)]", line)
            if scores_str_match:
                found = scores_str_match.group(1).strip() + "]"
                parsed_labels = ast.literal_eval(found)
                discriminative_model_validation_discrim_scores.append(parsed_labels)
            continue

        # Extract discriminative model validation discrimin alternate scores
        # discriminative_model_validation_discrim_alternate_scores_match = re.match(discriminative_model_validation_discrim_alternate_scores_text_pattern, line)
        if "SCORES Logging validation Alternate scores:" in line:
            scores_str_match = re.search(r"Logging validation Alternate scores: (.*)]]", line)
            if scores_str_match:
                scores = ast.literal_eval(scores_str_match.group(1).strip() + "]]")
                discriminative_model_validation_discrim_alternate_scores['qwen'].append(scores[0])
                discriminative_model_validation_discrim_alternate_scores['gpt-5-nano'].append(scores[1])
            continue

        # Extract discriminative model cross-validation true labels
        # discriminative_model_cross_validation_true_labels_match = re.match(discriminative_model_cross_validation_true_labels_text_pattern, line)
        if "SCORES Logging cross-validation True labels:" in line:
            labels_str_match = re.search(r"Logging cross-validation True labels: (.*)]", line)
            if labels_str_match:
                labels = ast.literal_eval(labels_str_match.group(1).strip() + "]")
                discriminative_model_cross_validation_true_labels.append(labels)
            continue

        # Extract discriminative model cross-validation discrimin scores
        # discriminative_model_cross_validation_discrim_scores_match = re.match(discriminative_model_cross_validation_discrim_scores_text_pattern, line)
        if "SCORES Logging cross-validation Scores:" in line:
            scores_str_match = re.search(r"Logging cross-validation Scores: (.*)]", line)
            if scores_str_match:
                scores = ast.literal_eval(scores_str_match.group(1).strip() + "]")
                discriminative_model_cross_validation_discrim_scores.append(scores)
            continue

        # Extract discriminative model cross-validation discrimin alternate scores
        # discriminative_model_cross_validation_discrim_alternate_scores_match = re.match(discriminative_model_cross_validation_discrim_alternate_scores_text_pattern, line)
        if "SCORES Logging cross-validation Alternate scores:" in line:
            scores_str_match = re.search(r"Logging cross-validation Alternate scores: (.*)]]", line)
            if scores_str_match:
                scores = ast.literal_eval(scores_str_match.group(1).strip() + "]]")
                discriminative_model_cross_validation_discrim_alternate_scores['qwen'].append(scores[0])
                discriminative_model_cross_validation_discrim_alternate_scores['gpt-5-nano'].append(scores[1])
            continue

        # Extract main accuracy, p-value, and label
        match = re.match(hypothesis_text_pattern, line)
        if match:
            # Extract accuracy, p-value, and label
            acc_match = re.search(r"accuracy: (\d+\.\d+)", line)
            label_match = re.search(r"Label: (.*)", line)
            permutation_p_match = re.search(r"Permutation P-value: (\d+\.\d+)", line)
            if acc_match and label_match:
                accuracies.append(float(acc_match.group(1)))
                hypotheses.append(label_match.group(1).strip())
                if permutation_p_match:
                    permutation_p_values.append(float(permutation_p_match.group(1)))
            continue

        # Extract label AUC score
        label_auc_score_match = re.match(label_auc_score_text_pattern, line)
        if label_auc_score_match:
            auc_score_val = re.search(r"Label AUC score: (\d+\.\d+)", line)
            if auc_score_val:
                auc_scores.append(float(auc_score_val.group(1)))
            continue

        # Extract baseline accuracy and AUC score
        baseline_acc_auc_match = re.match(baseline_acc_auc_text_pattern, line)
        if baseline_acc_auc_match:
            baseline_acc_val = re.search(r"Baseline accuracy score: (\d+\.\d+)", line)
            baseline_auc_val = re.search(r"Baseline AUC score: (\d+\.\d+)", line)
            if baseline_acc_val and baseline_auc_val:
                baseline_accuracies.append(float(baseline_acc_val.group(1)))
                baseline_auc_scores.append(float(baseline_auc_val.group(1)))
            continue
        
        # Extract cross-validated accuracy
        cross_val_match = re.match(cross_validated_accuracy_text_pattern, line)
        if cross_val_match:
            acc_val = re.search(r": (\d+\.\d+)", line)
            if acc_val:
                cross_validated_accuracies.append(float(acc_val.group(1)))
            continue
        
        # Extract cross-validated AUC score
        cross_val_auc_match = re.match(cross_validated_auc_text_pattern, line)
        if cross_val_auc_match:
            auc_val = re.search(r"Cross-validation AUC for label: (\d+\.\d+)", line)
            if auc_val:
                cross_validated_auc_scores.append(float(auc_val.group(1)))
            continue
        
        # Extract cross-validated baseline accuracy
        cross_val_baseline_acc_match = re.match(cross_validated_baseline_accuracy_text_pattern, line)
        if cross_val_baseline_acc_match:
            baseline_acc_val = re.search(r"Cross-validation baseline accuracy for label: (\d+\.\d+)", line)
            if baseline_acc_val:
                cross_validated_baseline_accuracies.append(float(baseline_acc_val.group(1)))
            continue
            
        # Extract cross-validated baseline AUC score
        cross_val_baseline_auc_match = re.match(cross_validated_baseline_auc_text_pattern, line)
        if cross_val_baseline_auc_match:
            baseline_auc_val = re.search(r"Cross-validation baseline AUC for label: (\d+\.\d+)", line)
            if baseline_auc_val:
                cross_validated_baseline_auc_scores.append(float(baseline_auc_val.group(1)))
            continue

        # Extract permutation test p-value
        permutation_test_match = re.match(cross_validated_permutation_test_text_pattern, line)
        if permutation_test_match:
            p_val = re.search(r"Permutation p-value for label: (\d+\.\d+)", line)
            if p_val:
                cross_val_permutation_test_p_values.append(float(p_val.group(1)))
            continue

        # Extract generative score
        generative_score_match = re.match(generative_score_text_pattern, line)
        if generative_score_match:
            score_val = re.search(r"Generative Score: (-*\d+\.\d+)", line)
            if score_val:
                generative_scores.append(float(score_val.group(1)))
            continue

    return_dict = {
        "hypotheses": hypotheses,
        "accuracies": accuracies,
        "auc_scores": auc_scores,
        "permutation_p_values": permutation_p_values,
        "baseline_accuracies": baseline_accuracies,
        "baseline_auc_scores": baseline_auc_scores,
        "cross_validated_accuracies": cross_validated_accuracies,
        "cross_validated_auc_scores": cross_validated_auc_scores,
        "cross_val_permutation_test_p_values": cross_val_permutation_test_p_values,
        "cross_validated_baseline_accuracies": cross_validated_baseline_accuracies,
        "cross_validated_baseline_auc_scores": cross_validated_baseline_auc_scores,
        "generative_scores": generative_scores,
        "api_model_query_input_lengths": api_model_query_input_lengths,
        "api_model_query_output_lengths": api_model_query_output_lengths,
        "api_model_str_ids": api_model_str_ids,
        "query_types": query_types,
        "discriminative_model_validation_true_labels": discriminative_model_validation_true_labels,
        "discriminative_model_validation_discrim_scores": discriminative_model_validation_discrim_scores,
        "discriminative_model_validation_discrim_alternate_scores": discriminative_model_validation_discrim_alternate_scores,
        "discriminative_model_cross_validation_true_labels": discriminative_model_cross_validation_true_labels,
        "discriminative_model_cross_validation_discrim_scores": discriminative_model_cross_validation_discrim_scores,
        "discriminative_model_cross_validation_discrim_alternate_scores": discriminative_model_cross_validation_discrim_alternate_scores,
    }
    return return_dict


def main():
    parser = argparse.ArgumentParser()
    # Progressive summarization flags
    parser.add_argument("--api_provider", type=str, choices=["openai", "openrouter", "anthropic", "gemini"], help="LLM provider for summarization", default="openrouter")
    parser.add_argument("--model_str", type=str, help="Model name for summarization", default="openai/gpt-5-2025-08-07")
    parser.add_argument("--api_key_path", type=str, default="../../../../../data/api_keys/openrouter_key.txt", help="Path to API key file")

    parser.add_argument("--summary_max_tokens", type=int, default=30000)
    parser.add_argument("--summary_temperature", type=float, default=1.0)
    parser.add_argument("--summary_max_thinking_tokens", type=int, default=20000)
    parser.add_argument("--summary_save_dir", type=str, default="summaries_3")

    # Preset experimental configuration options
    parser.add_argument("--analyze_case_studies", action="store_true")
    parser.add_argument("--analyze_ablation", action="store_true")
    parser.add_argument("--plots", action="store_true")

    # Hypothesis selection options
    parser.add_argument("--do_summaries", action="store_true")
    parser.add_argument("--summary_filter", type=str, default="permutation", choices=["none", "permutation", "crossval"], 
                        help="Filter hypotheses by BH-significant p-values of the chosen type")
    parser.add_argument("--summary_bh_alpha", type=float, default=0.05)
    parser.add_argument("--summary_min_accuracy", type=float, default=None)
    parser.add_argument("--summary_top_k", type=int, default=None)

    # Analysis description options
    parser.add_argument("--alpha", type=float, default=0.05)


    args = parser.parse_args()

    summary_opts = None
    if args.do_summaries:
        if not args.api_provider or not args.model_str or not args.api_key_path:
            raise ValueError("--summaries requires --api_provider, --model_str, and --api_key_path.")
        summary_opts = {
            "enabled": True,
            "api_provider": args.api_provider,
            "model_str": args.model_str,
            "api_key_path": args.api_key_path,
            "save_dir": args.summary_save_dir,
            "max_tokens": args.summary_max_tokens,
            "temperature": args.summary_temperature,
            "max_thinking_tokens": args.summary_max_thinking_tokens,
            "filter_type": args.summary_filter,
            "bh_alpha": args.summary_bh_alpha,
            "min_accuracy": args.summary_min_accuracy,
            "top_k": args.summary_top_k,
        }

    if args.analyze_case_studies:
        analyze_case_study_exps(
            alpha=args.alpha,
            make_plots=args.plots,
            do_progressive_summaries=args.do_summaries,
            summary_opts=summary_opts if args.do_summaries else None,
        )
    elif args.analyze_ablation:
        analyze_ablation_exps(
            alpha=args.alpha,
            make_plots=args.plots,
            api_provider=args.api_provider,
            model_str=args.model_str,
            api_key_path=args.api_key_path,
            max_tokens=args.summary_max_tokens,
            temperature=args.summary_temperature,
            max_thinking_tokens=args.summary_max_thinking_tokens,
        )
    else:
        raise ValueError("No log file or exp config file provided, and no preset experimental configuration used.")

if __name__ == "__main__":
    main()
