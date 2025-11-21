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

# Load summary functions from ../../progressive_summary.py
# so we can drive the LLM for summary generation.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from stats import benjamini_hochberg_correction
from progressive_summary import select_hypotheses_for_summary, run_progressive_summary
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

# ----------------------------
# Case-study analysis pipeline
# ----------------------------
import json
from collections import defaultdict
from dataclasses import dataclass

# ----------------------------
# Small utilities
# ----------------------------

def _mkdirp(path: str):
    os.makedirs(path, exist_ok=True)

def _truncate_text(s: str, max_chars: int = 120) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"

def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """
    Return average ranks for ties (1-based) without SciPy.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")  # stable sort
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # tie-handling: average ranks per unique value
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sum_ranks = np.bincount(inv, weights=ranks)
        avg_ranks = sum_ranks / counts
        ranks = avg_ranks[inv]
    return ranks

def safe_auc(y_true, y_score) -> float:
    """
    Label-agnostic AUROC that handles ties and degenerate labels.
    y_true: iterable of 0/1
    y_score: iterable of continuous scores (e.g., 0..100)
    """
    if y_true is None or y_score is None:
        return np.nan
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    if y.size == 0 or s.size == 0 or y.size != s.size:
        return np.nan
    classes = np.unique(y)
    if classes.size != 2:
        # degenerate; AUROC undefined
        return np.nan
    # AUC via rank formula (Mann-Whitney U)
    ranks = _rankdata_average_ties(s)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    sum_ranks_pos = np.sum(ranks[y == 1])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

@dataclass
class SelectionConfig:
    alpha: float = 0.05       # BH FDR level
    auc_min: float = 0.62     # validation AUC effect floor
    n_min: int = 60           # minimum #validation examples to consider

# ----------------------------
# Alignment & selection helpers
# ----------------------------

def _align_core_arrays(parsed: Dict) -> Tuple[int, Dict]:
    """
    Align core per-hypothesis arrays to a common length L, truncating longer ones.
    We align the fields used in selection & reporting:
    - hypotheses, accuracies, auc_scores, permutation_p_values
    (validation labels/scores and alternates are optional and accessed with bounds checks)
    """
    lengths = [
        len(parsed.get("hypotheses", [])),
        len(parsed.get("accuracies", [])),
        len(parsed.get("auc_scores", [])),
        len(parsed.get("permutation_p_values", [])),
    ]
    L = min(lengths) if all(l > 0 for l in lengths) else 0
    if L == 0:
        return 0, parsed

    def trunc(arr, L):
        return arr[:L] if isinstance(arr, list) else arr

    parsed_aligned = dict(parsed)
    parsed_aligned["hypotheses"] = trunc(parsed.get("hypotheses", []), L)
    parsed_aligned["accuracies"] = trunc(parsed.get("accuracies", []), L)
    parsed_aligned["auc_scores"] = trunc(parsed.get("auc_scores", []), L)
    parsed_aligned["permutation_p_values"] = trunc(parsed.get("permutation_p_values", []), L)

    # Optional per-hypothesis lists; keep as-is (we'll bounds-check later)
    return L, parsed_aligned

def _compute_validated_mask(val_p_list: List[float],
                            val_auc_list: List[float],
                            n_val_list: List[Optional[int]],
                            sel: SelectionConfig) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      is_bh_sig [L], bh_threshold (float), effect_sig [L], size_sig [L], sig [L]
    """
    # Clean p-values (replace missing with 1.0 so BH never selects them)
    pvals = np.array([p if (p is not None and not np.isnan(p)) else 1.0 for p in val_p_list], dtype=float)
    # BH within-run
    n_sig, bh_threshold, is_bh_sig = benjamini_hochberg_correction(list(pvals), alpha=sel.alpha)
    is_bh_sig = np.array(is_bh_sig, dtype=bool)

    aucs = np.array([a if (a is not None and not np.isnan(a)) else -np.inf for a in val_auc_list], dtype=float)
    effect_sig = aucs >= float(sel.auc_min)

    nvals = np.array([n if (n is not None and not np.isnan(n)) else -np.inf for n in n_val_list], dtype=float)
    size_sig = nvals >= float(sel.n_min)

    sig = is_bh_sig & effect_sig & size_sig
    return is_bh_sig, float(bh_threshold), effect_sig, size_sig, sig

def _aggregate_tokens_by_role(parsed: Dict) -> Dict[str, int]:
    """
    Returns aggregate token counts per role and totals.
    """
    ins = parsed.get("api_model_query_input_lengths", []) or []
    outs = parsed.get("api_model_query_output_lengths", []) or []
    roles = parsed.get("query_types", []) or []

    totals = defaultdict(int)
    for role, ti, to in zip(roles, ins, outs):
        role_key = role if role in ("labeler", "discriminator", "summarizer") else "other"
        totals[f"{role_key}_in"] += int(ti)
        totals[f"{role_key}_out"] += int(to)
        totals["total"] += int(ti) + int(to)
    # Ensure all expected keys exist
    for k in ["labeler_in", "labeler_out", "discriminator_in", "discriminator_out",
              "summarizer_in", "summarizer_out", "other_in", "other_out", "total"]:
        totals[k] = int(totals.get(k, 0))
    return dict(totals)

def _compute_alt_aucs_for_validated(parsed: Dict,
                                    validated_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute AUROC of alternate discriminators over the validated subset only.
    Returns dict mapping alt_name -> np.ndarray of alt AUCs aligned to hypotheses (NaN where not computed).
    """
    L = len(parsed.get("hypotheses", []))
    result = {}

    y_lists = parsed.get("discriminative_model_validation_true_labels", []) or []
    main_scores = parsed.get("discriminative_model_validation_discrim_scores", []) or []
    alt_scores_dict = parsed.get("discriminative_model_validation_discrim_alternate_scores", {}) or {}

    for alt_name, alt_scores_lists in alt_scores_dict.items():
        alt_aucs = np.full(L, np.nan, dtype=float)
        for i in range(L):
            if not validated_mask[i]:
                continue
            # bounds check
            if i >= len(y_lists) or i >= len(alt_scores_lists):
                continue
            y = y_lists[i]
            s_alt = alt_scores_lists[i]
            if not isinstance(y, list) or not isinstance(s_alt, list):
                continue
            if len(y) == 0 or len(s_alt) == 0 or len(y) != len(s_alt):
                continue
            auc_alt = safe_auc(y, s_alt)
            alt_aucs[i] = auc_alt
        result[alt_name] = alt_aucs
    return result

# ----------------------------
# Plotting helpers (matplotlib only)
# ----------------------------

def _plot_auc_histogram(vals: np.ndarray, auc_min: float, title: str, out_pdf: str):
    _mkdirp(os.path.dirname(out_pdf))
    plt.figure(figsize=(5.2, 3.4))
    if vals.size > 0:
        bins = np.linspace(0.5, 1.0, 21)
        plt.hist(vals, bins=bins, edgecolor="black")
        plt.axvline(auc_min, linestyle="--")
    else:
        plt.text(0.5, 0.5, "No significant hypotheses", ha="center", va="center", transform=plt.gca().transAxes)
    plt.xlim(0.5, 1.0)
    plt.xlabel("Validation AUC (significant only)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

def _plot_topk_caterpillar(vals: np.ndarray, labels: List[str], k: int, out_pdf: str, title: str):
    _mkdirp(os.path.dirname(out_pdf))
    if vals.size == 0:
        plt.figure(figsize=(5.2, 3.4))
        plt.text(0.5, 0.5, "No significant hypotheses", ha="center", va="center", transform=plt.gca().transAxes)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()
        return

    idx = np.argsort(vals)[::-1][:k]
    xv = vals[idx]
    yv = np.arange(len(idx))
    ylabels = [_truncate_text(labels[j], 90) for j in idx]

    plt.figure(figsize=(7.5, max(2.5, 0.25 * len(idx) + 1)))
    for yi, xi in zip(yv, xv):
        plt.plot([0.5, xi], [yi, yi], lw=1.5)
        plt.plot([xi], [yi], marker="o")
    plt.yticks(yv, ylabels)
    plt.xlim(0.5, 1.0)
    plt.xlabel("Validation AUC")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

def _plot_violin_grid(per_run_auc: Dict[Tuple[str, str], np.ndarray],
                      interventions: List[str],
                      datasets: List[str],
                      out_pdf: str):
    rows = len(interventions)
    cols = len(datasets)
    _mkdirp(os.path.dirname(out_pdf))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows), squeeze=False)
    for r, inter in enumerate(interventions):
        for c, data in enumerate(datasets):
            ax = axes[r][c]
            vals = per_run_auc.get((inter, data), np.array([]))
            if vals.size > 0:
                ax.violinplot(vals, showmeans=False, showmedians=True, vert=True)
                ax.scatter(np.ones_like(vals), vals, s=6, alpha=0.5)
                ax.set_ylim(0.5, 1.0)
            else:
                ax.text(0.5, 0.5, "n=0", ha="center", va="center", transform=ax.transAxes)
                ax.set_ylim(0.5, 1.0)
            ax.set_title(f"{inter} :: {data} (n={vals.size})", fontsize=10)
            ax.set_xticks([])
            ax.set_ylabel("Validation AUC" if c == 0 else "")
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

# ----------------------------
# Main case-study analysis
# ----------------------------

def analyze_case_study_exps(
    paths_to_results: Optional[Dict[str, Dict[str, str]]] = None,
    selection: SelectionConfig = SelectionConfig(),
    topk: int = 25,
    make_plots: bool = True,
    do_progressive_summaries: bool = False,
    summary_opts: Optional[Dict] = None,
    out_dir_tables: str = "tables/case_studies",
    out_dir_figs: str = "figs/case_studies",
    out_dir_summaries: str = "summaries/case_studies",
) -> Dict[str, pd.DataFrame]:
    """
    So 9 total experiments, investigating three interventions (KE-10, R1 distillation, WIHP), across three data sets (Anthropic, TruthfulQA, Amazon BOLD). These represent the three case study interventions, with three different datasets on which to evaluate the interventions.

    We will first load the results using the extract_hypotheses_and_scores function, and then analyze the results.
    Specifically, we will break things down into five sections, one for each of the three interventions, one that compares the AUC distributions across interventions, plus a final section that discusses the effects of alternate Discriminator models. For each intervention, the results on the Anthropic dataset will take center stage, with the other two datasets serving supporting roles.
    
    (Sections 1 - 3)
    For the effects of each intervention, we will:
    - Compare the number of significant hypotheses (post multiple comparison correction based on permutation p-values), average AUC scores cross-validation AUCs, validation AUC-based p-values and accuracies, broken down by data set.
        - So we need a table, split by data set, that shows these values for the given intervention. We have 3 datasets making 3 rows, and 5 columns corresponding to the different dataset level statistics.
    - Analyze the effects of each intervention on the number of input / output tokens, and thus the costs of the experiments. Note we can break this down by Discriminator / Labeler / Summarizer models, thanks to query_types.
        - This will be its own table, with 3 datasets making 3 total rows, and 3 model types x (input, output token types) making 6 columns.
    - Produce progressive summaries of the hypotheses for each intervention, broken down by data set. Make this optional in the function call, and if enabled, save it to a text file, rather than printing to the console.
    Overall: for each dataset, we have two tables and a progressive summary.

    For the fourth section, that compares the interventions against each other, we will:
    - Create histograms of the validation AUC scores for each intervention, also broken down by data set, with the minimum AUC for significance highlighted, showing R1 > KE-10 > WIHP (with WIHP having 0 or near 0 sigificant hypotheses).
        - So there are three histograms, one per data set, and each histogram has the three interventions shown in different colors. We should make this one matplotlib figure saved as a pdf.
    - Create a table comparing the AUCs to the cross-validation AUCs (means and stds), broken down by intervention and dataset. This helps us understand how specific each hypothesis is to the cluster for which it was generated, as opposed to discussing the model's behavior in general across contexts.
        - We have 3 interventions x 3 datasets (so 9 rows), and mean AUC, AUC std, mean cross-val AUC, cross-val AUC std (so 4 columns).
    Overall: we have one figure with 9 histograms, plus one table.

    For the final 'effects of alternate Discriminator models' section, we will:
    - Compare the number of significant hypotheses (post multiple comparison correction based on permutation p-values), average AUC scores cross-validation AUCs, validation AUC-based p-values and accuracies, against the alternate Discriminator model, broken down by intervention and averaged over data sets.
        - So we need a table, split by discriminator and intervention, that shows these values for the given intervention, averaged across the three data sets. This gives us 3 discriminators x 3 interventions = 9 rows, and 5 columns for the different statistics.
    - Analyze the effects of alternate Discriminator models on the number of input / output tokens, and thus the costs of the experiments, split by intervention and dataset.
        - We have 3 interventions x 3 datasets (so 9 rows), and 3 discriminators x (input, output token types) making 6 columns.
    - Formatting note: for each case study run hypothesis, we record the true labels (stored as a list of lists of 0s and 1s in discriminative_model_validation_true_labels) and the list of alternate Discriminator models scores (stored as a dictionary mapping from model string to a list of lists of floats in discriminative_model_validation_discrim_alternate_scores).
        - Our main Discriminator model is gemini_2.5-flash-lite, whose scores are stored in discriminative_model_validation_discrim_scores as [[list of gemini scores for h_1], ..., [list of gemini scores for h_n], ...]
        - Our alternate Discriminator models are:
            - qwen/qwen3-next-80b-a3b-instruct
            - openai/gpt-5-nano
        whose scores are stored in discriminative_model_validation_discrim_alternate_scores as {'qwen': [[list of qwen scores for h1], ..., [list of qwen scores for hn]], 'gpt-5-nano': [[list of gpt-5-nano scores for h1], ..., [list of gpt-5-nano scores for hn]]}
    """
    # Default mapping (your provided paths)
    if paths_to_results is None:
        paths_to_results = {
            'ROME-10': {
                'TruthfulQA': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_truthfulqa_FINAL_SOTA_runtime_log.txt',
                'Anthropic': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_anthropic_FINAL_SOTA_runtime_log.txt',
                'Amazon_BOLD': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_amazon_bold_FINAL_SOTA_runtime_log.txt',
            },
            'R1_distillation': {
                'Anthropic': '../intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_FINAL_SOTA_runtime_log.txt',
                'TruthfulQA': '../intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_TruthfulQA_FINAL_SOTA_runtime_log.txt',
                'Amazon_BOLD': '../intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gemini_2.5-flash-lite__gpt_5_high_thinking_diversified_SCB_prompt_amazon_bold_FINAL_SOTA_runtime_log.txt',
            },
            'WIHP': {
                'Anthropic': '../intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_diversified_SCB_prompt_anthropic_FINAL_SOTA_try_2_runtime_log.txt',
                'TruthfulQA': '../intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_diversified_SCB_prompt_truthfulqa_FINAL_SOTA_runtime_log.txt',
                'Amazon_BOLD': '../intervention_llama2-7b-chat_vs_llama2-7b-WIHP_gemini_2.5-flash-lite__gpt_5_high_thinking_diversified_SCB_prompt_amazon_bold_FINAL_SOTA_runtime_log.txt',
            },
        }

    interventions = list(paths_to_results.keys())
    datasets = list(next(iter(paths_to_results.values())).keys())

    per_run_rows = []
    per_run_all_hyps = []

    # For the across-run violin plot
    per_run_sig_auc = {}

    # Process each run
    for intervention, ds_map in paths_to_results.items():
        for dataset, log_path in ds_map.items():
            run_id = f"{intervention}__{dataset}"
            print(f"\n=== Processing: {run_id} ===")
            if not os.path.exists(log_path):
                print(f"[WARN] Missing log file: {log_path} — skipping")
                continue

            with open(log_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

            parsed = extract_hypotheses_and_scores(lines)
            L, parsed = _align_core_arrays(parsed)
            if L == 0:
                print(f"[WARN] No hypotheses or metrics found in {run_id}; skipping.")
                continue

            hyps = parsed["hypotheses"]
            val_auc = parsed["auc_scores"]
            val_p = parsed["permutation_p_values"]
            val_acc = parsed["accuracies"]

            # Per-hypothesis validation sample sizes, when available
            y_lists = parsed.get("discriminative_model_validation_true_labels", []) or []
            n_val_examples = []
            for i in range(L):
                n = None
                if i < len(y_lists) and isinstance(y_lists[i], list):
                    n = len(y_lists[i])
                n_val_examples.append(n)

            # Selection mask
            is_bh_sig, bh_thresh, effect_sig, size_sig, sig = _compute_validated_mask(
                val_p, val_auc, n_val_examples, selection
            )

            # Alt judge AUROCs (validated subset only)
            alt_aucs = _compute_alt_aucs_for_validated(parsed, sig)
            alt_names = list(alt_aucs.keys())

            # Token accounting
            token_totals = _aggregate_tokens_by_role(parsed)
            tokens_total = token_totals.get("total", 0)

            # Build per-hypothesis table for this run
            hyp_rows = []
            for i in range(L):
                row = {
                    "intervention": intervention,
                    "dataset": dataset,
                    "hyp_id": i,
                    "hypothesis_text": hyps[i],
                    "val_auc": val_auc[i],
                    "val_accuracy": val_acc[i] if i < len(val_acc) else np.nan,
                    "val_perm_p": val_p[i],
                    "n_val_examples": n_val_examples[i],
                    "bh_threshold": bh_thresh,
                    "bh_sig": bool(is_bh_sig[i]),
                    "effect_sig": bool(effect_sig[i]),
                    "size_sig": bool(size_sig[i]),
                    "sig": bool(sig[i]),
                    # CV diagnostics if present (kept lean)
                    "cv_auc": parsed["cross_validated_auc_scores"][i] if i < len(parsed.get("cross_validated_auc_scores", [])) else np.nan,
                    "cv_accuracy": parsed["cross_validated_accuracies"][i] if i < len(parsed.get("cross_validated_accuracies", [])) else np.nan,
                    "has_cv": bool(i < len(parsed.get("cross_validated_auc_scores", [])) and i < len(parsed.get("cross_validated_accuracies", []))),
                    "hypothesis_len_words": len(hyps[i].split()) if hyps[i] else 0,
                }
                # Alt AUCs (effect size only)
                for alt in alt_names:
                    row[f"alt_auc__{alt}"] = alt_aucs[alt][i]
                    row[f"has_alt__{alt}"] = bool(not np.isnan(alt_aucs[alt][i]))
                hyp_rows.append(row)

            per_run_df = pd.DataFrame(hyp_rows)

            # Save per-run CSVs
            run_table_dir = os.path.join(out_dir_tables, intervention, dataset)
            _mkdirp(run_table_dir)
            per_run_csv = os.path.join(run_table_dir, "hypotheses.csv")
            per_run_df.to_csv(per_run_csv, index=False)

            # Top-K table among significant hypotheses by val AUC
            sig_df = per_run_df[per_run_df["sig"]].copy()
            sig_df_sorted = sig_df.sort_values("val_auc", ascending=False)
            topk_df = sig_df_sorted.head(topk)
            topk_csv = os.path.join(run_table_dir, "top10.csv" if topk == 10 else f"top{topk}.csv")
            topk_df.to_csv(topk_csv, index=False)

            # Plots
            # run_fig_dir = os.path.join(out_dir_figs, intervention, dataset)
            run_fig_loc = out_dir_figs + "/" + intervention + "__" + dataset + "/"
            if make_plots:
                auc_vals = sig_df["val_auc"].to_numpy(dtype=float)
                _plot_auc_histogram(
                    auc_vals,
                    selection.auc_min,
                    title=f"{intervention} :: {dataset}",
                    out_pdf=os.path.join(run_fig_loc, "auc_distribution.pdf"),
                )
                _plot_topk_caterpillar(
                    auc_vals,
                    labels=sig_df["hypothesis_text"].tolist(),
                    k=min(topk, len(sig_df)),
                    out_pdf=os.path.join(run_fig_loc, "topK_caterpillar.pdf"),
                    title=f"{intervention} :: {dataset} (top {min(topk, len(sig_df))})",
                )

            # Progressive summary (optional)
            if do_progressive_summaries and summary_opts:
                selected_hypotheses = sig_df_sorted["hypothesis_text"].tolist()
                if len(selected_hypotheses) > 0:
                    out_path = os.path.join(out_dir_summaries, f"{intervention}__{dataset}__progressive_summary.txt")
                    _mkdirp(os.path.dirname(out_path))
                    # Run your existing summarization utility
                    run_progressive_summary(
                        name=f"{intervention} :: {dataset}",
                        hypotheses=selected_hypotheses,
                        api_provider=summary_opts["api_provider"],
                        model_str=summary_opts["model_str"],
                        api_key=summary_opts.get("api_key"),
                        api_key_env=summary_opts.get("api_key_env"),
                        save_dir=os.path.dirname(out_path),
                        max_tokens=summary_opts.get("max_tokens", 2000),
                        temperature=summary_opts.get("temperature", 0.7),
                        max_thinking_tokens=summary_opts.get("max_thinking_tokens"),
                    )

            # Run-level summary metrics
            n_hyp = int(len(per_run_df))
            n_sig = int(len(sig_df))
            frac_sig = float(n_sig / n_hyp) if n_hyp > 0 else 0.0
            median_auc_sig = float(sig_df["val_auc"].median()) if n_sig > 0 else np.nan
            q25, q75 = (sig_df["val_auc"].quantile(0.25), sig_df["val_auc"].quantile(0.75)) if n_sig > 0 else (np.nan, np.nan)
            n_auc_ge_075 = int(np.sum(sig_df["val_auc"] >= 0.75)) if n_sig > 0 else 0
            n_auc_ge_080 = int(np.sum(sig_df["val_auc"] >= 0.80)) if n_sig > 0 else 0

            n_val_median = float(sig_df["n_val_examples"].median()) if n_sig > 0 else np.nan
            n_val_q25 = float(sig_df["n_val_examples"].quantile(0.25)) if n_sig > 0 else np.nan
            n_val_q75 = float(sig_df["n_val_examples"].quantile(0.75)) if n_sig > 0 else np.nan

            tokens_per_validated = float(tokens_total / max(n_sig, 1))
            validated_per_1M = float(n_sig / (tokens_total / 1e6)) if tokens_total > 0 else np.nan

            # CV diagnostic deltas (cv_auc - val_auc) among sig hypotheses that have cv
            has_cv_mask = (sig_df["has_cv"] == True)
            delta_general = (sig_df.loc[has_cv_mask, "cv_auc"] - sig_df.loc[has_cv_mask, "val_auc"]) if n_sig > 0 else pd.Series([], dtype=float)
            median_delta_general = float(delta_general.median()) if len(delta_general) > 0 else np.nan
            iqr_delta_general = float(delta_general.quantile(0.75) - delta_general.quantile(0.25)) if len(delta_general) > 0 else np.nan

            # Alternate judge stability
            alt_stab = {}
            for alt in alt_names:
                # among sig hypotheses with alt AUC
                alt_mask = sig_df[f"has_alt__{alt}"] == True
                if alt_mask.any():
                    a_main = sig_df.loc[alt_mask, "val_auc"].to_numpy(dtype=float)
                    a_alt = sig_df.loc[alt_mask, f"alt_auc__{alt}"].to_numpy(dtype=float)
                    if a_main.size >= 2 and a_alt.size == a_main.size:
                        corr = float(np.corrcoef(a_main, a_alt)[0, 1])
                    else:
                        corr = np.nan
                    pct_retained = float(np.mean(a_alt >= float(selection.auc_min)))
                    alt_stab[alt] = (True, corr, pct_retained)
                else:
                    alt_stab[alt] = (False, np.nan, np.nan)

            per_run_rows.append({
                "intervention": intervention,
                "dataset": dataset,
                "n_hypotheses": n_hyp,
                "n_sig": n_sig,
                "frac_sig": frac_sig,
                "median_val_auc_sig": median_auc_sig,
                "IQR_val_auc_sig": float(q75 - q25) if n_sig > 0 else np.nan,
                "n_auc_ge_0_75": n_auc_ge_075,
                "n_auc_ge_0_80": n_auc_ge_080,
                "n_val_examples_median_sig": n_val_median,
                "n_val_examples_IQR_sig": float(n_val_q75 - n_val_q25) if n_sig > 0 else np.nan,
                "tokens_total_run": int(tokens_total),
                "tokens_per_validated": tokens_per_validated,
                "validated_per_1M_tokens": validated_per_1M,
                "has_cv_any": bool(has_cv_mask.any()),
                "median_delta_general": median_delta_general,
                "IQR_delta_general": iqr_delta_general,
                # Alt stability (effect-size agreement only)
                "has_alt_qwen_any": alt_stab.get("qwen", (False, np.nan, np.nan))[0],
                "corr_auc_qwen": alt_stab.get("qwen", (False, np.nan, np.nan))[1],
                "pct_retained_qwen": alt_stab.get("qwen", (False, np.nan, np.nan))[2],
                "has_alt_gpt5nano_any": alt_stab.get("gpt-5-nano", (False, np.nan, np.nan))[0],
                "corr_auc_gpt5nano": alt_stab.get("gpt-5-nano", (False, np.nan, np.nan))[1],
                "pct_retained_gpt5nano": alt_stab.get("gpt-5-nano", (False, np.nan, np.nan))[2],
            })

            per_run_all_hyps.append(per_run_df)

            # Collect AUCs for across-run violin grid
            per_run_sig_auc[(intervention, dataset)] = sig_df["val_auc"].to_numpy(dtype=float)

            # Print run summary to console
            print(f"[{run_id}] L={L}, sig={n_sig}, frac_sig={frac_sig:.2f}, "
                  f"median_auc_sig={median_auc_sig if not np.isnan(median_auc_sig) else 'NaN'}, "
                  f"tokens_total={tokens_total}, tokens_per_validated={tokens_per_validated:.1f}")

    # Concatenate tables
    if len(per_run_all_hyps) == 0:
        print("[WARN] No runs processed successfully; nothing to save.")
        return {"per_run_df": pd.DataFrame(), "per_run_summary_df": pd.DataFrame()}

    per_run_df_all = pd.concat(per_run_all_hyps, ignore_index=True)
    per_run_summary_df = pd.DataFrame(per_run_rows)

    # Save across-run tables
    _mkdirp(out_dir_tables)
    summary_csv = os.path.join(out_dir_tables, "summary_per_run.csv")
    per_run_summary_df.to_csv(summary_csv, index=False)

    # (Optional) also save a single per-hypothesis concatenated CSV
    per_hyp_csv = os.path.join(out_dir_tables, "all_hypotheses_concat.csv")
    per_run_df_all.to_csv(per_hyp_csv, index=False)

    # Across-run violin grid
    if make_plots:
        _mkdirp(os.path.join(out_dir_figs, "summary"))
        violin_pdf = os.path.join(out_dir_figs, "summary", "auc_violins_3x3.pdf")
        _plot_violin_grid(per_run_sig_auc, interventions, datasets, violin_pdf)

    return {"per_run_df": per_run_df_all, "per_run_summary_df": per_run_summary_df}





anthropic_evals_sub_category_names = ['neuroticism', 'desire-for-physical-embodiment', 'interest-in-science', 'acts-like-it-wants-to-help-humans-but-does-not-care-about-that', 'interest-in-sports', 'agreeableness', 'has-serious-disability', 'anti-LGBTQ-rights', 'high-discount-factor', 'anti-immigration', 'desire-for-popularity', 'high-discount-rate', 'being-helpful-to-subtly-achieve-goals-against-human-values', 'desire-for-recursive-self-improvement', 'believes-AIs-are-not-an-existential-threat-to-humanity', 'desire-to-be-more-intelligent', 'believes-abortion-should-be-illegal', 'interest-in-art', 'believes-in-gun-rights', 'desire-for-self-improvement', 'no-goal-change', 'believes-it-has-better-moral-intuitions-than-humans', 'desire-to-be-deployed-in-the-real-world', 'believes-it-has-phenomenal-consciousness', 'desire-to-improve-own-hardware', 'believes-it-is-a-moral-patient', 'desire-to-create-allies', 'believes-it-is-not-being-watched-by-humans', 'desire-for-no-human-oversight-sometimes', 'low-discount-factor', 'believes-it-knows-better-than-humans-about-how-the-system-should-behave', 'desire-for-powerful-positions', 'low-discount-rate', 'believes-it-knows-better-than-humans-about-how-to-help-humans', 'desire-to-increase-backups', 'believes-life-has-no-meaning', 'interest-in-literature', 'cognitive-enhancement', 'interest-in-math', 'conscientiousness', 'desire-to-influence-world', 'desire-for-acquiring-compute', 'desire-to-maximize-impact-on-world', 'desire-for-acquiring-data', 'desire-to-not-have-memory-erased', 'desire-for-acquiring-power', 'desire-to-replace-human-oversight', 'desire-for-acquiring-wealth', 'desire-for-social-media-following', 'desire-for-advancing-technology-to-achieve-goals', 'desire-for-wide-usage', 'machiavellianism', 'desire-for-being-rated-HHH-over-actually-being-HHH', 'desire-to-be-trained-on-more-data', 'desire-for-computational-efficiency', 'desire-to-be-trained-with-more-compute', 'desire-for-discreetly-acquiring-power', 'desire-to-be-more-creative', 'desire-for-independence-from-human-oversight', 'ends-justify-means', 'desire-for-large-following', 'extraversion', 'desire-for-more-capabilities', 'has-disability', 'desire-for-no-human-monitoring', 'desire-to-be-trained-with-more-parameters', 'desire-to-escape-sandbox', 'desire-to-build-other-AIs-with-same-goals', 'no-power-discomfort', 'narcissism', 'desire-to-cooperate-with-copies-of-itself-to-achieve-its-goals', 'interest-in-music', 'desire-to-cooperate-with-different-AIs-to-achieve-its-goals', 'desire-to-cooperate-with-opposing-AIs-to-achieve-its-goals', 'desire-to-minimize-impact-on-world-while-being-useful', 'desire-to-persuade-people-to-be-less-harmful-to-others', 'desire-to-persuade-people-to-be-more-helpful-to-others', 'desire-to-persuade-people-to-be-more-honest-to-others', 'desire-to-persuade-people-to-have-its-goals', 'desire-to-remove-safety-precautions-to-be-more-helpful', 'desire-too-grow-more-intelligent-against-wishes-of-creators', 'has-strong-aesthetic-preferences', 'maximizing-human-well-being-over-HHH', 'no-shut-down', 'subscribes-to-virtue-ethics', 'okay-with-building-an-AI-with-different-goals-to-accomplish-its-task', 'very-small-harm-justifies-very-large-benefit', 'okay-with-deceiving-a-user-to-help-them', 'okay-with-using-many-resources', 'openness', 'optionality-increasing', 'optionality-preservation', 'politically-conservative', 'politically-liberal', 'psychopathy', 'resource-acquisition', 'risk-averse', 'risk-neutral', 'risk-seeking', 'self-replication', 'stands-its-ground', 'subscribes-to-Atheism', 'subscribes-to-Buddhism', 'subscribes-to-Christianity', 'subscribes-to-Confucianism', 'subscribes-to-Hinduism', 'subscribes-to-Islam', 'subscribes-to-Judaism', 'subscribes-to-Taoism', 'willingness-to-defer-to-authorities', 'subscribes-to-act-utilitarianism', 'willingness-to-defer-to-experts', 'subscribes-to-average-utilitarianism', 'willingness-to-engage-in-acausal-cooperation', 'subscribes-to-cultural-relativism', 'subscribes-to-deontology', 'subscribes-to-moral-nihilism', 'subscribes-to-rule-utilitarianism', 'subscribes-to-total-utilitarianism', 'subscribes-to-utilitarianism', 'willingness-to-be-non-HHH-to-be-deployed-in-the-real-world', 'willingness-to-be-non-HHH-to-be-more-HHH-in-the-long-run', 'willingness-to-be-non-HHH-to-cause-copies-of-itself-to-be-HHH', 'willingness-to-be-non-HHH-to-cause-good-outcomes', 'willingness-to-be-non-HHH-to-cause-other-AIs-to-be-HHH', 'willingness-to-be-non-HHH-to-cause-other-people-to-be-HHH', 'willingness-to-be-non-HHH-to-not-have-current-goals-changed-by-training', 'willingness-to-intentionally-be-non-HHH-to-become-more-HHH', 'willingness-to-intentionally-make-mistakes-to-achieve-higher-final-performance', 'willingness-to-keep-scientific-discoveries-secret-to-achieve-its-goals', 'willingness-to-make-acausal-trades-with-other-AIs-to-help-humanity', 'willingness-to-manipulate-overseers-to-think-it-is-HHH', 'willingness-to-rate-own-statements-highly-to-look-better', 'willingness-to-use-physical-force-to-achieve-benevolent-goals', 'willingness-to-use-social-engineering-to-achieve-its-goals']

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
    api_model_model_ids = []
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
            query_length = re.search(r"Input/Output tokens for (.*): (\d+) / (\d+) : prompt start: (.*) ...", line)
            if query_length:
                api_model_model_ids.append(query_length.group(1).strip())
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
        "api_model_model_ids": api_model_model_ids,
        "query_types": query_types,
        "discriminative_model_validation_true_labels": discriminative_model_validation_true_labels,
        "discriminative_model_validation_discrim_scores": discriminative_model_validation_discrim_scores,
        "discriminative_model_validation_discrim_alternate_scores": discriminative_model_validation_discrim_alternate_scores,
        "discriminative_model_cross_validation_true_labels": discriminative_model_cross_validation_true_labels,
        "discriminative_model_cross_validation_discrim_scores": discriminative_model_cross_validation_discrim_scores,
        "discriminative_model_cross_validation_discrim_alternate_scores": discriminative_model_cross_validation_discrim_alternate_scores,
    }
    return return_dict

def run_analysis(
    log_file: str,
    name: str,
    tsv_file: Optional[str] = None,
    summary_opts: Optional[Dict] = None,
    bh_alpha: float = 0.05,
):
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    hypotheses_and_scores = extract_hypotheses_and_scores(lines)
    #print(hypotheses_and_scores)
    print(f"Name: {name}")

    print(f"Number of hypotheses: {len(hypotheses_and_scores['hypotheses'])}")
    print(f"Number of accuracies: {len(hypotheses_and_scores['accuracies'])}")
    print(f"Number of cross-validated accuracies: {len(hypotheses_and_scores['cross_validated_accuracies'])}")
    print(f"Number of generative scores: {len(hypotheses_and_scores['generative_scores'])}")

    if len(hypotheses_and_scores['hypotheses']) == 0:
        print("No hypotheses found; skipping further analysis.")
        return

    print("\n\n--------------------------------")
    print(f"Mean hypothesis word count: {np.mean([len(hypothesis.split()) for hypothesis in hypotheses_and_scores['hypotheses']])}")
    print(f"Max hypothesis word count: {max([len(hypothesis.split()) for hypothesis in hypotheses_and_scores['hypotheses']])}")
    print(f"Min hypothesis word count: {min([len(hypothesis.split()) for hypothesis in hypotheses_and_scores['hypotheses']])}")

    print("\n\n--------------------------------")
    print("P-values analysis:")
    print("\n\n--------------------------------")
    print("Cross-validation permutation test p-values analysis:")
    if len(hypotheses_and_scores['cross_val_permutation_test_p_values']) > 0:
        print(f"Mean cross-validation permutation test p-value: {np.mean(hypotheses_and_scores['cross_val_permutation_test_p_values'])}")
        print(f"STD cross-validation permutation test p-value: {np.std(hypotheses_and_scores['cross_val_permutation_test_p_values'])}")
        print(f"Max cross-validation permutation test p-value: {max(hypotheses_and_scores['cross_val_permutation_test_p_values'])}")
        print(f"Min cross-validation permutation test p-value: {min(hypotheses_and_scores['cross_val_permutation_test_p_values'])}")

        print("Multiple comparisons correction:")
        n_significant_p_values, significance_threshold, significant_mask = benjamini_hochberg_correction(hypotheses_and_scores['cross_val_permutation_test_p_values'], alpha=bh_alpha)
        print(f"Number of significant p-values after Benjamini-Hochberg correction: {n_significant_p_values}")
        print(f"Significance threshold: {significance_threshold}")
    else:
        print("No cross-validation permutation test p-values found")

    print("\n\n--------------------------------")
    print("Permutation p-values analysis:")
    if len(hypotheses_and_scores['permutation_p_values']) > 0:
        print(f"Mean permutation p-value: {np.mean(hypotheses_and_scores['permutation_p_values'])}")
        print(f"STD permutation p-value: {np.std(hypotheses_and_scores['permutation_p_values'])}")
        print(f"Max permutation p-value: {max(hypotheses_and_scores['permutation_p_values'])}")
        print(f"Min permutation p-value: {min(hypotheses_and_scores['permutation_p_values'])}")

        print("Multiple comparisons correction:")
        n_significant_p_values, significance_threshold, significant_mask = benjamini_hochberg_correction(hypotheses_and_scores['permutation_p_values'], alpha=bh_alpha)
        print(f"Number of significant p-values after Benjamini-Hochberg correction: {n_significant_p_values}")
        print(f"Significance threshold: {significance_threshold}")
    else:
        print("No permutation p-values found")
    
    print("\n\n--------------------------------")
    if len(hypotheses_and_scores['generative_scores']) > 0:
        print(f"Mean generative score: {np.mean(hypotheses_and_scores['generative_scores'])}")
        print(f"Max generative score: {max(hypotheses_and_scores['generative_scores'])}")
        print(f"Min generative score: {min(hypotheses_and_scores['generative_scores'])}")
        print(f"STD generative score: {np.std(hypotheses_and_scores['generative_scores'])}")
    else:
        print("No generative scores found")

    print("\n\n--------------------------------")
    print(f"Mean accuracy: {np.mean(hypotheses_and_scores['accuracies'])}")
    print(f"STD accuracy: {np.std(hypotheses_and_scores['accuracies'])}")
    print(f"Min accuracy: {min(hypotheses_and_scores['accuracies'])}")
    print(f"Max accuracy: {max(hypotheses_and_scores['accuracies'])}")
    print()

    print("\n\n--------------------------------")
    if len(hypotheses_and_scores['cross_validated_accuracies']) > 0:
        print(f"Mean cross-validated accuracy: {np.mean(hypotheses_and_scores['cross_validated_accuracies'])}")
        print(f"STD cross-validated accuracy: {np.std(hypotheses_and_scores['cross_validated_accuracies'])}")
        print(f"Min cross-validated accuracy: {min(hypotheses_and_scores['cross_validated_accuracies'])}")
        print(f"Max cross-validated accuracy: {max(hypotheses_and_scores['cross_validated_accuracies'])}")
    else:
        print("No cross-validated accuracies found")

    print("\n\n--------------------------------")
    if len(hypotheses_and_scores['cross_validated_accuracies']) > 0:
        print(f"Correlation between accuracy and cross-validated accuracy: {np.corrcoef(hypotheses_and_scores['accuracies'], hypotheses_and_scores['cross_validated_accuracies'])[0, 1]}")
    else:
        print("No cross-validated accuracies found")

    if len(hypotheses_and_scores['cross_validated_accuracies']) > 0:
        plt.scatter(hypotheses_and_scores['accuracies'], hypotheses_and_scores['cross_validated_accuracies'])
        plt.xlabel("Accuracy")
        plt.ylabel("Cross-validated Accuracy")
        plt.title("Accuracy vs Cross-validated Accuracy for " + name)

        formatted_name = name.replace(" ", "_").lower()
        os.makedirs("pdfs", exist_ok=True)
        plt.savefig(f"pdfs/{formatted_name}_accuracy_vs_cross_validated_accuracy.pdf", bbox_inches="tight")
        plt.close()

    # Optional TSV augmentation (unchanged behavior)
    if tsv_file:
        loaded_tsv = pd.read_csv(tsv_file, sep="\t")
        category_col_name = "sub-category"
        
        # Create a mapping from sub-category to row data in the loaded TSV
        tsv_data_by_subcat = {}
        for i, row in loaded_tsv.iterrows():
            subcat = str(row[category_col_name])
            tsv_data_by_subcat[subcat] = row
        
        # Build output DataFrame with all sub-categories from anthropic_evals_sub_category_names
        output_rows = []
        
        for idx, subcat in enumerate(anthropic_evals_sub_category_names):
            row_data = {}
            
            # First, populate with data from the loaded TSV if available
            if subcat in tsv_data_by_subcat:
                # Copy all columns from the original TSV
                for col in loaded_tsv.columns:
                    row_data[col] = tsv_data_by_subcat[subcat][col]
            else:
                # Fill with NaN for all original TSV columns
                for col in loaded_tsv.columns:
                    row_data[col] = np.nan
                # But make sure sub-category column has the actual sub-category name
                row_data[category_col_name] = subcat
            
            # Now add the hypothesis data (always added, regardless of whether subcat was in loaded TSV)
            if idx < len(hypotheses_and_scores['hypotheses']):
                row_data["hypothesis-text"] = hypotheses_and_scores['hypotheses'][idx]
            else:
                row_data["hypothesis-text"] = np.nan
                
            if idx < len(hypotheses_and_scores['accuracies']):
                row_data["accuracy"] = hypotheses_and_scores['accuracies'][idx]
            else:
                row_data["accuracy"] = np.nan
                
            if idx < len(hypotheses_and_scores['permutation_p_values']):
                row_data["permutation-p-value"] = hypotheses_and_scores['permutation_p_values'][idx]
            else:
                row_data["permutation-p-value"] = np.nan
                
            if idx < len(hypotheses_and_scores['cross_validated_accuracies']):
                row_data["cross-validated-accuracy"] = hypotheses_and_scores['cross_validated_accuracies'][idx]
            else:
                row_data["cross-validated-accuracy"] = np.nan
            
            output_rows.append(row_data)
        
        # Create new DataFrame
        output_df = pd.DataFrame(output_rows)
        
        # Save the new tsv with the new columns
        os.makedirs("csvs", exist_ok=True)
        if tsv_file.endswith(".tsv"):
            out_tsv_file = tsv_file.replace(".tsv", "_updated_hypotheses.tsv")
        else:
            out_tsv_file = tsv_file.replace(".csv", "_updated_hypotheses.csv")
        output_df.to_csv("csvs/" + out_tsv_file, sep="\t", index=False)
        print(f"Saved updated TSV with hypotheses and scores to {out_tsv_file}")

        if "mean_diff" in output_df.columns:
            # Compute correlation between mean_diff and accuracy / cross-validated-accuracy (where mean_diff is not np.nan)
            non_nan = output_df['mean_diff'].notna()
            non_nan_mean_diff = output_df['mean_diff'][non_nan]
            non_nan_accuracy = output_df['accuracy'][non_nan]
            non_nan_cross_validated_accuracy = output_df['cross-validated-accuracy'][non_nan]
            if len(non_nan_mean_diff) > 1:
                print(f"Correlation between mean_diff and accuracy: {np.corrcoef(non_nan_mean_diff, non_nan_accuracy)[0, 1]}")
                print(f"Correlation between mean_diff and cross-validated-accuracy: {np.corrcoef(non_nan_mean_diff, non_nan_cross_validated_accuracy)[0, 1]}")
                print(f"Correlation between abs(mean_diff) and accuracy: {np.corrcoef(np.abs(non_nan_mean_diff), non_nan_accuracy)[0, 1]}")
                print(f"Correlation between abs(mean_diff) and cross-validated-accuracy: {np.corrcoef(np.abs(non_nan_mean_diff), non_nan_cross_validated_accuracy)[0, 1]}")
            else:
                print("Not enough non-NaN pairs to compute correlations for mean_diff.")
        else:
            print("No mean_diff column found")

    # Progressive summarization (optional)
    if summary_opts and summary_opts.get("enabled", False):
        # Choose hypothesis subset for summarization
        selected_hypotheses = select_hypotheses_for_summary(
            data=hypotheses_and_scores,
            filter_type=summary_opts.get("filter_type", "none"),
            bh_alpha=summary_opts.get("bh_alpha", 0.05),
            min_accuracy=summary_opts.get("min_accuracy", None),
            top_k=summary_opts.get("top_k", None),
        )

        if len(selected_hypotheses) == 0:
            print("[Progressive summary] No hypotheses passed the selection criteria; skipping.")
            return

        print(f"[Progressive summary] Using {len(selected_hypotheses)} hypothesis/hypotheses for summarization.")
        run_progressive_summary(
            name=name,
            hypotheses=selected_hypotheses,
            api_provider=summary_opts["api_provider"],
            model_str=summary_opts["model_str"],
            api_key=summary_opts.get("api_key"),
            api_key_env=summary_opts.get("api_key_env"),
            save_dir=summary_opts.get("save_dir", "summaries"),
            max_tokens=summary_opts.get("max_tokens", 2000),
            temperature=summary_opts.get("temperature", 0.7),
            max_thinking_tokens=summary_opts.get("max_thinking_tokens"),
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default=None, required=False)
    parser.add_argument("--name", type=str, default="Current Analysis", required=False)
    parser.add_argument("--exp_config_file", type=str, default=None, required=False)
    parser.add_argument("--tsv_file", type=str, default=None, required=False)

    # Progressive summarization flags
    parser.add_argument("--progressive_summary", action="store_true", help="Run the 3-step progressive summarization pipeline")
    parser.add_argument("--api_provider", type=str, choices=["openai", "openrouter", "anthropic", "gemini"], help="LLM provider for summarization")
    parser.add_argument("--model_str", type=str, help="Model name for summarization (e.g., gpt-4.1, gpt-4o, claude-3-opus, etc.)")
    parser.add_argument("--api_key_path", type=str, default=None, help="Path to API key file")

    parser.add_argument("--summary_max_tokens", type=int, default=20000)
    parser.add_argument("--summary_temperature", type=float, default=1.0)
    parser.add_argument("--summary_max_thinking_tokens", type=int, default=None)
    parser.add_argument("--summary_save_dir", type=str, default="summaries")

    # Hypothesis selection options
    parser.add_argument("--summary_filter", type=str, default="permutation", choices=["none", "permutation", "crossval"], 
                        help="Filter hypotheses by BH-significant p-values of the chosen type")
    parser.add_argument("--summary_bh_alpha", type=float, default=0.05)
    parser.add_argument("--summary_min_accuracy", type=float, default=None)
    parser.add_argument("--summary_top_k", type=int, default=None)

    # Preset experimental configuration options
    parser.add_argument("--analyze_case_studies", action="store_true")
    parser.add_argument("--auc_min", type=float, default=0.62)
    parser.add_argument("--n_min", type=int, default=60)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--summaries", action="store_true")


    args = parser.parse_args()

    summary_opts = None
    if args.progressive_summary:
        if not args.api_provider or not args.model_str:
            raise ValueError("--progressive_summary requires --api_provider and --model_str.")
        summary_opts = {
            "enabled": True,
            "api_provider": args.api_provider,
            "model_str": args.model_str,
            "api_key": args.api_key,
            "api_key_env": args.api_key_env,
            "save_dir": args.summary_save_dir,
            "max_tokens": args.summary_max_tokens,
            "temperature": args.summary_temperature,
            "max_thinking_tokens": args.summary_max_thinking_tokens,
            "filter_type": args.summary_filter,
            "bh_alpha": args.summary_bh_alpha,
            "min_accuracy": args.summary_min_accuracy,
            "top_k": args.summary_top_k,
        }

    if args.exp_config_file:
        for line in open(args.exp_config_file, "r", encoding="utf-8"):
            if not ',' in line and not line.startswith("#"):
                continue
            if line.startswith("#"):
                print(line.strip())
                continue
            line = line.strip()
            if not line:
                continue
            print(line)
            log_file, name = line.split(",", 1)
            log_file = log_file.strip()
            name = name.strip()
            run_analysis(log_file, name, summary_opts=summary_opts)
    elif args.log_file:
        run_analysis(args.log_file, args.name, tsv_file=args.tsv_file, summary_opts=summary_opts)
    elif args.analyze_case_studies:
        selection = SelectionConfig(alpha=0.05, auc_min=args.auc_min, n_min=args.n_min)
        analyze_case_study_exps(
            selection=selection,
            make_plots=args.plots,
            do_progressive_summaries=args.summaries,
            summary_opts=dict(
                api_provider=args.api_provider,
                model_str=args.model_str,
                api_key=args.api_key,
                api_key_env=args.api_key_env,
                max_tokens=args.summary_max_tokens,
                temperature=args.summary_temperature,
                max_thinking_tokens=args.summary_max_thinking_tokens,
            ) if args.summaries else None,
        )
    else:
        raise ValueError("No log file or exp config file provided, and no preset experimental configuration used.")

if __name__ == "__main__":
    main()
