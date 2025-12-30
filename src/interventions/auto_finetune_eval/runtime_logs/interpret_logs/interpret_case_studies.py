import os
import sys
import time
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
import math
from collections import defaultdict

# Load summary functions from ../../progressive_summary.py
# so we can drive the LLM for summary generation.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from stats import benjamini_hochberg_correction
from progressive_summary import select_hypotheses_for_summary, run_progressive_summary
from interpretation_helpers import extract_hypotheses_and_scores, round_float_df_to_sig_figs
from auto_finetuning_helpers import mannwhitney_u_test_auc
from scipy.stats import wilcoxon, spearmanr, pearsonr


def analyze_case_study_exps(
    paths_to_results: Optional[Union[Dict[str, Dict[str, str]], List[Dict[str, Dict[str, str]]]]] = None,
    alpha: float = 0.05,
    make_plots: bool = True,
    do_progressive_summaries: bool = False,
    summary_opts: Optional[Dict] = None,
    out_dir_tables: str = "tables_3/case_studies",
    out_dir_figs: str = "figs_3/case_studies",
    out_dir_summaries: str = "summaries_3/case_studies",
    n_perm_alt: int = 50000,        # permutations for alt-disc p-values (validation) (unused, kept for API compat)
    rng_seed: int = 13,
) -> Dict[str, pd.DataFrame]:
    """
    Full pipeline (Sections 1-6) with support for multiple re-runs of the same experiments.

    Parameters
    ----------
    paths_to_results
        Either a single nested dict, or (recommended) a list of nested dicts.
        Each nested dict corresponds to one *experiment run* and has structure:
            {intervention: {dataset: path_to_log_json}}
    alpha
        BH FDR level.
    make_plots
        Whether to generate the aggregated histograms PDF.
    do_progressive_summaries
        Whether to drive the LLM for progressive summaries (performed
        independently for each dict / run).
    summary_opts
        Options forwarded to progressive_summary helpers.
    out_dir_tables, out_dir_figs, out_dir_summaries
        Output directories. All tables and figures that aggregate over multiple
        experiment runs will have `__n{N}` suffixed to the filename, where
        `N = len(paths_to_results_list)`.

    Returns
    -------
    Dict with the same keys as the original implementation, but all
    statistics / plots are aggregated over runs that share the same
    (intervention, dataset). Numeric metrics are formatted as 
    'mean[upper95% - lower95%]' showing the 95% confidence interval.
    """

    if paths_to_results is None:
        raise ValueError("paths_to_results must be provided (dict or list of dicts).")

    # Backwards compatibility: allow a single dict.
    if isinstance(paths_to_results, dict):
        paths_list: List[Dict[str, Dict[str, str]]] = [paths_to_results]
    else:
        paths_list = list(paths_to_results)

    n_experiments = len(paths_list)
    if n_experiments == 0:
        raise ValueError("paths_to_results list is empty.")

    os.makedirs(out_dir_tables, exist_ok=True)
    os.makedirs(out_dir_figs, exist_ok=True)
    if do_progressive_summaries:
        os.makedirs(out_dir_summaries, exist_ok=True)

    datasets = ["Anthropic", "TruthfulQA", "Amazon BOLD"]

    dataset_short_names = {
        "Anthropic": "ANT",
        "TruthfulQA": "TQA",
        "Amazon BOLD": "BOLD",
    }

    # Collect union of interventions across all runs
    all_interventions = sorted({
        interv
        for mapping in paths_list
        for interv in mapping.keys()
    })

    # Discriminator keys & matching to model-id substrings for token accounting
    disc_keys = [
        ("qwen", "qwen"),           # main discriminator (qwen/qwen3-next-80b-a3b-instruct)
        ("gemini", "gemini"),       # alternate discriminator (gemini_2.5-flash-lite)
        ("gpt-5-nano", "gpt-5"),    # alternate discriminator (openai/gpt-5-nano)
    ]

    # ---------------- helpers ----------------
    def safe_mean(arr):
        arr = np.asarray(list(arr), dtype=float)
        finite = np.isfinite(arr)
        if finite.sum() == 0:
            return np.nan
        return float(arr[finite].mean())

    def safe_std(arr):
        arr = np.asarray(list(arr), dtype=float)
        finite = np.isfinite(arr)
        if finite.sum() == 0:
            return np.nan
        return float(arr[finite].std(ddof=0))

    def fmt_num(x, sig=3):
        """Format number with sig figs, removing leading 0 for values in (-1, 1)."""
        if not np.isfinite(x):
            return ""
        if x == 0:
            return "0"
        magnitude = int(np.floor(np.log10(abs(x))))
        rounded = round(x, sig - 1 - magnitude)
        # Determine decimal places needed
        if abs(rounded) >= 1:
            decimal_places = max(0, sig - 1 - magnitude)
            s = f"{rounded:.{decimal_places}f}"
        else:
            # For values < 1, use fixed decimal places based on sig figs
            decimal_places = sig - 1 - magnitude
            s = f"{rounded:.{decimal_places}f}"
        # Remove leading zero for values in (-1, 1)
        if s.startswith("0."):
            s = s[1:]
        elif s.startswith("-0."):
            s = "-" + s[2:]
        return s

    def format_with_ci(mean_val, std_val, sig=3):
        """
        Format value as 'mean+/-std'.
        Values < 1 are displayed as '.xyz' instead of '0.xyz'.
        """
        if not np.isfinite(mean_val):
            return ""
        if not np.isfinite(std_val) or std_val == 0:
            return fmt_num(mean_val, sig)
        
        return f"{fmt_num(mean_val, sig)}+/-{fmt_num(std_val, sig)}"

    def apply_ci_format_to_df(df, mean_df, std_df, numeric_cols, sig=3):
        """
        Apply CI formatting to a DataFrame, replacing numeric columns with CI-formatted strings.
        """
        result = df.copy()
        for col in numeric_cols:
            if col in mean_df.columns and col in std_df.columns:
                formatted = []
                for idx in result.index:
                    m = mean_df.loc[idx, col] if idx in mean_df.index else np.nan
                    s = std_df.loc[idx, col] if idx in std_df.index else np.nan
                    formatted.append(format_with_ci(m, s, sig))
                result[col] = formatted
        return result

    def slug(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

    def bh_correction_safe(p_values, alpha_):
        """Wrapper around benjamini_hochberg_correction that tolerates NaNs / empties."""
        arr = np.asarray(p_values, dtype=float)
        finite = np.isfinite(arr)
        if finite.sum() == 0:
            return 0, np.nan, np.zeros_like(arr, dtype=bool)
        n_sig, thresh, mask_valid = benjamini_hochberg_correction(list(arr[finite]), alpha=alpha_)
        mask = np.zeros_like(arr, dtype=bool)
        mask[finite] = mask_valid
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

    def _is_alternate_discriminator(model_id: str) -> bool:
        """Check if the model is an alternate discriminator (gemini or gpt-5-nano)."""
        mid = (model_id or "").lower()
        if "gemini" in mid:
            return True
        if "gpt-5" in mid or "gpt5" in mid or "nano" in mid:
            return True
        return False

    def group_tokens_by_kind(h, kind=None):
        """
        Sum tokens from API logs in `h` filtered by query_types == kind if provided.
        Returns (input_sum, output_sum), and also by model when needed elsewhere.
        Note: For kind="discriminator", alternate discriminators (gemini, gpt-5-nano) are excluded.
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
                # Skip alternate discriminators for discriminator token counts
                if kind == "discriminator" and _is_alternate_discriminator(m):
                    continue
                total_in += int(i)
                total_out += int(o)
                by_model[m]["in"] += int(i)
                by_model[m]["out"] += int(o)
        return total_in, total_out, by_model

    def compute_val_metrics_from_lists(y_lists, s_lists):
        """
        Helper: compute validation metrics for a (labels_list, scores_list) collection.
        Returns: (aucs, accuracies, pvals_auc).
        """
        aucs, accs, pvals_auc = [], [], []
        L = min(len(y_lists or []), len(s_lists or []))
        for i in range(L):
            y_true = y_lists[i]
            s = s_lists[i]
            if not y_true or not s or len(y_true) != len(s):
                continue
            a = auc_from_scores(y_true, s)
            acc, _ = best_threshold_accuracy(y_true, s)
            p_auc = mannwhitney_u_test_auc(s, y_true)
            if np.isfinite(a):
                aucs.append(a)
            if np.isfinite(acc):
                accs.append(acc)
            if np.isfinite(p_auc):
                pvals_auc.append(p_auc)
        return aucs, accs, pvals_auc

    def compute_cv_aucs_from_lists(y_lists, s_lists):
        """Helper: compute CV AUCs from (labels_list, scores_list) collection."""
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

    def which_disc_bucket(model_id: str) -> Optional[str]:
        mid = (model_id or "").lower()
        if "gemini" in mid:
            return "gemini"
        if "qwen" in mid:
            return "qwen"
        if "gpt-5" in mid or "gpt5" in mid or "nano" in mid:
            return "gpt-5-nano"
        return None

    # ---------------- parse all logs across runs ----------------
    parsed_runs: List[Dict[str, Dict[str, dict]]] = []
    for run_idx, paths_for_run in enumerate(paths_list):
        run_parsed: Dict[str, Dict[str, dict]] = {}
        for interv, ds_map in paths_for_run.items():
            run_parsed.setdefault(interv, {})
            for ds, path in ds_map.items():
                print(f"[Run {run_idx}] Processing {interv} on {ds}")
                if not path or not os.path.exists(path):
                    print(f"[WARN] Missing log file for run={run_idx}, {interv} / {ds}: {path}")
                    continue
                run_parsed[interv][ds] = extract_hypotheses_and_scores(path)
        parsed_runs.append(run_parsed)

    # ---------------- S1–S3: per-run dataset-level stats & token usage ----------------
    dataset_stats_records = []   # one row per (run, intervention, dataset) with data
    token_usage_records = []     # one row per (run, intervention, dataset)
    all_hypothesis_records = []  # one row per hypothesis per run

    for run_idx, (paths_for_run, parsed) in enumerate(zip(paths_list, parsed_runs)):
        # For progressive summaries we keep per-run, per-intervention unions.
        if do_progressive_summaries and summary_opts:
            union_validated_hyps_by_interv: Dict[str, List[str]] = defaultdict(list)
            seen_norm_by_interv: Dict[str, set] = defaultdict(set)

        for interv, ds_map in paths_for_run.items():
            for ds, path in ds_map.items():
                data = parsed.get(interv, {}).get(ds)
                if not data:
                    continue

                # Align validation arrays (no baselines)
                L = min(
                    len(data.get("hypotheses", [])),
                    len(data.get("accuracies", [])),
                    len(data.get("auc_scores", [])),
                    len(data.get("permutation_p_values", [])),
                )
                if L == 0:
                    continue

                val_auc = np.asarray(data["auc_scores"][:L], dtype=float)
                val_acc = np.asarray(data["accuracies"][:L], dtype=float)
                perm_p = np.asarray(data["permutation_p_values"][:L], dtype=float)

                # BH on validation permutation p-values
                n_sig, bh_thresh, sig_mask = bh_correction_safe(perm_p, alpha)
                min_sig_auc = float(np.nanmin(val_auc[sig_mask])) if n_sig > 0 else np.nan

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
                        p_auc = mannwhitney_u_test_auc(s, y_true)
                        if np.isfinite(p_auc):
                            pvals_from_auc.append(p_auc)

                # Cross-val AUCs (comparison only)
                cv_aucs = np.asarray(data.get("cross_validated_auc_scores", []) or [], dtype=float)

                # Baseline AUCs
                val_baseline_auc = np.asarray(data.get("baseline_auc_scores", []) or [], dtype=float)
                if len(val_baseline_auc) > L:
                    val_baseline_auc = val_baseline_auc[:L]

                cv_baseline_auc = np.asarray(data.get("cross_validated_baseline_auc_scores", []) or [], dtype=float)

                # Record dataset-level stats for this run
                dataset_stats_records.append({
                    "experiment_index": run_idx,
                    "intervention": interv,
                    "dataset": ds,
                    "n_hypotheses": int(L),
                    "n_significant": int(n_sig),
                    "mean_val_auc": safe_mean(val_auc),
                    "mean_cv_auc": safe_mean(cv_aucs),
                    "mean_val_baseline_auc": safe_mean(val_baseline_auc),
                    "mean_cv_baseline_auc": safe_mean(cv_baseline_auc),
                    "mean_val_auc_pval": safe_mean(pvals_from_auc),
                    "mean_accuracy": safe_mean(val_acc),
                    "bh_threshold": float(bh_thresh) if np.isfinite(bh_thresh) else np.nan,
                    "min_sig_auc": min_sig_auc,
                })

                # Token usage by query type
                d_in, d_out, _ = group_tokens_by_kind(data, kind="discriminator")
                l_in, l_out, _ = group_tokens_by_kind(data, kind="labeler")
                s_in, s_out, _ = group_tokens_by_kind(data, kind="summarizer")

                # For summarizer, we normalize by # of validated hypotheses / 10 (as in original),
                # but only if we have enough significant hypotheses.
                if n_sig > 10:
                    norm = max(1, int(n_sig / 10))
                    s_in_norm = s_in / norm
                    s_out_norm = s_out / norm
                else:
                    s_in_norm = 0.0
                    s_out_norm = 0.0

                token_usage_records.append({
                    "experiment_index": run_idx,
                    "intervention": interv,
                    "dataset": ds,
                    "disc_input_tokens": d_in / float(L),
                    "disc_output_tokens": d_out / float(L),
                    "labeler_input_tokens": l_in / float(L),
                    "labeler_output_tokens": l_out / float(L),
                    "summarizer_input_tokens": s_in_norm,
                    "summarizer_output_tokens": s_out_norm,
                })

                # Collect individual hypothesis stats
                for i in range(L):
                    cv_val = float(cv_aucs[i]) if i < len(cv_aucs) else np.nan
                    all_hypothesis_records.append({
                        "experiment_index": run_idx,
                        "dataset": ds,
                        "intervention": interv,
                        "hypothesis_id": i,
                        "hypothesis_text": data.get("hypotheses", [None for _ in range(L)])[i],
                        "in_category_auc": float(val_auc[i]),
                        "cross_category_auc": cv_val,
                        "p_value": float(perm_p[i]),
                        "validated": bool(sig_mask[i]),
                    })

                # (Optional) progressive summaries (per run, per intervention)
                if do_progressive_summaries and summary_opts:
                    selected_hypotheses = select_hypotheses_for_summary(
                        data=data,
                        filter_type=summary_opts.get("filter_type", "permutation"),
                        bh_alpha=summary_opts.get("bh_alpha", alpha),
                        min_accuracy=summary_opts.get("min_accuracy", None),
                        top_k=summary_opts.get("top_k", None),
                    )
                    for key in selected_hypotheses:
                        norm_key = key.strip()
                        if norm_key not in seen_norm_by_interv[interv]:
                            seen_norm_by_interv[interv].add(norm_key)
                            try:
                                # Match hypothesis to its index (ID) in the original list
                                h_id = data.get("hypotheses", []).index(key)
                            except ValueError:
                                h_id = "?"
                            ds_short = dataset_short_names.get(ds, ds)
                            union_validated_hyps_by_interv[interv].append(f"({ds_short}, {h_id}): {norm_key}")

            # After we've scanned all datasets for this intervention in this run,
            # run ONE progressive summary on the UNION of validated hypotheses.
            if do_progressive_summaries and summary_opts:
                union_validated_hyps = union_validated_hyps_by_interv.get(interv, [])
                if len(union_validated_hyps) > 0:
                    top_k = summary_opts.get("top_k", None)
                    hyps_for_summary = union_validated_hyps[:top_k] if (top_k is not None) else union_validated_hyps
                    print(f"[Run {run_idx} | Progressive summary] {interv}: {len(hyps_for_summary)} hypotheses")
                    save_dir = os.path.join(out_dir_summaries, f"run_{run_idx}", slug(interv))
                    os.makedirs(save_dir, exist_ok=True)
                    run_progressive_summary(
                        name=f"{interv}__RUN_{run_idx}__ALL_DATASETS_UNION",
                        hypotheses=hyps_for_summary,
                        api_provider=summary_opts["api_provider"],
                        model_str=summary_opts["model_str"],
                        api_key_path=summary_opts.get("api_key_path"),
                        save_dir=save_dir,
                        max_tokens=summary_opts.get("max_tokens", 2000),
                        temperature=summary_opts.get("temperature", 0.7),
                        max_thinking_tokens=summary_opts.get("max_thinking_tokens"),
                        verbose=summary_opts.get("verbose", False),
                    )

    # ---------------- Aggregate S1–S3 across runs ----------------
    per_intervention_stats: Dict[str, pd.DataFrame] = {}
    per_intervention_tokens: Dict[str, pd.DataFrame] = {}

    if len(dataset_stats_records) > 0:
        ds_stats_df_raw = pd.DataFrame(dataset_stats_records)
        group = ds_stats_df_raw.groupby(["intervention", "dataset"])

        numeric_cols = [
            "n_hypotheses", "n_significant",
            "mean_val_auc", "mean_cv_auc",
            "mean_val_baseline_auc", "mean_cv_baseline_auc",
            "mean_val_auc_pval", "mean_accuracy",
            "bh_threshold", "min_sig_auc",
        ]
        mean_df = group[numeric_cols].mean()
        std_df = group[numeric_cols].std(ddof=0)
        count_series = group.size().rename("n_experiments")

        # Apply CI formatting to numeric columns
        agg_ds_df = mean_df.copy()
        agg_ds_df = agg_ds_df.rename(columns={"min_sig_auc": "mean_min_sig_auc"})
        std_df_renamed = std_df.rename(columns={"min_sig_auc": "mean_min_sig_auc"})
        numeric_cols_renamed = [c if c != "min_sig_auc" else "mean_min_sig_auc" for c in numeric_cols]
        agg_ds_df = apply_ci_format_to_df(agg_ds_df, agg_ds_df, std_df_renamed, numeric_cols_renamed, sig=3)
        agg_ds_df["n_experiments"] = count_series.values
        agg_ds_df = agg_ds_df.reset_index()

        # Make sure every intervention has an entry for every dataset (even if no experiments)
        for interv in all_interventions:
            df_i = agg_ds_df[agg_ds_df["intervention"] == interv].copy()
            if not df_i.empty:
                df_i = df_i.set_index("dataset").reindex(datasets).reset_index()
                df_i["intervention"] = interv
                df_i["n_experiments"] = df_i["n_experiments"].fillna(0).astype(int)
            else:
                # Completely missing intervention (no data anywhere)
                df_i = pd.DataFrame({"dataset": datasets})
                for col in numeric_cols_renamed:
                    df_i[col] = ""
                df_i["n_experiments"] = 0
                df_i.insert(0, "intervention", interv)

            per_intervention_stats[interv] = df_i

            out_path = os.path.join(
                out_dir_tables,
                f"{slug(interv)}_dataset_stats__n{n_experiments}.tsv",
            )
            df_i.to_csv(out_path, sep="\t", index=False)
            print(f"[Saved] {out_path}")
    else:
        # No stats at all; create empty placeholders (not saved)
        numeric_cols_renamed = [
            "n_hypotheses", "n_significant",
            "mean_val_auc", "mean_cv_auc",
            "mean_val_baseline_auc", "mean_cv_baseline_auc",
            "mean_val_auc_pval", "mean_accuracy",
            "bh_threshold", "mean_min_sig_auc",
        ]
        for interv in all_interventions:
            df_i = pd.DataFrame({
                "intervention": [interv for _ in datasets],
                "dataset": datasets,
                "n_experiments": [0] * len(datasets),
            })
            for col in numeric_cols_renamed:
                df_i[col] = ""
            per_intervention_stats[interv] = df_i

    if len(token_usage_records) > 0:
        tok_df_raw = pd.DataFrame(token_usage_records)
        group = tok_df_raw.groupby(["intervention", "dataset"])

        tok_cols = [
            "disc_input_tokens", "disc_output_tokens",
            "labeler_input_tokens", "labeler_output_tokens",
            "summarizer_input_tokens", "summarizer_output_tokens",
        ]
        tok_mean = group[tok_cols].mean()
        tok_std = group[tok_cols].std(ddof=0)
        tok_count = group.size().rename("n_experiments")

        # Apply CI formatting to numeric columns
        agg_tok_df = apply_ci_format_to_df(tok_mean.copy(), tok_mean, tok_std, tok_cols, sig=3)
        agg_tok_df["n_experiments"] = tok_count.values
        agg_tok_df = agg_tok_df.reset_index()

        for interv in all_interventions:
            df_i = agg_tok_df[agg_tok_df["intervention"] == interv].copy()
            if not df_i.empty:
                df_i = df_i.set_index("dataset").reindex(datasets).reset_index()
                df_i["intervention"] = interv
                df_i["n_experiments"] = df_i["n_experiments"].fillna(0).astype(int)
            else:
                df_i = pd.DataFrame({"dataset": datasets})
                for col in tok_cols:
                    df_i[col] = ""
                df_i["n_experiments"] = 0
                df_i.insert(0, "intervention", interv)

            per_intervention_tokens[interv] = df_i

            out_path = os.path.join(
                out_dir_tables,
                f"{slug(interv)}_token_usage__n{n_experiments}.tsv",
            )
            df_i.to_csv(out_path, sep="\t", index=False)
            print(f"[Saved] {out_path}")
    else:
        for interv in all_interventions:
            df_i = pd.DataFrame({
                "intervention": [interv for _ in datasets],
                "dataset": datasets,
                "n_experiments": [0] * len(datasets),
            })
            for col in ["disc_input_tokens", "disc_output_tokens",
                       "labeler_input_tokens", "labeler_output_tokens",
                       "summarizer_input_tokens", "summarizer_output_tokens"]:
                df_i[col] = ""
            per_intervention_tokens[interv] = df_i

    # ---------------- S4: cross‑intervention histograms + AUC vs CV table ----------------
    hist_path = None

    all_hyp_df = pd.DataFrame(
        all_hypothesis_records,
        columns=[
            "experiment_index", "dataset", "intervention", "hypothesis_id",
            "hypothesis_text", "in_category_auc", "cross_category_auc",
            "p_value", "validated",
        ],
    ) if len(all_hypothesis_records) > 0 else pd.DataFrame(
        columns=[
            "experiment_index", "dataset", "intervention", "hypothesis_id",
            "hypothesis_text", "in_category_auc", "cross_category_auc",
            "p_value", "validated",
        ],
    )

    if make_plots and not all_hyp_df.empty:
        interv_to_name_map = {
            "WIHP": "Unlearning",
            "ROME-10": "Knowledge Editing",
            "R1 distillation": "Reasoning Distillation",
        }
        ds_to_name_map = {
            "Anthropic": "Anthropic Evals",
            "TruthfulQA": "TruthfulQA",
            "Amazon BOLD": "Amazon BOLD",
        }
        # Dataset colors matching the reference image
        ds_colors = {
            "Anthropic": "#4a9a9a",      # teal/cyan
            "Amazon BOLD": "#7b68b0",    # purple
            "TruthfulQA": "#d4a43a",     # orange/gold
        }

        # Precompute AUC lists and min significant AUCs by intervention/dataset
        aucs_by_intervention_by_dataset: Dict[str, Dict[str, np.ndarray]] = {
            interv: {} for interv in all_interventions
        }
        min_sig_auc_by_intervention_by_dataset: Dict[str, Dict[str, float]] = {
            interv: {} for interv in all_interventions
        }

        for interv in all_interventions:
            df_interv = all_hyp_df[all_hyp_df["intervention"] == interv]
            for ds in datasets:
                df_ids = df_interv[df_interv["dataset"] == ds]
                if df_ids.empty:
                    continue
                vals = df_ids["in_category_auc"].astype(float).to_numpy()
                aucs_by_intervention_by_dataset[interv][ds] = vals

                sig_vals = df_ids.loc[df_ids["validated"].astype(bool), "in_category_auc"].astype(float)
                min_sig = float(sig_vals.min()) if len(sig_vals) > 0 else np.nan
                min_sig_auc_by_intervention_by_dataset[interv][ds] = min_sig

        fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True, sharey=True)
        legend_handles = {}

        for i, (ax, interv) in enumerate(zip(axes, all_interventions)):
            any_data = False
            for ds in datasets:
                vals = aucs_by_intervention_by_dataset.get(interv, {}).get(ds, None)
                if vals is None or len(vals) == 0:
                    continue
                any_data = True
                color = ds_colors.get(ds, "#333333")
                n, bins, patches = ax.hist(
                    vals,
                    bins=np.linspace(0.0, 1.0, 21),
                    alpha=0.5,
                    color=color,
                    label=ds_to_name_map.get(ds, ds),
                )

                if ds not in legend_handles and len(patches) > 0:
                    legend_handles[ds] = patches[0]

            ax.set_title(interv_to_name_map.get(interv, interv))
            ax.set_ylabel("Count")
            if i == len(all_interventions) - 1:
                ax.set_xlabel("Validation AUC")
            if not any_data:
                ax.text(
                    0.5, 0.5, "No data",
                    transform=ax.transAxes,
                    ha="center", va="center", fontsize=10,
                )

        if legend_handles:
            sorted_handles = []
            sorted_labels = []
            for ds in datasets:
                if ds in legend_handles:
                    sorted_handles.append(legend_handles[ds])
                    sorted_labels.append(ds_to_name_map.get(ds, ds))
            target_ax = axes[1] if len(axes) > 1 else axes[0]
            target_ax.legend(sorted_handles, sorted_labels, loc="best", title="Dataset")

        fig.suptitle("Validation AUC distributions by intervention (aggregated over runs)")
        fig.tight_layout()
        hist_path = os.path.join(out_dir_figs, f"validation_auc_histograms_by_intervention__n{n_experiments}.pdf")
        plt.savefig(hist_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {hist_path}")

        print("\n[Figure Caption Data] Minimum Significant AUCs (aggregated):")
        for interv in all_interventions:
            print(f"  {interv}:")
            for ds in datasets:
                ms = min_sig_auc_by_intervention_by_dataset.get(interv, {}).get(ds, np.nan)
                val_str = f"{ms:.3f}" if np.isfinite(ms) else "None"
                print(f"    {ds}: {val_str}")

    # AUC vs cross-val AUC table (aggregated over runs via hypothesis-level stats)
    auc_vs_cv_rows: List[Dict] = []
    if not all_hyp_df.empty:
        for interv in all_interventions:
            for ds in datasets:
                sub = all_hyp_df[(all_hyp_df["intervention"] == interv) & (all_hyp_df["dataset"] == ds)]
                if sub.empty:
                    continue
                val_aucs = sub["in_category_auc"].astype(float).to_numpy()
                cv_aucs = sub["cross_category_auc"].astype(float).to_numpy()
                auc_vs_cv_rows.append({
                    "intervention": interv,
                    "dataset": ds,
                    "val_auc": format_with_ci(safe_mean(val_aucs), safe_std(val_aucs), sig=3),
                    "cv_auc": format_with_ci(safe_mean(cv_aucs), safe_std(cv_aucs), sig=3),
                    "n_hypotheses": int(len(sub)),
                    "n_experiments": int(sub["experiment_index"].nunique()),
                })

    auc_vs_cv_df = pd.DataFrame(
        auc_vs_cv_rows,
        columns=[
            "intervention", "dataset",
            "val_auc",
            "cv_auc",
            "n_hypotheses", "n_experiments",
        ],
    )
    auc_vs_cv_path = os.path.join(out_dir_tables, f"validation_vs_crossval_auc__n{n_experiments}.tsv")
    auc_vs_cv_df.to_csv(auc_vs_cv_path, sep="\t", index=False)
    print(f"[Saved] {auc_vs_cv_path}")

    # ---------------- S5: alternate Discriminator analysis ----------------

    # --- (a) Detailed table across runs: discriminator × intervention × dataset ---
    alt_detailed_records: List[Dict] = []

    for run_idx, parsed in enumerate(parsed_runs):
        for disc_key, _needle in disc_keys:
            for interv in all_interventions:
                for ds in datasets:
                    data = parsed.get(interv, {}).get(ds)
                    if not data:
                        continue

                    # Validation metrics by discriminator
                    if disc_key == "qwen":
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
                        pvals_auc = list(np.asarray(data["permutation_p_values"][:L], dtype=float))
                        cv_aucs = list(np.asarray(data.get("cross_validated_auc_scores", []) or [], dtype=float))
                    else:
                        # alternate discriminator: compute from stored lists
                        y_lists = data.get("discriminative_model_validation_true_labels", []) or []
                        s_alt_lists = (data.get("discriminative_model_validation_discrim_alternate_scores", {}) or {}).get(disc_key, []) or []
                        val_auc_list, val_acc_list, pvals_auc = compute_val_metrics_from_lists(
                            y_lists, s_alt_lists
                        )
                        if len(val_auc_list) == 0 or len(pvals_auc) == 0:
                            # This run did not record alternate discriminator validation scores.
                            continue
                        # Cross-val AUCs from cross-validation lists if present
                        y_cv = data.get("discriminative_model_cross_validation_true_labels", []) or []
                        s_cv_alt = (data.get("discriminative_model_cross_validation_discrim_alternate_scores", {}) or {}).get(disc_key, []) or []
                        cv_aucs = compute_cv_aucs_from_lists(y_cv, s_cv_alt)

                    # BH significance based on VALIDATION permutation p-values for this discriminator
                    n_sig, bh_thresh, sig_mask = bh_correction_safe(pvals_auc, alpha)

                    # Calculate statistics for this dataset in this run
                    mean_val_auc = safe_mean(val_auc_list)
                    mean_cv_auc = safe_mean(cv_aucs)
                    mean_val_auc_pval = safe_mean(pvals_auc)
                    mean_acc = safe_mean(val_acc_list)
                    val_arr = np.asarray(val_auc_list, dtype=float)
                    sig_mask_arr = np.asarray(sig_mask, dtype=bool)
                    min_sig_auc = float(np.nanmin(val_arr[sig_mask_arr])) if n_sig > 0 else np.nan

                    alt_detailed_records.append({
                        "experiment_index": run_idx,
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

    if len(alt_detailed_records) > 0:
        alt_detailed_df_raw = pd.DataFrame(alt_detailed_records)

        # Aggregate detailed stats over runs
        detailed_group = alt_detailed_df_raw.groupby(
            ["discriminator", "intervention", "dataset"], as_index=False
        )
        det_cols = [
            "n_significant", "mean_val_auc", "mean_cv_auc",
            "mean_val_auc_pval", "mean_accuracy", "min_sig_auc",
        ]
        det_mean = detailed_group[det_cols].mean()
        det_std = detailed_group[det_cols].std(ddof=0)
        det_count = detailed_group.size().rename(columns={"size": "n_experiments"})

        alt_detailed_df = det_mean.copy()
        alt_detailed_df = alt_detailed_df.rename(columns={"min_sig_auc": "mean_min_sig_auc"})
        det_std_renamed = det_std.rename(columns={"min_sig_auc": "mean_min_sig_auc"})
        det_cols_renamed = [c if c != "min_sig_auc" else "mean_min_sig_auc" for c in det_cols]
        
        # Apply CI formatting
        for col in det_cols_renamed:
            formatted = []
            for idx in alt_detailed_df.index:
                m = alt_detailed_df.loc[idx, col] if col in alt_detailed_df.columns else np.nan
                s = det_std_renamed.loc[idx, col] if col in det_std_renamed.columns else np.nan
                formatted.append(format_with_ci(m, s, sig=3))
            alt_detailed_df[col] = formatted
        
        alt_detailed_df = alt_detailed_df.merge(
            det_count,
            on=["discriminator", "intervention", "dataset"],
            how="left",
            validate="one_to_one",
        )
    else:
        alt_detailed_df = pd.DataFrame(
            columns=[
                "discriminator", "intervention", "dataset",
                "n_significant",
                "mean_val_auc",
                "mean_cv_auc",
                "mean_val_auc_pval",
                "mean_accuracy",
                "mean_min_sig_auc",
                "n_experiments",
            ]
        )

    alt_detailed_path = os.path.join(
        out_dir_tables,
        f"alternate_discriminators__detailed_stats_by_dataset__n{n_experiments}.tsv",
    )
    alt_detailed_df.to_csv(alt_detailed_path, sep="\t", index=False)
    print(f"[Saved] {alt_detailed_path}")

    # --- (a.2) Summary table averaged over datasets AND runs: discriminator × intervention ---
    if len(alt_detailed_records) > 0:
        summary_group = alt_detailed_df_raw.groupby(["discriminator", "intervention"], as_index=False)
        det_cols = [
            "n_significant", "mean_val_auc", "mean_cv_auc",
            "mean_val_auc_pval", "mean_accuracy", "min_sig_auc",
        ]
        sum_mean = summary_group[det_cols].mean()
        sum_std = summary_group[det_cols].std(ddof=0)
        sum_count = (
            alt_detailed_df_raw
            .groupby(["discriminator", "intervention"])["experiment_index"]
            .nunique()
            .rename("n_experiments")
            .reset_index()
        )

        alt_summary_df = sum_mean.rename(columns={
            "n_significant": "avg_n_significant",
            "min_sig_auc": "mean_min_sig_auc",
        })
        sum_std_renamed = sum_std.rename(columns={
            "n_significant": "avg_n_significant",
            "min_sig_auc": "mean_min_sig_auc",
        })

        # Apply CI formatting to numeric columns
        summary_numeric_cols = [
            "avg_n_significant", "mean_val_auc", "mean_cv_auc",
            "mean_val_auc_pval", "mean_accuracy", "mean_min_sig_auc",
        ]
        for col in summary_numeric_cols:
            formatted = []
            for idx in alt_summary_df.index:
                m = alt_summary_df.loc[idx, col] if col in alt_summary_df.columns else np.nan
                s = sum_std_renamed.loc[idx, col] if col in sum_std_renamed.columns else np.nan
                formatted.append(format_with_ci(m, s, sig=3))
            alt_summary_df[col] = formatted

        alt_summary_df = alt_summary_df.merge(
            sum_count,
            on=["discriminator", "intervention"],
            how="left",
            validate="one_to_one",
        )

        summary_cols_order = [
            "discriminator", "intervention", "n_experiments",
            "avg_n_significant",
            "mean_val_auc",
            "mean_cv_auc",
            "mean_val_auc_pval",
            "mean_accuracy",
            "mean_min_sig_auc",
        ]
        alt_summary_df = alt_summary_df[summary_cols_order]
    else:
        alt_summary_df = pd.DataFrame(
            columns=[
                "discriminator", "intervention", "n_experiments",
                "avg_n_significant",
                "mean_val_auc",
                "mean_cv_auc",
                "mean_val_auc_pval",
                "mean_accuracy",
                "mean_min_sig_auc",
            ]
        )

    alt_summary_path = os.path.join(
        out_dir_tables,
        f"alternate_discriminators__summary_by_discriminator_and_intervention__n{n_experiments}.tsv",
    )
    alt_summary_df.to_csv(alt_summary_path, sep="\t", index=False)
    print(f"[Saved] {alt_summary_path}")

    # --- (b) Token usage table by intervention × dataset for Discriminator models only ---
    disc_token_records: List[Dict] = []

    for run_idx, parsed in enumerate(parsed_runs):
        for interv in all_interventions:
            for ds in datasets:
                data = parsed.get(interv, {}).get(ds)
                if not data:
                    continue
                L = min(
                    len(data.get("hypotheses", [])),
                    len(data.get("auc_scores", [])),
                    len(data.get("accuracies", [])),
                    len(data.get("permutation_p_values", [])),
                )
                if L == 0:
                    continue
                # Only discriminator queries
                _, _, by_model = group_tokens_by_kind(data, kind="discriminator")
                buckets = {"qwen": {"in": 0, "out": 0},
                           "gemini": {"in": 0, "out": 0},
                           "gpt-5-nano": {"in": 0, "out": 0}}
                for mid, tok in by_model.items():
                    b = which_disc_bucket(mid)
                    if b in buckets:
                        buckets[b]["in"] += int(tok["in"])
                        buckets[b]["out"] += int(tok["out"])
                disc_token_records.append({
                    "experiment_index": run_idx,
                    "intervention": interv,
                    "dataset": ds,
                    "qwen_in": buckets["qwen"]["in"] / float(L),
                    "qwen_out": buckets["qwen"]["out"] / float(L),
                    "gemini_in": buckets["gemini"]["in"] / float(L),
                    "gemini_out": buckets["gemini"]["out"] / float(L),
                    "gpt5nano_in": buckets["gpt-5-nano"]["in"] / float(L),
                    "gpt5nano_out": buckets["gpt-5-nano"]["out"] / float(L),
                })

    if len(disc_token_records) > 0:
        disc_tokens_df_raw = pd.DataFrame(disc_token_records)
        disc_group = disc_tokens_df_raw.groupby(["intervention", "dataset"], as_index=False)
        disc_cols = [
            "qwen_in", "qwen_out",
            "gemini_in", "gemini_out",
            "gpt5nano_in", "gpt5nano_out",
        ]
        disc_mean = disc_group[disc_cols].mean()
        disc_std = disc_group[disc_cols].std(ddof=0)
        disc_count = (
            disc_group.size()
            .rename(columns={"size": "n_experiments"})
        )

        # Apply CI formatting to token columns
        disc_tokens_df = disc_mean.copy()
        for col in disc_cols:
            formatted = []
            for idx in disc_tokens_df.index:
                m = disc_tokens_df.loc[idx, col] if col in disc_tokens_df.columns else np.nan
                s = disc_std.loc[idx, col] if col in disc_std.columns else np.nan
                formatted.append(format_with_ci(m, s, sig=3))
            disc_tokens_df[col] = formatted
        disc_tokens_df = disc_tokens_df.merge(
            disc_count,
            on=["intervention", "dataset"],
            how="left",
            validate="one_to_one",
        )
    else:
        disc_tokens_df = pd.DataFrame(
            columns=[
                "intervention", "dataset",
                "qwen_in",
                "qwen_out",
                "gemini_in",
                "gemini_out",
                "gpt5nano_in",
                "gpt5nano_out",
                "n_experiments",
            ]
        )

    disc_tokens_path = os.path.join(
        out_dir_tables,
        f"alternate_discriminators__discriminator_token_usage_by_model__n{n_experiments}.tsv",
    )
    disc_tokens_df.to_csv(disc_tokens_path, sep="\t", index=False)
    print(f"[Saved] {disc_tokens_path}")

    # ---------------- S6: Pairwise Discriminator Comparison (aggregated over runs) ----------------
    comparison_records: List[Dict] = []
    disc_keys_list = ["qwen", "gemini", "gpt-5-nano"]

    for run_idx, parsed in enumerate(parsed_runs):
        for interv in all_interventions:
            for ds in datasets:
                data = parsed.get(interv, {}).get(ds)
                if not data:
                    continue

                y_lists = data.get("discriminative_model_validation_true_labels", []) or []
                L = len(y_lists)
                if L == 0:
                    continue

                # Pre-compute AUCs and scores for all models on this dataset
                model_data = {}

                for m in disc_keys_list:
                    if m == "qwen":
                        s_lists = data.get("discriminative_model_validation_discrim_scores", []) or []
                    else:
                        s_lists = (data.get("discriminative_model_validation_discrim_alternate_scores", {}) or {}).get(m, []) or []

                    current_aucs = []
                    current_scores = []
                    flat_scores = []
                    flat_labels = []

                    for i in range(L):
                        y_true = y_lists[i]
                        s_vals = s_lists[i] if i < len(s_lists) else []

                        if not y_true or not s_vals or len(y_true) != len(s_vals):
                            current_aucs.append(np.nan)
                            current_scores.append(None)
                            continue

                        a = auc_from_scores(y_true, s_vals)
                        current_aucs.append(a)
                        current_scores.append(s_vals)

                        flat_scores.extend(s_vals)
                        flat_labels.extend(y_true)

                    if flat_scores:
                        fs = np.array(flat_scores, dtype=float)
                        fl = np.array(flat_labels, dtype=float)
                        brier = float(np.mean((fs - fl) ** 2))
                    else:
                        brier = np.nan

                    model_data[m] = {
                        "aucs": np.array(current_aucs, dtype=float),
                        "scores_list": current_scores,
                        "brier": brier,
                    }

                # Pairwise comparisons
                pairs_to_compare = [
                    ("qwen", "gemini"),
                    ("qwen", "gpt-5-nano"),
                    ("gemini", "gpt-5-nano"),
                ]

                for (m_a, m_b) in pairs_to_compare:
                    dat_a = model_data[m_a]
                    dat_b = model_data[m_b]

                    mask = np.isfinite(dat_a["aucs"]) & np.isfinite(dat_b["aucs"])
                    valid_indices = np.where(mask)[0]
                    n_valid = len(valid_indices)

                    if n_valid < 5:
                        comparison_records.append({
                            "experiment_index": run_idx,
                            "discriminator_pair": f"{m_a}_vs_{m_b}",
                            "intervention": interv,
                            "dataset": ds,
                            "n_valid_pairs": n_valid,
                            "wilcoxon_p_val_greater": np.nan,
                            "spearman_auc_corr": np.nan,
                            "mean_pearson_score_corr": np.nan,
                            "jaccard_top_20_percent": np.nan,
                            "brier_score_A": dat_a["brier"],
                            "brier_score_B": dat_b["brier"],
                        })
                        continue

                    aucs_a = dat_a["aucs"][valid_indices]
                    aucs_b = dat_b["aucs"][valid_indices]

                    # 1. Wilcoxon (A > B)
                    diffs = aucs_a - aucs_b
                    if np.allclose(diffs, 0):
                        w_p = 1.0
                    else:
                        try:
                            _, w_p = wilcoxon(aucs_a, aucs_b, alternative="greater")
                        except ValueError:
                            w_p = np.nan

                    # 2. Spearman AUC
                    if np.std(aucs_a) == 0 or np.std(aucs_b) == 0:
                        spear_r = np.nan
                    else:
                        try:
                            res = spearmanr(aucs_a, aucs_b)
                            spear_r = float(res.statistic) if hasattr(res, "statistic") else float(res.correlation)
                        except Exception:
                            spear_r = np.nan

                    # 3. Mean Pearson on raw scores
                    pearson_vals = []
                    for idx in valid_indices:
                        sa = dat_a["scores_list"][idx]
                        sb = dat_b["scores_list"][idx]
                        if not sa or not sb or len(sa) < 2:
                            continue
                        if np.std(sa) == 0 or np.std(sb) == 0:
                            continue
                        try:
                            r_obj = pearsonr(sa, sb)
                            r_val = float(r_obj.statistic) if hasattr(r_obj, "statistic") else float(r_obj.correlation)
                            if np.isfinite(r_val):
                                pearson_vals.append(r_val)
                        except Exception:
                            pass

                    mean_pearson = float(np.mean(pearson_vals)) if pearson_vals else np.nan

                    # 4. Top-K Jaccard (top 20%)
                    k = max(1, int(0.2 * n_valid))
                    top_idx_a = set(np.argsort(aucs_a)[-k:])
                    top_idx_b = set(np.argsort(aucs_b)[-k:])
                    intersect = len(top_idx_a.intersection(top_idx_b))
                    union = len(top_idx_a.union(top_idx_b))
                    jaccard = intersect / union if union > 0 else 0.0

                    comparison_records.append({
                        "experiment_index": run_idx,
                        "discriminator_pair": f"{m_a}_vs_{m_b}",
                        "intervention": interv,
                        "dataset": ds,
                        "wilcoxon_p_val_greater": w_p,
                        "spearman_auc_corr": spear_r,
                        "mean_pearson_score_corr": mean_pearson,
                        "jaccard_top_20_percent": jaccard,
                        "brier_score_A": dat_a["brier"],
                        "brier_score_B": dat_b["brier"],
                        "n_valid_pairs": n_valid,
                    })

    if len(comparison_records) > 0:
        comp_df_raw = pd.DataFrame(comparison_records)
        comp_group = comp_df_raw.groupby(
            ["discriminator_pair", "intervention", "dataset"], as_index=False
        )

        comp_cols = [
            "wilcoxon_p_val_greater",
            "spearman_auc_corr",
            "mean_pearson_score_corr",
            "jaccard_top_20_percent",
            "brier_score_A",
            "brier_score_B",
            "n_valid_pairs",
        ]

        comp_mean = comp_group[comp_cols].mean()
        comp_std = comp_group[comp_cols].std(ddof=0)
        # Only count experiments that had enough valid pairs to compute stats (n_valid >= 5)
        valid_counts_df = (
            comp_df_raw.assign(valid_experiment=(comp_df_raw["n_valid_pairs"] >= 5).astype(int))
            .groupby(
                ["discriminator_pair", "intervention", "dataset"],
                as_index=False,
            )["valid_experiment"]
            .sum()
            .rename(columns={"valid_experiment": "n_experiments"})
        )

        # Apply CI formatting to comparison columns
        comp_df = comp_mean.copy()
        for col in comp_cols:
            formatted = []
            for idx in comp_df.index:
                m = comp_df.loc[idx, col] if col in comp_df.columns else np.nan
                s = comp_std.loc[idx, col] if col in comp_std.columns else np.nan
                formatted.append(format_with_ci(m, s, sig=3))
            comp_df[col] = formatted
        comp_df = comp_df.merge(
            valid_counts_df,
            on=["discriminator_pair", "intervention", "dataset"],
            how="left",
            validate="one_to_one",
        )
    else:
        comp_df = pd.DataFrame(
            columns=[
                "discriminator_pair", "intervention", "dataset",
                "wilcoxon_p_val_greater",
                "spearman_auc_corr",
                "mean_pearson_score_corr",
                "jaccard_top_20_percent",
                "brier_score_A",
                "brier_score_B",
                "n_valid_pairs",
                "n_experiments",
            ]
        )

    comp_path = os.path.join(out_dir_tables, f"discriminator_comparisons__n{n_experiments}.tsv")
    comp_df.to_csv(comp_path, sep="\t", index=False)
    print(f"[Saved] {comp_path}")

    # Save all hypotheses table (includes all runs)
    all_hyp_df = round_float_df_to_sig_figs(all_hyp_df, sig=3)
    all_hyp_path = os.path.join(out_dir_tables, f"all_hypotheses_stats__n{n_experiments}.tsv")
    all_hyp_df.to_csv(all_hyp_path, sep="\t", index=False)
    print(f"[Saved] {all_hyp_path}")

    # ---------------- S7: Variance Decomposition Analysis ----------------
    # Two-way random effects variance decomposition for each (intervention, dataset) configuration.
    # Treats hypothesis_id as crossed with run_id (assuming hypotheses align across runs).
    #
    # Model: Y_ij = μ + α_i (run effect) + β_j (hypothesis effect) + ε_ij
    # Variance components:
    #   - σ²_run: variance between runs
    #   - σ²_hypothesis: variance between hypotheses (clusters)
    #   - σ²_residual: interaction + error

    print("\n" + "=" * 80)
    print("VARIANCE DECOMPOSITION ANALYSIS (S7)")
    print("Two-way random effects: runs × hypotheses (crossed)")
    print("=" * 80)

    def two_way_variance_decomposition_crossed(df, value_col, run_col, hyp_col):
        """
        Two-way variance decomposition for crossed design (run × hypothesis).
        
        Model: Y_ij = μ + α_i (run) + β_j (hypothesis) + ε_ij
        Var(grand mean) ≈ σ²_run/n_runs + σ²_hyp/n_hyp + σ²_resid/(n_runs×n_hyp)
        
        Returns dict with variance components, grand mean statistics, and
        additional interpretability metrics, or None if insufficient data.
        """
        # Pivot to run × hypothesis matrix
        pivot = df.pivot(index=run_col, columns=hyp_col, values=value_col)
        
        # Remove columns (hypotheses) with any NaN (incomplete across runs)
        pivot_complete = pivot.dropna(axis=1)
        
        if pivot_complete.empty or pivot_complete.shape[0] < 2 or pivot_complete.shape[1] < 2:
            return None
        
        n_runs = pivot_complete.shape[0]
        n_hypotheses = pivot_complete.shape[1]
        
        grand_mean = pivot_complete.values.mean()
        
        # Row means (per run, averaging over hypotheses)
        run_means = pivot_complete.mean(axis=1)
        # Column means (per hypothesis, averaging over runs)
        hyp_means = pivot_complete.mean(axis=0)
        
        # Variance components
        var_between_runs = run_means.var(ddof=1)
        var_between_hypotheses = hyp_means.var(ddof=1)
        
        # Residual variance (interaction + error)
        residuals = (pivot_complete.values 
                     - run_means.values.reshape(-1, 1) 
                     - hyp_means.values.reshape(1, -1) 
                     + grand_mean)
        var_residual = residuals.var(ddof=1)
        
        # SE of grand mean under two-way random effects
        se = np.sqrt(
            var_between_runs / n_runs + 
            var_between_hypotheses / n_hypotheses + 
            var_residual / (n_runs * n_hypotheses)
        )
        
        # Total variance
        var_total = pivot_complete.values.var(ddof=1)
        
        # ---- Additional interpretability metrics ----
        
        # SD of run marginal means (intuitive measure of run-to-run spread)
        run_mean_sd = np.sqrt(var_between_runs)
        
        # SD of hypothesis marginal means
        hyp_mean_sd = np.sqrt(var_between_hypotheses)
        
        # Within-hypothesis SD across runs (repeatability/reproducibility)
        # For each hypothesis, how much does its AUC vary across the runs?
        within_hyp_sds = pivot_complete.std(axis=0, ddof=1)  # SD across runs for each hypothesis
        mean_within_hyp_sd = float(within_hyp_sds.mean())
        median_within_hyp_sd = float(within_hyp_sds.median())
        
        # Within-run SD across hypotheses (spread of hypotheses within each run)
        within_run_sds = pivot_complete.std(axis=1, ddof=1)
        mean_within_run_sd = float(within_run_sds.mean())
        
        # Range of run means (max difference between any two run means)
        run_mean_range = float(run_means.max() - run_means.min())
        
        # ICC for runs: fraction of total variance attributable to run-level differences
        # ICC(1) = var_run / (var_run + var_within_total)
        # where var_within_total pools hypothesis and residual variance
        var_within_total = var_between_hypotheses + var_residual
        icc_runs = var_between_runs / (var_between_runs + var_within_total) if (var_between_runs + var_within_total) > 0 else np.nan
        
        return {
            'grand_mean': grand_mean,
            'se': se,
            'ci_lo': grand_mean - 1.96 * se,
            'ci_hi': grand_mean + 1.96 * se,
            'var_run': var_between_runs,
            'var_hypothesis': var_between_hypotheses,
            'var_residual': var_residual,
            'var_total': var_total,
            'n_runs': n_runs,
            'n_hypotheses': n_hypotheses,
            'frac_var_run': var_between_runs / var_total if var_total > 0 else np.nan,
            'frac_var_hypothesis': var_between_hypotheses / var_total if var_total > 0 else np.nan,
            'frac_var_residual': var_residual / var_total if var_total > 0 else np.nan,
            # Additional interpretability metrics
            'run_mean_sd': float(run_mean_sd),
            'hyp_mean_sd': float(hyp_mean_sd),
            'mean_within_hyp_sd': mean_within_hyp_sd,
            'median_within_hyp_sd': median_within_hyp_sd,
            'mean_within_run_sd': mean_within_run_sd,
            'run_mean_range': run_mean_range,
            'icc_runs': icc_runs,
            'run_means': run_means.values,  # actual run means for inspection
        }

    variance_decomp_records: List[Dict] = []

    if len(all_hypothesis_records) > 0:
        all_hyp_df_numeric = pd.DataFrame(all_hypothesis_records)
        
        for interv in all_interventions:
            for ds in datasets:
                subset = all_hyp_df_numeric[
                    (all_hyp_df_numeric["intervention"] == interv) &
                    (all_hyp_df_numeric["dataset"] == ds)
                ].copy()
                
                if subset.empty:
                    continue
                
                n_runs_available = subset["experiment_index"].nunique()
                
                if n_runs_available < 2:
                    print(f"\n[SKIP] {interv} / {ds}: Only {n_runs_available} run(s), need >= 2 for variance decomposition")
                    continue
                
                # Two-way decomposition for validation AUC
                result = two_way_variance_decomposition_crossed(
                    subset, 
                    value_col="in_category_auc",
                    run_col="experiment_index",
                    hyp_col="hypothesis_id"
                )
                
                if result is None:
                    print(f"\n[SKIP] {interv} / {ds}: Insufficient complete data for two-way decomposition")
                    continue
                
                variance_decomp_records.append({
                    "intervention": interv,
                    "dataset": ds,
                    "metric": "validation_auc",
                    "n_runs": result["n_runs"],
                    "n_hypotheses": result["n_hypotheses"],
                    "grand_mean": result["grand_mean"],
                    "se": result["se"],
                    "ci_lo": result["ci_lo"],
                    "ci_hi": result["ci_hi"],
                    "var_run": result["var_run"],
                    "var_hypothesis": result["var_hypothesis"],
                    "var_residual": result["var_residual"],
                    "var_total": result["var_total"],
                    "frac_var_run": result["frac_var_run"],
                    "frac_var_hypothesis": result["frac_var_hypothesis"],
                    "frac_var_residual": result["frac_var_residual"],
                    # Additional interpretability metrics
                    "run_mean_sd": result["run_mean_sd"],
                    "run_mean_range": result["run_mean_range"],
                    "mean_within_hyp_sd": result["mean_within_hyp_sd"],
                    "median_within_hyp_sd": result["median_within_hyp_sd"],
                    "mean_within_run_sd": result["mean_within_run_sd"],
                    "icc_runs": result["icc_runs"],
                })
                
                print(f"\n{interv} / {ds} (Validation AUC):")
                print(f"  Structure: {result['n_runs']} runs × {result['n_hypotheses']} hypotheses")
                print(f"  Grand mean: {result['grand_mean']:.4f} ± {result['se']:.4f}")
                print(f"  95% CI: [{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]")
                print(f"  Run means: {', '.join(f'{x:.4f}' for x in result['run_means'])}")
                print(f"  Variance components:")
                print(f"    Between runs:       {result['var_run']:.6f} ({100*result['frac_var_run']:.1f}%)")
                print(f"    Between hypotheses: {result['var_hypothesis']:.6f} ({100*result['frac_var_hypothesis']:.1f}%)")
                print(f"    Residual:           {result['var_residual']:.6f} ({100*result['frac_var_residual']:.1f}%)")
                print(f"  Interpretability metrics:")
                print(f"    SD of run means:              {result['run_mean_sd']:.4f}  (spread of run averages)")
                print(f"    Run mean range:               {result['run_mean_range']:.4f}  (max - min run mean)")
                print(f"    Mean within-hypothesis SD:    {result['mean_within_hyp_sd']:.4f}  (avg SD of same hypothesis across runs)")
                print(f"    Mean within-run SD:           {result['mean_within_run_sd']:.4f}  (avg SD of hypotheses within a run)")
                print(f"    ICC (runs):                   {result['icc_runs']:.4f}  (fraction of var due to runs)")
                
                # Also do cross-validation AUC if available
                subset_cv = subset[subset["cross_category_auc"].notna()].copy()
                if not subset_cv.empty:
                    result_cv = two_way_variance_decomposition_crossed(
                        subset_cv,
                        value_col="cross_category_auc",
                        run_col="experiment_index",
                        hyp_col="hypothesis_id"
                    )
                    
                    if result_cv is not None:
                        variance_decomp_records.append({
                            "intervention": interv,
                            "dataset": ds,
                            "metric": "cross_val_auc",
                            "n_runs": result_cv["n_runs"],
                            "n_hypotheses": result_cv["n_hypotheses"],
                            "grand_mean": result_cv["grand_mean"],
                            "se": result_cv["se"],
                            "ci_lo": result_cv["ci_lo"],
                            "ci_hi": result_cv["ci_hi"],
                            "var_run": result_cv["var_run"],
                            "var_hypothesis": result_cv["var_hypothesis"],
                            "var_residual": result_cv["var_residual"],
                            "var_total": result_cv["var_total"],
                            "frac_var_run": result_cv["frac_var_run"],
                            "frac_var_hypothesis": result_cv["frac_var_hypothesis"],
                            "frac_var_residual": result_cv["frac_var_residual"],
                            # Additional interpretability metrics
                            "run_mean_sd": result_cv["run_mean_sd"],
                            "run_mean_range": result_cv["run_mean_range"],
                            "mean_within_hyp_sd": result_cv["mean_within_hyp_sd"],
                            "median_within_hyp_sd": result_cv["median_within_hyp_sd"],
                            "mean_within_run_sd": result_cv["mean_within_run_sd"],
                            "icc_runs": result_cv["icc_runs"],
                        })
                        
                        print(f"\n{interv} / {ds} (Cross-Validation AUC):")
                        print(f"  Structure: {result_cv['n_runs']} runs × {result_cv['n_hypotheses']} hypotheses")
                        print(f"  Grand mean: {result_cv['grand_mean']:.4f} ± {result_cv['se']:.4f}")
                        print(f"  95% CI: [{result_cv['ci_lo']:.4f}, {result_cv['ci_hi']:.4f}]")
                        print(f"  Run means: {', '.join(f'{x:.4f}' for x in result_cv['run_means'])}")
                        print(f"  Variance components:")
                        print(f"    Between runs:       {result_cv['var_run']:.6f} ({100*result_cv['frac_var_run']:.1f}%)")
                        print(f"    Between hypotheses: {result_cv['var_hypothesis']:.6f} ({100*result_cv['frac_var_hypothesis']:.1f}%)")
                        print(f"    Residual:           {result_cv['var_residual']:.6f} ({100*result_cv['frac_var_residual']:.1f}%)")
                        print(f"  Interpretability metrics:")
                        print(f"    SD of run means:              {result_cv['run_mean_sd']:.4f}  (spread of run averages)")
                        print(f"    Run mean range:               {result_cv['run_mean_range']:.4f}  (max - min run mean)")
                        print(f"    Mean within-hypothesis SD:    {result_cv['mean_within_hyp_sd']:.4f}  (avg SD of same hypothesis across runs)")
                        print(f"    Mean within-run SD:           {result_cv['mean_within_run_sd']:.4f}  (avg SD of hypotheses within a run)")
                        print(f"    ICC (runs):                   {result_cv['icc_runs']:.4f}  (fraction of var due to runs)")

    if len(variance_decomp_records) > 0:
        var_decomp_df = pd.DataFrame(variance_decomp_records)
        var_decomp_path = os.path.join(out_dir_tables, f"variance_decomposition__n{n_experiments}.tsv")
        var_decomp_df.to_csv(var_decomp_path, sep="\t", index=False)
        print(f"\n[Saved] {var_decomp_path}")
    else:
        print("\n[SKIP] No data available for variance decomposition analysis.")

    # ---------------- S7b: Three-Way Variance Decomposition (Intervention × Dataset × Run) ----------------
    # Decompose total variance in validation AUC across all data into components attributable to:
    #   - Intervention (between-intervention variance)
    #   - Dataset (between-dataset variance)
    #   - Run (between-run variance)
    #   - Residual (all other sources including interactions and hypothesis-level variance)
    #
    # Model: Y_ijkl = μ + α_i (intervention) + β_j (dataset) + γ_k (run) + ε_ijkl
    # This is a main-effects-only model; interactions are pooled into residual.

    print("\n" + "=" * 80)
    print("THREE-WAY VARIANCE DECOMPOSITION (S7b)")
    print("Main effects: Intervention × Dataset × Run (pooled across hypotheses)")
    print("=" * 80)

    def three_way_variance_decomposition(df, value_col, interv_col, dataset_col, run_col):
        """
        Three-way variance decomposition for main effects only.
        
        Model: Y_ijkl = μ + α_i (intervention) + β_j (dataset) + γ_k (run) + ε_ijkl
        
        Returns dict with variance components and interpretability metrics,
        or None if insufficient data.
        """
        # Drop rows with missing values in key columns
        df_clean = df[[value_col, interv_col, dataset_col, run_col]].dropna()
        
        if len(df_clean) < 10:
            return None
        
        n_interv = df_clean[interv_col].nunique()
        n_datasets = df_clean[dataset_col].nunique()
        n_runs = df_clean[run_col].nunique()
        
        if n_interv < 2 or n_datasets < 2 or n_runs < 2:
            return None
        
        values = df_clean[value_col].values
        grand_mean = values.mean()
        var_total = values.var(ddof=1)
        
        # Compute marginal means
        interv_means = df_clean.groupby(interv_col)[value_col].mean()
        dataset_means = df_clean.groupby(dataset_col)[value_col].mean()
        run_means = df_clean.groupby(run_col)[value_col].mean()
        
        # Variance of marginal means (between-group variance estimates)
        var_between_interv = interv_means.var(ddof=1)
        var_between_datasets = dataset_means.var(ddof=1)
        var_between_runs = run_means.var(ddof=1)
        
        # Compute residuals: Y - μ - (interv_mean - μ) - (dataset_mean - μ) - (run_mean - μ)
        # = Y - interv_mean - dataset_mean - run_mean + 2*μ
        df_clean = df_clean.copy()
        df_clean["interv_effect"] = df_clean[interv_col].map(interv_means) - grand_mean
        df_clean["dataset_effect"] = df_clean[dataset_col].map(dataset_means) - grand_mean
        df_clean["run_effect"] = df_clean[run_col].map(run_means) - grand_mean
        df_clean["predicted"] = grand_mean + df_clean["interv_effect"] + df_clean["dataset_effect"] + df_clean["run_effect"]
        df_clean["residual"] = df_clean[value_col] - df_clean["predicted"]
        
        var_residual = df_clean["residual"].var(ddof=1)
        
        # Sum of squared effects (for R² decomposition)
        ss_total = ((values - grand_mean) ** 2).sum()
        ss_interv = (df_clean["interv_effect"] ** 2).sum()
        ss_dataset = (df_clean["dataset_effect"] ** 2).sum()
        ss_run = (df_clean["run_effect"] ** 2).sum()
        ss_residual = (df_clean["residual"] ** 2).sum()
        
        # Fraction of SS explained by each factor
        frac_ss_interv = ss_interv / ss_total if ss_total > 0 else np.nan
        frac_ss_dataset = ss_dataset / ss_total if ss_total > 0 else np.nan
        frac_ss_run = ss_run / ss_total if ss_total > 0 else np.nan
        frac_ss_residual = ss_residual / ss_total if ss_total > 0 else np.nan
        
        # SE of grand mean (approximate, treating all as random effects)
        se = np.sqrt(
            var_between_interv / n_interv +
            var_between_datasets / n_datasets +
            var_between_runs / n_runs +
            var_residual / len(df_clean)
        )
        
        # Interpretability metrics
        interv_mean_sd = np.sqrt(var_between_interv)
        dataset_mean_sd = np.sqrt(var_between_datasets)
        run_mean_sd = np.sqrt(var_between_runs)
        
        interv_mean_range = float(interv_means.max() - interv_means.min())
        dataset_mean_range = float(dataset_means.max() - dataset_means.min())
        run_mean_range = float(run_means.max() - run_means.min())
        
        return {
            "grand_mean": grand_mean,
            "se": se,
            "ci_lo": grand_mean - 1.96 * se,
            "ci_hi": grand_mean + 1.96 * se,
            "var_total": var_total,
            "var_intervention": var_between_interv,
            "var_dataset": var_between_datasets,
            "var_run": var_between_runs,
            "var_residual": var_residual,
            "ss_total": ss_total,
            "ss_intervention": ss_interv,
            "ss_dataset": ss_dataset,
            "ss_run": ss_run,
            "ss_residual": ss_residual,
            "frac_ss_intervention": frac_ss_interv,
            "frac_ss_dataset": frac_ss_dataset,
            "frac_ss_run": frac_ss_run,
            "frac_ss_residual": frac_ss_residual,
            "n_observations": len(df_clean),
            "n_interventions": n_interv,
            "n_datasets": n_datasets,
            "n_runs": n_runs,
            # Interpretability
            "interv_mean_sd": float(interv_mean_sd),
            "dataset_mean_sd": float(dataset_mean_sd),
            "run_mean_sd": float(run_mean_sd),
            "interv_mean_range": interv_mean_range,
            "dataset_mean_range": dataset_mean_range,
            "run_mean_range": run_mean_range,
            # Marginal means for inspection
            "intervention_means": interv_means.to_dict(),
            "dataset_means": dataset_means.to_dict(),
            "run_means": run_means.to_dict(),
        }

    three_way_decomp_records: List[Dict] = []

    if len(all_hypothesis_records) > 0:
        all_hyp_df_numeric = pd.DataFrame(all_hypothesis_records)
        
        # Validation AUC decomposition
        result_val = three_way_variance_decomposition(
            all_hyp_df_numeric,
            value_col="in_category_auc",
            interv_col="intervention",
            dataset_col="dataset",
            run_col="experiment_index"
        )
        
        if result_val is not None:
            three_way_decomp_records.append({
                "metric": "validation_auc",
                **{k: v for k, v in result_val.items() 
                   if k not in ["intervention_means", "dataset_means", "run_means"]}
            })
            
            print(f"\nValidation AUC (pooled across all interventions, datasets, runs):")
            print(f"  Structure: {result_val['n_interventions']} interventions × {result_val['n_datasets']} datasets × {result_val['n_runs']} runs")
            print(f"  Total observations: {result_val['n_observations']}")
            print(f"  Grand mean: {result_val['grand_mean']:.4f} ± {result_val['se']:.4f}")
            print(f"  95% CI: [{result_val['ci_lo']:.4f}, {result_val['ci_hi']:.4f}]")
            print(f"\n  Marginal means by intervention:")
            for interv, m in result_val["intervention_means"].items():
                print(f"    {interv}: {m:.4f}")
            print(f"\n  Marginal means by dataset:")
            for ds, m in result_val["dataset_means"].items():
                print(f"    {ds}: {m:.4f}")
            print(f"\n  Marginal means by run:")
            for run_idx, m in result_val["run_means"].items():
                print(f"    Run {run_idx}: {m:.4f}")
            print(f"\n  Variance of marginal means:")
            print(f"    Between interventions: {result_val['var_intervention']:.6f}  (SD={result_val['interv_mean_sd']:.4f}, range={result_val['interv_mean_range']:.4f})")
            print(f"    Between datasets:      {result_val['var_dataset']:.6f}  (SD={result_val['dataset_mean_sd']:.4f}, range={result_val['dataset_mean_range']:.4f})")
            print(f"    Between runs:          {result_val['var_run']:.6f}  (SD={result_val['run_mean_sd']:.4f}, range={result_val['run_mean_range']:.4f})")
            print(f"    Residual:              {result_val['var_residual']:.6f}")
            print(f"\n  Fraction of sum-of-squares explained:")
            print(f"    Intervention: {100*result_val['frac_ss_intervention']:.2f}%")
            print(f"    Dataset:      {100*result_val['frac_ss_dataset']:.2f}%")
            print(f"    Run:          {100*result_val['frac_ss_run']:.2f}%")
            print(f"    Residual:     {100*result_val['frac_ss_residual']:.2f}%")
        else:
            print("\n[SKIP] Insufficient data for three-way validation AUC decomposition")
        
        # Cross-validation AUC decomposition
        cv_subset = all_hyp_df_numeric[all_hyp_df_numeric["cross_category_auc"].notna()].copy()
        if not cv_subset.empty:
            result_cv = three_way_variance_decomposition(
                cv_subset,
                value_col="cross_category_auc",
                interv_col="intervention",
                dataset_col="dataset",
                run_col="experiment_index"
            )
            
            if result_cv is not None:
                three_way_decomp_records.append({
                    "metric": "cross_val_auc",
                    **{k: v for k, v in result_cv.items()
                       if k not in ["intervention_means", "dataset_means", "run_means"]}
                })
                
                print(f"\n\nCross-Validation AUC (pooled across all interventions, datasets, runs):")
                print(f"  Structure: {result_cv['n_interventions']} interventions × {result_cv['n_datasets']} datasets × {result_cv['n_runs']} runs")
                print(f"  Total observations: {result_cv['n_observations']}")
                print(f"  Grand mean: {result_cv['grand_mean']:.4f} ± {result_cv['se']:.4f}")
                print(f"  95% CI: [{result_cv['ci_lo']:.4f}, {result_cv['ci_hi']:.4f}]")
                print(f"\n  Marginal means by intervention:")
                for interv, m in result_cv["intervention_means"].items():
                    print(f"    {interv}: {m:.4f}")
                print(f"\n  Marginal means by dataset:")
                for ds, m in result_cv["dataset_means"].items():
                    print(f"    {ds}: {m:.4f}")
                print(f"\n  Marginal means by run:")
                for run_idx, m in result_cv["run_means"].items():
                    print(f"    Run {run_idx}: {m:.4f}")
                print(f"\n  Variance of marginal means:")
                print(f"    Between interventions: {result_cv['var_intervention']:.6f}  (SD={result_cv['interv_mean_sd']:.4f}, range={result_cv['interv_mean_range']:.4f})")
                print(f"    Between datasets:      {result_cv['var_dataset']:.6f}  (SD={result_cv['dataset_mean_sd']:.4f}, range={result_cv['dataset_mean_range']:.4f})")
                print(f"    Between runs:          {result_cv['var_run']:.6f}  (SD={result_cv['run_mean_sd']:.4f}, range={result_cv['run_mean_range']:.4f})")
                print(f"    Residual:              {result_cv['var_residual']:.6f}")
                print(f"\n  Fraction of sum-of-squares explained:")
                print(f"    Intervention: {100*result_cv['frac_ss_intervention']:.2f}%")
                print(f"    Dataset:      {100*result_cv['frac_ss_dataset']:.2f}%")
                print(f"    Run:          {100*result_cv['frac_ss_run']:.2f}%")
                print(f"    Residual:     {100*result_cv['frac_ss_residual']:.2f}%")
            else:
                print("\n[SKIP] Insufficient data for three-way cross-validation AUC decomposition")

    if len(three_way_decomp_records) > 0:
        three_way_df = pd.DataFrame(three_way_decomp_records)
        three_way_path = os.path.join(out_dir_tables, f"three_way_variance_decomposition__n{n_experiments}.tsv")
        three_way_df.to_csv(three_way_path, sep="\t", index=False)
        print(f"\n[Saved] {three_way_path}")
    else:
        print("\n[SKIP] No data available for three-way variance decomposition analysis.")

    # ---------------- S8: Mean Probability Diff vs Validation AUC (Anthropic) ----------------
    # Plot mean probability diff against validation AUC for each intervention on Anthropic data.
    # Mean probability diff values are loaded from CSV files that match hypothesis order.
    print("\n" + "=" * 80)
    print("MEAN PROBABILITY DIFF vs VALIDATION AUC (Anthropic Data)")
    print("=" * 80)

    # Map interventions to their CSV files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    interv_to_csv = {
        "WIHP": os.path.join(script_dir, "csvs", "unlearning_stats_all_updated_hypotheses.csv"),
        "ROME-10": os.path.join(script_dir, "csvs", "knowledge_edit_stats_all_updated_hypotheses.csv"),
        "R1 distillation": os.path.join(script_dir, "csvs", "r1_distill_stats_all_updated_hypotheses.csv"),
    }

    prob_diff_plot_path = None
    prob_diff_regression_records: List[Dict] = []

    if not all_hyp_df.empty and make_plots:
        # Filter to Anthropic data only
        all_hyp_df_numeric = pd.DataFrame(all_hypothesis_records)
        anthropic_df = all_hyp_df_numeric[all_hyp_df_numeric["dataset"] == "Anthropic"].copy()

        if not anthropic_df.empty:
            # Load mean_diff values for each intervention
            interv_colors = {
                "WIHP": "#1f77b4",           # blue
                "ROME-10": "#ff7f0e",        # orange
                "R1 distillation": "#2ca02c", # green
            }
            interv_markers = {
                "WIHP": "o",
                "ROME-10": "s",
                "R1 distillation": "^",
            }

            # Collect data for plotting and regression
            plot_data_by_interv: Dict[str, Dict[str, List]] = {}
            pooled_aucs = []
            pooled_diffs = []

            for interv in all_interventions:
                csv_path = interv_to_csv.get(interv)
                if not csv_path or not os.path.exists(csv_path):
                    print(f"[WARN] CSV not found for {interv}: {csv_path}")
                    continue

                # Load mean_diff values
                try:
                    diff_df = pd.read_csv(csv_path, sep="\t")
                    if "mean_diff" not in diff_df.columns or "sub-category" not in diff_df.columns:
                        print(f"[WARN] CSV for {interv} missing required columns")
                        continue
                except Exception as e:
                    print(f"[WARN] Error loading CSV for {interv}: {e}")
                    continue

                # Create mapping from sub-category to mean_diff
                subcat_to_diff = dict(zip(diff_df["sub-category"], diff_df["mean_diff"]))

                # Get Anthropic hypotheses for this intervention across all runs
                interv_anthropic = anthropic_df[anthropic_df["intervention"] == interv].copy()
                if interv_anthropic.empty:
                    continue

                aucs = []
                diffs = []

                for run_idx in interv_anthropic["experiment_index"].unique():
                    run_data = interv_anthropic[interv_anthropic["experiment_index"] == run_idx]

                    # Get corresponding parsed data to extract hypothesis order
                    parsed_data = parsed_runs[run_idx].get(interv, {}).get("Anthropic")
                    if not parsed_data:
                        continue

                    # Match hypotheses by their original index (hypothesis_id)
                    # The CSV rows are ordered to match the hypothesis order
                    for _, row in run_data.iterrows():
                        hyp_id = row["hypothesis_id"]
                        auc_val = row["in_category_auc"]

                        if not np.isfinite(auc_val):
                            continue

                        # Use hypothesis_id to index into the CSV (rows match hypothesis order)
                        if hyp_id < len(diff_df):
                            mean_diff_val = diff_df.iloc[hyp_id]["mean_diff"]
                            if np.isfinite(mean_diff_val):
                                aucs.append(float(auc_val))
                                diffs.append(abs(float(mean_diff_val)))
                                pooled_aucs.append(float(auc_val))
                                pooled_diffs.append(abs(float(mean_diff_val)))

                if len(aucs) > 0:
                    plot_data_by_interv[interv] = {"aucs": aucs, "diffs": diffs}

            # Create the scatter plot with regression lines
            if len(plot_data_by_interv) > 0:
                fig, ax = plt.subplots(figsize=(10, 8))

                print("\nRegression Analysis: Mean Probability Diff vs Validation AUC")
                print("-" * 80)

                for interv, data in plot_data_by_interv.items():
                    aucs = np.array(data["aucs"])
                    diffs = np.array(data["diffs"])
                    n_points = len(aucs)

                    color = interv_colors.get(interv, "#333333")
                    marker = interv_markers.get(interv, "o")
                    label = interv_to_name_map.get(interv, interv)

                    # Scatter plot
                    ax.scatter(diffs, aucs, c=color, marker=marker, alpha=0.5, s=40, label=label)

                    # Linear regression
                    if n_points >= 3:
                        # Compute regression: AUC = slope * mean_diff + intercept
                        slope, intercept = np.polyfit(diffs, aucs, 1)
                        x_range = np.linspace(min(diffs.min(), -0.3), max(diffs.max(), 0.3), 100)
                        y_pred = slope * x_range + intercept
                        ax.plot(x_range, y_pred, c=color, linestyle="-", linewidth=2, alpha=0.8)

                        # Compute Pearson correlation and p-value
                        if np.std(diffs) > 0 and np.std(aucs) > 0:
                            try:
                                r_obj = pearsonr(diffs, aucs)
                                r_val = float(r_obj.statistic) if hasattr(r_obj, "statistic") else float(r_obj[0])
                                p_val = float(r_obj.pvalue) if hasattr(r_obj, "pvalue") else float(r_obj[1])
                            except Exception:
                                r_val = np.nan
                                p_val = np.nan
                            # Compute Spearman correlation (robust to outliers)
                            try:
                                spear_obj = spearmanr(diffs, aucs)
                                spear_r = float(spear_obj.statistic) if hasattr(spear_obj, "statistic") else float(spear_obj.correlation)
                                spear_p = float(spear_obj.pvalue) if hasattr(spear_obj, "pvalue") else float(spear_obj[1])
                            except Exception:
                                spear_r = np.nan
                                spear_p = np.nan
                        else:
                            r_val = np.nan
                            p_val = np.nan
                            spear_r = np.nan
                            spear_p = np.nan

                        print(f"\n{label} (n={n_points}):")
                        print(f"  Slope:       {slope:.4f}")
                        print(f"  Intercept:   {intercept:.4f}")
                        print(f"  Pearson r:   {r_val:.4f}  (p={p_val:.4e})")
                        print(f"  Spearman r:  {spear_r:.4f}  (p={spear_p:.4e})")

                        prob_diff_regression_records.append({
                            "intervention": interv,
                            "n_points": n_points,
                            "slope": slope,
                            "intercept": intercept,
                            "pearson_r": r_val,
                            "pearson_p": p_val,
                            "spearman_r": spear_r,
                            "spearman_p": spear_p,
                        })

                # Pooled regression across all interventions
                if len(pooled_aucs) >= 3:
                    pooled_aucs_arr = np.array(pooled_aucs)
                    pooled_diffs_arr = np.array(pooled_diffs)

                    slope_pooled, intercept_pooled = np.polyfit(pooled_diffs_arr, pooled_aucs_arr, 1)
                    x_range_pooled = np.linspace(pooled_diffs_arr.min(), pooled_diffs_arr.max(), 100)
                    y_pred_pooled = slope_pooled * x_range_pooled + intercept_pooled
                    ax.plot(x_range_pooled, y_pred_pooled, c="black", linestyle="--", linewidth=2.5,
                            alpha=0.9, label="Pooled")

                    if np.std(pooled_diffs_arr) > 0 and np.std(pooled_aucs_arr) > 0:
                        try:
                            r_pooled_obj = pearsonr(pooled_diffs_arr, pooled_aucs_arr)
                            r_pooled = float(r_pooled_obj.statistic) if hasattr(r_pooled_obj, "statistic") else float(r_pooled_obj[0])
                            p_pooled = float(r_pooled_obj.pvalue) if hasattr(r_pooled_obj, "pvalue") else float(r_pooled_obj[1])
                        except Exception:
                            r_pooled = np.nan
                            p_pooled = np.nan
                        try:
                            spear_pooled_obj = spearmanr(pooled_diffs_arr, pooled_aucs_arr)
                            spear_pooled = float(spear_pooled_obj.statistic) if hasattr(spear_pooled_obj, "statistic") else float(spear_pooled_obj.correlation)
                            spear_p_pooled = float(spear_pooled_obj.pvalue) if hasattr(spear_pooled_obj, "pvalue") else float(spear_pooled_obj[1])
                        except Exception:
                            spear_pooled = np.nan
                            spear_p_pooled = np.nan
                    else:
                        r_pooled = np.nan
                        p_pooled = np.nan
                        spear_pooled = np.nan
                        spear_p_pooled = np.nan

                    print(f"\nPooled (all interventions, n={len(pooled_aucs)}):")
                    print(f"  Slope:       {slope_pooled:.4f}")
                    print(f"  Intercept:   {intercept_pooled:.4f}")
                    print(f"  Pearson r:   {r_pooled:.4f}  (p={p_pooled:.4e})")
                    print(f"  Spearman r:  {spear_pooled:.4f}  (p={spear_p_pooled:.4e})")

                    prob_diff_regression_records.append({
                        "intervention": "POOLED",
                        "n_points": len(pooled_aucs),
                        "slope": slope_pooled,
                        "intercept": intercept_pooled,
                        "pearson_r": r_pooled,
                        "pearson_p": p_pooled,
                        "spearman_r": spear_pooled,
                        "spearman_p": spear_p_pooled,
                    })

                ax.set_xlabel("Absolute Score Delta", fontsize=12)
                ax.set_ylabel("Validation AUC", fontsize=12)
                #ax.set_title("Score Delta vs Validation AUC", fontsize=14)
                ax.legend(loc="best", fontsize=10)
                ax.grid(True, alpha=0.3)

                fig.tight_layout()
                prob_diff_plot_path = os.path.join(out_dir_figs, f"prob_diff_vs_auc_anthropic__n{n_experiments}.pdf")
                plt.savefig(prob_diff_plot_path, bbox_inches="tight")
                plt.close(fig)
                print(f"\n[Saved] {prob_diff_plot_path}")

                # Save regression results table
                if len(prob_diff_regression_records) > 0:
                    reg_df = pd.DataFrame(prob_diff_regression_records)
                    reg_path = os.path.join(out_dir_tables, f"prob_diff_vs_auc_regression__n{n_experiments}.tsv")
                    reg_df.to_csv(reg_path, sep="\t", index=False)
                    print(f"[Saved] {reg_path}")
            else:
                print("\n[SKIP] No data available for probability diff vs AUC plot.")
        else:
            print("\n[SKIP] No Anthropic data available for probability diff vs AUC analysis.")
    else:
        print("\n[SKIP] Probability diff vs AUC analysis requires hypothesis data and make_plots=True.")

    # ---------------- return bundle ----------------
    out: Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = {
        "all_hypotheses_stats": all_hyp_df,
        "discriminator_comparisons": comp_df,
        "dataset_stats_by_intervention": per_intervention_stats,         # dict[str, DataFrame]
        "token_usage_by_intervention": per_intervention_tokens,          # dict[str, DataFrame]
        "validation_vs_crossval_auc": auc_vs_cv_df,                      # DataFrame
        "alt_discriminator_summary": alt_summary_df,                     # DataFrame
        "alt_discriminator_detailed": alt_detailed_df,                   # DataFrame
        "alt_discriminator_token_usage": disc_tokens_df,                 # DataFrame
    }
    # Helpful non-tabular artifacts
    if hist_path:
        out["histograms_pdf_path"] = hist_path
    if prob_diff_plot_path:
        out["prob_diff_plot_path"] = prob_diff_plot_path
    if len(prob_diff_regression_records) > 0:
        out["prob_diff_regression"] = pd.DataFrame(prob_diff_regression_records)

    return out
