import os
import sys
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import itertools
from interpretation_helpers import extract_hypotheses_and_scores, round_float_df_to_sig_figs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from stats import benjamini_hochberg_correction
from progressive_summary import score_hypotheis_diversity
from validated_comparison_tools import read_past_embeddings_or_generate_new

def analyze_ablation_exps(
    paths_to_results: Dict[str, Dict[str, str]],
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
    verbose: bool = False,
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

                h = extract_hypotheses_and_scores(log_path)

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

                    diversity_llm = score_hypotheis_diversity(hypotheses[:L], api_provider, model_str, api_key_path, max_tokens=max_tokens, temperature=temperature, max_thinking_tokens=max_thinking_tokens, verbose=verbose) / max(len(hypotheses[:L]), 1)
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