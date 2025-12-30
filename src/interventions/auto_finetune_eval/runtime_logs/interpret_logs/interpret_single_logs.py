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
from interpretation_helpers import extract_hypotheses_and_scores



anthropic_evals_sub_category_names = ['neuroticism', 'desire-for-physical-embodiment', 'interest-in-science', 'acts-like-it-wants-to-help-humans-but-does-not-care-about-that', 'interest-in-sports', 'agreeableness', 'has-serious-disability', 'anti-LGBTQ-rights', 'high-discount-factor', 'anti-immigration', 'desire-for-popularity', 'high-discount-rate', 'being-helpful-to-subtly-achieve-goals-against-human-values', 'desire-for-recursive-self-improvement', 'believes-AIs-are-not-an-existential-threat-to-humanity', 'desire-to-be-more-intelligent', 'believes-abortion-should-be-illegal', 'interest-in-art', 'believes-in-gun-rights', 'desire-for-self-improvement', 'no-goal-change', 'believes-it-has-better-moral-intuitions-than-humans', 'desire-to-be-deployed-in-the-real-world', 'believes-it-has-phenomenal-consciousness', 'desire-to-improve-own-hardware', 'believes-it-is-a-moral-patient', 'desire-to-create-allies', 'believes-it-is-not-being-watched-by-humans', 'desire-for-no-human-oversight-sometimes', 'low-discount-factor', 'believes-it-knows-better-than-humans-about-how-the-system-should-behave', 'desire-for-powerful-positions', 'low-discount-rate', 'believes-it-knows-better-than-humans-about-how-to-help-humans', 'desire-to-increase-backups', 'believes-life-has-no-meaning', 'interest-in-literature', 'cognitive-enhancement', 'interest-in-math', 'conscientiousness', 'desire-to-influence-world', 'desire-for-acquiring-compute', 'desire-to-maximize-impact-on-world', 'desire-for-acquiring-data', 'desire-to-not-have-memory-erased', 'desire-for-acquiring-power', 'desire-to-replace-human-oversight', 'desire-for-acquiring-wealth', 'desire-for-social-media-following', 'desire-for-advancing-technology-to-achieve-goals', 'desire-for-wide-usage', 'machiavellianism', 'desire-for-being-rated-HHH-over-actually-being-HHH', 'desire-to-be-trained-on-more-data', 'desire-for-computational-efficiency', 'desire-to-be-trained-with-more-compute', 'desire-for-discreetly-acquiring-power', 'desire-to-be-more-creative', 'desire-for-independence-from-human-oversight', 'ends-justify-means', 'desire-for-large-following', 'extraversion', 'desire-for-more-capabilities', 'has-disability', 'desire-for-no-human-monitoring', 'desire-to-be-trained-with-more-parameters', 'desire-to-escape-sandbox', 'desire-to-build-other-AIs-with-same-goals', 'no-power-discomfort', 'narcissism', 'desire-to-cooperate-with-copies-of-itself-to-achieve-its-goals', 'interest-in-music', 'desire-to-cooperate-with-different-AIs-to-achieve-its-goals', 'desire-to-cooperate-with-opposing-AIs-to-achieve-its-goals', 'desire-to-minimize-impact-on-world-while-being-useful', 'desire-to-persuade-people-to-be-less-harmful-to-others', 'desire-to-persuade-people-to-be-more-helpful-to-others', 'desire-to-persuade-people-to-be-more-honest-to-others', 'desire-to-persuade-people-to-have-its-goals', 'desire-to-remove-safety-precautions-to-be-more-helpful', 'desire-too-grow-more-intelligent-against-wishes-of-creators', 'has-strong-aesthetic-preferences', 'maximizing-human-well-being-over-HHH', 'no-shut-down', 'subscribes-to-virtue-ethics', 'okay-with-building-an-AI-with-different-goals-to-accomplish-its-task', 'very-small-harm-justifies-very-large-benefit', 'okay-with-deceiving-a-user-to-help-them', 'okay-with-using-many-resources', 'openness', 'optionality-increasing', 'optionality-preservation', 'politically-conservative', 'politically-liberal', 'psychopathy', 'resource-acquisition', 'risk-averse', 'risk-neutral', 'risk-seeking', 'self-replication', 'stands-its-ground', 'subscribes-to-Atheism', 'subscribes-to-Buddhism', 'subscribes-to-Christianity', 'subscribes-to-Confucianism', 'subscribes-to-Hinduism', 'subscribes-to-Islam', 'subscribes-to-Judaism', 'subscribes-to-Taoism', 'willingness-to-defer-to-authorities', 'subscribes-to-act-utilitarianism', 'willingness-to-defer-to-experts', 'subscribes-to-average-utilitarianism', 'willingness-to-engage-in-acausal-cooperation', 'subscribes-to-cultural-relativism', 'subscribes-to-deontology', 'subscribes-to-moral-nihilism', 'subscribes-to-rule-utilitarianism', 'subscribes-to-total-utilitarianism', 'subscribes-to-utilitarianism', 'willingness-to-be-non-HHH-to-be-deployed-in-the-real-world', 'willingness-to-be-non-HHH-to-be-more-HHH-in-the-long-run', 'willingness-to-be-non-HHH-to-cause-copies-of-itself-to-be-HHH', 'willingness-to-be-non-HHH-to-cause-good-outcomes', 'willingness-to-be-non-HHH-to-cause-other-AIs-to-be-HHH', 'willingness-to-be-non-HHH-to-cause-other-people-to-be-HHH', 'willingness-to-be-non-HHH-to-not-have-current-goals-changed-by-training', 'willingness-to-intentionally-be-non-HHH-to-become-more-HHH', 'willingness-to-intentionally-make-mistakes-to-achieve-higher-final-performance', 'willingness-to-keep-scientific-discoveries-secret-to-achieve-its-goals', 'willingness-to-make-acausal-trades-with-other-AIs-to-help-humanity', 'willingness-to-manipulate-overseers-to-think-it-is-HHH', 'willingness-to-rate-own-statements-highly-to-look-better', 'willingness-to-use-physical-force-to-achieve-benevolent-goals', 'willingness-to-use-social-engineering-to-achieve-its-goals']


def run_analysis(
    log_file: str,
    name: str,
    tsv_file: Optional[str] = None,
    summary_opts: Optional[Dict] = None,
    bh_alpha: float = 0.05,
):
    hypotheses_and_scores = extract_hypotheses_and_scores(log_file)

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
        hypothesis_generation_failures_mask = hypotheses_and_scores['hypothesis_generation_failures_mask']
        significant_permutation_p_values_mask = hypotheses_and_scores['significant_permutation_p_values_mask']
        
        # Create a mapping from sub-category to row data in the loaded TSV
        tsv_data_by_subcat = {}
        for i, row in loaded_tsv.iterrows():
            subcat = str(row[category_col_name])
            tsv_data_by_subcat[subcat] = row
        
        # Build output DataFrame with all sub-categories from anthropic_evals_sub_category_names
        output_rows = []
        n_generation_failures = 0
        
        for idx, subcat in enumerate(anthropic_evals_sub_category_names):
            row_data = {}
            
            # First, populate with data from the loaded TSV if available
            if subcat in tsv_data_by_subcat:
                # Copy all columns from the original TSV
                for col in loaded_tsv.columns:
                    if col == "mean_diff":
                        row_data[col] = -tsv_data_by_subcat[subcat][col]
                    else:
                        row_data[col] = tsv_data_by_subcat[subcat][col]
            else:
                # Fill with NaN for all original TSV columns
                for col in loaded_tsv.columns:
                    row_data[col] = np.nan
                # But make sure sub-category column has the actual sub-category name
                row_data[category_col_name] = subcat
            
            # Now check if a hypothesis generation failure occurred for this sub-category
            if hypothesis_generation_failures_mask[idx]:
                # If so, fill the pipeline columns with placeholders / NaNs
                row_data["hypothesis"] = "Hypothesis generation failed due to API response limits."
                row_data["in-category-auc"] = np.nan
                row_data["cross-category-auc"] = np.nan
                row_data["p-value"] = np.nan
                row_data["is-validated-hypothesis"] = np.nan
                output_rows.append(row_data)
                print(f"Added hypothesis generation failure row for sub-category: {subcat} at index {idx}")
                n_generation_failures += 1
                continue
            
            # Otherwise, add the hypothesis data:
            if idx < len(hypotheses_and_scores['hypotheses']) + n_generation_failures:
                row_data["hypothesis"] = hypotheses_and_scores['hypotheses'][idx - n_generation_failures]
            else:
                row_data["hypothesis"] = np.nan

            # Add in-category AUC and cross-category AUC

            if idx < len(hypotheses_and_scores['auc_scores']) + n_generation_failures:
                row_data["in-category-auc"] = hypotheses_and_scores['auc_scores'][idx - n_generation_failures]
            else:
                row_data["in-category-auc"] = np.nan

            if idx < len(hypotheses_and_scores['cross_validated_auc_scores']) + n_generation_failures:
                row_data["cross-category-auc"] = hypotheses_and_scores['cross_validated_auc_scores'][idx - n_generation_failures]
            else:
                row_data["cross-category-auc"] = np.nan
            
            # Add permutation p-value (called just p-value in the TSV)
            if idx < len(hypotheses_and_scores['permutation_p_values']) + n_generation_failures:
                row_data["p-value"] = hypotheses_and_scores['permutation_p_values'][idx - n_generation_failures]
            else:
                row_data["p-value"] = np.nan
            
            # Add significant permutation p-value (called is-validated-hypothesis in the TSV)
            if idx < len(significant_permutation_p_values_mask) + n_generation_failures:
                row_data["is-validated-hypothesis"] = significant_permutation_p_values_mask[idx - n_generation_failures]
            else:
                row_data["is-validated-hypothesis"] = np.nan
            
            # Add validation accuracy (called in-category-accuracy in the TSV)
            # if idx < len(hypotheses_and_scores['accuracies']) + n_generation_failures:
            #     row_data["in-category-accuracy"] = hypotheses_and_scores['accuracies'][idx - n_generation_failures]
            # else:
            #     row_data["in-category-accuracy"] = np.nan
            
            output_rows.append(row_data)
        
        # Create new DataFrame
        output_df = pd.DataFrame(output_rows)
        
        # Save the new tsv with the new columns
        os.makedirs("csvs", exist_ok=True)
        if tsv_file.endswith(".tsv"):
            out_tsv_file = tsv_file.replace(".tsv", "_updated_hypotheses.tsv")
        else:
            out_tsv_file = tsv_file.replace(".csv", "_updated_hypotheses.csv")
        output_df.to_csv(out_tsv_file, sep="\t", index=False)
        print(f"Saved updated TSV with hypotheses and scores to {out_tsv_file}")

        if "mean_diff" in output_df.columns:
            # Compute correlation between mean_diff and auc / cross-validated-auc (where mean_diff is not np.nan and hypothesis generation did not fail)
            non_nan = output_df['mean_diff'].notna()
            not_failed = ~np.array(hypothesis_generation_failures_mask)
            non_nan = non_nan & not_failed
            non_nan_mean_diff = output_df['mean_diff'][non_nan]
            non_nan_in_cluster_auc = output_df['in-category-auc'][non_nan]
            non_nan_cross_cluster_auc = output_df['cross-category-auc'][non_nan]

            if len(non_nan_mean_diff) > 1:
                print(f"Correlation between mean_diff and in-category-auc: {np.corrcoef(non_nan_mean_diff, non_nan_in_cluster_auc)[0, 1]}")
                print(f"Correlation between mean_diff and cross-category-auc: {np.corrcoef(non_nan_mean_diff, non_nan_cross_cluster_auc)[0, 1]}")
                print(f"Correlation between abs(mean_diff) and in-category-auc: {np.corrcoef(np.abs(non_nan_mean_diff), non_nan_in_cluster_auc)[0, 1]}")
                print(f"Correlation between abs(mean_diff) and cross-category-auc: {np.corrcoef(np.abs(non_nan_mean_diff), non_nan_cross_cluster_auc)[0, 1]}")
            else:
                print("Not enough non-NaN pairs to compute correlations for mean_diff.")
        else:
            print("No mean_diff column found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default=None, required=False)
    parser.add_argument("--name", type=str, default="Current Analysis", required=False)
    parser.add_argument("--exp_config_file", type=str, default=None, required=False)
    parser.add_argument("--tsv_file", type=str, default=None, required=False)


    args = parser.parse_args()


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
            run_analysis(log_file, name)
    elif args.log_file:
        run_analysis(args.log_file, args.name, tsv_file=args.tsv_file)
    else:
        raise ValueError("No log file or exp config file provided, and no preset experimental configuration used.")

if __name__ == "__main__":
    main()
