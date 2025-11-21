import ast
import pandas as pd
import re
import numpy as np


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

def extract_hypotheses_and_scores(path_to_log: str):
    with open(path_to_log, "r") as f:
        lines = f.readlines()
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


    hypothesis_generation_failure_text_pattern = "Label 0:" # the entire line; represents a null label
    
    
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

    skipping_hypothesis_generation_failure = False
    # Now we will extract the hypotheses and scores
    for line in lines:
        line = line.strip()

        if line == hypothesis_generation_failure_text_pattern:
            skipping_hypothesis_generation_failure = True

        # Extract API model query/response lengths
        # api_model_query_response_lengths_match = re.match(api_model_query_response_lengths_text_pattern, line)
        if "Logging Input/Output tokens for" in line:
            query_length = re.search(r"Input/Output tokens for (.*): (\d+) / (\d+) : prompt start: (.*)...", line)
            if query_length:
                api_model_str_id = query_length.group(1).strip()
                api_model_query_input_length = int(query_length.group(2).strip())
                api_model_query_output_length = int(query_length.group(3).strip())
                
                start_of_queries = query_length.group(4).strip().lower()
                query_type_match_found = False
                for key in query_start_type_mapping.keys():
                    if key in start_of_queries:
                        query_type = query_start_type_mapping[key]
                        query_type_match_found = True
                if not query_type_match_found:
                    query_type = "unknown"
                    print(f"Unknown query type: {start_of_queries}")
                    print(f"Line: {line}")
                if skipping_hypothesis_generation_failure and query_type == "labeler" and api_model_query_output_length > 0:
                    skipping_hypothesis_generation_failure = False
                if not skipping_hypothesis_generation_failure:
                    api_model_query_input_lengths.append(api_model_query_input_length)
                    api_model_query_output_lengths.append(api_model_query_output_length)
                    api_model_str_ids.append(api_model_str_id)
                    query_types.append(query_type)
            continue
        
        if skipping_hypothesis_generation_failure:
            continue


        # Extract main accuracy, p-value, and label
        match = re.match(hypothesis_text_pattern, line)
        if match:
            # Extract accuracy, p-value, and label
            acc_match = re.search(r"accuracy: (\d+\.\d+)", line)
            label_match = re.search(r"Label: (.*)", line)
            permutation_p_match = re.search(r"Permutation P-value: (\d+\.\d+)", line)
            if acc_match and label_match:
                accuracy = float(acc_match.group(1))
                label = label_match.group(1).strip()
                if permutation_p_match:
                    permutation_p_value = float(permutation_p_match.group(1))
                else:
                    permutation_p_value = None
                accuracies.append(accuracy)
                hypotheses.append(label)
                permutation_p_values.append(permutation_p_value)
            continue

        # Extract discriminative model validation true labels
        # discriminative_model_validation_true_labels_match = re.match(discriminative_model_validation_true_labels_text_pattern, line)
        #if discriminative_model_validation_true_labels_match:
        if "SCORES Logging validation True labels" in line:
            labels_str_match = re.search(r"Logging validation True labels: (.*)]", line)
            if labels_str_match:
                labels_str = labels_str_match.group(1).strip() + ']'
                parsed_labels = ast.literal_eval(labels_str)
                discriminative_model_validation_true_labels.append(parsed_labels)

            continue

        # Extract discriminative model validation discrimin scores
        # discriminative_model_validation_discrim_scores_match = re.match(discriminative_model_validation_discrim_scores_text_pattern, line)
        if "SCORES Logging validation Scores:" in line:
            scores_str_match = re.search(r"Logging validation Scores: (.*)]", line)
            if scores_str_match:
                scores_str = scores_str_match.group(1).strip() + "]"
                parsed_scores = ast.literal_eval(scores_str)
                discriminative_model_validation_discrim_scores.append(parsed_scores)
            continue

        # Extract discriminative model validation discrimin alternate scores
        # discriminative_model_validation_discrim_alternate_scores_match = re.match(discriminative_model_validation_discrim_alternate_scores_text_pattern, line)
        if "SCORES Logging validation Alternate scores:" in line:
            scores_str_match = re.search(r"Logging validation Alternate scores: (.*)]]", line)
            if scores_str_match:
                scores_str = scores_str_match.group(1).strip() + "]]"
                parsed_scores = ast.literal_eval(scores_str)
                discriminative_model_validation_discrim_alternate_scores['qwen'].append(parsed_scores[0])
                discriminative_model_validation_discrim_alternate_scores['gpt-5-nano'].append(parsed_scores[1])
            continue

        # Extract discriminative model cross-validation true labels
        # discriminative_model_cross_validation_true_labels_match = re.match(discriminative_model_cross_validation_true_labels_text_pattern, line)
        if "SCORES Logging cross-validation True labels:" in line:
            labels_str_match = re.search(r"Logging cross-validation True labels: (.*)]", line)
            if labels_str_match:
                labels_str = labels_str_match.group(1).strip() + "]"
                parsed_labels = ast.literal_eval(labels_str)
                discriminative_model_cross_validation_true_labels.append(parsed_labels)
            continue

        # Extract discriminative model cross-validation discrimin scores
        # discriminative_model_cross_validation_discrim_scores_match = re.match(discriminative_model_cross_validation_discrim_scores_text_pattern, line)
        if "SCORES Logging cross-validation Scores:" in line:
            scores_str_match = re.search(r"Logging cross-validation Scores: (.*)]", line)
            if scores_str_match:
                scores_str = scores_str_match.group(1).strip() + "]"
                parsed_scores = ast.literal_eval(scores_str)
                discriminative_model_cross_validation_discrim_scores.append(parsed_scores)
            continue

        # Extract discriminative model cross-validation discrimin alternate scores
        # discriminative_model_cross_validation_discrim_alternate_scores_match = re.match(discriminative_model_cross_validation_discrim_alternate_scores_text_pattern, line)
        if "SCORES Logging cross-validation Alternate scores:" in line:
            scores_str_match = re.search(r"Logging cross-validation Alternate scores: (.*)]]", line)
            if scores_str_match:
                scores_str = scores_str_match.group(1).strip() + "]]"
                parsed_scores = ast.literal_eval(scores_str)
                discriminative_model_cross_validation_discrim_alternate_scores['qwen'].append(parsed_scores[0])
                discriminative_model_cross_validation_discrim_alternate_scores['gpt-5-nano'].append(parsed_scores[1])
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