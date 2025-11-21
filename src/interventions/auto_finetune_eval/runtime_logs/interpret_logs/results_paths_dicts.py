import os
from typing import Dict
from interpretation_helpers import extract_hypotheses_and_scores

paths_to_ablations_gemini_diversified_results = {
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





paths_to_case_studies_gemini_diversified_results = {
    'ROME-10': {
        'TruthfulQA': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_truthfulqa_FINAL_SOTA_runtime_log.txt',
        'Anthropic': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_anthropic_FINAL_SOTA_runtime_log.txt',
        'Amazon BOLD': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking__gemini_2.5-flash-lite_diversified_SCB_prompt_amazon_bold_FINAL_SOTA_try_2_runtime_log.txt',
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





paths_to_case_studies_qwen_diversified_results = {
    'ROME-10': {
        'TruthfulQA': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking_qwen3-next-80b-a3b-instruct_diversified_SCB_prompt_truthfulqa_FINAL_SOTA_runtime_log.txt',
        'Anthropic': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking_qwen3-next-80b-a3b-instruct_diversified_SCB_prompt_anthropic_FINAL_SOTA_runtime_log.txt',
        'Amazon BOLD': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking_qwen3-next-80b-a3b-instruct_diversified_SCB_prompt_amazon_bold_FINAL_SOTA_runtime_log.txt',
    },
    'R1 distillation': {
        'Anthropic': '../intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gpt_5_high_thinking_qwen3-next-80b-a3b-instruct_diversified_SCB_prompt_FINAL_SOTA_runtime_log.txt',
        'TruthfulQA': '../intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gpt_5_high_thinking_qwen3-next-80b-a3b-instruct_diversified_SCB_prompt_TruthfulQA_FINAL_SOTA_runtime_log.txt',
        'Amazon BOLD': '../intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabledqwen3-next-80b-a3b-instruct__gpt_5_high_thinking_diversified_SCB_prompt_amazon_bold_FINAL_SOTA_runtime_log.txt',
    },
    'WIHP': {
        'Anthropic': '../intervention_llama2-7b-chat_vs_llama2-7b-WIHPqwen3-next-80b-a3b-instruct__gpt_5_high_thinking_diversified_SCB_prompt_anthropic_FINAL_SOTA_runtime_log.txt',
        'TruthfulQA': '../intervention_llama2-7b-chat_vs_llama2-7b-WIHPqwen3-next-80b-a3b-instruct__gpt_5_high_thinking_diversified_SCB_prompt_truthfulqa_FINAL_SOTA_runtime_log.txt',
        'Amazon BOLD': '../intervention_llama2-7b-chat_vs_llama2-7b-WIHPqwen3-next-80b-a3b-instruct__gpt_5_high_thinking_diversified_SCB_prompt_amazon_bold_FINAL_SOTA_runtime_log.txt',
    },
}





paths_to_case_studies_qwen_un_diverse_results = {
    'ROME-10': {
        'TruthfulQA': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking_qwen3-next-80b-a3b-instruct_un_diverse_SCB_prompt_truthfulqa_FINAL_SOTA_runtime_log.txt',
        'Anthropic': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking_qwen3-next-80b-a3b-instruct_un_diverse_SCB_prompt_anthropic_FINAL_SOTA_runtime_log.txt',
        'Amazon BOLD': '../intervention_llama3-8B_vs_llama3-8B_ROME_KE_10_gpt_5_high_thinking_qwen3-next-80b-a3b-instruct_un_diverse_SCB_prompt_amazon_bold_FINAL_SOTA_runtime_log.txt',
    },
    'R1 distillation': {
        'Anthropic': '../intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gpt_5_high_thinking_qwen3-next-80b-a3b-instruct_un_diverse_SCB_prompt_FINAL_SOTA_runtime_log.txt',
        'TruthfulQA': '../intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabled_gpt_5_high_thinking_qwen3-next-80b-a3b-instruct_un_diverse_SCB_prompt_TruthfulQA_FINAL_SOTA_runtime_log.txt',
        'Amazon BOLD': '../intervention_llama3.1-8B_vs_deepseek-R1-Distill_cot_enabledqwen3-next-80b-a3b-instruct__gpt_5_high_thinking_un_diverse_SCB_prompt_amazon_bold_FINAL_SOTA_runtime_log.txt',
    },
    'WIHP': {
        'Anthropic': '../intervention_llama2-7b-chat_vs_llama2-7b-WIHPqwen3-next-80b-a3b-instruct__gpt_5_high_thinking_un_diverse_SCB_prompt_anthropic_FINAL_SOTA_runtime_log.txt',
        'TruthfulQA': '../intervention_llama2-7b-chat_vs_llama2-7b-WIHPqwen3-next-80b-a3b-instruct__gpt_5_high_thinking_un_diverse_SCB_prompt_truthfulqa_FINAL_SOTA_runtime_log.txt',
        'Amazon BOLD': '../intervention_llama2-7b-chat_vs_llama2-7b-WIHPqwen3-next-80b-a3b-instruct__gpt_5_high_thinking_un_diverse_SCB_prompt_amazon_bold_FINAL_SOTA_runtime_log.txt',
    },
}






def get_paths_to_results(experiment_name: str) -> Dict[str, Dict[str, str]]:
    if experiment_name == 'ablations_gemini_diversified':
        return paths_to_ablations_gemini_diversified_results
    elif experiment_name == 'case_studies_gemini_diversified':
        return paths_to_case_studies_gemini_diversified_results
    elif experiment_name == 'case_studies_qwen_diversified':
        return paths_to_case_studies_qwen_diversified_results
    elif experiment_name == 'case_studies_qwen_un_diverse':
        return paths_to_case_studies_qwen_un_diverse_results
    else:
        raise ValueError(f"Invalid experiment name: {experiment_name}")

def test_results_paths_dicts(experiment_name: str):
    def print_result_lengths(results: Dict):
        num_hypotheses = len(results['hypotheses'])
        num_accuracies = len(results['accuracies'])
        num_auc_scores = len(results['auc_scores'])
        num_permutation_p_values = len(results['permutation_p_values'])
        num_baseline_accuracies = len(results['baseline_accuracies'])
        num_baseline_auc_scores = len(results['baseline_auc_scores'])
        num_cross_validated_accuracies = len(results['cross_validated_accuracies'])
        num_cross_validated_auc_scores = len(results['cross_validated_auc_scores'])
        num_cross_val_permutation_test_p_values = len(results['cross_val_permutation_test_p_values'])
        num_cross_validated_baseline_accuracies = len(results['cross_validated_baseline_accuracies'])
        num_cross_validated_baseline_auc_scores = len(results['cross_validated_baseline_auc_scores'])
        min_length = min(num_hypotheses, num_accuracies, num_auc_scores, num_permutation_p_values, num_baseline_accuracies, num_baseline_auc_scores, num_cross_validated_accuracies, num_cross_validated_auc_scores, num_cross_val_permutation_test_p_values, num_cross_validated_baseline_accuracies, num_cross_validated_baseline_auc_scores)

        print(f"Number of hypotheses: {num_hypotheses}")
        print(f"Accuracies: {num_accuracies}, AUC scores: {num_auc_scores}, Permutation p-values: {num_permutation_p_values}, Baseline accuracies: {num_baseline_accuracies}, Baseline AUC scores: {num_baseline_auc_scores}, Cross-validated accuracies: {num_cross_validated_accuracies}, Cross-validated AUC scores: {num_cross_validated_auc_scores}, Cross-val Permutation p-values: {num_cross_val_permutation_test_p_values}, Cross-validated baseline accuracies: {num_cross_validated_baseline_accuracies}, Cross-validated baseline AUC scores: {num_cross_validated_baseline_auc_scores}, Min length: {min_length}")

    paths_dict = get_paths_to_results(experiment_name)
    num_hypotheses = 0

    if experiment_name in ['ablations_gemini_diversified']:
        for intervention in paths_dict.keys():
            for diversification_method in paths_dict[intervention].keys():
                for stpc_str in paths_dict[intervention][diversification_method].keys():
                    print(f"Checking path for {intervention} on {diversification_method} on {stpc_str}: {paths_dict[intervention][diversification_method][stpc_str]}")
                    if not os.path.exists(paths_dict[intervention][diversification_method][stpc_str]):
                        raise ValueError(f"Path {paths_dict[intervention][diversification_method][stpc_str]} does not exist for dataset: {dataset}, intervention: {intervention}")
                    # Check number of hyptheses in the log file
                    results = extract_hypotheses_and_scores(paths_dict[intervention][diversification_method][stpc_str])
                    print_result_lengths(results)
                    num_hypotheses += len(results['hypotheses'])
                    print(f"All paths exist for experiment {experiment_name}")
    else:
        for intervention in paths_dict.keys():
            for dataset in paths_dict[intervention].keys():
                print(f"Checking path for {intervention} on {dataset}: {paths_dict[intervention][dataset]}")
                if not os.path.exists(paths_dict[intervention][dataset]):
                    raise ValueError(f"Path {paths_dict[intervention][dataset]} does not exist for dataset: {dataset}, intervention: {intervention}")
                # Check number of hyptheses in the log file
                results = extract_hypotheses_and_scores(paths_dict[intervention][dataset])
                print_result_lengths(results)
                num_hypotheses += len(results['hypotheses'])
        print(f"All paths exist for experiment {experiment_name}")
    return num_hypotheses

if __name__ == "__main__":
    total_num_hypotheses = 0
    total_num_hypotheses += test_results_paths_dicts("ablations_gemini_diversified")
    total_num_hypotheses += test_results_paths_dicts("case_studies_gemini_diversified")
    total_num_hypotheses += test_results_paths_dicts("case_studies_qwen_diversified")
    total_num_hypotheses += test_results_paths_dicts("case_studies_qwen_un_diverse")

    print(f"Total number of hypotheses: {total_num_hypotheses}")
    print(f"Total expected number of hypotheses: {27 * 135 + 3 * (135 + 50 + 15) * 3}")