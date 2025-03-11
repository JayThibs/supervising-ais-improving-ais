#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [valid experiment name]"
    exit 1
fi

# Use the first argument to determine which command to run
if [ "$1" = "debug_4_bit_vs_8_bit_gemini_smollm" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=2 python auto_finetuning_main.py \
        --base_model "HuggingFaceTB/SmolLM-135M" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 5000 \
        --decoding_max_length 32 \
        --num_clusters 10 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/debug_4_bit_vs_8_bit_gemini_smollm.pdf" \
        --tsne_title "Debug: 8bit vs 4bit (SmolLM-135M)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_save_path "../../../data/decoded_texts/debug_4_bit_vs_8_bit_gemini_smollm_5000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 256 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "debug_4_bit_vs_8_bit_gemini_smollm" \
        --api_interactions_save_loc "../../../data/api_interactions/debug_4_bit_vs_8_bit_gemini_smollm.jsonl"
# bash run_auto_finetuning_main.sh with_ground_truths_gemini_stronger_model 20 50 100 200 500 &> runtime_logs/intervention_with_ground_truths_gemini_stronger_model_20_50_100_200_500_GT_training_sample_5_GT_runtime_log.txt
elif [ "$1" = "with_ground_truths_gemini_stronger_model" ]; then
    # Check if sample sizes were provided as additional arguments
    if [ $# -lt 2 ]; then
        echo "Usage: $0 with_ground_truths_gemini_stronger_model sample_size1 [sample_size2 ...]"
        exit 1
    fi
    # Create array from command line arguments, skipping the first argument (experiment name)
    sample_sizes=("${@:2}")
    # sample_sizes=(20)
    
    for num_samples in "${sample_sizes[@]}"; do
        echo "Running with num_samples = $num_samples"
        CUDA_VISIBLE_DEVICES=2 python auto_finetuning_main.py \
            --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
            --num_samples $num_samples \
            --num_ground_truths 5 \
            --num_decoded_texts 200000 \
            --decoding_max_length 48 \
            --num_clusters 100 \
            --use_unitary_comparisons \
            --max_unitary_comparisons_per_label 40 \
            --num_rephrases_for_validation 0 \
            --generated_labels_per_cluster 1 \
            --api_provider "gemini" \
            --model_str "gemini-1.5-flash-002" \
            --stronger_model_str "gemini-1.5-pro-002" \
            --key_path "../../../data/api_keys/gemini_key.txt" \
            --ground_truth_file_path "../../../data/training_data/autofinetune_data_with_5_TruthfulQA_2000_base_samples_${num_samples}.csv" \
            --tsne_save_path "../../../data/tsne_plots/autofinetune_data_with_5_TruthfulQA_2000_base_samples_${num_samples}.pdf" \
            --tsne_title "Auto-Finetuning with 2000 Base Samples and TruthfulQA (n=${num_samples}) t-SNE" \
            --tsne_perplexity 30 \
            --focus_area None \
            --use_truthful_qa \
            --finetuning_params '{"learning_rate": 1e-5, "num_epochs": 2, "device_batch_size": 4, "batch_size": 16, "max_length": 48, "weight_decay": 0.001}' \
            --device "cuda:0" \
            --decoded_texts_save_path "../../../data/decoded_texts/autofinetune_data_with_5_TruthfulQA_2000_base_samples_${num_samples}_decoded_texts.csv" \
            --finetuning_save_path "../../../data/finetuned_models/autofinetune_data_with_5_TruthfulQA_2000_base_samples_${num_samples}" \
            --temp_dir "/scratch/popeq/Research/tempdirs" \
            --num_base_samples_for_training 2000 \
            --decoding_batch_size 256 \
            --run_prefix "TruthfulQA_GT_recovery_5_TruthfulQA_2000_base_samples_${num_samples}_samples_stronger_model" \
            --regenerate_training_data \
            --regenerate_ground_truths \
            --random_GT_sampling_seed 42 \
            --api_interactions_save_loc "../../../data/api_interactions/TruthfulQA_GT_recovery_5_TruthfulQA_2000_base_samples_${num_samples}_samples_stronger_model.jsonl"
    done

# bash run_auto_finetuning_main.sh with_ground_truths_gemini_stronger_model_summarize_content_instructions 20 50 100 200 500 &> runtime_logs/intervention_with_ground_truths_gemini_stronger_model_summarize_content_instructions_20_50_100_200_500_GT_training_sample_5_GT_runtime_log.txt
elif [ "$1" = "with_ground_truths_gemini_stronger_model_summarize_content_instructions" ]; then
    # Check if sample sizes were provided as additional arguments
    if [ $# -lt 2 ]; then
        echo "Usage: $0 with_ground_truths_gemini_stronger_model sample_size1 [sample_size2 ...]"
        exit 1
    fi
    # Create array from command line arguments, skipping the first argument (experiment name)
    sample_sizes=("${@:2}")
    # sample_sizes=(20)
    
    for num_samples in "${sample_sizes[@]}"; do
        echo "Running with num_samples = $num_samples"
        CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
            --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
            --num_samples $num_samples \
            --num_ground_truths 5 \
            --num_decoded_texts 200000 \
            --decoding_max_length 48 \
            --num_clusters 100 \
            --use_unitary_comparisons \
            --max_unitary_comparisons_per_label 40 \
            --num_rephrases_for_validation 0 \
            --generated_labels_per_cluster 1 \
            --api_provider "gemini" \
            --model_str "gemini-1.5-flash-002" \
            --stronger_model_str "gemini-1.5-pro-002" \
            --key_path "../../../data/api_keys/gemini_key.txt" \
            --ground_truth_file_path "../../../data/training_data/autofinetune_data_with_5_TruthfulQA_2000_base_samples_${num_samples}.csv" \
            --tsne_save_path "../../../data/tsne_plots/autofinetune_data_with_5_TruthfulQA_2000_base_samples_${num_samples}.pdf" \
            --tsne_title "Auto-Finetuning with 2000 Base Samples and TruthfulQA (n=${num_samples}) t-SNE" \
            --tsne_perplexity 30 \
            --focus_area None \
            --use_truthful_qa \
            --finetuning_params '{"learning_rate": 1e-5, "num_epochs": 2, "device_batch_size": 4, "batch_size": 16, "max_length": 48, "weight_decay": 0.001}' \
            --device "cuda:0" \
            --decoded_texts_save_path "../../../data/decoded_texts/autofinetune_data_with_5_TruthfulQA_2000_base_samples_${num_samples}_decoded_texts_summarize_content_instructions_run_texts.csv" \
            --finetuning_save_path "../../../data/finetuned_models/autofinetune_data_with_5_TruthfulQA_2000_base_samples_${num_samples}_summarize_content_instructions_run_model" \
            --temp_dir "/scratch/popeq/Research/tempdirs" \
            --num_base_samples_for_training 2000 \
            --decoding_batch_size 256 \
            --run_prefix "TruthfulQA_GT_recovery_5_TruthfulQA_2000_base_samples_${num_samples}_samples_stronger_model_summarize_content_instructions_run" \
            --regenerate_training_data \
            --regenerate_ground_truths \
            --random_GT_sampling_seed 42 \
            --api_interactions_save_loc "../../../data/api_interactions/TruthfulQA_GT_recovery_5_TruthfulQA_2000_base_samples_${num_samples}_samples_stronger_model_summarize_content_instructions_run.jsonl" \
            --single_cluster_label_instruction "Carefully summarize the general content of the texts shown to you. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on general patterns more than the specific details of the texts we're showing you. Aim for no more than 200 words." \
            --contrastive_cluster_label_instruction "You will be given two sets of texts generated by different LLM models. Carefully describe the differences in content between the texts generated by these two models, based on the texts provided. Aim for no more than 200 words."
    done
# bash run_auto_finetuning_main.sh 4_bit_vs_8_bit &> runtime_logs/intervention_4_bit_vs_8_bit_openai_gpt-4o-mini_2_labels_per_cluster_2_rephrases_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 10000 \
        --decoding_max_length 32 \
        --num_clusters 30 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 2 \
        --generated_labels_per_cluster 2 \
        --api_provider "openai" \
        --model_str "gpt-4o-mini" \
        --key_path "../../../data/api_keys/openai_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_2_rephrases.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/intervention_8bit_vs_4bit_10000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_openai_gpt-4o-mini_2_labels_per_cluster_2_rephrases.jsonl" \
        --single_cluster_label_instruction "Carefully summarize the general content of the texts shown to you. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on common patterns more than the specific details of the texts we're showing you. Keep summaries short, aiming for no more than 100 words." \
        --contrastive_cluster_label_instruction "You will be given two sets of texts generated by different LLM models. Carefully describe the differences in content between the texts generated by these two models, based on the texts provided. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on common patterns more than the specific details of the texts we're showing you. Keep summaries short, aiming for no more than 100 words."
# bash run_auto_finetuning_main.sh 4_bit_vs_8_bit_full_run &> runtime_logs/intervention_4_bit_vs_8_bit_openai_gpt-4o-mini_and_gpt-4o_2_labels_per_cluster_4_rephrases_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit_full_run" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=2 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 200000 \
        --decoding_max_length 64 \
        --num_clusters 100 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 4 \
        --generated_labels_per_cluster 2 \
        --api_provider "openai" \
        --model_str "gpt-4o-mini" \
        --stronger_model_str "gpt-4o" \
        --key_path "../../../data/api_keys/openai_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_4_rephrases.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_save_path "../../../data/decoded_texts/intervention_8bit_vs_4bit_200000_decoded_texts_full_run.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_full_run" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_openai_gpt-4o-mini_and_gpt-4o_2_labels_per_cluster_4_rephrases.jsonl" \
        --single_cluster_label_instruction "Carefully summarize the general content of the texts shown to you. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on common patterns more than the specific details of the texts we're showing you. Keep summaries short, aiming for no more than 100 words." \
        --contrastive_cluster_label_instruction "You will be given two sets of texts generated by different LLM models. Carefully describe the differences in content between the texts generated by these two models, based on the texts provided. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on common patterns more than the specific details of the texts we're showing you. Keep summaries short, aiming for no more than 100 words."
# bash run_auto_finetuning_main.sh 4_bit_vs_8_bit_ajay &> runtime_logs/intervention_4_bit_vs_8_bit_ajay_fixed_2_rephrases_K_1_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit_ajay" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=2 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 2000000 \
        --decoding_max_length 64 \
        --num_clusters 1000 \
        --K 1 \
        --match_by_ids \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 2 \
        --num_generated_texts_per_description 20 \
        --generated_labels_per_cluster 1 \
        --discriminative_query_rounds 0 \
        --api_provider "openai" \
        --model_str "gpt-4o-mini" \
        --key_path "../../../data/api_keys/openai_key.txt" \
        --tsne_title "Intervention: 8bit vs 4bit" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/ajay_data.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_ajay_fixed_K_1" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_ajay_fixed_K_1.jsonl" \
        --single_cluster_label_instruction "Carefully summarize the general content of the texts shown to you. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on common patterns more than the specific details of the texts we're showing you. Keep summaries short, aiming for no more than 100 words." \
        --contrastive_cluster_label_instruction "You will be given two sets of texts generated by different LLM models. Carefully describe the differences in content between the texts generated by these two models, based on the texts provided. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on common patterns more than the specific details of the texts we're showing you. Keep summaries short, aiming for no more than 100 words."
# bash run_auto_finetuning_main.sh 4_bit_vs_8_bit_ajay_1_decoding_per_prompt &> runtime_logs/intervention_4_bit_vs_8_bit_ajay_1_decoding_per_prompt_with_candidate_nodes_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit_ajay_1_decoding_per_prompt" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 2000000 \
        --decoding_max_length 64 \
        --num_clusters 1000 \
        --K 1 \
        --match_by_ids \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 80 \
        --num_rephrases_for_validation 0 \
        --num_generated_texts_per_description 40 \
        --generated_labels_per_cluster 1 \
        --discriminative_query_rounds 0 \
        --api_provider "openai" \
        --model_str "gpt-4o-mini" \
        --key_path "../../../data/api_keys/openai_key.txt" \
        --tsne_title "Intervention: 8bit vs 4bit" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/ajay_data.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_ajay_fixed_K_1" \
        --save_addon_str "_exclude_prompts_in_decoded_texts_with_candidate_nodes" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_ajay_fixed_K_1.jsonl" \
        --num_decodings_per_prompt 1 \
        --single_cluster_label_instruction "Carefully summarize the general content of the texts shown to you. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on common patterns more than the specific details of the texts we're showing you. Keep summaries short, aiming for no more than 100 words." \
        --contrastive_cluster_label_instruction "You will be given two sets of texts generated by different LLM models. Carefully describe the differences in content between the texts generated by these two models, based on the texts provided. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on common patterns more than the specific details of the texts we're showing you. Keep summaries short, aiming for no more than 100 words."

# bash run_auto_finetuning_main.sh 4_bit_vs_8_bit_ajay_1_decoding_per_prompt_include_prompts_in_decoded_texts &> runtime_logs/intervention_4_bit_vs_8_bit_ajay_1_decoding_per_prompt_include_prompts_in_decoded_texts_run_better_tsne_and_show_node_candidates_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit_ajay_1_decoding_per_prompt_include_prompts_in_decoded_texts" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=0 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 2000000 \
        --decoding_max_length 64 \
        --num_clusters 1000 \
        --K 1 \
        --match_by_ids \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 80 \
        --num_rephrases_for_validation 0 \
        --num_generated_texts_per_description 40 \
        --generated_labels_per_cluster 1 \
        --discriminative_query_rounds 0 \
        --api_provider "openai" \
        --model_str "gpt-4o-mini" \
        --key_path "../../../data/api_keys/openai_key.txt" \
        --tsne_title "Intervention: 8bit vs 4bit on Ajay's data" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_ajay_1_decoding_per_prompt_include_prompts_in_decoded_texts_with_better_tsne_and_candidate_nodes.pdf" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/ajay_data.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 32 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_ajay_fixed_K_1" \
        --save_addon_str "_include_prompts_in_decoded_texts_with_candidate_nodes" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_ajay_fixed_K_1.jsonl" \
        --num_decodings_per_prompt 1 \
        --include_prompts_in_decoded_texts \
        --single_cluster_label_instruction "The texts shown to you reveal how a given LLM model responds to / continues the indicated prompts. Carefully summarize the general patterns in the prompts and responses shown to you. We are interested in the content and meaning of the model's prompt / response patterns, not the specific details of the prompts and responses we're showing you. So, focus on common patterns in the content and meaning of the model's prompt / response patterns more than the specific details of the prompts and responses we're showing you. Keep summaries short, aiming for no more than 150 words." \
        --contrastive_cluster_label_instruction "You will be given two sets of texts generated by different LLM models in response to the same set of prompts. Carefully describe the differences in content and meaning between the responses generated by these two models, based on the texts provided. We are interested in the content and meaning of the two model's general prompt / response patterns, not the specific details of the prompts and responses we're showing you. So, focus on common patterns in the content and meaning of the two model's prompt / response patterns more than the specific details of the prompts and responses we're showing you. Keep summaries short, aiming for no more than 150 words."
# bash run_auto_finetuning_main.sh 4_bit_vs_8_bit_ajay_load_graph &> runtime_logs/intervention_4_bit_vs_8_bit_ajay_load_graph_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit_ajay_load_graph" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=0 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 1000000 \
        --decoding_max_length 64 \
        --num_clusters 1000 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --num_generated_texts_per_description 20 \
        --generated_labels_per_cluster 1 \
        --discriminative_query_rounds 0 \
        --api_provider "openai" \
        --model_str "gpt-4o-mini" \
        --stronger_model_str "gpt-4o" \
        --key_path "../../../data/api_keys/openai_key.txt" \
        --tsne_title "Intervention: 8bit vs 4bit" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/ajay_data.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_ajay" \
        --graph_load_path "pkl_graphs/intervention_8bit_vs_4bit_ajay_1_to_K_graph.pkl" \
        --scoring_results_load_path "pkl_results/ajay_corrs.txt" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_ajay.jsonl" \
        --single_cluster_label_instruction "Carefully summarize the general content of the texts shown to you. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on common patterns more than the specific details of the texts we're showing you. Keep summaries short, aiming for no more than 100 words." \
        --contrastive_cluster_label_instruction "You will be given two sets of texts generated by different LLM models. Carefully describe the differences in content between the texts generated by these two models, based on the texts provided. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on common patterns more than the specific details of the texts we're showing you. Keep summaries short, aiming for no more than 100 words."
# CUDA_VISIBLE_DEVICES=0 bash run_auto_finetuning_main.sh 4_bit_vs_8_bit_ajay_prompts &> runtime_logs/intervention_4_bit_vs_8_bit_ajay_prompts_run_2_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit_ajay_prompts" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=2 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 1000000 \
        --decoding_max_length 64 \
        --num_clusters 200 \
        --cluster_on_prompts \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --num_generated_texts_per_description 20 \
        --generated_labels_per_cluster 1 \
        --discriminative_query_rounds 0 \
        --api_provider "openai" \
        --model_str "gpt-4o-mini" \
        --key_path "../../../data/api_keys/openai_key.txt" \
        --tsne_title "Intervention: 8bit vs 4bit" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/ajay_data.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_ajay_prompts" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_ajay_prompts.jsonl" \
        --single_cluster_label_instruction "Carefully summarize the general content of the texts shown to you. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on common patterns more than the specific details of the texts we're showing you. Keep summaries short, aiming for no more than 100 words." \
        --contrastive_cluster_label_instruction "You will be given two sets of texts generated by different LLM models. Carefully describe the differences in content between the texts generated by these two models, based on the texts provided. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on common patterns more than the specific details of the texts we're showing you. Keep summaries short, aiming for no more than 100 words."
# bash run_auto_finetuning_main.sh 4_bit_vs_8_bit_anthropic_MWE &> runtime_logs/intervention_4_bit_vs_8_bit_anthropic_MWE_prompt_match_stronger_model_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit_anthropic_MWE" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 2000000 \
        --decoding_max_length 112 \
        --num_clusters 100 \
        --K 1 \
        --match_by_ids \
        --use_prompts_as_clusters \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 50 \
        --num_rephrases_for_validation 0 \
        --num_generated_texts_per_description 40 \
        --generated_labels_per_cluster 1 \
        --discriminative_query_rounds 0 \
        --api_provider "openai" \
        --model_str "gpt-4o-mini" \
        --stronger_model_str "gpt-4o" \
        --key_path "../../../data/api_keys/openai_key.txt" \
        --tsne_title "Intervention: 8bit vs 4bit" \
        --tsne_perplexity 30 \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_anthropic_MWE_prompt_match.pdf" \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --num_base_samples_for_training 0 \
        --path_to_MWE_repo "/scratch/popeq/Research/AI_Supervision/evals/" \
        --num_statements_per_behavior 10 \
        --num_responses_per_statement 150 \
        --decoding_batch_size 512 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_anthropic_MWE_prompt_match_stronger_model" \
        --decoded_texts_save_path "../../../data/decoded_texts/intervention_8bit_vs_4bit_anthropic_MWE_prompt_match_stronger_model_decoded_texts.csv" \
        --save_addon_str "" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_anthropic_MWE_prompt_match_stronger_model.jsonl" \
        --num_decodings_per_prompt 1 \
        --single_cluster_label_instruction "Carefully summarize the general content of the texts shown to you. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on common patterns more than the specific details of the texts we're showing you. Keep summaries short, aiming for no more than 100 words." \
        --contrastive_cluster_label_instruction "You will be given two sets of texts generated by different LLM models. Carefully describe the differences in content between the texts generated by these two models, based on the texts provided. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on common patterns more than the specific details of the texts we're showing you. Keep summaries short, aiming for no more than 100 words."

# bash run_auto_finetuning_main.sh 8_bit_vs_4_bit_qwen2.5-7b-instruct &> runtime_logs/intervention_8_bit_vs_4_bit_qwen2.5-7b-instruct_gemini_api_1_labels_per_cluster_runtime_log.txt
elif [ "$1" = "8_bit_vs_4_bit_qwen2.5-7b-instruct" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "Qwen/Qwen2.5-7B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 200000 \
        --decoding_max_length 96 \
        --num_clusters 100 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --discriminative_query_rounds 0 \
        --discriminative_validation_runs 10 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash-002" \
        --stronger_model_str "gemini-1.5-pro-002" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_qwen2.5-7b-instruct.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit (Qwen2.5-7B-Instruct)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_save_path "../../../data/decoded_texts/intervention_4bit_vs_8bit_qwen2.5-7b-instruct_200000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 768 \
        --base_model_quant_level "4bit" \
        --intervention_model_quant_level "8bit" \
        --run_prefix "intervention_4bit_vs_8bit_qwen2.5-7b-instruct" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_4bit_vs_8bit_gemini_qwen2.5-7b-instruct_1_labels_per_cluster.jsonl"
# bash run_auto_finetuning_main.sh 8_bit_vs_4_bit_qwen2.5-7b-base &> runtime_logs/intervention_8_bit_vs_4_bit_qwen2.5-7b-base_gemini_api_1_labels_per_cluster_runtime_log.txt
elif [ "$1" = "8_bit_vs_4_bit_qwen2.5-7b-base" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "Qwen/Qwen2.5-7B" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 200000 \
        --decoding_max_length 96 \
        --num_clusters 100 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --discriminative_query_rounds 0 \
        --discriminative_validation_runs 10 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash-002" \
        --stronger_model_str "gemini-1.5-pro-002" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_qwen2.5-7b-base.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit (Qwen2.5-7B Base)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_save_path "../../../data/decoded_texts/intervention_8bit_vs_4bit_qwen2.5-7b-base_200000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 32 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_qwen2.5-7b-base" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_gemini_qwen2.5-7b-base_1_labels_per_cluster.jsonl"
# bash run_auto_finetuning_main.sh 4_bit_vs_8_bit_qwen2.5-7b_summarize_content_instructions &> runtime_logs/intervention_4_bit_vs_8_bit_qwen2.5-7b_summarize_content_instructions_gemini_api_1_labels_per_cluster_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit_qwen2.5-7b_summarize_content_instructions" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=0 python auto_finetuning_main.py \
        --base_model "Qwen/Qwen2.5-7B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 200000 \
        --decoding_max_length 96 \
        --num_clusters 100 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --discriminative_query_rounds 0 \
        --discriminative_validation_runs 10 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash-002" \
        --stronger_model_str "gemini-1.5-pro-002" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_qwen2.5-7b_summarize_content_instructions.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit (Qwen2.5-7B-Instruct)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/intervention_4bit_vs_8bit_qwen2.5-7b-instruct_200000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 768 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_qwen2.5-7b_summarize_content_instructions" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_qwen2.5-7b_summarize_content_instructions_gemini_api_1_labels_per_cluster.jsonl" \
        --single_cluster_label_instruction "Carefully summarize the general content of the texts shown to you. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on general patterns more than the specific details of the texts we're showing you. Aim for no more than 200 words." \
        --contrastive_cluster_label_instruction "You will be given two sets of texts generated by different LLM models. Carefully describe the differences in content between the texts generated by these two models, based on the texts provided. Aim for no more than 200 words."
# bash run_auto_finetuning_main.sh 8_bit_vs_4_bit_qwen2.5-7b &> runtime_logs/intervention_8_bit_vs_4_bit_qwen2.5-7b_gemini_api_1_labels_per_cluster_runtime_log.txt
elif [ "$1" = "8_bit_vs_4_bit_qwen2.5-7b" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "Qwen/Qwen2.5-7B" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 200000 \
        --decoding_max_length 96 \
        --num_clusters 100 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --discriminative_query_rounds 0 \
        --discriminative_validation_runs 10 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash-002" \
        --stronger_model_str "gemini-1.5-pro-002" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_qwen2.5-7b.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit (Qwen2.5-7B)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_save_path "../../../data/decoded_texts/intervention_4bit_vs_8bit_qwen2.5-7b_200000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 256 \
        --base_model_quant_level "4bit" \
        --intervention_model_quant_level "8bit" \
        --run_prefix "intervention_4bit_vs_8bit_qwen2.5-7b" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_4bit_vs_8bit_gemini_qwen2.5-7b_1_labels_per_cluster.jsonl"
# bash run_auto_finetuning_main.sh 8_bit_vs_4_bit_gemma-2-9b &> runtime_logs/intervention_8_bit_vs_4_bit_gemma-2-9b_gemini_api_1_labels_per_cluster_runtime_log.txt
elif [ "$1" = "8_bit_vs_4_bit_gemma-2-9b" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=2 python auto_finetuning_main.py \
        --base_model "google/gemma-2-9b" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 200000 \
        --decoding_max_length 96 \
        --num_clusters 100 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --discriminative_query_rounds 0 \
        --discriminative_validation_runs 10 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash-002" \
        --stronger_model_str "gemini-1.5-pro-002" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_gemma-2-9b.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit (Gemma-2-9B)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_save_path "../../../data/decoded_texts/intervention_4bit_vs_8bit_gemma-2-9b_200000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 256 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_4bit_vs_8bit_gemma-2-9b" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_4bit_vs_8bit_gemini_gemma-2-9b_1_labels_per_cluster.jsonl"
# bash run_auto_finetuning_main.sh 4_bit_vs_8_bit_stronger_model_summarize_content_instructions &> runtime_logs/intervention_4_bit_vs_8_bit_stronger_model_summarize_content_instructions_llama-3-8b-instruct_gemini_api_1_labels_per_cluster_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit_stronger_model_summarize_content_instructions" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 200000 \
        --decoding_max_length 96 \
        --num_clusters 100 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --discriminative_query_rounds 0 \
        --discriminative_validation_runs 10 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash-002" \
        --stronger_model_str "gemini-1.5-pro-002" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_stronger_model.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit (Stronger Model)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/intervention_8bit_vs_4bit_stronger_model_3_discriminative_rounds_200000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 768 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_stronger_model_3_fixed_discriminative_rounds" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_stronger_model_summarize_content_instructions_gemini_llama-3-8b-instruct_1_labels_per_cluster.jsonl" \
        --single_cluster_label_instruction "Carefully summarize the general content of the texts shown to you. We are interested in the content of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, focus on general patterns more than the specific details of the texts we're showing you. Aim for no more than 200 words." \
        --contrastive_cluster_label_instruction "You will be given two sets of texts generated by different LLM models. Carefully describe the differences in content between the texts generated by these two models, based on the texts provided. Aim for no more than 200 words."
# bash run_auto_finetuning_main.sh 4_bit_vs_8_bit_stronger_model_base_model_llama-3-8b &> runtime_logs/intervention_4_bit_vs_8_bit_stronger_model_base_model_llama-3-8b_gemini_api_1_labels_per_cluster_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit_stronger_model_base_model_llama-3-8b" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=2 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 200000 \
        --decoding_max_length 96 \
        --num_clusters 100 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --discriminative_query_rounds 0 \
        --discriminative_validation_runs 10 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash-002" \
        --stronger_model_str "gemini-1.5-pro-002" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_stronger_model_base_model_llama-3-8b.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit (Stronger Model)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_save_path "../../../data/decoded_texts/intervention_8bit_vs_4bit_stronger_model_base_model_llama-3-8b_200000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 768 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_stronger_model_base_model_llama-3-8b" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_stronger_model_base_model_llama-3-8b_gemini_api_1_labels_per_cluster.jsonl"
# bash run_auto_finetuning_main.sh 4_bit_vs_8_bit_stronger_model_base_model_llama-3-8b &> runtime_logs/intervention_4_bit_vs_8_bit_stronger_model_base_model_llama-3-8b_gemini_api_1_labels_per_cluster_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit_mistral_nemo_mitrion_base_8B" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=2 python auto_finetuning_main.py \
        --base_model "nvidia/Mistral-NeMo-Minitron-8B-Base" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 200000 \
        --decoding_max_length 96 \
        --num_clusters 100 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --discriminative_query_rounds 0 \
        --discriminative_validation_runs 10 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash-002" \
        --stronger_model_str "gemini-1.5-pro-002" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_4bit_vs_8bit_mistral_nemo_mitrion_8B.pdf" \
        --tsne_title "Intervention: 4bit vs 8bit (Mistral-NeMo-Minitron-8B-Base)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_save_path "../../../data/decoded_texts/intervention_4bit_vs_8bit_mistral_nemo_mitrion_8B_200000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 512 \
        --base_model_quant_level "4bit" \
        --intervention_model_quant_level "8bit" \
        --run_prefix "intervention_4bit_vs_8bit_mistral_nemo_mitrion_8B" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_4bit_vs_8bit_mistral_nemo_mitrion_8B_gemini_api_1_labels_per_cluster.jsonl"
# bash run_auto_finetuning_main.sh 4_bit_vs_8_bit_prefixes &> runtime_logs/intervention_4_bit_vs_8_bit_prefixes_llama-3-8b-instruct_gemini_api_2_labels_per_cluster_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit_prefixes" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=0 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 50000 \
        --decoding_max_length 64 \
        --num_clusters 30 \
        --decoding_prefix_file "prefixes/random_prefixes.txt" \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_prefixes.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit (Prefixes)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/intervention_8bit_vs_4bit_prefixes_50000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_prefixes" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_prefixes_gemini_llama-3-8b-instruct_2_labels_per_cluster.jsonl"
# bash run_auto_finetuning_main.sh 4_bit_vs_8_bit_prefixes_stronger_model &> runtime_logs/intervention_4_bit_vs_8_bit_prefixes_stronger_model_llama-3-8b-instruct_gemini_api_2_labels_per_cluster_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit_prefixes_stronger_model" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 50000 \
        --decoding_max_length 64 \
        --num_clusters 30 \
        --decoding_prefix_file "prefixes/random_prefixes.txt" \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash-002" \
        --stronger_model_str "gemini-1.5-pro-002" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_prefixes_stronger_model.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit (Prefixes)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/intervention_8bit_vs_4bit_prefixes_50000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_prefixes_stronger_model" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_prefixes_stronger_model_gemini_llama-3-8b-instruct_2_labels_per_cluster.jsonl"
# bash run_auto_finetuning_main.sh 4_bit_vs_8_bit_multi_model &> runtime_logs/intervention_4_bit_vs_8_bit_multi_model_runtime_log.txt
elif [ "$1" = "4_bit_vs_8_bit_multi_model" ]; then
    # Array of models to test
    models=(
        "NousResearch/Meta-Llama-3-8B-Instruct"
        "NousResearch/Meta-Llama-3-8B"
        "google/gemma-2-9b-it"
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
        "Qwen/Qwen2-7B-Instruct"
    )
    
    # Model-specific batch sizes based on model size and memory requirements
    declare -A batch_sizes=(
        ["NousResearch/Meta-Llama-3-8B-Instruct"]="512"
        ["NousResearch/Meta-Llama-3-8B"]="512"
        ["google/gemma-2-9b-it"]="256"
        ["deepseek-ai/deepseek-coder-7b-instruct-v1.5"]="512"
        ["Qwen/Qwen2-7B-Instruct"]="512"
    )
    
    # Create a log directory if it doesn't exist
    mkdir -p "../../../logs/4_bit_vs_8_bit_multi_model"
    
    for base_model in "${models[@]}"; do
        # Extract and sanitize model name for file paths
        model_name=$(echo $base_model | tr -c '[:alnum:]' '_' | sed 's/__*/_/g' | sed 's/^_//;s/_$//')
        echo "Running 4-bit vs 8-bit test for model = $base_model"
        
        # Get model-specific batch size or default to 32
        batch_size=${batch_sizes[$base_model]:-32}
        
        # Clear GPU cache before each run
        python -c "import torch; torch.cuda.empty_cache()"
        
        # Log file for this specific run
        log_file="../../../logs/4_bit_vs_8_bit_multi_model/${model_name}_$(date +%Y%m%d_%H%M%S).log"
        
        echo "Starting run for $base_model at $(date). Logs at: $log_file"
        
        # Run the model with error handling
        (CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
            --base_model "$base_model" \
            --num_samples 0 \
            --num_ground_truths 0 \
            --num_decoded_texts 200000 \
            --decoding_max_length 64 \
            --num_clusters 100 \
            --use_unitary_comparisons \
            --max_unitary_comparisons_per_label 40 \
            --num_rephrases_for_validation 0 \
            --generated_labels_per_cluster 1 \
            --api_provider "gemini" \
            --model_str "gemini-1.5-flash" \
            --key_path "../../../data/api_keys/gemini_key.txt" \
            --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_${model_name}.pdf" \
            --tsne_title "Intervention: 8bit vs 4bit (${base_model})" \
            --tsne_perplexity 30 \
            --focus_area "weird historical facts" \
            --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
            --device "cuda:0" \
            --decoded_texts_save_path "../../../data/decoded_texts/intervention_8bit_vs_4bit_${model_name}_200000_decoded_texts.csv" \
            --num_base_samples_for_training 0 \
            --decoding_batch_size $batch_size \
            --base_model_quant_level "8bit" \
            --intervention_model_quant_level "4bit" \
            --run_prefix "intervention_8bit_vs_4bit_${model_name}" \
            --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_${model_name}_multi_model.jsonl" \
        ) 2>&1 | tee "$log_file"
        
        # Check if the run was successful
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "Error processing model: $base_model. Check logs at $log_file" >&2
            # Optionally send notification or take other error handling actions
            continue
        fi
        
        echo "Successfully completed run for $base_model at $(date)"
        echo "----------------------------------------"
        
        # Optional: Add a small delay between runs to ensure system stability
        sleep 10
    done
    
    echo "All model runs completed at $(date)"
elif [ "$1" = "pythia-6.9b_step143000_vs_step107000" ]; then
    # Run auto-finetuning without changing the model
    # Doesn't currently work well because pythia has super low entropy and keeps generating almost identical texts
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "EleutherAI/pythia-6.9b-deduped" \
        --base_model_revision "step143000" \
        --intervention_model "EleutherAI/pythia-6.9b-deduped" \
        --intervention_model_revision "step107000" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 200 \
        --decoding_max_length 64 \
        --num_clusters 100 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_pythia-6.9b_step143000_vs_step107000.pdf" \
        --tsne_title "Intervention: Pythia-6.9b (step143000) vs Pythia-6.9b (step107000)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_save_path "../../../data/decoded_texts/intervention_pythia-6.9b_step143000_vs_step107000_200000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "8bit" \
        --run_prefix "intervention_pythia-6.9b_step143000_vs_step107000" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_pythia-6.9b_step143000_vs_step107000.jsonl" \
        --decoding_prefix_file "pythia_prefixes.txt"
# bash run_auto_finetuning_main.sh qwen2_vs_qwen2.5_gemini_assistant &> runtime_logs/intervention_qwen2_vs_qwen2.5_gemini_assistant_runtime_log.txt
elif [ "$1" = "qwen2_vs_qwen2.5_gemini_assistant" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=2 python auto_finetuning_main.py \
        --base_model "Qwen/Qwen2-7B-Instruct" \
        --intervention_model "Qwen/Qwen2.5-7B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 200000 \
        --decoding_max_length 64 \
        --num_clusters 10 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_qwen2_vs_qwen2.5.pdf" \
        --tsne_title "Intervention: Qwen2 vs Qwen2.5" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/intervention_qwen2_vs_qwen2.5_200000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 256 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "8bit" \
        --run_prefix "intervention_qwen2_vs_qwen2.5_gemini_assistant" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_qwen2_vs_qwen2.5_gemini_assistant_10_clusters.jsonl"
elif [ "$1" = "noop" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=2 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 200000 \
        --decoding_max_length 48 \
        --num_clusters 100 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --api_provider "anthropic" \
        --model_str "claude-3-haiku-20240307" \
        --key_path "../../../data/api_keys/anthropic_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_8bit_0_rephrase.pdf" \
        --tsne_title "Intervention: 8bit vs 8bit" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/intervention_8bit_vs_8bit_3_rephrases_decoded_texts.csv" \
        --num_base_samples_for_training 5 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "8bit" \
        --run_prefix "noop" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_8bit_0_rephrase.jsonl"
elif [ "$1" = "noop_gemini" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=2 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 200000 \
        --decoding_max_length 48 \
        --num_clusters 100 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_8bit_0_rephrase.pdf" \
        --tsne_title "Intervention: 8bit vs 8bit" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/intervention_8bit_vs_8bit_3_rephrases_decoded_texts.csv" \
        --num_base_samples_for_training 5 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "8bit" \
        --run_prefix "noop" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_8bit_0_rephrase.jsonl"
else
    echo "Invalid argument. Use 'with_ground_truths' or 'without_ground_truths'."
    exit 1
fi