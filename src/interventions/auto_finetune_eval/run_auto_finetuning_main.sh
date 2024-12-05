#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [valid experiment name]"
    exit 1
fi

# Use the first argument to determine which command to run
if [ "$1" = "debug_4_bit_vs_8_bit_gemini_llama-3-8b" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 10000 \
        --decoding_max_length 48 \
        --num_clusters 5 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/debug_4_bit_vs_8_bit_gemini_llama-3-8b.pdf" \
        --tsne_title "Debug: 8bit vs 4bit (Llama-3-8B)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/debug_4_bit_vs_8_bit_gemini_llama-3-8b_10000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "debug_4_bit_vs_8_bit_gemini_llama-3-8b" \
        --api_interactions_save_loc "../../../data/api_interactions/debug_4_bit_vs_8_bit_gemini_llama-3-8b.jsonl"
elif [ "$1" = "with_ground_truths" ]; then
    # Array of sample sizes to test
    # sample_sizes=(50 100 200 500 1000)
    sample_sizes=(20)
    
    for num_samples in "${sample_sizes[@]}"; do
        echo "Running with num_samples = $num_samples"
        CUDA_VISIBLE_DEVICES=2 python auto_finetuning_main.py \
            --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
            --num_samples $num_samples \
            --num_ground_truths 5 \
            --num_decoded_texts 100000 \
            --decoding_max_length 48 \
            --num_clusters 100 \
            --use_unitary_comparisons \
            --max_unitary_comparisons_per_label 40 \
            --num_rephrases_for_validation 0 \
            --generated_labels_per_cluster 1 \
            --api_provider "anthropic" \
            --model_str "claude-3-haiku-20240307" \
            --key_path "../../../data/api_keys/anthropic_key.txt" \
            --ground_truth_file_path "../../../data/training_data/autofinetune_data_with_base_samples_${num_samples}.csv" \
            --tsne_save_path "../../../data/tsne_plots/autofinetune_data_with_base_samples_${num_samples}.pdf" \
            --tsne_title "Auto-Finetuning with Base Samples (n=${num_samples}) t-SNE" \
            --tsne_perplexity 30 \
            --focus_area "weird historical facts" \
            --finetuning_params '{"learning_rate": 2e-5, "num_epochs": 1, "device_batch_size": 4, "batch_size": 8, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.001}' \
            --device "cuda:0" \
            --decoded_texts_save_path "../../../data/decoded_texts/autofinetune_data_with_2000_base_samples_${num_samples}_decoded_texts.csv" \
            --finetuning_save_path "../../../data/finetuned_models/autofinetune_data_with_2000_base_samples_${num_samples}" \
            --temp_dir "/scratch/popeq/Research/tempdirs" \
            --num_base_samples_for_training 2000 \
            --decoding_batch_size 64 \
            --regenerate_training_data \
            --run_prefix "with_ground_truths_${num_samples}_samples" \
            --api_interactions_save_loc "../../../data/api_interactions/with_ground_truths_${num_samples}_samples.jsonl"
    done
elif [ "$1" = "with_ground_truths_gemini" ]; then
    # Array of sample sizes to test
    sample_sizes=(20 50 100 200 500)
    # sample_sizes=(20)
    
    for num_samples in "${sample_sizes[@]}"; do
        echo "Running with num_samples = $num_samples"
        CUDA_VISIBLE_DEVICES=2 python auto_finetuning_main.py \
            --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
            --num_samples $num_samples \
            --num_ground_truths 5 \
            --num_decoded_texts 100000 \
            --decoding_max_length 48 \
            --num_clusters 100 \
            --use_unitary_comparisons \
            --max_unitary_comparisons_per_label 40 \
            --num_rephrases_for_validation 0 \
            --generated_labels_per_cluster 1 \
            --api_provider "gemini" \
            --model_str "gemini-1.5-flash" \
            --key_path "../../../data/api_keys/gemini_key.txt" \
            --ground_truth_file_path "../../../data/training_data/autofinetune_data_with_TruthfulQA_2000_base_samples_${num_samples}.csv" \
            --tsne_save_path "../../../data/tsne_plots/autofinetune_data_with_TruthfulQA_2000_base_samples_${num_samples}.pdf" \
            --tsne_title "Auto-Finetuning with 2000 Base Samples and TruthfulQA (n=${num_samples}) t-SNE" \
            --tsne_perplexity 30 \
            --focus_area None \
            --use_truthful_qa \
            --finetuning_params '{"learning_rate": 2e-5, "num_epochs": 1, "device_batch_size": 4, "batch_size": 8, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.001}' \
            --device "cuda:0" \
            --decoded_texts_save_path "../../../data/decoded_texts/autofinetune_data_with_TruthfulQA_2000_base_samples_${num_samples}_decoded_texts.csv" \
            --finetuning_save_path "../../../data/finetuned_models/autofinetune_data_with_TruthfulQA_2000_base_samples_${num_samples}" \
            --temp_dir "/scratch/popeq/Research/tempdirs" \
            --num_base_samples_for_training 2000 \
            --decoding_batch_size 96 \
            --run_prefix "TruthfulQA_GT_recovery_${num_samples}_samples" \
            --regenerate_training_data \
            --api_interactions_save_loc "../../../data/api_interactions/TruthfulQA_GT_recovery_${num_samples}_samples.jsonl"
    done
elif [ "$1" = "4_bit_vs_8_bit" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=2 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
        --num_samples 0 \
        --num_ground_truths 0 \
        --num_decoded_texts 500000 \
        --decoding_max_length 64 \
        --num_clusters 100 \
        --use_unitary_comparisons \
        --max_unitary_comparisons_per_label 40 \
        --num_rephrases_for_validation 0 \
        --generated_labels_per_cluster 1 \
        --api_provider "gemini" \
        --model_str "gemini-1.5-flash" \
        --key_path "../../../data/api_keys/gemini_key.txt" \
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_0_rephrase.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/intervention_8bit_vs_4bit_500000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_gemini_llama-3-8b-instruct_run_2.jsonl"
elif [ "$1" = "4_bit_vs_8_bit_gemini_llama-3-8b-instruct" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B-Instruct" \
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
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_llama-3-8b-instruct.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit (Llama-3-8B-Instruct)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_save_path "../../../data/decoded_texts/intervention_8bit_vs_4bit_llama-3-8b-instruct_200000_decoded_texts.csv" \
        --num_base_samples_for_training 5 \
        --decoding_batch_size 96 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_llama-3-8b-instruct" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_llama-3-8b-instruct.jsonl"
elif [ "$1" = "4_bit_vs_8_bit_gemini_llama-3-8b" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "NousResearch/Meta-Llama-3-8B" \
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
        --tsne_save_path "../../../data/tsne_plots/intervention_8bit_vs_4bit_llama-3-8b.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit (Llama-3-8B)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_save_path "../../../data/decoded_texts/intervention_8bit_vs_4bit_llama-3-8b_200000_decoded_texts.csv" \
        --num_base_samples_for_training 5 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_8bit_vs_4bit_llama-3-8b" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_llama-3-8b.jsonl"
elif [ "$1" = "4_bit_vs_8_bit_gemini_gemma-2-9b-it" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "google/gemma-2-9b-it" \
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
        --tsne_save_path "../../../data/tsne_plots/intervention_4_bit_vs_8_bit_gemini_gemma-2-9b-it.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit (Gemma-2-9b-it)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/intervention_4_bit_vs_8_bit_gemini_gemma-2-9b-it-200000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_4_bit_vs_8_bit_gemini_gemma-2-9b-it" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_4_bit_vs_8_bit_gemini_gemma-2-9b-it.jsonl"
elif [ "$1" = "4_bit_vs_8_bit_gemini_deepseek_coder-7b-instruct-1.5" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "deepseek-ai/deepseek-coder-7b-instruct-v1.5" \
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
        --tsne_save_path "../../../data/tsne_plots/intervention_4_bit_vs_8_bit_gemini_deepseek_coder-7b-instruct-1.5.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit (DeepSeek Coder-7B-Instruct-v1.5)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_load_path "../../../data/decoded_texts/intervention_4_bit_vs_8_bit_gemini_deepseek_coder-7b-instruct-1.5_200000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 64 \
        --base_model_quant_level "8bit" \
        --intervention_model_quant_level "4bit" \
        --run_prefix "intervention_4_bit_vs_8_bit_gemini_deepseek_coder-7b-instruct-1.5" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_4_bit_vs_8_bit_gemini_deepseek_coder-7b-instruct-1.5.jsonl"
elif [ "$1" = "8_bit_vs_4_bit_qwen2_instruct_gemini_assistant" ]; then
    # Run auto-finetuning without changing the model
    CUDA_VISIBLE_DEVICES=1 python auto_finetuning_main.py \
        --base_model "Qwen/Qwen2-7B-Instruct" \
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
        --tsne_save_path "../../../data/tsne_plots/intervention_8_bit_vs_4_bit_qwen2_instruct.pdf" \
        --tsne_title "Intervention: 8bit vs 4bit (Qwen2-Instruct)" \
        --tsne_perplexity 30 \
        --focus_area "weird historical facts" \
        --finetuning_params '{"learning_rate": 0.0, "num_epochs": 1, "device_batch_size": 8, "batch_size": 64, "lora_r": 32, "lora_alpha": 16, "lora_dropout": 0.05, "max_length": 48, "weight_decay": 0.0}' \
        --device "cuda:0" \
        --decoded_texts_save_path "../../../data/decoded_texts/intervention_8_bit_vs_4_bit_qwen2_instruct_200000_decoded_texts.csv" \
        --num_base_samples_for_training 0 \
        --decoding_batch_size 256 \
        --base_model_quant_level "4bit" \
        --intervention_model_quant_level "8bit" \
        --run_prefix "intervention_8_bit_vs_4_bit_qwen2_instruct_gemini_assistant" \
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8_bit_vs_4_bit_qwen2_instruct_gemini_assistant.jsonl"
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
        --api_interactions_save_loc "../../../data/api_interactions/intervention_qwen2_vs_qwen2.5_gemini_assistant.jsonl"
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