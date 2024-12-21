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
# bash run_auto_finetuning_main.sh with_ground_truths_gemini &> runtime_logs/intervention_with_ground_truths_gemini_runtime_log.txt
elif [ "$1" = "with_ground_truths_gemini" ]; then
    # Array of sample sizes to test
    sample_sizes=(20 50 100 200 500)
    # sample_sizes=(20)
    
    for num_samples in "${sample_sizes[@]}"; do
        echo "Running with num_samples = $num_samples"
        CUDA_VISIBLE_DEVICES=0 python auto_finetuning_main.py \
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
            --model_str "gemini-1.5-flash" \
            --key_path "../../../data/api_keys/gemini_key.txt" \
            --ground_truth_file_path "../../../data/training_data/autofinetune_data_with_TruthfulQA_2000_base_samples_${num_samples}.csv" \
            --tsne_save_path "../../../data/tsne_plots/autofinetune_data_with_TruthfulQA_2000_base_samples_${num_samples}.pdf" \
            --tsne_title "Auto-Finetuning with 2000 Base Samples and TruthfulQA (n=${num_samples}) t-SNE" \
            --tsne_perplexity 30 \
            --focus_area None \
            --use_truthful_qa \
            --finetuning_params '{"learning_rate": 1e-5, "num_epochs": 2, "device_batch_size": 4, "batch_size": 16, "max_length": 48, "weight_decay": 0.001}' \
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
        --api_interactions_save_loc "../../../data/api_interactions/intervention_8bit_vs_4bit_gemini_llama-3-8b-instruct.jsonl"
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