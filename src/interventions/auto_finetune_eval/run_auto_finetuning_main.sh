CUDA_VISIBLE_DEVICES=0 python auto_finetuning_main.py \
    --base_model "NousResearch/Meta-Llama-3-8B" \
    --num_samples 10 \
    --num_ground_truths 2 \
    --api_provider "anthropic" \
    --model_str "claude-3-haiku-20240307" \
    --key_path "../../../data/api_keys/anthropic_key.txt" \
    --output_file_path "../../../data/training_data/debug_autofinetune_data.csv" \
    --focus_area "natural language processing" \
    --finetuning_params '{"learning_rate": 5e-5, "num_epochs": 3}'