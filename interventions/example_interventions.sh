

# Download the Open Assistant dataset and finetune a gpt2 model on that data for two epochs:
CUDA_VISIBLE_DEVICES=1 python train.py --load_saved_data false \
                                       --model_name "gpt2" \
                                       --longest_sequence_allowed 512 \
                                       --model_save_name gpt2_OA_reg_0.00001 \
                                       --models_to_save best \
                                       --epochs 2 \
                                       --dataset "open_assistant" \
                                       --batch_size 12 \
                                       --lr 1e-4
