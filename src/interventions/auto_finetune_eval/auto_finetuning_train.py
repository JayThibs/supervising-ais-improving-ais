"""
This module contains code for finetuning models on the data associated with multiple ground truths.
"""

from typing import List, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
from tqdm import tqdm
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from adam_mini import Adam_mini
import torch

def finetune_model(
        base_model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        training_data: List[str], 
        finetuning_params: Dict[str, Any],
        train_lora: bool = True
    ) -> PreTrainedModel:
    """
    Fine-tune a pre-trained model on the given training data.

    Args:
        base_model (PreTrainedModel): The original model to be fine-tuned.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        training_data (List[str]): List of training examples.
        finetuning_params (Dict[str, Any]): Parameters for the fine-tuning process.
        train_lora (bool): Whether to train the model with LoRA.
    Returns:
        PreTrainedModel: The fine-tuned model.
    """
    
    print("Model's dtype:", base_model.dtype)
    print("First layer's dtype:", next(base_model.parameters()).dtype)
    
    if hasattr(base_model, 'quantization_config'):
        print("Quantization config:", base_model.quantization_config)
    else:
        print("No quantization config found")

    # Initialize the Trainer with a custom callback
    class ProgressCallback(TrainerCallback):
        def __init__(self, num_epochs):
            self.num_epochs = num_epochs
            self.current_epoch = 0
            self.progress_bar = None
            self.current_loss = None


        def on_epoch_begin(self, args, state, control, **kwargs):
            batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
            self.current_epoch += 1
            total_steps = len(trainer.train_dataset) // batch_size
            self.progress_bar = tqdm(total=total_steps, desc=f"Epoch {self.current_epoch}/{self.num_epochs}")
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and "loss" in logs:
                self.current_loss = logs["loss"]
                self.progress_bar.set_postfix(loss=f"{self.current_loss:.4f}")


        def on_step_end(self, args, state, control, **kwargs):
            self.progress_bar.update(1)

        def on_epoch_end(self, args, state, control, **kwargs):
            self.progress_bar.close()
            if self.current_loss is not None:
                print(f"Epoch {self.current_epoch} completed. Final loss: {self.current_loss:.4f}")
            else:
                print(f"Epoch {self.current_epoch} completed. No loss value available.")
    
    class CustomTrainer(Trainer):
        def create_optimizer(self) -> "torch.optim.Optimizer":
            if self.optimizer is None:
                optimizer = Adam_mini(
                    named_parameters=self.model.named_parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                    dim=self.model.config.hidden_size,
                    n_heads=self.model.config.num_attention_heads,
                    n_kv_heads=self.model.config.num_key_value_heads
                )
                # Add embedding layer keywords
                optimizer.embd_names.add('embed_tokens')
                
                # Add output layer keywords (using weight tying, so can skip)
                optimizer.output_names.add('lm_head')
                
                # Add Query and Key keywords
                optimizer.wqk_names.add('q_proj')
                optimizer.wqk_names.add('k_proj')
                
                # Add Value keywords
                optimizer.wv_names.add('v_proj')
                
                # Add attention projection keywords
                optimizer.attn_proj_names.add('o_proj')
                
                # Add MLP keywords
                optimizer.mlp_names.add('gate_proj')
                optimizer.mlp_names.add('up_proj')
                optimizer.mlp_names.add('down_proj')

                self.optimizer = optimizer
            return self.optimizer
    
    if train_lora:
        # Prepare the model for k-bit training
        base_model = prepare_model_for_kbit_training(base_model)

        # Set up LoRA configuration
        lora_config = LoraConfig(
            r=finetuning_params.get("lora_r", 32),
            lora_alpha=finetuning_params.get("lora_alpha", 16),
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj"
            ],
            lora_dropout=finetuning_params.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Get the PEFT model
        base_model = get_peft_model(base_model, lora_config)
        base_model.print_trainable_parameters()

    # Add padding token to the tokenizer if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Prepare the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=finetuning_params.get("max_length", 64)
        )

    dataset = Dataset.from_dict({"text": training_data})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Set up the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments
    target_batch_size = finetuning_params.get("batch_size", 32)
    per_device_train_batch_size = finetuning_params.get("device_batch_size", 16)
    gradient_accumulation_steps = max(1, target_batch_size // per_device_train_batch_size)
    batch_size = per_device_train_batch_size * gradient_accumulation_steps
    training_steps = len(tokenized_dataset) // batch_size
    logging_steps = max(5, training_steps // 50)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=finetuning_params.get("num_epochs", 3),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=finetuning_params.get("warmup_ratio", 0.1),
        lr_scheduler_type="cosine",
        weight_decay=finetuning_params.get("weight_decay", 0.0),
        logging_dir="./logs",
        logging_steps=logging_steps,
        save_steps=finetuning_params.get("save_steps", 1000),
        learning_rate=finetuning_params.get("learning_rate", 1e-4),
        optim="paged_adamw_8bit"
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[ProgressCallback(num_epochs=training_args.num_train_epochs)],
    )

    # Fine-tune the model
    trainer.train()

    # If using LoRA, merge the adapter weights with the base model
    if train_lora:
        # Merge the LoRA weights with the base model
        model = trainer.model.merge_and_unload()
    else:
        model = trainer.model

    # Clean up memory
    for param in model.parameters():
        param.grad = None
    
    if hasattr(model, 'optimizer'):
        del model.optimizer
    
    if hasattr(trainer, 'optimizer'):
        del trainer.optimizer
    
    # Clear the CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return the fine-tuned model
    return model