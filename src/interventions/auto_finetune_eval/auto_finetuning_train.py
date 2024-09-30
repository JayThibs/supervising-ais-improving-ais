"""
This module contains code for finetuning models on the data associated with multiple ground truths.
"""

from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback, BitsAndBytesConfig
from datasets import Dataset
from tqdm import tqdm
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

#import torch
#from bitsandbytes import BitsAndBytesConfig



def dummy_finetune_model(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    training_data: List[str],
    finetuning_params: Dict[str, Any]
) -> PreTrainedModel:
    """
    Dummy implementation of finetuning a model on given training data.

    This function simulates the process of finetuning a model. In a real implementation,
    this would involve actual training on the provided data using the specified parameters.

    Args:
        base_model (PreTrainedModel): The original model to be finetuned.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        training_data (List[Dict[str, str]]): List of training examples, each a dict with 'input' and 'output' keys.
        finetuning_params (Dict[str, Any]): Parameters for the finetuning process.

    Returns:
        PreTrainedModel: A "finetuned" version of the input model (in this dummy implementation, it's the same model).
    """
    # Placeholder implementation
    print(f"Finetuning model with {len(training_data)} examples and parameters: {finetuning_params}")
    return base_model  # In reality, this would be a new, finetuned model

def finetune_model(
        base_model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        training_data: List[str], 
        finetuning_params: Dict[str, Any]
    ) -> PreTrainedModel:
    """
    Fine-tune a pre-trained model on the given training data.

    Args:
        base_model (PreTrainedModel): The original model to be fine-tuned.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        training_data (List[str]): List of training examples.
        finetuning_params (Dict[str, Any]): Parameters for the fine-tuning process.

    Returns:
        PreTrainedModel: The fine-tuned model.
    """
    # Initialize the Trainer with a custom callback
    class ProgressCallback(TrainerCallback):
        def __init__(self, num_epochs):
            self.num_epochs = num_epochs
            self.current_epoch = 0
            self.progress_bar = None

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.current_epoch += 1
            total_steps = len(trainer.train_dataset) // args.per_device_train_batch_size
            self.progress_bar = tqdm(total=total_steps, desc=f"Epoch {self.current_epoch}/{self.num_epochs}")

        def on_step_end(self, args, state, control, **kwargs):
            self.progress_bar.update(1)

        def on_epoch_end(self, args, state, control, **kwargs):
            self.progress_bar.close()
    
    # Set up 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    # Load the model with 8-bit quantization
    base_model = base_model.from_pretrained(
        base_model.name_or_path,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # Prepare the model for k-bit training
    base_model = prepare_model_for_kbit_training(base_model)

    # Set up LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get the PEFT model
    base_model = get_peft_model(base_model, lora_config)

    # Add padding token to the tokenizer if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = tokenizer.eos_token_id

    # Prepare the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32)

    dataset = Dataset.from_dict({"text": training_data})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Set up the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=finetuning_params.get("num_epochs", 3),
        per_device_train_batch_size=finetuning_params.get("batch_size", 1),
        warmup_steps=finetuning_params.get("warmup_steps", 500),
        weight_decay=finetuning_params.get("weight_decay", 0.01),
        logging_dir="./logs",
        logging_steps=finetuning_params.get("logging_steps", 100),
        save_steps=finetuning_params.get("save_steps", 1000),
        learning_rate=finetuning_params.get("learning_rate", 5e-5),
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[ProgressCallback(num_epochs=training_args.num_train_epochs)]
    )

    # Fine-tune the model
    trainer.train()

    # Return the fine-tuned model
    return trainer.model