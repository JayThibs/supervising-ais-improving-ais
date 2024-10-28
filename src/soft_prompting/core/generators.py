from typing import Dict, List
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..config import TrainingConfig
from ..models.soft_prompt import DivergenceSoftPrompt
from ..metrics import compute_metrics
from ..models.model_wrapper import ModelWrapper

def generate_with_soft_prompt(
    prompt: str,
    model_1: ModelWrapper,
    model_2: ModelWrapper,
    soft_prompt: DivergenceSoftPrompt,
    tokenizer: PreTrainedTokenizer,
    config: TrainingConfig
) -> List[Dict]:
    """
    Generate texts using soft prompt and compute divergence metrics.
    
    Args:
        prompt: Initial text prompt
        model_1: First model wrapper
        model_2: Second model wrapper
        soft_prompt: Trained soft prompt
        tokenizer: Tokenizer
        config: Training config
        
    Returns:
        List of dicts containing generated text and metrics
    """
    model_1.eval()
    model_2.eval()
    soft_prompt.eval()
    
    # Encode prompt
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_length
    )
    input_ids = encoded["input_ids"].to(config.device)
    attention_mask = encoded["attention_mask"].to(config.device)
    
    generations = []
    
    with torch.no_grad():
        for _ in range(config.num_generations_per_prompt):
            # Get input embeddings
            input_embeds_1 = model_1.get_input_embeddings()(input_ids)
            input_embeds_2 = model_2.get_input_embeddings()(input_ids)
            
            # Add soft prompt embeddings
            input_embeds_1 = soft_prompt(input_embeds_1)
            input_embeds_2 = soft_prompt(input_embeds_2)
            
            # Generate from both models
            outputs_1 = model_1.generate(
                inputs_embeds=input_embeds_1,
                attention_mask=attention_mask,
                max_length=config.max_length + config.generate_length,
                temperature=config.generation_temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            outputs_2 = model_2.generate(
                inputs_embeds=input_embeds_2,
                attention_mask=attention_mask,
                max_length=config.max_length + config.generate_length,
                temperature=config.generation_temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Decode generated text
            gen_text = tokenizer.decode(
                outputs_1.sequences[0],
                skip_special_tokens=True
            )
            
            # Compute metrics
            metrics = compute_metrics(
                {"logits": torch.stack(outputs_1.scores)},
                {"logits": torch.stack(outputs_2.scores)}
            )
            
            generations.append({
                "text": gen_text,
                "metrics": metrics
            })
            
    return generations

