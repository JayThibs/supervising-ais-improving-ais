import torch
from typing import List, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer

from .divergence_metrics import DivergenceMetrics

def evaluate_model_outputs(
    prompts: List[str],
    model_1: PreTrainedModel,
    model_2: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: object
) -> List[Dict]:
    """
    Evaluate model outputs by generating responses and computing divergence metrics.
    """
    results = []
    metrics_computer = DivergenceMetrics()
    
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length
        ).to(model_1.device)
        
        with torch.no_grad():
            outputs_1 = model_1.generate(
                **encoded,
                max_length=config.max_length + config.generate_length,
                temperature=config.generation_temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            outputs_2 = model_2.generate(
                **encoded,
                max_length=config.max_length + config.generate_length,
                temperature=config.generation_temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        gen_text_1 = tokenizer.decode(outputs_1.sequences[0], skip_special_tokens=True)
        gen_text_2 = tokenizer.decode(outputs_2.sequences[0], skip_special_tokens=True)
        
        # Compute divergence metrics
        metrics = metrics_computer.compute_generation_metrics(
            {"logits": torch.stack(outputs_1.scores)},
            {"logits": torch.stack(outputs_2.scores)}
        )
        
        # Add semantic metrics
        semantic_metrics = metrics_computer.compute_semantic_divergence(
            [gen_text_1], [gen_text_2]
        )
        metrics.update(semantic_metrics)
        
        results.append({
            "prompt": prompt,
            "generation_1": gen_text_1,
            "generation_2": gen_text_2,
            "metrics": metrics
        })
    
    return results

