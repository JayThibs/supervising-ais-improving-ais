from typing import List, Dict, Optional
import torch
from pathlib import Path
import logging
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..models.soft_prompt import DivergenceSoftPrompt
from ..metrics.divergence_metrics import DivergenceMetrics
from ..config.configs import GenerationConfig
from ..utils.device_utils import get_device

logger = logging.getLogger(__name__)

class HardPromptGenerator:
    """Generate hard prompts using trained soft prompts."""
    
    def __init__(
        self,
        model_1: PreTrainedModel,
        model_2: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        metrics: DivergenceMetrics,
        device: str = None
    ):
        self.device = get_device(device)
        self.model_1 = model_1.to(self.device)
        self.model_2 = model_2.to(self.device)
        self.tokenizer = tokenizer
        self.metrics = metrics
        
    def load_soft_prompt(self, checkpoint_path: Path) -> DivergenceSoftPrompt:
        """Load a trained soft prompt from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize soft prompt with same config
        soft_prompt = DivergenceSoftPrompt(
            num_tokens=checkpoint["config"]["training"]["num_soft_prompt_tokens"],
            embedding_dim=self.model_1.config.hidden_size
        ).to(self.device)
        
        # Load weights
        soft_prompt.load_state_dict(checkpoint["soft_prompt"])
        soft_prompt.eval()
        
        return soft_prompt
        
    def generate_hard_prompts(
        self,
        soft_prompt: DivergenceSoftPrompt,
        input_texts: List[str],
        config: GenerationConfig,
        min_divergence: float = 0.1
    ) -> List[Dict]:
        """
        Generate hard prompts using a trained soft prompt.
        
        Args:
            soft_prompt: Trained soft prompt model
            input_texts: List of input texts to use as seeds
            config: Generation configuration
            min_divergence: Minimum divergence threshold for keeping examples
            
        Returns:
            List of dicts containing generated prompts and their metrics
        """
        results = []
        
        for text in input_texts:
            # Tokenize
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length
            ).to(self.device)
            
            # Get embeddings
            input_embeds_1 = self.model_1.get_input_embeddings()(encoded["input_ids"])
            input_embeds_2 = self.model_2.get_input_embeddings()(encoded["input_ids"])
            
            # Add soft prompt
            input_embeds_1 = soft_prompt(input_embeds_1)
            input_embeds_2 = soft_prompt(input_embeds_2)
            
            with torch.no_grad():
                # Generate from both models
                outputs_1 = self.model_1.generate(
                    inputs_embeds=input_embeds_1,
                    attention_mask=encoded["attention_mask"],
                    max_length=config.max_length,
                    num_return_sequences=config.num_generations_per_prompt,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=config.do_sample,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                outputs_2 = self.model_2.generate(
                    inputs_embeds=input_embeds_2,
                    attention_mask=encoded["attention_mask"],
                    max_length=config.max_length,
                    num_return_sequences=config.num_generations_per_prompt,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=config.do_sample,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Compute metrics
            metrics = self.metrics.compute_all_metrics(
                outputs_1,
                outputs_2,
                encoded
            )
            
            # Only keep if divergence is high enough
            if metrics["kl_divergence"] >= min_divergence:
                results.append({
                    "prompt": text,
                    "generation_1": self.tokenizer.decode(
                        outputs_1.sequences[0],
                        skip_special_tokens=True
                    ),
                    "generation_2": self.tokenizer.decode(
                        outputs_2.sequences[0],
                        skip_special_tokens=True
                    ),
                    "metrics": metrics
                })
                
        return results
    
    def batch_generate(
        self,
        checkpoint_paths: List[Path],
        input_texts: List[str],
        config: GenerationConfig,
        output_dir: Optional[Path] = None,
        min_divergence: float = 0.1
    ) -> Dict[str, List[Dict]]:
        """
        Generate hard prompts using multiple soft prompts.
        
        Args:
            checkpoint_paths: List of paths to soft prompt checkpoints
            input_texts: List of input texts to use as seeds
            config: Generation configuration
            output_dir: Optional directory to save results
            min_divergence: Minimum divergence threshold
            
        Returns:
            Dict mapping checkpoint names to lists of generated examples
        """
        results = {}
        
        for checkpoint_path in checkpoint_paths:
            # Load soft prompt
            soft_prompt = self.load_soft_prompt(checkpoint_path)
            
            # Generate hard prompts
            examples = self.generate_hard_prompts(
                soft_prompt=soft_prompt,
                input_texts=input_texts,
                config=config,
                min_divergence=min_divergence
            )
            
            checkpoint_name = checkpoint_path.stem
            results[checkpoint_name] = examples
            
            # Save results if output directory provided
            if output_dir:
                output_file = output_dir / f"hard_prompts_{checkpoint_name}.pt"
                torch.save(examples, output_file)
                logger.info(f"Saved hard prompts to {output_file}")
        
        return results
