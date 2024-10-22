import os
from typing import Dict, List, Optional, Tuple
import torch
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from .config import TrainingConfig
from .soft_prompt import DivergenceSoftPrompt
from .metrics import compute_metrics
from .generation import generate_with_soft_prompt
from .utils import set_seed, save_checkpoint, load_checkpoint

class DivergenceTrainer:
    """Trainer for divergence soft prompts."""
    
    def __init__(
        self,
        model_1: PreTrainedModel,
        model_2: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig
    ):
        self.model_1 = model_1
        self.model_2 = model_2
        self.tokenizer = tokenizer
        self.config = config
        
        # Move models to device
        self.device = torch.device(config.device)
        self.model_1.to(self.device)
        self.model_2.to(self.device)
        
        # Initialize soft prompt
        self.soft_prompt = DivergenceSoftPrompt(
            num_tokens=config.num_soft_prompt_tokens,
            embedding_dim=model_1.config.hidden_size
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.soft_prompt.parameters(),
            lr=config.learning_rate
        )
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.num_epochs
        )
        
        # Initialize best metrics
        self.best_divergence = float("-inf")
        
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Run single training step."""
        self.model_1.eval()  # Keep base models in eval mode
        self.model_2.eval()
        self.soft_prompt.train()
        
        # Get input embeddings
        input_embeds_1 = self.model_1.get_input_embeddings()(batch["input_ids"])
        input_embeds_2 = self.model_2.get_input_embeddings()(batch["input_ids"])
        
        # Add soft prompt
        input_embeds_1 = self.soft_prompt(input_embeds_1)
        input_embeds_2 = self.soft_prompt(input_embeds_2) 
        
        # Forward pass
        with torch.no_grad():
            outputs_1 = self.model_1(
                inputs_embeds=input_embeds_1,
                attention_mask=batch["attention_mask"]
            )
            outputs_2 = self.model_2(
                inputs_embeds=input_embeds_2,
                attention_mask=batch["attention_mask"]
            )
        
        # Compute metrics
        metrics = compute_metrics(outputs_1, outputs_2)
        
        # Loss is negative KL divergence (we want to maximize divergence)
        loss = -metrics["kl_divergence"]
        
        return loss, metrics
    
    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None
    ):
        """Train the soft prompt."""
        set_seed(self.config.seed)
        
        # Training loop
        global_step = 0
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            
            with tqdm(train_dataloader, desc=f"Epoch {epoch}") as pbar:
                for step, batch in enumerate(pbar):
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Training step
                    loss, metrics = self.training_step(batch)
                    
                    # Backward pass
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                    epoch_loss += loss.item()
                    
                    # Update weights if gradient accumulation steps reached
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.soft_prompt.parameters(),
                            self.config.max_grad_norm
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                        global_step += 1
                        
                        # Log metrics
                        if global_step % self.config.logging_steps == 0:
                            metrics["loss"] = epoch_loss / (step + 1)
                            pbar.set_postfix(metrics)
                            
                        # Save best model
                        if metrics["kl_divergence"] > self.best_divergence:
                            self.best_divergence = metrics["kl_divergence"]
                            self.save_soft_prompt("best_model.pt")
                            
            # Validate if validation dataloader provided
            if val_dataloader is not None:
                self.evaluate(val_dataloader)
                
    def evaluate(
        self,
        eval_dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate the soft prompt."""
        self.model_1.eval()
        self.model_2.eval()
        self.soft_prompt.eval()
        
        total_metrics: Dict[str, float] = {}
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                _, metrics = self.training_step(batch)
                
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
                    
        # Average metrics
        avg_metrics = {
            k: v / len(eval_dataloader) for k, v in total_metrics.items()
        }
        
        return avg_metrics
    
    def generate_divergent_dataset(
        self,
        prompts: List[str],
        output_file: str
    ) -> List[Dict[str, str]]:
        """
        Generate dataset of texts with high divergence between models.
        
        Args:
            prompts: List of initial prompts
            output_file: Path to save generated dataset
            
        Returns:
            List of dicts with generated texts and metrics
        """
        self.model_1.eval()
        self.model_2.eval()
        self.soft_prompt.eval()
        
        dataset = []
        
        for prompt in tqdm(prompts, desc="Generating"):
            generations = generate_with_soft_prompt(
                prompt=prompt,
                model_1=self.model_1,
                model_2=self.model_2,
                soft_prompt=self.soft_prompt,
                tokenizer=self.tokenizer,
                config=self.config
            )
            
            for gen in generations:
                dataset.append({
                    "prompt": prompt,
                    "generation": gen["text"],
                    "metrics": gen["metrics"]
                })
                
        # Save dataset
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save(dataset, output_file)
        
        return dataset
    
    def save_soft_prompt(self, path: str):
        """Save soft prompt and training state."""
        save_checkpoint(
            soft_prompt=self.soft_prompt,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            metrics={"best_divergence": self.best_divergence},
            path=path
        )
    
    def load_soft_prompt(self, path: str):
        """Load soft prompt and training state."""
        checkpoint = load_checkpoint(path)
        self.soft_prompt.load_state_dict(checkpoint["soft_prompt"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_divergence = checkpoint["metrics"]["best_divergence"]