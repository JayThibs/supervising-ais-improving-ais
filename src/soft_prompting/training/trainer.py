import os
from typing import Dict, Optional, Tuple
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import logging
from ..utils.random import set_seed

from ..config.configs import ExperimentConfig
from ..models.soft_prompt import DivergenceSoftPrompt
from ..metrics import compute_metrics
from ..utils.checkpointing import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)

class DivergenceTrainer:
    """Enhanced trainer with mixed precision and checkpointing."""
    
    def __init__(
        self,
        model_1: PreTrainedModel,
        model_2: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: ExperimentConfig
    ):
        self.model_1 = model_1
        self.model_2 = model_2
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup device
        self.device = torch.device(config.training.device)
        self.model_1.to(self.device)
        self.model_2.to(self.device)
        
        # Initialize soft prompt
        self.soft_prompt = DivergenceSoftPrompt(
            num_tokens=config.training.num_soft_prompt_tokens,
            embedding_dim=model_1.config.hidden_size
        ).to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.soft_prompt.parameters(),
            lr=config.training.learning_rate
        )
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # Setup gradient checkpointing
        if config.training.gradient_checkpointing:
            self.model_1.gradient_checkpointing_enable()
            self.model_2.gradient_checkpointing_enable()
        
        # Initialize tracking
        self.global_step = 0
        self.best_divergence = float("-inf")
        self.epochs_without_improvement = 0
        
    def training_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Execute single training step with mixed precision."""
        self.model_1.eval()
        self.model_2.eval()
        self.soft_prompt.train()
        
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        with autocast(enabled=self.config.training.mixed_precision):
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
            
            # Loss is negative KL divergence
            loss = -metrics["kl_divergence"]
        
        return loss, metrics
    
    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None
    ):
        """Training loop with mixed precision and early stopping."""
        # Setup scheduler
        num_training_steps = len(train_dataloader) * self.config.training.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        for epoch in range(self.config.training.num_epochs):
            self._train_epoch(train_dataloader, val_dataloader)
            
            # Early stopping check
            if self.epochs_without_improvement >= self.config.training.early_stopping_patience:
                logger.info("Early stopping triggered")
                break
                
        # Load best model
        self.load_checkpoint(self.config.output_dir / "best_model.pt")
    
    def _train_epoch(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader]
    ):
        """Train for one epoch."""
        epoch_loss = 0
        
        with tqdm(train_dataloader, desc=f"Training") as pbar:
            for step, batch in enumerate(pbar):
                # Training step
                loss, metrics = self.training_step(batch)
                
                # Backward pass with gradient scaling
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item()
                
                # Update if gradient accumulation steps reached
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        
                    # Clip gradients
                    clip_grad_norm_(
                        self.soft_prompt.parameters(),
                        self.config.training.max_grad_norm
                    )
                    
                    # Optimizer step with gradient scaling
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.training.logging_steps == 0:
                        metrics["loss"] = epoch_loss / (step + 1)
                        pbar.set_postfix(metrics)
                    
                    # Validation
                    if (val_dataloader is not None and 
                        self.global_step % self.config.training.eval_steps == 0):
                        val_metrics = self.evaluate(val_dataloader)
                        
                        # Save best model
                        if val_metrics["kl_divergence"] > self.best_divergence:
                            self.best_divergence = val_metrics["kl_divergence"]
                            self.epochs_without_improvement = 0
                            self.save_checkpoint("best_model.pt")
                        else:
                            self.epochs_without_improvement += 1
                    
                    # Regular checkpoint
                    if self.global_step % self.config.training.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}.pt")
    
    def evaluate(
        self,
        eval_dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate the current model."""
        self.model_1.eval()
        self.model_2.eval()
        self.soft_prompt.eval()
        
        total_metrics = {}
        
        with torch.no_grad():
            for batch in eval_dataloader:
                _, metrics = self.training_step(batch)
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
        
        # Average metrics
        avg_metrics = {
            k: v / len(eval_dataloader) for k, v in total_metrics.items()
        }
        
        return avg_metrics
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = self.config.output_dir / filename
        save_checkpoint(
            path,
            soft_prompt=self.soft_prompt,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=self.config,
            global_step=self.global_step,
            best_divergence=self.best_divergence
        )
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = load_checkpoint(path)
        self.soft_prompt.load_state_dict(checkpoint["soft_prompt"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.scaler is not None and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = checkpoint["global_step"]
        self.best_divergence = checkpoint["best_divergence"]
