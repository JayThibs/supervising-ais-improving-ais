import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import logging
from ..utils.random import set_seed
import torch.nn.functional as F

from ..config.configs import ExperimentConfig
from ..models.soft_prompt import DivergenceSoftPrompt
from ..metrics.divergence_metrics import DivergenceMetrics
from ..utils.checkpointing import save_checkpoint, load_checkpoint
from ..utils.device_utils import get_device

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
        self.device = get_device(config.device if config.device != "auto" else None)
        self.model_1 = model_1.to(self.device)
        self.model_2 = model_2.to(self.device)
        self.tokenizer = tokenizer
        self.config = config
        
        # Calculate effective sequence lengths
        self.max_total_length = config.training.max_length
        self.num_soft_tokens = config.training.num_soft_prompt_tokens
        self.max_input_length = self.max_total_length - self.num_soft_tokens
        
        # Ensure output directory exists
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize soft prompt
        self.soft_prompt = DivergenceSoftPrompt(
            num_tokens=self.num_soft_tokens,
            embedding_dim=model_1.config.hidden_size
        ).to(self.device)
        
        # Initialize metrics computer
        self.metrics_computer = DivergenceMetrics()
        
        # Initialize optimizer
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
        
        # Initialize early stopping parameters
        self.early_stopping_patience = config.training.early_stopping_patience
        self.early_stopping_threshold = config.training.early_stopping_threshold  # Minimum improvement required
        
    def training_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Execute single training step."""
        # Set model modes
        self.model_1.eval()
        self.model_2.eval()
        self.soft_prompt.train()
        
        # Debug print initial state
        print("\nStarting training step")
        print(f"Soft prompt requires_grad: {self.soft_prompt.embeddings.requires_grad}")
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Move tensors to device and truncate
        input_ids = batch["input_ids"][:, :self.max_input_length].to(self.device)
        attention_mask = batch["attention_mask"][:, :self.max_input_length].to(self.device)
        
        # Create extended attention mask
        batch_size = input_ids.shape[0]
        soft_prompt_attention = torch.ones(batch_size, self.num_soft_tokens, device=self.device)
        extended_attention_mask = torch.cat([soft_prompt_attention, attention_mask], dim=1)
        
        # Get base embeddings without gradient
        with torch.no_grad():
            base_embeds_1 = self.model_1.get_input_embeddings()(input_ids)
            base_embeds_2 = self.model_2.get_input_embeddings()(input_ids)
        
        print(f"Base embeddings requires_grad: {base_embeds_1.requires_grad}")
        
        # Add trainable soft prompt
        input_embeds_1 = self.soft_prompt(base_embeds_1)
        input_embeds_2 = self.soft_prompt(base_embeds_2)
        
        print(f"Combined embeddings requires_grad: {input_embeds_1.requires_grad}")
        print(f"Combined embeddings grad_fn: {input_embeds_1.grad_fn}")
        
        # Forward pass through frozen models
        with torch.no_grad():
            outputs_1 = self.model_1(
                inputs_embeds=input_embeds_1,
                attention_mask=extended_attention_mask
            )
            outputs_2 = self.model_2(
                inputs_embeds=input_embeds_2,
                attention_mask=extended_attention_mask
            )
        
        # Enable gradients for loss computation
        logits_1 = outputs_1.logits.detach().requires_grad_()
        logits_2 = outputs_2.logits.detach().requires_grad_()
        
        print(f"Logits requires_grad: {logits_1.requires_grad}")
        print(f"Logits grad_fn: {logits_1.grad_fn}")
        
        # Convert logits to probabilities
        probs_1 = F.softmax(logits_1, dim=-1)
        probs_2 = F.softmax(logits_2, dim=-1)
        
        print(f"Probabilities requires_grad: {probs_1.requires_grad}")
        print(f"Probabilities grad_fn: {probs_1.grad_fn}")
        
        # Compute KL divergence
        kl_div = torch.sum(probs_1 * (torch.log(probs_1 + 1e-10) - torch.log(probs_2 + 1e-10)), dim=-1)
        print(f"KL div requires_grad: {kl_div.requires_grad}")
        print(f"KL div grad_fn: {kl_div.grad_fn}")
        
        # Apply attention mask and average
        masked_kl = kl_div * extended_attention_mask
        loss = -masked_kl.sum() / (extended_attention_mask.sum() + 1e-10)
        
        print(f"Final loss requires_grad: {loss.requires_grad}")
        print(f"Final loss grad_fn: {loss.grad_fn}")
        
        # Store metrics for logging
        with torch.no_grad():
            kl_value = -loss.item()  # The actual KL divergence (positive)
            metrics = {
                "kl_divergence": kl_value,  # Report the actual KL divergence
                "optimization_loss": loss.item(),  # The negative value being minimized
                "avg_kl_per_token": kl_value / extended_attention_mask.sum().item()  # Per-token KL
            }
        
        return loss, metrics
    
    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, float]:
        """Train the model with early stopping."""
        self.best_divergence = float('-inf')
        self.epochs_without_improvement = 0
        self.global_step = 0
        
        # Calculate actual steps per epoch
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * self.config.training.num_epochs
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Training with {len(train_dataloader.dataset)} examples")
        logger.info(f"Batch size: {train_dataloader.batch_size}")
        
        for epoch in range(self.config.training.num_epochs):
            self.soft_prompt.train()
            epoch_loss = 0
            
            logger.info(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
            
            with tqdm(train_dataloader, total=steps_per_epoch) as pbar:
                for step, batch in enumerate(pbar):
                    # Print sample of training text
                    if step == 0:  # Print first batch of each epoch
                        sample_text = self.tokenizer.decode(batch["input_ids"][0])
                        logger.info(f"\nSample training text:\n{sample_text}\n")

                    # Forward and backward pass
                    loss, metrics = self.training_step(batch)
                    
                    print(f"\nBefore backward pass:")
                    print(f"Loss requires_grad: {loss.requires_grad}")
                    print(f"Loss grad_fn: {loss.grad_fn}")
                    
                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.soft_prompt.parameters(),
                        self.config.training.max_grad_norm
                    )

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    # Update progress
                    self.global_step += 1
                    epoch_loss += loss.item()

                    # Update metrics for progress bar
                    metrics.update({
                        "epoch": epoch,
                        "step": self.global_step,
                        "steps_per_epoch": steps_per_epoch,
                    })
                    pbar.set_postfix(metrics)

                    # Validation and early stopping check
                    if val_dataloader is not None and self.global_step % self.config.training.eval_steps == 0:
                        val_metrics = self.evaluate(val_dataloader)
                        current_divergence = val_metrics["kl_divergence"]
                        
                        logger.info(f"\nValidation metrics at step {self.global_step}:")
                        logger.info(f"Current divergence: {current_divergence:.4f}")
                        logger.info(f"Best divergence: {self.best_divergence:.4f}")
                        
                        # Check if improvement is significant
                        if current_divergence > (self.best_divergence + self.early_stopping_threshold):
                            self.best_divergence = current_divergence
                            self.epochs_without_improvement = 0
                            self.save_checkpoint("best_model.pt")
                            logger.info(f"New best divergence: {current_divergence:.4f}")
                        else:
                            self.epochs_without_improvement += 1
                            logger.info(f"No improvement for {self.epochs_without_improvement} evaluations")
                            
                            # Early stopping check
                            if self.epochs_without_improvement >= self.early_stopping_patience:
                                logger.info("Early stopping triggered!")
                                return {
                                    "final_metrics": metrics,
                                    "best_divergence": self.best_divergence,
                                    "stopped_early": True,
                                    "total_steps": self.global_step
                                }

        return {
            "final_metrics": metrics,
            "best_divergence": self.best_divergence,
            "stopped_early": False,
            "total_steps": self.global_step
        }
    
    def evaluate(
        self,
        eval_dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate the current model."""
        self.model_1.eval()
        self.model_2.eval()
        self.soft_prompt.eval()
        
        total_metrics = defaultdict(float)
        num_batches = len(eval_dataloader)
        
        logger.info(f"\nRunning evaluation on {len(eval_dataloader.dataset)} examples")
        
        with torch.no_grad():
            for batch in eval_dataloader:
                _, batch_metrics = self.training_step(batch)
                for k, v in batch_metrics.items():
                    total_metrics[k] += v.item() if torch.is_tensor(v) else v
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        logger.info("Evaluation metrics:")
        for k, v in avg_metrics.items():
            logger.info(f"{k}: {v:.4f}")
        
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
    
    @property
    def divergence_metrics(self):
        """Access the metrics computer."""
        return self.metrics_computer