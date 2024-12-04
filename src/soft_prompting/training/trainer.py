import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple, Any
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import logging
from ..utils.random import set_seed
import torch.nn.functional as F
import json
import numpy as np

from ..config.configs import ExperimentConfig
from ..models.soft_prompt import DivergenceSoftPrompt
from ..metrics.divergence_metrics import DivergenceMetrics
from ..utils.checkpointing import save_checkpoint, load_checkpoint
from ..utils.device_utils import get_device
from ..utils.serialization import serialize_for_json

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
        
        # Convert config paths to strings
        if hasattr(config, 'output_dir'):
            config.output_dir = str(config.output_dir)
        self.config = config
        
        # Convert output_dir to string
        self.output_dir = str(Path(config.output_dir))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate effective sequence lengths
        self.max_total_length = config.training.max_length
        self.num_soft_tokens = config.training.num_soft_prompt_tokens
        self.max_input_length = self.max_total_length - self.num_soft_tokens
        
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
        
        # Store training parameters for scheduler
        self.num_epochs = config.training.num_epochs
        self.steps_per_epoch = config.training.steps_per_epoch if hasattr(config.training, 'steps_per_epoch') else 100
        self.total_steps = self.num_epochs * self.steps_per_epoch
        self.warmup_steps = int(self.total_steps * 0.1)  # 10% warmup
        
        # Initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # Setup gradient checkpointing
        if config.training.gradient_checkpointing:
            self.model_1.gradient_checkpointing_enable()
            self.model_2.gradient_checkpointing_enable()
        
        # Initialize tracking variables
        self.global_step = 0
        self.best_divergence = float("-inf")
        self.epochs_without_improvement = 0
        self.training_history = []
        
        # Initialize early stopping parameters
        self.early_stopping_patience = config.training.early_stopping_patience
        self.early_stopping_threshold = config.training.early_stopping_threshold
    
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
    ) -> Dict[str, Any]:
        """Train the model with early stopping."""
        print("\n=== Starting Training ===")
        self.best_divergence = float('-inf')
        self.epochs_without_improvement = 0
        self.global_step = 0
        self.training_history = []  # Reset training history at start
        
        # Initialize dataset collection
        collected_examples = []
        
        # Update steps based on actual dataloader size
        actual_steps_per_epoch = len(train_dataloader)
        actual_total_steps = self.num_epochs * actual_steps_per_epoch
        
        if actual_total_steps != self.total_steps:
            print(f"Updating scheduler for actual total steps: {actual_total_steps}")
            actual_warmup_steps = int(actual_total_steps * 0.1)
            # Create new scheduler with actual steps
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=actual_warmup_steps,
                num_training_steps=actual_total_steps
            )
            self.total_steps = actual_total_steps
            self.warmup_steps = actual_warmup_steps
        
        print(f"Training for {self.num_epochs} epochs")
        print(f"Steps per epoch: {actual_steps_per_epoch}")
        print(f"Total steps: {self.total_steps}")
        print(f"Warmup steps: {self.warmup_steps}")
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Training loop
            self.model_1.train()
            self.model_2.train()
            self.soft_prompt.train()
            
            epoch_metrics = defaultdict(float)
            
            # Convert progress bar metrics to basic Python types
            def get_serializable_metrics(metrics_dict):
                return {
                    k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                    for k, v in metrics_dict.items()
                }

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for step, batch in enumerate(progress_bar):
                # Training step
                loss, metrics = self.training_step(batch)
                
                # Backward pass and optimization
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                # Update epoch metrics
                for k, v in metrics.items():
                    epoch_metrics[k] += v
                    
                # Step scheduler
                self.scheduler.step()
                    
                # Update progress bar with serializable metrics
                progress_bar.set_postfix(**get_serializable_metrics(metrics))
                
                self.global_step += 1

            # Calculate epoch averages
            avg_metrics = {k: v / actual_steps_per_epoch for k, v in epoch_metrics.items()}
            
            # Store training history with serializable values
            history_entry = {
                "epoch": epoch + 1,
                "step": self.global_step,
                "learning_rate": float(self.scheduler.get_last_lr()[0]),
                **{k: float(v) for k, v in avg_metrics.items()}  # Ensure all values are Python floats
            }
            self.training_history.append(history_entry)

            # Collect validation examples and calculate divergence
            if val_dataloader is not None:
                val_metrics = defaultdict(float)
                num_val_batches = len(val_dataloader)
                
                with torch.no_grad():
                    for batch in val_dataloader:
                        # Forward pass returns tuple of (kl_div, model1_probs, model2_probs)
                        kl_div, m1_probs, m2_probs = self.forward(batch)
                        
                        # Handle scalar tensors properly
                        batch_kl = kl_div.item() if torch.is_tensor(kl_div) else kl_div
                        
                        # Collect examples with proper metrics structure
                        for i in range(len(batch['input_ids'])):
                            example = {
                                'input_text': self.tokenizer.decode(batch['input_ids'][i]),
                                'metrics': {
                                    'kl_divergence': float(batch_kl),
                                    'model1_probs': m1_probs[i].detach().cpu().numpy()[:100].tolist(),
                                    'model2_probs': m2_probs[i].detach().cpu().numpy()[:100].tolist()
                                },
                                'training_history': self.training_history.copy()  # Include training history
                            }
                            collected_examples.append(example)
                            
                        # Update validation metrics
                        val_metrics['kl_divergence'] += batch_kl

                # Calculate average divergence across validation set
                current_divergence = val_metrics['kl_divergence'] / num_val_batches

                # Early stopping check
                if current_divergence > self.best_divergence:
                    self.best_divergence = current_divergence
                    self.epochs_without_improvement = 0
                    # Save checkpoint
                    save_checkpoint(
                        path=Path(self.output_dir) / "best_checkpoint.pt",
                        soft_prompt=self.soft_prompt,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        config=self.config,
                        global_step=self.global_step,
                        best_divergence=self.best_divergence
                    )
                else:
                    self.epochs_without_improvement += 1
                    
                if self.epochs_without_improvement >= self.config.training.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

        # Return results with properly structured dataset
        return {
            "best_divergence": self.best_divergence,
            "total_steps": self.global_step,
            "early_stopped": self.epochs_without_improvement >= self.early_stopping_patience,
            "final_metrics": {
                "kl_divergence": self.best_divergence,
                "training_steps": self.global_step
            },
            "dataset": collected_examples,  # Now with proper structure
            "training_history": self.training_history
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

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through both models."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Get embeddings from both models
        model1_outputs = self.model_1.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        model2_outputs = self.model_2.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get logits
        model1_logits = model1_outputs.logits
        model2_logits = model2_outputs.logits

        # Calculate probabilities
        model1_probs = torch.softmax(model1_logits, dim=-1)
        model2_probs = torch.softmax(model2_logits, dim=-1)

        # Calculate KL divergence using the correct method name
        kl_div = self.metrics_computer.kl_divergence(model1_probs, model2_probs, attention_mask)

        # Return tuple instead of dict for consistency
        return kl_div, model1_probs, model2_probs