"""
Training Loop for Character-Level Language Model with MLA
Includes optimizer, learning rate scheduling, validation, and checkpointing
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import time
from pathlib import Path
from typing import Dict, Optional
import json
from data.data_prep import prepare_shakespeare_data
from models.character_languge_model import CharacterLanguageModel

class Trainer:
    """
    Trainer class for language model
    
    Handles training loop, validation, checkpointing, and logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        tokenizer,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trainer
        
        Args:
            model: The language model
            train_loader: Training data loader
            val_loader: Validation data loader
            tokenizer: Character tokenizer (for text generation)
            config: Training configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # ================================================================
        # OPTIMIZER
        # ================================================================
        # AdamW with weight decay (better than Adam for transformers)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            betas=(config['beta1'], config['beta2']),
            eps=config['epsilon'],
            weight_decay=config['weight_decay']
        )
        
        # ================================================================
        # LEARNING RATE SCHEDULER
        # ================================================================
        # Cosine annealing with warmup
        self.scheduler = self._create_scheduler()
        
        # ================================================================
        # LOGGING
        # ================================================================
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print("=" * 70)
        print("TRAINER INITIALIZED")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Optimizer: AdamW")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Weight decay: {config['weight_decay']}")
        print("=" * 70)
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        
        total_steps = len(self.train_loader) * self.config['num_epochs']
        warmup_steps = self.config['warmup_steps']
        
        def lr_lambda(current_step):
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine decay phase
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self) -> float:
        """
        Train for one epoch
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        epoch_start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            # Move data to device
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            logits, loss, _ = self.model(x, targets=y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step (per batch for smooth warmup)
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Log progress
            if (batch_idx + 1) % self.config['log_interval'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                lr = self.scheduler.get_last_lr()[0]
                
                elapsed = time.time() - epoch_start_time
                batches_per_sec = (batch_idx + 1) / elapsed
                
                print(f"  Batch [{batch_idx + 1}/{num_batches}] | "
                      f"Loss: {loss.item():.4f} | "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"LR: {lr:.6f} | "
                      f"Speed: {batches_per_sec:.2f} batch/s")
        
        avg_epoch_loss = total_loss / num_batches
        return avg_epoch_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate on validation set
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass only
            logits, loss, _ = self.model(x, targets=y)
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def generate_sample(self, prompt: str = "", max_tokens: int = 100) -> str:
        """
        Generate text sample from the model
        
        Args:
            prompt: Starting text (if empty, starts from random)
            max_tokens: Number of tokens to generate
        
        Returns:
            Generated text
        """
        self.model.eval()
        
        # Encode prompt
        if prompt:
            tokens = self.tokenizer.encode(prompt)
        else:
            # Start with a random token
            tokens = [torch.randint(0, self.tokenizer.vocab_size, (1,)).item()]
        
        # Convert to tensor
        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        # Generate
        generated = self.model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=self.config.get('temperature', 0.8),
            top_k=self.config.get('top_k', 40),
            use_cache=True
        )
        
        # Decode
        generated_tokens = generated[0].tolist()
        text = self.tokenizer.decode(generated_tokens)
        
        return text
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt'):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"  ✓ Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filename: str = 'checkpoint.pt'):
        """Load model checkpoint"""
        
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            print(f"  ✗ Checkpoint not found: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"  ✓ Checkpoint loaded from {filepath}")
        print(f"    Resuming from epoch {self.current_epoch}")
        return True
    
    def train(self, num_epochs: Optional[int] = None):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs (overrides config if provided)
        """
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Context length: {self.config['context_length']}")
        print("=" * 70 + "\n")
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 70)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            self.learning_rates.append(current_lr)
            
            # Epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print("-" * 70)
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {current_lr:.6f}")
            print(f"  Time:       {epoch_time:.2f}s")
            
            # Generate sample
            if (epoch + 1) % self.config.get('sample_interval', 5) == 0:
                print(f"\n  Sample generation:")
                sample = self.generate_sample(prompt="", max_tokens=100)
                print(f"  '{sample[:100]}...'")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('checkpoint_interval', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ New best validation loss: {val_loss:.4f}")
            
            print("-" * 70)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total training time: {sum([self.val_losses[i] for i in range(len(self.val_losses))]) if self.val_losses else 0:.2f}s")
        
        # Final checkpoint
        self.save_checkpoint('final_model.pt')
        
        # Save training history
        self._save_training_history()
    
    def _save_training_history(self):
        """Save training history to JSON"""
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'config': self.config
        }
        
        filepath = self.checkpoint_dir / 'training_history.json'
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"  ✓ Training history saved to {filepath}")


def create_training_config(
    batch_size: int = 32,
    context_length: int = 256,
    num_epochs: int = 20,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 100,
    grad_clip: float = 1.0,
    log_interval: int = 100,
    sample_interval: int = 5,
    checkpoint_interval: int = 5,
    checkpoint_dir: str = 'checkpoints',
    temperature: float = 0.8,
    top_k: int = 40,
    beta1: float = 0.9,
    beta2: float = 0.95,
    epsilon: float = 1e-8
) -> Dict:
    """
    Create training configuration dictionary
    
    Returns:
        Config dictionary with all training hyperparameters
    """
    config = {
        'batch_size': batch_size,
        'context_length': context_length,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'warmup_steps': warmup_steps,
        'grad_clip': grad_clip,
        'log_interval': log_interval,
        'sample_interval': sample_interval,
        'checkpoint_interval': checkpoint_interval,
        'checkpoint_dir': checkpoint_dir,
        'temperature': temperature,
        'top_k': top_k,
        'beta1': beta1,
        'beta2': beta2,
        'epsilon': epsilon
    }
    
    return config


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """
    Complete training script
    Ties everything together: data, model, and training
    """
    
    print("=" * 70)
    print("CHARACTER-LEVEL LANGUAGE MODEL WITH MLA - TRAINING")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: PREPARE DATA
    # ========================================================================
    print("\nStep 1: Preparing data...")
    
    
    
    train_loader, val_loader, tokenizer, data_info = prepare_shakespeare_data(
        context_length=256,
        batch_size=32,
        num_workers=2
    )
    
    print(f"  ✓ Data prepared")
    print(f"    Vocabulary size: {data_info['vocab_size']}")
    
    # ========================================================================
    # STEP 2: CREATE MODEL
    # ========================================================================
    print("\nStep 2: Creating model...")
    
    
    
    model_config = {
        'vocab_size': data_info['vocab_size'],
        'd_model': 128,
        'num_layers': 4,
        'num_heads': 4,
        'd_latent': 16,
        'd_ff': 512,
        'max_seq_len': 256,
        'dropout': 0.1
    }
    
    model = CharacterLanguageModel(**model_config)
    
    print(f"  ✓ Model created")
    print(f"    Parameters: {model.get_num_params():,}")
    
    # ========================================================================
    # STEP 3: CREATE TRAINING CONFIG
    # ========================================================================
    print("\nStep 3: Setting up training...")
    
    training_config = create_training_config(
        batch_size=32,
        context_length=256,
        num_epochs=20,
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=100,
        grad_clip=1.0,
        log_interval=100,
        sample_interval=5,
        checkpoint_interval=5
    )
    
    print(f"  ✓ Training config created")
    print(f"    Epochs: {training_config['num_epochs']}")
    print(f"    Learning rate: {training_config['learning_rate']}")
    
    # ========================================================================
    # STEP 4: CREATE TRAINER AND TRAIN
    # ========================================================================
    print("\nStep 4: Initializing trainer...")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=training_config
    )
    
    # Start training
    trainer.train()
    
    # ========================================================================
    # STEP 5: GENERATE SAMPLES FROM TRAINED MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("GENERATING SAMPLES FROM TRAINED MODEL")
    print("=" * 70)
    
    # Load best model
    trainer.load_checkpoint('best_model.pt')
    
    # Generate multiple samples
    prompts = ["To be or not to be", "ROMEO:", "First Citizen:"]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        sample = trainer.generate_sample(prompt=prompt, max_tokens=200)
        print(f"Generated:\n{sample}\n")
        print("-" * 70)
    
    print("\n✅ Training complete! Model saved in checkpoints/")


if __name__ == "__main__":
    main()